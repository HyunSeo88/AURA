import argparse
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from aura_v1 import AuraTeacher
from aura_v2 import AuraV2
from dataloader import get_train_val_dataloaders

utils_logger = importlib.import_module("utils.logger")
utils_losses = importlib.import_module("utils.losses")
utils_metrics = importlib.import_module("utils.metrics")

append_epoch_metrics = utils_logger.append_epoch_metrics
create_experiment_dirs = utils_logger.create_experiment_dirs
save_experiment_config = utils_logger.save_experiment_config
setup_experiment_logger = utils_logger.setup_experiment_logger

aura_loss = utils_losses.aura_loss

AverageMeter = utils_metrics.AverageMeter
as_float_metrics = utils_metrics.as_float_metrics
compute_audio_metrics = utils_metrics.compute_audio_metrics


def _is_distributed_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    if _is_distributed_available_and_initialized():
        return dist.get_rank()
    return 0


def _is_main_process() -> bool:
    return _get_rank() == 0


def _init_distributed_if_needed(train_cfg: Dict):
    dist_cfg = train_cfg.get("distributed", {})
    enabled = bool(dist_cfg.get("enabled", False))
    if not enabled:
        return False, 0, 1

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = str(dist_cfg.get("backend", "nccl"))

    dist.init_process_group(backend=backend, init_method="env://")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size


def _cleanup_distributed():
    if _is_distributed_available_and_initialized():
        dist.destroy_process_group()


def _parse_args():
    parser = argparse.ArgumentParser(description="AURA training entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/project/configs/default_experiment.json",
        help="Path to experiment config JSON",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (optional). Overrides config training.resume_path.",
    )
    return parser.parse_args()


def _load_config(config_path: str):
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _apply_gpu_visibility(train_cfg: dict):
    gpu_ids = train_cfg.get("gpu_ids")
    if gpu_ids is None:
        return
    if isinstance(gpu_ids, list) and len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)


def _align_sequence_length(x, target_len):
    if x.size(-1) == target_len:
        return x
    x = x.unsqueeze(1)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.squeeze(1)


def _set_requires_grad(module: Optional[torch.nn.Module], requires_grad: bool):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def _apply_fine_tuning_policy(model: torch.nn.Module, epoch: int, policy: Dict):
    enabled = bool(policy.get("enabled", False))
    if not enabled:
        return "disabled"

    freeze_encoder_epochs = int(policy.get("freeze_encoder_epochs", 0))
    freeze_masknet_epochs = int(policy.get("freeze_masknet_epochs", 0))
    freeze_decoder_epochs = int(policy.get("freeze_decoder_epochs", 0))
    train_pitchnet = bool(policy.get("train_pitchnet", True))

    _set_requires_grad(model.encoder, epoch > freeze_encoder_epochs)
    _set_requires_grad(model.masknet, epoch > freeze_masknet_epochs)
    _set_requires_grad(model.decoder, epoch > freeze_decoder_epochs)
    _set_requires_grad(getattr(model, "pitch_net", None), train_pitchnet)

    return " ".join(
        [
            f"encoder={'train' if epoch > freeze_encoder_epochs else 'freeze'}",
            f"masknet={'train' if epoch > freeze_masknet_epochs else 'freeze'}",
            f"decoder={'train' if epoch > freeze_decoder_epochs else 'freeze'}",
            f"pitchnet={'train' if train_pitchnet else 'freeze'}",
        ]
    )


def _build_scheduler(optimizer: optim.Optimizer, scheduler_cfg: Dict, total_epochs: int):
    sched_type = str(scheduler_cfg.get("type", "none")).lower()
    if sched_type in {"none", "off", "disabled"}:
        return None, None

    if sched_type == "cosine":
        t_max = int(scheduler_cfg.get("t_max", total_epochs))
        eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max),
            eta_min=eta_min,
        )
        return scheduler, "epoch"

    if sched_type == "step":
        step_size = int(scheduler_cfg.get("step_size", 10))
        gamma = float(scheduler_cfg.get("gamma", 0.5))
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, step_size),
            gamma=gamma,
        )
        return scheduler, "epoch"

    if sched_type == "plateau":
        factor = float(scheduler_cfg.get("factor", 0.5))
        patience = int(scheduler_cfg.get("patience", 3))
        min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=max(1, patience),
            min_lr=min_lr,
        )
        return scheduler, "plateau"

    raise ValueError(f"Unsupported lr_scheduler.type: {sched_type}")


def _get_linear_warmup_cfg(scheduler_cfg: Dict):
    warmup_cfg = scheduler_cfg.get("warmup", {})
    enabled = bool(warmup_cfg.get("enabled", False))
    epochs = int(warmup_cfg.get("epochs", 0))
    start_factor = float(warmup_cfg.get("start_factor", 0.1))
    return enabled and epochs > 0, max(0, epochs), start_factor


def _apply_linear_warmup(
    optimizer: optim.Optimizer,
    base_lrs,
    epoch: int,
    warmup_epochs: int,
    start_factor: float,
):
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return False

    progress = epoch / warmup_epochs
    factor = start_factor + (1.0 - start_factor) * progress
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * float(factor)
    return True


def _strip_ddp_prefix(name: str) -> str:
    if name.startswith("module."):
        return name[len("module.") :]
    return name


def _build_optimizer_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    train_cfg: Dict,
):
    opt_cfg = train_cfg.get("optimizer", {})
    group_cfg = opt_cfg.get("param_groups", {})
    enabled = bool(group_cfg.get("enabled", False))
    if not enabled:
        return None, {
            "enabled": False,
            "group_names": [],
            "group_sizes": {},
            "group_lrs": {},
        }

    backbone_lr_scale = float(group_cfg.get("backbone_lr_scale", 0.1))
    body_lr_scale = float(group_cfg.get("body_lr_scale", 1.0))
    backbone_prefixes = tuple(
        str(x) for x in group_cfg.get("backbone_prefixes", ["encoder.", "decoder.", "masknet."])
    )
    body_name_keywords = tuple(str(x) for x in group_cfg.get("body_name_keywords", [".film."]))
    model_ref = model.module if isinstance(model, DDP) else model
    backbone_param_ids = set()
    for attr in ("encoder", "decoder", "masknet"):
        module = getattr(model_ref, attr, None)
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                backbone_param_ids.add(id(p))

    body_hint_param_ids = set()
    for module in model_ref.modules():
        for attr in ("film", "cross_attn", "enroll_proj", "gate_proj"):
            submodule = getattr(module, attr, None)
            if isinstance(submodule, torch.nn.Module):
                for p in submodule.parameters():
                    body_hint_param_ids.add(id(p))

    backbone_params: List[torch.nn.Parameter] = []
    body_params: List[torch.nn.Parameter] = []

    for raw_name, param in model.named_parameters():
        name = _strip_ddp_prefix(raw_name)
        is_backbone_by_ref = id(param) in backbone_param_ids
        is_backbone_by_name = len(backbone_prefixes) > 0 and name.startswith(backbone_prefixes)
        is_backbone = is_backbone_by_ref or is_backbone_by_name

        is_body_hint_by_ref = id(param) in body_hint_param_ids
        is_body_hint_by_name = any(kw and (kw in name) for kw in body_name_keywords)
        force_body = is_body_hint_by_ref or is_body_hint_by_name

        if is_backbone and not force_body:
            backbone_params.append(param)
        else:
            body_params.append(param)

    if len(backbone_params) == 0:
        raise ValueError(
            "No parameters were assigned to backbone group. "
            "Check model encoder/decoder/masknet registration or "
            "training.optimizer.param_groups.backbone_prefixes."
        )
    if len(body_params) == 0:
        raise ValueError(
            "No parameters were assigned to body group. "
            "Check training.optimizer.param_groups.body_name_keywords."
        )

    param_groups = [
        {
            "name": "body",
            "params": body_params,
            "lr": float(base_lr) * body_lr_scale,
        },
        {
            "name": "backbone",
            "params": backbone_params,
            "lr": float(base_lr) * backbone_lr_scale,
        },
    ]
    return param_groups, {
        "enabled": True,
        "group_names": ["body", "backbone"],
        "group_sizes": {
            "body": len(body_params),
            "backbone": len(backbone_params),
        },
        "group_lrs": {
            "body": float(base_lr) * body_lr_scale,
            "backbone": float(base_lr) * backbone_lr_scale,
        },
    }


def _build_optimizer(
    model: torch.nn.Module,
    train_cfg: Dict,
    lr: float,
) -> Tuple[optim.Optimizer, Dict]:
    opt_cfg = train_cfg.get("optimizer", {})
    opt_type = str(opt_cfg.get("type", "adam")).lower()
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    eps = float(opt_cfg.get("eps", 1e-8))

    raw_betas = opt_cfg.get("betas", [0.9, 0.999])
    if not isinstance(raw_betas, (list, tuple)) or len(raw_betas) != 2:
        raise ValueError("training.optimizer.betas must be a list/tuple of length 2")
    betas = (float(raw_betas[0]), float(raw_betas[1]))

    param_groups, group_meta = _build_optimizer_param_groups(
        model=model,
        base_lr=lr,
        train_cfg=train_cfg,
    )
    optimizer_params = param_groups if param_groups is not None else model.parameters()

    if opt_type == "adam":
        optimizer = optim.Adam(
            optimizer_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif opt_type == "adamw":
        optimizer = optim.AdamW(
            optimizer_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer.type: {opt_type}. Use 'adam' or 'adamw'.")

    group_meta["type"] = opt_type
    group_meta["weight_decay"] = weight_decay
    group_meta["eps"] = eps
    group_meta["betas"] = [betas[0], betas[1]]
    return optimizer, group_meta


def _collect_lr_metrics(optimizer: optim.Optimizer) -> Dict[str, float]:
    lr_row = {"lr": float(optimizer.param_groups[0]["lr"])}
    for idx, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", f"group{idx}")).strip().lower()
        key = f"lr_{group_name}"
        lr_row[key] = float(group["lr"])
    return lr_row


def _resolve_backbone_lr_warmup_cfg(train_cfg: Dict) -> Dict:
    opt_cfg = train_cfg.get("optimizer", {})
    group_cfg = opt_cfg.get("param_groups", {})
    warm_cfg = group_cfg.get("backbone_lr_warmup", {})

    end_scale = float(group_cfg.get("backbone_lr_scale", 1.0))
    start_scale = float(warm_cfg.get("start_scale", end_scale))
    epochs = max(1, int(warm_cfg.get("epochs", 1)))
    start_epoch = max(1, int(warm_cfg.get("start_epoch", 1)))
    enabled = bool(group_cfg.get("enabled", False)) and bool(warm_cfg.get("enabled", False))

    return {
        "enabled": enabled,
        "start_scale": start_scale,
        "end_scale": end_scale,
        "epochs": epochs,
        "start_epoch": start_epoch,
    }


def _current_backbone_lr_scale(epoch: int, cfg: Dict) -> float:
    start_scale = float(cfg["start_scale"])
    end_scale = float(cfg["end_scale"])
    if not bool(cfg["enabled"]):
        return end_scale

    start_epoch = int(cfg["start_epoch"])
    ramp_epochs = int(cfg["epochs"])
    if epoch <= start_epoch:
        return start_scale

    denom = max(1, ramp_epochs - 1)
    progress = min(1.0, max(0.0, float(epoch - start_epoch) / float(denom)))
    return start_scale + (end_scale - start_scale) * progress


def _apply_backbone_lr_scale_from_body(optimizer: optim.Optimizer, backbone_scale: float) -> bool:
    body_lr = None
    for group in optimizer.param_groups:
        if str(group.get("name", "")).strip().lower() == "body":
            body_lr = float(group["lr"])
            break

    if body_lr is None:
        return False

    updated = False
    for group in optimizer.param_groups:
        if str(group.get("name", "")).strip().lower() == "backbone":
            group["lr"] = body_lr * float(backbone_scale)
            updated = True
    return updated


def _all_reduce_mean_dict(local_avg: Dict[str, float], meter_state: Dict[str, Dict[str, float]], device):
    if not _is_distributed_available_and_initialized():
        return local_avg

    result = {}
    keys = sorted(set(meter_state["sum"].keys()) | set(local_avg.keys()))
    for key in keys:
        local_sum = float(meter_state["sum"].get(key, 0.0))
        local_count = float(meter_state["count"].get(key, 0.0))
        vec = torch.tensor([local_sum, local_count], dtype=torch.float64, device=device)
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        global_sum = float(vec[0].item())
        global_count = float(vec[1].item())
        result[key] = (global_sum / global_count) if global_count > 0 else float("nan")
    return result


def _resolve_precision_cfg(train_cfg: Dict, device: torch.device):
    prec_cfg = train_cfg.get("precision", {})
    amp_enabled = bool(prec_cfg.get("amp_enabled", True)) and device.type == "cuda"
    amp_dtype_cfg = str(prec_cfg.get("amp_dtype", "auto")).lower()
    grad_accum_steps = max(1, int(prec_cfg.get("gradient_accumulation_steps", 1)))
    allow_tf32 = bool(prec_cfg.get("allow_tf32", True))

    if allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not amp_enabled:
        return {
            "amp_enabled": False,
            "amp_dtype": None,
            "use_grad_scaler": False,
            "grad_accum_steps": grad_accum_steps,
            "allow_tf32": allow_tf32,
        }

    if amp_dtype_cfg == "bf16":
        amp_dtype = torch.bfloat16
        use_grad_scaler = False
    elif amp_dtype_cfg == "fp16":
        amp_dtype = torch.float16
        use_grad_scaler = True
    else:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            use_grad_scaler = False
        else:
            amp_dtype = torch.float16
            use_grad_scaler = True

    return {
        "amp_enabled": True,
        "amp_dtype": amp_dtype,
        "use_grad_scaler": use_grad_scaler,
        "grad_accum_steps": grad_accum_steps,
        "allow_tf32": allow_tf32,
    }


def _resolve_model_state_dict(ckpt_obj: Dict):
    state_dict = ckpt_obj.get("model_state_dict", ckpt_obj)
    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint format: model_state_dict is not a dict")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _optimizer_to_device(optimizer: optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=True)


def _resolve_resume_path(
    resume_arg: Optional[str],
    train_cfg: Dict,
    project_root: str,
) -> Optional[Path]:
    raw = resume_arg if resume_arg else train_cfg.get("resume_path")
    if not raw:
        return None

    p = Path(str(raw))
    if p.is_absolute():
        return p

    if p.exists():
        return p.resolve()

    return (Path(project_root) / p).resolve()


def train(config_path: str, resume_path: Optional[str] = None):
    cfg = _load_config(config_path)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    val_cfg = cfg["validation"]
    ckpt_cfg = cfg["checkpoint"]
    log_cfg = cfg["logging"]
    fine_tune_cfg = train_cfg.get("fine_tuning", {"enabled": False})
    scheduler_cfg = train_cfg.get("lr_scheduler", {"type": "none"})

    _apply_gpu_visibility(train_cfg)
    distributed_enabled, rank, world_size = _init_distributed_if_needed(train_cfg)

    epochs = int(train_cfg["epochs"])
    batch_size = int(data_cfg["batch_size"])
    lr = float(train_cfg["learning_rate"])
    grad_clip_norm = float(train_cfg["gradient_clip_norm"])
    if distributed_enabled:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = _resolve_device(train_cfg.get("device", "auto"))

    exp_name = str(exp_cfg["name"])
    project_root = str(exp_cfg.get("project_root", "/workspace/project"))
    resume_ckpt_path = _resolve_resume_path(resume_path, train_cfg, project_root)
    exp_dirs = create_experiment_dirs(project_root=project_root, exp_name=exp_name)
    logger = setup_experiment_logger(exp_dirs["logs_dir"] / "train.log")
    save_dir = str(exp_dirs["checkpoints_dir"])

    use_pitchnet = bool(model_cfg["use_pitchnet"])
    precision_cfg = _resolve_precision_cfg(train_cfg, device)
    val_ratio = float(data_cfg["val_ratio"])
    best_metric_name = str(val_cfg.get("best_metric_name", "val_total_loss"))
    best_mode = str(val_cfg.get("best_mode", "min"))
    save_best = bool(ckpt_cfg.get("save_best", True))
    save_every_n_epochs = int(ckpt_cfg.get("save_every_n_epochs", 10))
    log_interval = max(1, int(log_cfg.get("log_interval", 10)))

    eval_metrics_cfg = val_cfg.get("eval_metrics", {})
    use_pesq = bool(eval_metrics_cfg.get("pesq", True))
    use_stoi = bool(eval_metrics_cfg.get("stoi", True))
    use_sdr = bool(eval_metrics_cfg.get("sdr", True))
    audio_metrics_enabled = bool(eval_metrics_cfg.get("enabled", True))
    audio_metrics_every_n_epochs = max(1, int(eval_metrics_cfg.get("every_n_epochs", 1)))
    train_audio_metrics_enabled = bool(eval_metrics_cfg.get("include_train", False))

    w_target = float(loss_cfg["w_target"])
    w_consistency = float(loss_cfg["w_consistency"])
    w_pitch = float(loss_cfg["w_pitch"]) if use_pitchnet else 0.0
    w_residual = float(loss_cfg["w_residual"])
    w_absent_silence = float(loss_cfg.get("w_absent_silence", 0.5))
    w_absent_residual = float(loss_cfg.get("w_absent_residual", 0.5))

    os.makedirs(save_dir, exist_ok=True)
    if _is_main_process():
        logger.info(f"[Train] Device: {device}")
        logger.info(
            f"[Train] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}"
        )
        if distributed_enabled:
            logger.info(f"[DDP] enabled world_size={world_size} rank={rank}")

    model_arch = str(
        model_cfg.get(
            "arch",
            model_cfg.get("type", model_cfg.get("name", "aura_v1")),
        )
    ).lower()

    model_kwargs = {
        "enroll_dim": int(model_cfg["enroll_dim"]),
        "device": device,
        "use_pitchnet": use_pitchnet,
        "sepformer_source": str(model_cfg.get("sepformer_source", "speechbrain/sepformer-wsj02mix")),
        "sepformer_savedir": str(
            model_cfg.get("sepformer_savedir", "pretrained_models/sepformer-wsj02mix")
        ),
    }

    if model_arch in {"aura_v2", "v2"}:
        model = AuraV2(**model_kwargs)
    else:
        model = AuraTeacher(**model_kwargs)
    use_enroll_mask = model_arch in {"aura_v2", "v2"}

    if _is_main_process():
        logger.info(f"[Model] arch={model_arch}")

    resume_ckpt = None
    start_epoch = 0
    resumed_metric_name = None
    resumed_metric_value = None
    if resume_ckpt_path is not None:
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt_path}")
        resume_ckpt = torch.load(str(resume_ckpt_path), map_location="cpu")
        model_state = _resolve_model_state_dict(resume_ckpt)
        incompatible = model.load_state_dict(model_state, strict=False)
        start_epoch = max(0, int(resume_ckpt.get("epoch", 0)))
        resumed_metric_name = resume_ckpt.get("metric_name")
        resumed_metric_value = resume_ckpt.get("metric_value")

        if _is_main_process():
            logger.info(
                f"[Resume] Loaded checkpoint: {resume_ckpt_path} "
                f"(start_epoch={start_epoch + 1})"
            )
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            if missing:
                logger.warning(f"[Resume] missing_keys={len(missing)}")
            if unexpected:
                logger.warning(f"[Resume] unexpected_keys={len(unexpected)}")

    if distributed_enabled:
        find_unused = bool(train_cfg.get("distributed", {}).get("find_unused_parameters", False))
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=find_unused,
        )

    train_loader, val_loader = get_train_val_dataloaders(
        batch_size=batch_size,
        num_workers=int(data_cfg.get("num_workers", 0)),
        manifest_path=str(data_cfg["manifest_path"]),
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        duration_sec=float(data_cfg.get("duration_sec", 6.0)),
        enroll_dim=int(data_cfg.get("enroll_dim", 128)),
        audio_base_dir=str(data_cfg["audio_base_dir"]),
        pitch_base_dir=str(data_cfg.get("pitch_base_dir", data_cfg["audio_base_dir"])),
        strict_audio_length=bool(data_cfg.get("strict_audio_length", False)),
        use_zero_enroll_fallback=bool(data_cfg.get("use_zero_enroll_fallback", True)),
        random_crop_train=bool(data_cfg.get("random_crop_train", data_cfg.get("random_crop", True))),
        random_crop_val=bool(data_cfg.get("random_crop_val", False)),
        partial_audio_load=bool(data_cfg.get("partial_audio_load", True)),
        normalize_enroll=bool(data_cfg.get("normalize_enroll", False)),
        enroll_cache_size=int(data_cfg.get("enroll_cache_size", 4096)),
        audio_info_cache_size=int(data_cfg.get("audio_info_cache_size", 16384)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        persistent_workers=(
            None
            if data_cfg.get("persistent_workers", None) is None
            else bool(data_cfg.get("persistent_workers"))
        ),
        pin_memory=(
            None
            if data_cfg.get("pin_memory", None) is None
            else bool(data_cfg.get("pin_memory"))
        ),
        val_ratio=val_ratio,
        split_seed=int(data_cfg.get("split_seed", 42)),
        split_mode=str(data_cfg.get("split_mode", "random")),
        distributed_train=distributed_enabled,
        distributed_val=distributed_enabled,
        world_size=world_size,
        rank=rank,
    )

    optimizer, optimizer_meta = _build_optimizer(model=model, train_cfg=train_cfg, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=precision_cfg["use_grad_scaler"])
    scheduler, scheduler_mode = _build_scheduler(optimizer, scheduler_cfg, epochs)

    if resume_ckpt is not None:
        opt_state = resume_ckpt.get("optimizer_state_dict")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
                _optimizer_to_device(optimizer, device)
                if _is_main_process():
                    logger.info("[Resume] Restored optimizer state.")
            except ValueError as exc:
                if _is_main_process():
                    logger.warning(
                        "[Resume] optimizer_state_dict mismatch. "
                        f"Using freshly initialized optimizer. detail={exc}"
                    )

        scaler_state = resume_ckpt.get("scaler_state_dict")
        if scaler_state is not None and precision_cfg["use_grad_scaler"]:
            scaler.load_state_dict(scaler_state)
            if _is_main_process():
                logger.info("[Resume] Restored GradScaler state.")

        sched_state = resume_ckpt.get("scheduler_state_dict")
        if sched_state is not None and scheduler is not None:
            try:
                scheduler.load_state_dict(sched_state)
                if _is_main_process():
                    logger.info("[Resume] Restored scheduler state.")
            except ValueError as exc:
                if _is_main_process():
                    logger.warning(
                        "[Resume] scheduler_state_dict mismatch. "
                        f"Using freshly initialized scheduler. detail={exc}"
                    )

    warmup_enabled, warmup_epochs, warmup_start_factor = _get_linear_warmup_cfg(scheduler_cfg)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    backbone_lr_warmup_cfg = _resolve_backbone_lr_warmup_cfg(train_cfg)

    if _is_main_process() and warmup_enabled:
        logger.info(
            f"[LR] Linear warmup enabled: epochs={warmup_epochs}, start_factor={warmup_start_factor}"
        )
    if _is_main_process():
        amp_dtype_name = (
            str(precision_cfg["amp_dtype"]).replace("torch.", "")
            if precision_cfg["amp_dtype"] is not None
            else "off"
        )
        logger.info(
            "[Precision] "
            f"amp_enabled={precision_cfg['amp_enabled']} "
            f"amp_dtype={amp_dtype_name} "
            f"grad_accum_steps={precision_cfg['grad_accum_steps']} "
            f"allow_tf32={precision_cfg['allow_tf32']}"
        )
        if optimizer_meta.get("enabled", False):
            logger.info(
                "[Optimizer] "
                f"type={optimizer_meta['type']} "
                f"body_lr={optimizer_meta['group_lrs']['body']:.6g} "
                f"backbone_lr={optimizer_meta['group_lrs']['backbone']:.6g} "
                f"body_params={optimizer_meta['group_sizes']['body']} "
                f"backbone_params={optimizer_meta['group_sizes']['backbone']}"
            )
        else:
            logger.info(
                "[Optimizer] "
                f"type={optimizer_meta['type']} "
                f"lr={float(optimizer.param_groups[0]['lr']):.6g}"
            )
        if backbone_lr_warmup_cfg["enabled"]:
            logger.info(
                "[Optimizer] backbone_lr_warmup "
                f"start_scale={backbone_lr_warmup_cfg['start_scale']:.6g} "
                f"end_scale={backbone_lr_warmup_cfg['end_scale']:.6g} "
                f"start_epoch={backbone_lr_warmup_cfg['start_epoch']} "
                f"epochs={backbone_lr_warmup_cfg['epochs']}"
            )

    resolved_config = {
        **cfg,
        "training": {**train_cfg, "device": str(device)},
        "loss": {
            **loss_cfg,
            "w_pitch": w_pitch,
            "w_absent_silence": w_absent_silence,
            "w_absent_residual": w_absent_residual,
        },
        "artifact_paths": {
            "run_dir": str(exp_dirs["run_dir"]),
            "checkpoints_dir": str(exp_dirs["checkpoints_dir"]),
            "metrics_dir": str(exp_dirs["metrics_dir"]),
            "logs_dir": str(exp_dirs["logs_dir"]),
            "config_dir": str(exp_dirs["config_dir"]),
        },
    }
    if _is_main_process():
        save_experiment_config(exp_dirs["config_dir"] / "resolved_config.json", resolved_config)
        save_experiment_config(exp_dirs["config_dir"] / "input_config_snapshot.json", cfg)

    metrics_jsonl = exp_dirs["metrics_dir"] / "epoch_metrics.jsonl"
    metrics_csv = exp_dirs["metrics_dir"] / "epoch_metrics.csv"
    best_value = float("inf") if best_mode == "min" else float("-inf")
    if resumed_metric_name == best_metric_name and resumed_metric_value is not None:
        best_value = float(resumed_metric_value)
    prev_ft_state = None

    if start_epoch >= epochs:
        if _is_main_process():
            logger.info(
                f"[Resume] start_epoch={start_epoch + 1} >= configured epochs={epochs}. "
                "Nothing to train."
            )
        _cleanup_distributed()
        return

    for epoch in range(start_epoch, epochs):
        if distributed_enabled and isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if distributed_enabled and isinstance(getattr(val_loader, "sampler", None), DistributedSampler):
            val_loader.sampler.set_epoch(epoch)

        model_for_policy = model.module if isinstance(model, DDP) else model
        in_warmup = _apply_linear_warmup(
            optimizer=optimizer,
            base_lrs=base_lrs,
            epoch=epoch + 1,
            warmup_epochs=warmup_epochs if warmup_enabled else 0,
            start_factor=warmup_start_factor,
        )
        backbone_scale = _current_backbone_lr_scale(epoch=epoch + 1, cfg=backbone_lr_warmup_cfg)
        if backbone_lr_warmup_cfg["enabled"]:
            _apply_backbone_lr_scale_from_body(optimizer=optimizer, backbone_scale=backbone_scale)

        phase = _apply_fine_tuning_policy(model_for_policy, epoch + 1, fine_tune_cfg)
        if phase != prev_ft_state and _is_main_process():
            logger.info(f"[FineTune] Epoch {epoch + 1}: {phase}")
        prev_ft_state = phase

        model.train()
        train_meter = AverageMeter()
        train_audio_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            should_step = (
                ((batch_idx + 1) % precision_cfg["grad_accum_steps"] == 0)
                or ((batch_idx + 1) == len(train_loader))
            )
            ddp_no_sync = (
                distributed_enabled
                and isinstance(model, DDP)
                and (not should_step)
            )
            sync_context = model.no_sync() if ddp_no_sync else nullcontext()

            mix = batch["mix"].to(device)
            enroll = batch["enroll"].to(device)
            enroll_mask = batch.get("enroll_mask")
            if enroll_mask is not None:
                enroll_mask = enroll_mask.to(device)
            gt_target = batch["gt_target"].to(device)
            target_present = batch["is_target_present"].to(device)

            gt_res = batch.get("gt_residual")
            if gt_res is not None:
                gt_res = gt_res.to(device)

            with sync_context:
                with torch.cuda.amp.autocast(
                    enabled=precision_cfg["amp_enabled"],
                    dtype=precision_cfg["amp_dtype"],
                ):
                    if use_enroll_mask:
                        est_target, est_res, aux = model(
                            mix,
                            enroll,
                            return_aux=True,
                            enroll_mask=enroll_mask,
                        )
                    else:
                        est_target, est_res, aux = model(mix, enroll, return_aux=True)

                    min_len = min(mix.shape[-1], est_target.shape[-1], gt_target.shape[-1])
                    est_target = est_target[:, :min_len]
                    est_res = est_res[:, :min_len]
                    gt_target = gt_target[:, :min_len]
                    mix_wave = mix[:, :min_len]
                    if gt_res is not None:
                        gt_res = gt_res[:, :min_len]

                    pred_pitch = aux.get("pitch_pred")
                    gt_pitch = batch.get("pitch_gt")
                    pitch_mask = batch.get("pitch_voiced_mask")
                    if pred_pitch is not None and gt_pitch is not None:
                        gt_pitch = _align_sequence_length(gt_pitch.to(device), pred_pitch.shape[-1])
                    else:
                        gt_pitch = None

                    if pred_pitch is not None and pitch_mask is not None:
                        pitch_mask = _align_sequence_length(
                            pitch_mask.float().to(device), pred_pitch.shape[-1]
                        )
                    else:
                        pitch_mask = None

                    loss, loss_dict = aura_loss(
                        est_target=est_target,
                        est_residual=est_res,
                        gt_target=gt_target,
                        mix_input=mix_wave,
                        target_present_mask=target_present,
                        pred_pitch=pred_pitch,
                        gt_pitch=gt_pitch,
                        pitch_voiced_mask=pitch_mask,
                        gt_residual=gt_res,
                        w_target=w_target,
                        w_consistency=w_consistency,
                        w_pitch=w_pitch,
                        w_residual=w_residual,
                        w_absent_silence=w_absent_silence,
                        w_absent_residual=w_absent_residual,
                    )

                    loss_to_backward = loss / float(precision_cfg["grad_accum_steps"])

                if precision_cfg["use_grad_scaler"]:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

            if should_step:
                if precision_cfg["use_grad_scaler"]:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if precision_cfg["use_grad_scaler"]:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_meter.update(as_float_metrics(loss, loss_dict))

            do_audio_metrics = (
                audio_metrics_enabled
                and train_audio_metrics_enabled
                and ((epoch + 1) % audio_metrics_every_n_epochs == 0)
            )
            if do_audio_metrics:
                train_audio_meter.update(
                    compute_audio_metrics(
                        est_target=est_target,
                        gt_target=gt_target,
                        sample_rate=int(data_cfg.get("sample_rate", 16000)),
                        use_pesq=use_pesq,
                        use_stoi=use_stoi,
                        use_sdr=use_sdr,
                    )
                )

            if batch_idx % log_interval == 0 and _is_main_process():
                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}] Step [{batch_idx}] "
                    f"Loss: {loss.item():.4f} "
                    f"(T: {loss_dict['target'].item():.2f}, "
                    f"C: {loss_dict['consist'].item():.4f}, "
                    f"P: {loss_dict['pitch'].item():.4f}, "
                    f"R: {loss_dict['res'].item():.2f}, "
                    f"AS: {loss_dict['abs_sil'].item():.4f}, "
                    f"AR: {loss_dict['abs_res'].item():.4f})"
                )

        model.eval()
        val_meter = AverageMeter()
        val_audio_meter = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                mix = batch["mix"].to(device)
                enroll = batch["enroll"].to(device)
                enroll_mask = batch.get("enroll_mask")
                if enroll_mask is not None:
                    enroll_mask = enroll_mask.to(device)
                gt_target = batch["gt_target"].to(device)
                target_present = batch["is_target_present"].to(device)

                gt_res = batch.get("gt_residual")
                if gt_res is not None:
                    gt_res = gt_res.to(device)

                with torch.cuda.amp.autocast(
                    enabled=precision_cfg["amp_enabled"],
                    dtype=precision_cfg["amp_dtype"],
                ):
                    if use_enroll_mask:
                        est_target, est_res, aux = model(
                            mix,
                            enroll,
                            return_aux=True,
                            enroll_mask=enroll_mask,
                        )
                    else:
                        est_target, est_res, aux = model(mix, enroll, return_aux=True)

                    min_len = min(mix.shape[-1], est_target.shape[-1], gt_target.shape[-1])
                    est_target = est_target[:, :min_len]
                    est_res = est_res[:, :min_len]
                    gt_target = gt_target[:, :min_len]
                    mix_wave = mix[:, :min_len]
                    if gt_res is not None:
                        gt_res = gt_res[:, :min_len]

                    pred_pitch = aux.get("pitch_pred")
                    gt_pitch = batch.get("pitch_gt")
                    pitch_mask = batch.get("pitch_voiced_mask")
                    if pred_pitch is not None and gt_pitch is not None:
                        gt_pitch = _align_sequence_length(gt_pitch.to(device), pred_pitch.shape[-1])
                    else:
                        gt_pitch = None
                    if pred_pitch is not None and pitch_mask is not None:
                        pitch_mask = _align_sequence_length(
                            pitch_mask.float().to(device), pred_pitch.shape[-1]
                        )
                    else:
                        pitch_mask = None

                    val_loss, val_loss_dict = aura_loss(
                        est_target=est_target,
                        est_residual=est_res,
                        gt_target=gt_target,
                        mix_input=mix_wave,
                        target_present_mask=target_present,
                        pred_pitch=pred_pitch,
                        gt_pitch=gt_pitch,
                        pitch_voiced_mask=pitch_mask,
                        gt_residual=gt_res,
                        w_target=w_target,
                        w_consistency=w_consistency,
                        w_pitch=w_pitch,
                        w_residual=w_residual,
                        w_absent_silence=w_absent_silence,
                        w_absent_residual=w_absent_residual,
                    )
                val_meter.update(as_float_metrics(val_loss, val_loss_dict))

                do_audio_metrics = audio_metrics_enabled and ((epoch + 1) % audio_metrics_every_n_epochs == 0)
                if do_audio_metrics:
                    val_audio_meter.update(
                        compute_audio_metrics(
                            est_target=est_target,
                            gt_target=gt_target,
                            sample_rate=int(data_cfg.get("sample_rate", 16000)),
                            use_pesq=use_pesq,
                            use_stoi=use_stoi,
                            use_sdr=use_sdr,
                        )
                    )

        train_avg = _all_reduce_mean_dict(train_meter.average(), train_meter.state(), device)
        train_audio_avg = _all_reduce_mean_dict(
            train_audio_meter.average(), train_audio_meter.state(), device
        )
        val_avg = _all_reduce_mean_dict(val_meter.average(), val_meter.state(), device)
        val_audio_avg = _all_reduce_mean_dict(val_audio_meter.average(), val_audio_meter.state(), device)

        lr_row = _collect_lr_metrics(optimizer)
        epoch_row = {
            "epoch": epoch + 1,
            **lr_row,
            "backbone_lr_scale_applied": float(backbone_scale),
            "train_total_loss": train_avg.get("total_loss", 0.0),
            "train_target_loss": train_avg.get("target_loss", 0.0),
            "train_consist_loss": train_avg.get("consist_loss", 0.0),
            "train_pitch_loss": train_avg.get("pitch_loss", 0.0),
            "train_residual_loss": train_avg.get("residual_loss", 0.0),
            "train_abs_silence_loss": train_avg.get("abs_silence_loss", 0.0),
            "train_abs_residual_loss": train_avg.get("abs_residual_loss", 0.0),
            "train_pesq": train_audio_avg.get("pesq", float("nan")),
            "train_stoi": train_audio_avg.get("stoi", float("nan")),
            "train_sdr": train_audio_avg.get("sdr", float("nan")),
            "val_total_loss": val_avg.get("total_loss", 0.0),
            "val_target_loss": val_avg.get("target_loss", 0.0),
            "val_consist_loss": val_avg.get("consist_loss", 0.0),
            "val_pitch_loss": val_avg.get("pitch_loss", 0.0),
            "val_residual_loss": val_avg.get("residual_loss", 0.0),
            "val_abs_silence_loss": val_avg.get("abs_silence_loss", 0.0),
            "val_abs_residual_loss": val_avg.get("abs_residual_loss", 0.0),
            "val_pesq": val_audio_avg.get("pesq", float("nan")),
            "val_stoi": val_audio_avg.get("stoi", float("nan")),
            "val_sdr": val_audio_avg.get("sdr", float("nan")),
        }

        if _is_main_process():
            append_epoch_metrics(metrics_jsonl, metrics_csv, epoch_row)

        if best_metric_name not in epoch_row:
            if _is_main_process():
                logger.warning(
                    f"best_metric_name='{best_metric_name}' not found in epoch metrics. "
                    "Fallback to 'val_total_loss'."
                )
            best_metric_name = "val_total_loss"

        if _is_main_process():
            logger.info(
                f"==> Epoch {epoch + 1} Finished | "
                f"train_total={epoch_row['train_total_loss']:.4f} "
                f"val_total={epoch_row['val_total_loss']:.4f}"
            )

        current_metric = epoch_row[best_metric_name]
        is_better = current_metric < best_value if best_mode == "min" else current_metric > best_value
        if save_best and is_better and _is_main_process():
            best_value = current_metric
            best_path = Path(save_dir) / "best_model.pth"
            model_to_save = model.module if isinstance(model, DDP) else model
            torch.save(
                {
                    "epoch": epoch + 1,
                    "metric_name": best_metric_name,
                    "metric_value": current_metric,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "scaler_state_dict": scaler.state_dict() if precision_cfg["use_grad_scaler"] else None,
                },
                str(best_path),
            )
            logger.info(
                f"[Checkpoint] Updated best model at epoch {epoch + 1} "
                f"({best_metric_name}={current_metric:.6f})"
            )

        if save_every_n_epochs > 0 and (epoch + 1) % save_every_n_epochs == 0 and _is_main_process():
            ckpt_path = os.path.join(save_dir, f"aura_teacher_ep{epoch + 1}.pth")
            model_to_save = model.module if isinstance(model, DDP) else model
            torch.save(
                {
                    "epoch": epoch + 1,
                    "metric_name": best_metric_name,
                    "metric_value": current_metric,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "scaler_state_dict": scaler.state_dict() if precision_cfg["use_grad_scaler"] else None,
                    "metrics": epoch_row,
                },
                ckpt_path,
            )
            logger.info(f"[Checkpoint] Saved periodic checkpoint: {ckpt_path}")

        if scheduler is not None and not in_warmup:
            if scheduler_mode == "plateau":
                scheduler.step(epoch_row["val_total_loss"])
            else:
                scheduler.step()

    if _is_main_process():
        logger.info(f"Training done. Artifacts saved in: {exp_dirs['run_dir']}")

    _cleanup_distributed()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config, args.resume)
