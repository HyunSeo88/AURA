from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbone_load import sepformer_load
from src.utils.aura_pa_temporal import build_chunk_grid, build_latent_chunk_ranges, chunk_mean_pool_sequence


def _default_sepformer_savedir() -> str:
    project_root = Path(__file__).resolve().parents[1]
    return str((project_root / "pretrained_models" / "sepformer-wsj02mix").resolve())


def _normalize_enroll_mask(
    enroll_mask: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if enroll_mask is None:
        return None

    if enroll_mask.dim() == 1:
        if batch_size != 1 or enroll_mask.numel() != seq_len:
            raise ValueError(
                f"1D enroll_mask requires batch_size=1 and seq_len={seq_len}, "
                f"got batch_size={batch_size}, shape={tuple(enroll_mask.shape)}"
            )
        enroll_mask = enroll_mask.unsqueeze(0)
    elif enroll_mask.dim() == 3 and enroll_mask.size(-1) == 1:
        enroll_mask = enroll_mask.squeeze(-1)

    if enroll_mask.dim() != 2:
        raise ValueError(f"enroll_mask must have shape [B, K], got {tuple(enroll_mask.shape)}")
    if enroll_mask.size(0) != batch_size or enroll_mask.size(1) != seq_len:
        raise ValueError(
            f"enroll_mask shape mismatch: expected ({batch_size}, {seq_len}), got {tuple(enroll_mask.shape)}"
        )

    return enroll_mask.to(device=device, dtype=torch.bool)


def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return sequence.mean(dim=1)
    valid = mask.to(dtype=sequence.dtype).unsqueeze(-1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (sequence * valid).sum(dim=1) / denom


class EnrollmentProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim is None or int(hidden_dim) <= 0:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnrollmentConditionedDualPathBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        *,
        enroll_dim: int,
        feature_dim: int,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        attn_scale_init: float = 0.1,
        projector_hidden_dim: int | None = None,
        projector_dropout: float = 0.0,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.feature_dim = int(feature_dim)
        self.use_gate = bool(use_gate)

        self.enroll_projector = EnrollmentProjector(
            input_dim=int(enroll_dim),
            output_dim=self.feature_dim,
            hidden_dim=projector_hidden_dim,
            dropout=projector_dropout,
        )
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.kv_norm = nn.LayerNorm(self.feature_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=int(num_heads),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.attn_out_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Dropout(float(proj_dropout)),
        )
        if self.use_gate:
            self.gate_query = nn.Linear(self.feature_dim, self.feature_dim)
            self.gate_attn = nn.Linear(self.feature_dim, self.feature_dim)
            nn.init.constant_(self.gate_query.bias, 0.0)
            nn.init.constant_(self.gate_attn.bias, -1.0)

        self.attn_scale = nn.Parameter(torch.tensor(float(attn_scale_init), dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        enrollment: torch.Tensor,
        enroll_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.original_block(x)
        bsz, channels, chunk_len, num_chunks = base.shape

        if enrollment.dim() != 3:
            raise ValueError(f"enrollment must have shape [B, K, D], got {tuple(enrollment.shape)}")
        if enrollment.size(0) != bsz:
            raise ValueError(
                f"batch mismatch between features and enrollment: {bsz} vs {enrollment.size(0)}"
            )

        key_padding_mask = _normalize_enroll_mask(
            enroll_mask,
            batch_size=bsz,
            seq_len=enrollment.size(1),
            device=enrollment.device,
        )
        if key_padding_mask is not None:
            key_padding_mask = ~key_padding_mask

        kv = self.enroll_projector(enrollment)
        kv = self.kv_norm(kv)

        query = base.permute(0, 2, 3, 1).contiguous().view(bsz, chunk_len * num_chunks, channels)
        query_norm = self.query_norm(query)

        attn_out, _ = self.cross_attn(
            query=query_norm,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = self.attn_out_proj(attn_out)

        if self.use_gate:
            gate = torch.sigmoid(self.gate_query(query_norm) + self.gate_attn(attn_out))
            attn_out = gate * attn_out

        attn_out = attn_out.view(bsz, chunk_len, num_chunks, channels).permute(0, 3, 1, 2).contiguous()
        return base + self.attn_scale * attn_out


class TemporalPresenceHead(nn.Module):
    def __init__(
        self,
        *,
        encoder_dim: int,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_similarity: bool = True,
        use_mask_feature: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.feature_dim = int(feature_dim)
        self.use_similarity = bool(use_similarity)
        self.use_mask_feature = bool(use_mask_feature)

        self.target_projector = nn.Linear(self.encoder_dim, self.feature_dim)
        input_dim = (2 * self.feature_dim)
        if self.use_mask_feature:
            input_dim += 1
        if self.use_similarity:
            input_dim += 1

        self.pre_norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        *,
        target_chunk_latent: torch.Tensor,
        target_chunk_mask: torch.Tensor,
        enroll_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_chunk_latent.dim() != 3:
            raise ValueError(
                f"target_chunk_latent must have shape [B, S, C], got {tuple(target_chunk_latent.shape)}"
            )
        if target_chunk_mask.dim() != 2:
            raise ValueError(
                f"target_chunk_mask must have shape [B, S], got {tuple(target_chunk_mask.shape)}"
            )
        if enroll_summary.dim() != 2:
            raise ValueError(f"enroll_summary must have shape [B, D], got {tuple(enroll_summary.shape)}")

        target_proj = self.target_projector(target_chunk_latent)
        enroll_feat = enroll_summary.unsqueeze(1).expand(-1, target_proj.size(1), -1)

        feature_parts = [target_proj, enroll_feat]
        if self.use_mask_feature:
            feature_parts.append(target_chunk_mask.unsqueeze(-1))
        if self.use_similarity:
            similarity = F.cosine_similarity(
                F.normalize(target_proj, p=2, dim=-1, eps=1e-8),
                F.normalize(enroll_feat, p=2, dim=-1, eps=1e-8),
                dim=-1,
                eps=1e-8,
            ).unsqueeze(-1)
            feature_parts.append(similarity)

        presence_feat_chunk = torch.cat(feature_parts, dim=-1)
        logits = self.net(self.pre_norm(presence_feat_chunk)).squeeze(-1)
        return logits, presence_feat_chunk


class AuraPA(nn.Module):
    def __init__(
        self,
        *,
        enroll_dim: int = 192,
        sepformer_source: str = "speechbrain/sepformer-wsj02mix",
        sepformer_savedir: str | None = None,
        condition_last_n: int = 1,
        conditioning_num_heads: int = 8,
        conditioning_attn_dropout: float = 0.1,
        conditioning_proj_dropout: float = 0.1,
        conditioning_scale_init: float = 0.1,
        conditioning_use_gate: bool = True,
        enroll_projector_hidden_dim: int | None = None,
        enroll_projector_dropout: float = 0.0,
        presence_hidden_dim: int = 256,
        presence_dropout: float = 0.1,
        presence_use_similarity: bool = True,
        presence_use_mask_feature: bool = True,
        presence_use_energy_ratio: bool | None = None,
        presence_chunk_length_ms: int = 100,
        presence_chunk_hop_ms: int = 50,
        presence_clip_aggregate: str = "logsumexp_mean",
        return_masks_in_aux: bool = False,
    ) -> None:
        super().__init__()
        savedir = sepformer_savedir or _default_sepformer_savedir()

        backbone = sepformer_load(source=sepformer_source, savedir=savedir)
        self.encoder = backbone.mods.encoder
        self.decoder = backbone.mods.decoder
        self.masknet = backbone.mods.masknet

        self.encoder_dim = int(self.masknet.conv1d.in_channels)
        self.feature_dim = int(self.masknet.conv1d.out_channels)
        self.enroll_dim = int(enroll_dim)
        self.condition_last_n = max(0, int(condition_last_n))
        self.return_masks_in_aux = bool(return_masks_in_aux)
        self.presence_chunk_length_ms = int(presence_chunk_length_ms)
        self.presence_chunk_hop_ms = int(presence_chunk_hop_ms)
        self.presence_clip_aggregate = str(presence_clip_aggregate).lower()
        self.presence_use_energy_ratio = presence_use_energy_ratio

        if int(self.masknet.num_spks) != 2:
            raise ValueError(
                f"AuraPA expects a 2-output masknet for target/residual decomposition, got {self.masknet.num_spks}"
            )

        self._wrap_masknet_dual_path_blocks(
            num_heads=int(conditioning_num_heads),
            attn_dropout=float(conditioning_attn_dropout),
            proj_dropout=float(conditioning_proj_dropout),
            attn_scale_init=float(conditioning_scale_init),
            projector_hidden_dim=enroll_projector_hidden_dim,
            projector_dropout=float(enroll_projector_dropout),
            use_gate=bool(conditioning_use_gate),
        )

        self.presence_enroll_projector = EnrollmentProjector(
            input_dim=self.enroll_dim,
            output_dim=self.feature_dim,
            hidden_dim=enroll_projector_hidden_dim,
            dropout=float(enroll_projector_dropout),
        )
        self.presence_head = TemporalPresenceHead(
            encoder_dim=self.encoder_dim,
            feature_dim=self.feature_dim,
            hidden_dim=int(presence_hidden_dim),
            dropout=float(presence_dropout),
            use_similarity=bool(presence_use_similarity),
            use_mask_feature=bool(presence_use_mask_feature),
        )

    def _wrap_masknet_dual_path_blocks(
        self,
        *,
        num_heads: int,
        attn_dropout: float,
        proj_dropout: float,
        attn_scale_init: float,
        projector_hidden_dim: int | None,
        projector_dropout: float,
        use_gate: bool,
    ) -> None:
        original_blocks = list(self.masknet.dual_mdl)
        total_blocks = len(original_blocks)
        first_conditioned = max(0, total_blocks - self.condition_last_n)
        wrapped_blocks = nn.ModuleList()

        for idx, block in enumerate(original_blocks):
            if idx >= first_conditioned and self.condition_last_n > 0:
                wrapped_blocks.append(
                    EnrollmentConditionedDualPathBlock(
                        original_block=block,
                        enroll_dim=self.enroll_dim,
                        feature_dim=self.feature_dim,
                        num_heads=num_heads,
                        attn_dropout=attn_dropout,
                        proj_dropout=proj_dropout,
                        attn_scale_init=attn_scale_init,
                        projector_hidden_dim=projector_hidden_dim,
                        projector_dropout=projector_dropout,
                        use_gate=use_gate,
                    )
                )
            else:
                wrapped_blocks.append(block)

        self.masknet.dual_mdl = wrapped_blocks

    def _prepare_enrollment(
        self,
        enrollment: torch.Tensor,
        enroll_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if enrollment.dim() == 2:
            enrollment = enrollment.unsqueeze(1)
        if enrollment.dim() != 3:
            raise ValueError(
                f"enrollment must have shape [B, D] or [B, K, D], got {tuple(enrollment.shape)}"
            )
        if enrollment.size(-1) != self.enroll_dim:
            raise ValueError(
                f"enrollment dim mismatch: expected {self.enroll_dim}, got {enrollment.size(-1)}"
            )
        enroll_mask = _normalize_enroll_mask(
            enroll_mask,
            batch_size=enrollment.size(0),
            seq_len=enrollment.size(1),
            device=enrollment.device,
        )
        enrollment = F.normalize(enrollment, p=2, dim=-1, eps=1e-8)
        return enrollment, enroll_mask

    def _forward_masknet_conditioned(
        self,
        x: torch.Tensor,
        enrollment: torch.Tensor,
        enroll_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.masknet.norm(x)
        x = self.masknet.conv1d(x)

        if getattr(self.masknet, "use_global_pos_enc", False):
            x = self.masknet.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (x.size(1) ** 0.5)

        x, gap = self.masknet._Segmentation(x, self.masknet.K)
        y = x
        for block in self.masknet.dual_mdl:
            if isinstance(block, EnrollmentConditionedDualPathBlock):
                y = block(y, enrollment=enrollment, enroll_mask=enroll_mask)
            else:
                y = block(y)

        conditioned_chunks = y

        y = self.masknet.prelu(y)
        y = self.masknet.conv2d(y)
        bsz, _, k, s = y.shape
        y = y.contiguous().view(bsz * self.masknet.num_spks, -1, k, s)
        y = self.masknet._over_add(y, gap)
        y = self.masknet.output(y) * self.masknet.output_gate(y)
        y = self.masknet.end_conv1x1(y)

        _, channels, length = y.shape
        y = y.view(bsz, self.masknet.num_spks, channels, length)
        masks = y.permute(0, 2, 3, 1).contiguous()
        masks = torch.softmax(masks, dim=-1)
        return masks, conditioned_chunks

    def _aggregate_chunk_logits(self, chunk_logits: torch.Tensor) -> torch.Tensor:
        if chunk_logits.dim() != 2:
            raise ValueError(f"chunk_logits must have shape [B, S], got {tuple(chunk_logits.shape)}")
        num_chunks = max(1, int(chunk_logits.size(-1)))
        if self.presence_clip_aggregate == "logsumexp_mean":
            return torch.logsumexp(chunk_logits, dim=-1) - math.log(float(num_chunks))
        if self.presence_clip_aggregate == "mean":
            return chunk_logits.mean(dim=-1)
        if self.presence_clip_aggregate == "max":
            return chunk_logits.max(dim=-1).values
        raise ValueError(f"unsupported presence_clip_aggregate: {self.presence_clip_aggregate}")

    def forward(
        self,
        mix: torch.Tensor,
        enrollment: torch.Tensor,
        *,
        enroll_mask: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if mix.dim() != 2:
            raise ValueError(f"mix must have shape [B, T], got {tuple(mix.shape)}")

        enrollment, enroll_mask = self._prepare_enrollment(enrollment, enroll_mask)
        encoded_mix = self.encoder(mix)
        masks, conditioned_chunks = self._forward_masknet_conditioned(
            encoded_mix,
            enrollment=enrollment,
            enroll_mask=enroll_mask,
        )

        target_mask = masks[:, :, :, 0]
        residual_mask = masks[:, :, :, 1]
        target_latent = encoded_mix * target_mask
        residual_latent = encoded_mix * residual_mask

        est_target = self.decoder(target_latent)
        est_residual = self.decoder(residual_latent)

        enroll_summary = _masked_mean(
            self.presence_enroll_projector(enrollment),
            enroll_mask,
        )

        chunk_grid = build_chunk_grid(
            int(mix.size(-1)),
            sample_rate=16000,
            chunk_length_ms=self.presence_chunk_length_ms,
            chunk_hop_ms=self.presence_chunk_hop_ms,
            device=mix.device,
        )
        step_starts, step_ends = build_latent_chunk_ranges(
            num_samples=int(mix.size(-1)),
            num_steps=int(target_latent.size(-1)),
            grid=chunk_grid,
            device=target_latent.device,
        )
        target_chunk_latent = chunk_mean_pool_sequence(target_latent, step_starts=step_starts, step_ends=step_ends)
        target_chunk_latent = target_chunk_latent.transpose(1, 2).contiguous()
        target_mask_summary = chunk_mean_pool_sequence(
            target_mask.mean(dim=1, keepdim=True),
            step_starts=step_starts,
            step_ends=step_ends,
        ).squeeze(1)

        presence_logit_chunk, presence_feat_chunk = self.presence_head(
            target_chunk_latent=target_chunk_latent,
            target_chunk_mask=target_mask_summary,
            enroll_summary=enroll_summary,
        )
        presence_logit_clip = self._aggregate_chunk_logits(presence_logit_chunk)

        aux: dict[str, torch.Tensor] = {
            "presence_logit_chunk": presence_logit_chunk,
            "presence_logit_clip": presence_logit_clip,
            "chunk_center_sec": chunk_grid.center_sec,
        }
        if return_intermediates:
            aux["presence_feat_chunk"] = presence_feat_chunk
            if self.return_masks_in_aux:
                aux["target_mask"] = target_mask
                aux["residual_mask"] = residual_mask
            aux["target_latent"] = target_latent
            aux["conditioned_chunks"] = conditioned_chunks
            aux["enroll_summary"] = enroll_summary
            aux["target_chunk_mask"] = target_mask_summary

        return est_target, est_residual, aux


AuraPATemporal = AuraPA
