import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone_load import sepformer_load


def overlap_and_add(signal, frame_step):
    """
    Reconstruct a signal from framed representation.

    Args:
        signal: Tensor shaped [..., frames, frame_length]
        frame_step: Frame hop size
    Returns:
        Tensor shaped [..., output_size]
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(
        0, subframes_per_frame, subframe_step
    )
    frame = frame.clone().detach().to(signal.device).long()
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation using global conditioning vector."""

    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, feature_dim * 2)

        # Start near identity: gamma=1, beta=0
        nn.init.constant_(self.cond_proj.weight, 0.0)
        nn.init.constant_(self.cond_proj.bias, 0.0)
        self.cond_proj.bias.data[:feature_dim] = 1.0

    def forward(self, x, condition):
        params = self.cond_proj(condition)
        gamma, beta = torch.chunk(params, 2, dim=-1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class PitchNet(nn.Module):
    """
    Estimate pitch features from mixture latent + enrollment.

    Outputs:
    - fused frame feature: [B, H, L]
    - global pitch embedding: [B, H]
    - frame-level pitch prediction: [B, L]
    """

    @staticmethod
    def _make_group_norm(num_channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
        groups = min(int(preferred_groups), int(num_channels))
        while groups > 1 and (num_channels % groups) != 0:
            groups -= 1
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels)

    def __init__(self, latent_dim, enroll_dim, hidden_dim=128, norm_groups=8):
        super().__init__()
        self.mix_proj = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        self.enroll_proj = nn.Linear(enroll_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            self._make_group_norm(hidden_dim, preferred_groups=norm_groups),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            self._make_group_norm(hidden_dim, preferred_groups=norm_groups),
            nn.ReLU(),
        )

        self.pitch_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, z, enroll):
        mix_feat = self.mix_proj(z)
        enroll_feat = self.enroll_proj(enroll).unsqueeze(-1).expand(-1, -1, z.size(-1))

        fused = self.fusion(torch.cat([mix_feat, enroll_feat], dim=1))
        pitch_pred = self.pitch_head(fused).squeeze(1)
        pitch_global = self.global_pool(fused).squeeze(-1)
        return fused, pitch_global, pitch_pred


class ConditionalDualPathBlock(nn.Module):
    """Wrap SpeechBrain dual-path block and inject FiLM before inter-path."""

    def __init__(self, original_block, cond_dim, feature_dim):
        super().__init__()
        self.original_block = original_block
        self.film = FiLMLayer(cond_dim, feature_dim)

    def forward(self, x, condition):
        # x: [B, C, T1, T2]
        bsz, channels, t1, t2 = x.size()

        # 1) Intra-chunk path
        intra_in = x.permute(0, 3, 2, 1).contiguous().view(bsz * t2, t1, channels)
        intra_out = self.original_block.intra_mdl(intra_in)
        intra_out = intra_out.view(bsz, t2, t1, channels).permute(0, 3, 2, 1)
        intra_out = self.original_block.intra_norm(intra_out + x)

        # 2) Condition injection
        intra_out_conditioned = self.film(intra_out, condition)

        # 3) Inter-chunk path
        inter_in = (
            intra_out_conditioned.permute(0, 2, 3, 1).contiguous().view(bsz * t1, t2, channels)
        )
        inter_out = self.original_block.inter_mdl(inter_in)
        inter_out = inter_out.view(bsz, t1, t2, channels).permute(0, 3, 2, 1)

        # Keep residual connection against original intra output for stability.
        out = self.original_block.inter_norm(inter_out + intra_out)
        return out


class AuraTeacher(nn.Module):
    def __init__(self, enroll_dim=128, device="cpu", use_pitchnet=True):
        super().__init__()

        print("[AURA] Initializing Teacher Model...")
        self.backbone = sepformer_load()
        self.encoder = self.backbone.mods.encoder
        self.decoder = self.backbone.mods.decoder
        self.masknet = self.backbone.mods.masknet

        self.feat_dim = 256
        self.pitch_dim = 128
        self.use_pitchnet = use_pitchnet
        self.cond_dim = enroll_dim + (self.pitch_dim if self.use_pitchnet else 0)

        self.pitch_net = None
        if self.use_pitchnet:
            self.pitch_net = PitchNet(self.feat_dim, enroll_dim, hidden_dim=self.pitch_dim)

        new_dual_mdl = nn.ModuleList()
        for block in self.masknet.dual_mdl:
            new_dual_mdl.append(ConditionalDualPathBlock(block, self.cond_dim, self.feat_dim))
        self.masknet.dual_mdl = new_dual_mdl

        self.device = device
        self.to(device)

    def forward(self, mix, enrollment, return_aux=False):
        """
        Args:
            mix: [B, T]
            enrollment: [B, enroll_dim]
            return_aux: returns dict with pitch prediction when True
        """
        enrollment = F.normalize(enrollment, p=2, dim=-1, eps=1e-8)
        w = self.encoder(mix)

        pitch_pred = None
        if self.use_pitchnet:
            _, pitch_global, pitch_pred = self.pitch_net(w, enrollment)
            condition = torch.cat([enrollment, pitch_global], dim=-1)
        else:
            condition = enrollment

        m = self._forward_masknet_conditioned(w, condition)
        m_target = m[:, :, :, 0]
        m_residual = m[:, :, :, 1]

        est_target = self.decoder(w * m_target)
        est_residual = self.decoder(w * m_residual)

        if not return_aux:
            return est_target, est_residual
        return est_target, est_residual, {"pitch_pred": pitch_pred}

    def _forward_masknet_conditioned(self, x, condition):
        # x: [B, C, L]
        x = self.masknet.norm(x)
        x = self.masknet.conv1d(x)

        k, s = 250, 125
        bsz, channels, length = x.shape

        rest = (length - k) % s
        if rest > 0:
            pad = torch.zeros(bsz, channels, s - rest, device=x.device)
            x = torch.cat([x, pad], dim=2)

        x_unfolded = x.unfold(2, k, s).permute(0, 1, 3, 2)

        y = x_unfolded
        for block in self.masknet.dual_mdl:
            y = block(y, condition)

        y = self.masknet.conv2d(y)
        y = self.masknet.prelu(y)
        y = self.masknet.end_conv1x1(y)

        y_for_ola = y.permute(0, 1, 3, 2)
        y_folded = overlap_and_add(y_for_ola, s)
        y_folded = y_folded[:, :, :length]

        # Build 2-head logits and normalize with softmax.
        # This enforces non-negative masks and per-bin partition:
        # m_target + m_residual = 1
        out = self.masknet.output(y_folded)
        gate = self.masknet.output_gate(y_folded)
        logits = torch.stack([out, gate], dim=-1)  # [B, C, L, 2]
        m = torch.softmax(logits, dim=-1)
        return m


if __name__ == "__main__":
    model = AuraTeacher(use_pitchnet=True)
    mix = torch.randn(2, 16000)
    enroll = torch.randn(2, 128)
    tgt, res, aux = model(mix, enroll, return_aux=True)
    pitch_shape = None if aux["pitch_pred"] is None else aux["pitch_pred"].shape
    print(f"Target: {tgt.shape}, Residual: {res.shape}, Pitch: {pitch_shape}")
