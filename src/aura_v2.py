import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbone_load import sepformer_load


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


class EnrollmentCrossAttentionBlock(nn.Module):
    """Wrap SpeechBrain dual-path block and inject FiLM for pitch and CrossAttention for Enrollment."""

    def __init__(self, original_block, pitch_cond_dim, enroll_dim, feature_dim, num_heads=8):
        super().__init__()
        self.original_block = original_block
        if pitch_cond_dim > 0:
             self.film = FiLMLayer(pitch_cond_dim, feature_dim)
        else:
             self.film = None

        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        # linear projection to align enrollment dim with feature dim
        self.enroll_proj = nn.Linear(enroll_dim, feature_dim)
        
        # Gating mechanism for the attention output
        self.gate_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, pitch_condition, enrollment, enroll_mask=None):
        # x: [B, C, T1, T2]
        # enrollment: [B, T_enroll, enroll_dim] -> 2D sequence of enrollment features
        bsz, channels, t1, t2 = x.size()

        # 1) Intra-chunk path
        intra_in = x.permute(0, 3, 2, 1).contiguous().view(bsz * t2, t1, channels)
        intra_out = self.original_block.intra_mdl(intra_in)
        intra_out = intra_out.view(bsz, t2, t1, channels).permute(0, 3, 2, 1)
        intra_out = self.original_block.intra_norm(intra_out + x)

        # 2) Inter-chunk path
        inter_in = (
            intra_out.permute(0, 2, 3, 1).contiguous().view(bsz * t1, t2, channels)
        )
        inter_out = self.original_block.inter_mdl(inter_in)
        inter_out = inter_out.view(bsz, t1, t2, channels).permute(0, 3, 1, 2)
        
        # Residual connection against original intra output
        inter_out = self.inter_norm_forward(self.original_block.inter_norm, inter_out + intra_out)

        # 3) Condition injection & Cross Attention (Applied after Inter-chunk block)
        
        # FiLM for pitch (if applicable)
        if self.film is not None and pitch_condition is not None:
             inter_out_conditioned = self.film(inter_out, pitch_condition)
        else:
             inter_out_conditioned = inter_out
             
        # Cross Attention for enrollment
        # Reshape mixture features to [B, T1*T2, C] for attention (Query)
        query = inter_out_conditioned.permute(0, 2, 3, 1).contiguous().view(bsz, t1 * t2, channels)
        
        # Process enrollment to be the Key/Value: [B, T_enroll, C]
        # Assume enrollment is already [B, T_enroll, enroll_dim]
        # If it's 1D [B, enroll_dim] by mistake, we unsqueeze it to [B, 1, enroll_dim] to prevent crashing
        if enrollment.dim() == 2:
            enrollment = enrollment.unsqueeze(1)
            
        kv_seq = self.enroll_proj(enrollment)

        key_padding_mask = None
        if enroll_mask is not None:
            if enroll_mask.dim() == 1:
                enroll_mask = enroll_mask.unsqueeze(0)
            if enroll_mask.dim() == 3 and enroll_mask.size(-1) == 1:
                enroll_mask = enroll_mask.squeeze(-1)
            valid_mask = enroll_mask.bool()
            if valid_mask.dim() == 2 and valid_mask.size(1) == kv_seq.size(1):
                key_padding_mask = ~valid_mask
        
        # Attending to the enrollment features using mixture as Query
        # output is [B, T1*T2, C]
        attn_out, _ = self.cross_attn(
            query=query,
            key=kv_seq,
            value=kv_seq,
            key_padding_mask=key_padding_mask,
        )
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate_proj(attn_out))
        attn_out_gated = gate * attn_out
        
        # Reshape attended feature back to [B, C, T1, T2]
        attn_out_gated = attn_out_gated.view(bsz, t1, t2, channels).permute(0, 3, 1, 2)
        
        # Combine the original conditioned output and attended features
        final_out = inter_out_conditioned + attn_out_gated
        
        return final_out

    def inter_norm_forward(self, module, x):
        """Helper to handle norm layer correctly based on SpeechBrain's block structure"""
        # x is [B, C, T1, T2]
        # Need to permute to [B, T1, T2, C] for LayerNorm if it expects C at the end
        if isinstance(module, nn.LayerNorm) or hasattr(module, 'normalized_shape'):
            x_norm = x.permute(0, 2, 3, 1)
            x_norm = module(x_norm)
            return x_norm.permute(0, 3, 1, 2)
        return module(x)

class AuraV2(nn.Module):
    def __init__(
        self,
        enroll_dim=128,
        device="cpu",
        use_pitchnet=True,
        sepformer_source="speechbrain/sepformer-wsj02mix",
        sepformer_savedir="pretrained_models/sepformer-wsj02mix",
    ):
        super().__init__()

        print("[AURA] Initializing V2 Model...")
        self.backbone = sepformer_load(source=sepformer_source, savedir=sepformer_savedir)
        self.encoder = self.backbone.mods.encoder
        self.decoder = self.backbone.mods.decoder
        self.masknet = self.backbone.mods.masknet

        self.feat_dim = self.masknet.conv1d.out_channels
        self.pitch_dim = 128    
        self.use_pitchnet = use_pitchnet
        self.enroll_dim = enroll_dim

        self.pitch_net = None
        if self.use_pitchnet:
            self.pitch_net = PitchNet(self.feat_dim, enroll_dim, hidden_dim=self.pitch_dim)

        new_dual_mdl = nn.ModuleList()
        # V2 uses pitch for film and enroll for cross attention
        pitch_cond_dim = self.pitch_dim if self.use_pitchnet else 0
        
        for block in self.masknet.dual_mdl:
            new_dual_mdl.append(EnrollmentCrossAttentionBlock(
                block, 
                pitch_cond_dim=pitch_cond_dim, 
                enroll_dim=enroll_dim, 
                feature_dim=self.feat_dim
            ))
        self.masknet.dual_mdl = new_dual_mdl

        self.device = device
        self.to(device)
        
    def _forward_masknet_conditioned(self, x, pitch_condition, enrollment, enroll_mask=None):
        # x: [B, C, L]
        x = self.masknet.norm(x)
        x = self.masknet.conv1d(x)

        if getattr(self.masknet, "use_global_pos_enc", False):
            x = self.masknet.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
                x.size(1) ** 0.5
            )

        x, gap = self.masknet._Segmentation(x, self.masknet.K)

        y = x
        for block in self.masknet.dual_mdl:
            y = block(y, pitch_condition, enrollment, enroll_mask=enroll_mask)

        y = self.masknet.prelu(y)
        y = self.masknet.conv2d(y)
        bsz, _, k, s = y.shape

        y = y.contiguous().view(bsz * self.masknet.num_spks, -1, k, s)
        y = self.masknet._over_add(y, gap)
        y = self.masknet.output(y) * self.masknet.output_gate(y)
        y = self.masknet.end_conv1x1(y)

        _, channels, length = y.shape
        y = y.view(bsz, self.masknet.num_spks, channels, length)
        m = y.permute(0, 2, 3, 1).contiguous()  # [B, C, L, spks]
        m = torch.softmax(m, dim=-1)
        return m

    def _masked_mean_enrollment(self, enrollment, enroll_mask=None):
        if enrollment.dim() == 2:
            return enrollment
        if enroll_mask is None:
            return enrollment.mean(dim=1)

        valid = enroll_mask.float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (enrollment * valid).sum(dim=1) / denom

    def forward(self, mix, enrollment, return_aux=False, enroll_mask=None):
        """
        Args:
            mix: [B, T]
            enrollment: [B, enroll_dim] or [B, T_enroll, enroll_dim]
            return_aux: returns dict with pitch prediction when True
        """
        enrollment = F.normalize(enrollment, p=2, dim=-1, eps=1e-8)
        w = self.encoder(mix)

        pitch_pred = None
        pitch_global = None
        if self.use_pitchnet:
            enroll_for_pitch = self._masked_mean_enrollment(enrollment, enroll_mask=enroll_mask)
            _, pitch_global, pitch_pred = self.pitch_net(w, enroll_for_pitch)

        m = self._forward_masknet_conditioned(
            w,
            pitch_global,
            enrollment,
            enroll_mask=enroll_mask,
        )
        m_target = m[:, :, :, 0]
        m_residual = m[:, :, :, 1]

        est_target = self.decoder(w * m_target)
        est_residual = self.decoder(w * m_residual)

        if not return_aux:
            return est_target, est_residual

        return est_target, est_residual, {"pitch_pred": pitch_pred}
