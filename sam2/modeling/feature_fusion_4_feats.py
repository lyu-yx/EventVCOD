import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalContextGate(nn.Module):
    """Gate mechanism with adaptive filtering for irrelevant information."""
    def __init__(self, channels):
        super().__init__()
        self.conv_static = nn.Sequential(
            nn.Conv2d(channels, channels//2, 1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True)
        )
        self.conv_dynamic = nn.Sequential(
            nn.Conv2d(channels, channels//2, 1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        # Additional channel-wise attention for selective fusion
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, static, dynamic):
        static_feat = self.conv_static(static)
        dynamic_feat = self.conv_dynamic(dynamic)
        gate = self.gate(torch.cat([static_feat, dynamic_feat], dim=1))
        
        # Channel-wise attention to filter out misleading features
        channel_mask = self.channel_attention(dynamic)
        filtered_dynamic = dynamic * channel_mask

        return static * gate + filtered_dynamic * (1 - gate)


class MotionEnhancedAttention(nn.Module):
    """Attention mechanism with additional masking for temporal noise."""
    def __init__(self, channels):
        super().__init__()
        self.conv_q = nn.Conv2d(channels, channels, 1)
        self.conv_k = nn.Conv2d(channels, channels, 1)
        self.conv_v = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
        
        # Masking mechanism for filtering misleading features
        self.feature_mask = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, motion_info):
        B, C, H, W = x.shape
        
        # Apply feature masking to reduce irrelevant features
        masked_motion_info = motion_info * self.feature_mask(motion_info)
        
        q = self.conv_q(x).view(B, C, -1)
        k = self.conv_k(masked_motion_info).view(B, C, -1)
        v = self.conv_v(masked_motion_info).view(B, C, -1)
        
        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.permute(0, 2, 1))
        return out.view(B, C, H, W)


class AdaptiveTemporalFusion(nn.Module):
    """Fusion module with selective filtering and residual connection."""
    def __init__(self, in_channels=256):
        super().__init__()
        
        # Temporal context gating with adaptive filtering
        self.temporal_gate = TemporalContextGate(in_channels)
        
        # Motion-enhanced attention with feature masking
        self.motion_attn = MotionEnhancedAttention(in_channels)
        
        # Multi-scale temporal analysis
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels//4, 3, padding=r, dilation=r),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(inplace=True)
            ) for r in [1, 2, 4, 8]
        ])
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Temporal feature fusion
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, pix_feat, pix_feat_short_long):
        # 1. Adaptive temporal-spatial balancing with selective filtering
        balanced_feat = self.temporal_gate(pix_feat, pix_feat_short_long)
        
        # 2. Motion-enhanced feature attention with feature masking
        motion_enhanced = self.motion_attn(pix_feat_short_long, pix_feat)
        
        # 3. Multi-scale temporal analysis
        temporal_feats = []
        for branch in self.temporal_branches:
            temporal_feats.append(branch(pix_feat_short_long))
        temporal_feat = torch.cat(temporal_feats, dim=1)
        temporal_feat = self.temporal_fusion(temporal_feat)
        
        # 4. Combine and refine all features
        combined_feat = torch.cat([balanced_feat, motion_enhanced, temporal_feat, pix_feat], dim=1)
        refined_output = self.refinement(combined_feat)
        
        # 5. Residual connection to retain original spatial features
        output = refined_output + pix_feat  # Adding residual connection
        
        return output

# Example usage
if __name__ == "__main__":
    B, C, H, W = 2, 256, 64, 64
    pix_feat = torch.randn(B, C, H, W)
    pix_feat_short_long = torch.randn(B, C, H, W)
    
    fusion_module = AdaptiveTemporalFusion(in_channels=C)
    output = fusion_module(pix_feat, pix_feat_short_long)
    print(f"Output shape: {output.shape}")
