import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Type

# ----------------------------
# Auxiliary Modules (unchanged)
# ----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: Type[nn.Module]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = activation()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.activation(x)

class PyramidPooling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(bin_size) for bin_size in [1, 2, 3, 6]
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        h, w = x.shape[2:]
        for pool in self.pools:
            feat = pool(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            features.append(feat)
        return torch.cat(features, dim=1)

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, shape: Tuple[int, int]) -> torch.Tensor:
        h, w = shape
        return torch.rand(self.num_pos_feats * 2, h, w)

def initialize_embedding_generator(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class TemporalFeatureAggregator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[2] == 1:
            x = x.squeeze(2)
        out, hidden = self.gru(x)
        final_state = hidden[-1]
        summary_vector = self.out_fc(final_state)
        summary_vector = summary_vector.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return summary_vector

class SpatialTemporalFeatureAggregator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, spatial_size: Tuple[int, int]):
        super().__init__()
        self.spatial_size = spatial_size
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S, T, B, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * S, T, F)
        _, hidden = self.gru(x)
        final_hidden = hidden[-1]
        proj = self.out_fc(final_hidden)
        proj = proj.view(B, S, -1)
        H, W = self.spatial_size
        proj = proj.view(B, H, W, -1)
        proj = proj.permute(0, 3, 1, 2).contiguous()
        return proj

class SimpleHighResFusion(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.adapter_32 = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.adapter_64 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, fused_accumulator):
        in_channels = feat.shape[1]
        if in_channels == 32:
            feat = self.adapter_32(feat)
        elif in_channels == 64:
            feat = self.adapter_64(feat)
        else:
            raise ValueError(f"Unsupported input channels: {in_channels}")
        fused_accumulator += feat
        return fused_accumulator

# ----------------------------
# Modified Self-Attention with Windowed Attention
# ----------------------------
class SelfAttention2D(nn.Module):
    """
    Applies multi-head self-attention over local windows of a 2D feature map.
    Instead of flattening the full [H*W] tokens (which can be heavy), we partition
    the feature map into windows of size (window_size x window_size) and perform
    self-attention within each window independently.
    """
    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 16):
        super().__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Calculate necessary padding to make H and W divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        new_H, new_W = x.shape[2], x.shape[3]
        num_h = new_H // self.window_size
        num_w = new_W // self.window_size
        # Partition into windows: [B, C, num_h, num_w, window_size, window_size]
        x_windows = x.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size)
        # Rearrange to [B, num_h, num_w, window_size, window_size, C]
        x_windows = x_windows.contiguous().permute(0,2,3,4,5,1)
        # Flatten windows to sequences: [B*num_h*num_w, window_size*window_size, C]
        windows = x_windows.reshape(-1, self.window_size * self.window_size, C)
        # Apply layer normalization on each window
        windows_norm = self.norm(windows)
        attn_out, _ = self.attn(windows_norm, windows_norm, windows_norm)
        # Residual connection
        windows = windows + attn_out
        # Reshape back to [B, num_h, num_w, window_size, window_size, C]
        windows = windows.reshape(B, num_h, num_w, self.window_size, self.window_size, C)
        # Permute back to [B, C, num_h, window_size, num_w, window_size] and merge windows
        windows = windows.permute(0, 5, 1, 3, 2, 4).contiguous()
        x_out = windows.reshape(B, C, num_h * self.window_size, num_w * self.window_size)
        # Crop to original spatial dimensions
        x_out = x_out[:, :, :H, :W]
        return x_out

# ----------------------------
# Modified EmbeddingGenerator with Windowed Self-Attention
# ----------------------------
class EmbeddingGenerator(nn.Module):
    """
    A modified embedding generator that:
    1) Processes backbone and event features.
    2) Fuses high-resolution features.
    3) Optionally incorporates future-frame features.
    4) Introduces a windowed 2D self-attention block for enhanced global context,
       while keeping memory usage manageable.
    5) Produces sparse and dense embeddings.
    """
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.activation = activation()

        # 1. Backbone processing
        self.backbone_block = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
        )

        # 2. Event processing
        self.event_block = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
        )

        # 3. High-resolution feature fusion
        self.highres_fusion_conv = SimpleHighResFusion(out_channels=256)

        # 4. Aggregator for future-frame features (using spatial-temporal GRU)
        self.video_feature_rnn = SpatialTemporalFeatureAggregator(
            input_dim=256,
            hidden_dim=mask_in_chans,
            output_dim=mask_in_chans,
            spatial_size=image_embedding_size
        )

        # 5. New windowed self-attention block for global context
        self.self_attention = SelfAttention2D(channels=mask_in_chans, num_heads=4, window_size=16)

        # 6. Attention modules
        self.channel_attention = ChannelAttention(mask_in_chans)
        self.spatial_attention = SpatialAttention()
        self.region_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=1),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=3, padding=1)
        )

        # 7. Dense & Sparse embeddings
        self.dense_embedder = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        
        self.sparse_embedder = nn.Sequential(
            PyramidPooling(mask_in_chans),
            nn.Conv2d(mask_in_chans * 5, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim + 1, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            self.activation
        )

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

    def forward(
        self, 
        backbone_features: torch.Tensor, 
        event_features: torch.Tensor,
        high_res_features: List[torch.Tensor],
        high_res_event_features: List[torch.Tensor],
        cur_video: Dict[str, List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = backbone_features.shape

        # Backbone and event processing
        backbone_processed = self.backbone_block(backbone_features)
        event_processed = self.event_block(event_features)

        # High-res fusion
        fused_highres = torch.zeros((B, 256, H, W), device=backbone_features.device)
        for feat in high_res_features:
            if feat.shape[-2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            fused_highres = self.highres_fusion_conv(feat, fused_highres)

        for feat in high_res_event_features:
            if feat.shape[-2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            fused_highres = self.highres_fusion_conv(feat, fused_highres)

        combined_features = backbone_processed + event_processed + fused_highres

        # Incorporate future-frame features if provided
        if cur_video is not None:
            vision_feats = cur_video.get("vision_feats", None)
            if vision_feats is not None and len(vision_feats) > 0:
                feats_stack = torch.stack(
                    [feat[2].unsqueeze(1) if feat[2].dim() == 2 else feat[2] for feat in vision_feats], dim=1
                )
                future_summary = self.video_feature_rnn(feats_stack)
                combined_features += future_summary

        # Apply windowed self-attention block
        combined_features = self.self_attention(combined_features)

        # Apply channel and spatial attention
        features = self.channel_attention(combined_features)
        features = self.spatial_attention(features)

        # Region attention and embedding generation
        region_attention = torch.sigmoid(self.region_attention(features))
        dense_embeddings = self.dense_embedder(features)
        dense_embeddings = self.refinement(torch.cat([dense_embeddings, region_attention], dim=1))
        sparse_embeddings = self.sparse_embedder(features)
        sparse_embeddings = sparse_embeddings.flatten(2).transpose(1, 2)

        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self) -> torch.Tensor:
        pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)
        device = next(self.parameters()).device
        pe = pe.to(device)
        return pe
