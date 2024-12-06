import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Type
from prompt_gen.backbone.position_encoding import PositionEmbeddingRandom


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: Type[nn.Module]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = activation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.act(out + residual)

class PyramidPooling(nn.Module):
    def __init__(self, in_channels: int, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=in_channels),  # Replacing BatchNorm2d
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pyramids = [x]
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            pyramids.append(out)
        return torch.cat(pyramids, dim=1)

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, shape: Tuple[int, int]) -> torch.Tensor:
        # Simple random positional embeddings for demonstration
        h, w = shape
        pos_embed = torch.randn(1, self.num_pos_feats * 2, h, w)
        return pos_embed

class CrossScaleTransformer(nn.Module):
    """
    A lightweight transformer block for spatial feature refinement.
    Uses multi-head self-attention to capture long-range dependencies.
    """
    def __init__(self, dim: int, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        N = H * W
        # Flatten spatial dims
        x = x.view(B, C, N).transpose(1, 2)  # [B, N, C]

        # Self-attention
        x2 = self.norm1(x)
        x_attn, _ = self.attn(x2, x2, x2)
        x = x + x_attn

        # MLP block
        x2 = self.norm2(x)
        x_mlp = self.mlp(x2)
        x = x + x_mlp

        # Reshape back to [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class EmbeddingGenerator(nn.Module):
    """
    A promptless embedding generator that derives dense and sparse embeddings
    directly from visual (backbone) and event features. This version integrates
    a CrossScaleTransformer for more sophisticated spatial context modeling.
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
        
        # Backbone feature processing (primary/visual path)
        self.backbone_processor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=rate, dilation=rate),
                nn.BatchNorm2d(mask_in_chans),
                self.activation
            ) for rate in [1, 2, 4]
        ])
        
        # Event feature processing (secondary/auxiliary path)
        self.event_processor = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=1),
            nn.BatchNorm2d(mask_in_chans),
            self.activation
        )
        
        # High-res fusion for backbone (2 levels)
        self.high_res_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_channels, mask_in_chans, kernel_size=1),
                nn.BatchNorm2d(mask_in_chans),
                self.activation
            ) for feat_channels in [32, 64]
        ])
        
        # High-res fusion for event (2 levels)
        self.high_res_event_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_channels, mask_in_chans, kernel_size=1),
                nn.BatchNorm2d(mask_in_chans),
                self.activation
            ) for feat_channels in [32, 64]
        ])
        
        # Feature normalization layers
        self.backbone_norm = nn.BatchNorm2d(mask_in_chans)
        self.event_norm = nn.BatchNorm2d(mask_in_chans)
        self.high_res_norm = nn.BatchNorm2d(mask_in_chans)
        self.high_res_event_norm = nn.BatchNorm2d(mask_in_chans)
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(mask_in_chans)
        self.spatial_attention = SpatialAttention()
        
        # Dense embedder
        self.dense_embedder = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        )
        
        # Sparse embedder (global)
        self.sparse_embedder = nn.Sequential(
            PyramidPooling(mask_in_chans),
            nn.Conv2d(mask_in_chans * 5, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Region attention layer
        self.region_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=1),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=3, padding=1)
        )
        
        # Refinement module
        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim + 1, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            self.activation
        )

        # Positional embedding
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        
        # Learnable scaling factors
        self.backbone_scale = nn.Parameter(torch.ones(1))
        self.event_scale = nn.Parameter(torch.full((1,), 0.1))
        self.high_res_scale = nn.Parameter(torch.full((1,), 0.1))
        self.high_res_event_scale = nn.Parameter(torch.full((1,), 0.05))
        
        # Cross-scale transformer for refined spatial modeling
        self.cross_scale_transformer = CrossScaleTransformer(dim=mask_in_chans, num_heads=4, mlp_ratio=4.0, dropout=0.1)

    def forward(self, 
                backbone_features: torch.Tensor, 
                event_features: torch.Tensor,
                high_res_features: List[torch.Tensor],
                high_res_event_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Process backbone (visual) features
        backbone_processed = sum(processor(backbone_features) for processor in self.backbone_processor)
        backbone_processed = self.backbone_norm(backbone_processed)
        
        # 2. Process event features
        event_processed = self.event_processor(event_features)
        event_processed = self.event_norm(event_processed)
        
        # 3. Fuse high-resolution features for backbone and event
        high_res_fused = torch.zeros_like(backbone_processed)
        high_res_event_fused = torch.zeros_like(backbone_processed)
        
        # Fuse backbone high-res
        for feat, fusion in zip(high_res_features, self.high_res_fusion):
            if feat.shape[-2:] != backbone_processed.shape[-2:]:
                feat = F.interpolate(feat, size=backbone_processed.shape[-2:], mode='bilinear', align_corners=False)
            high_res_fused += fusion(feat)
            
        # Fuse event high-res
        for feat, fusion in zip(high_res_event_features, self.high_res_event_fusion):
            if feat.shape[-2:] != backbone_processed.shape[-2:]:
                feat = F.interpolate(feat, size=backbone_processed.shape[-2:], mode='bilinear', align_corners=False)
            high_res_event_fused += fusion(feat)
            
        high_res_fused = self.high_res_norm(high_res_fused)
        high_res_event_fused = self.high_res_event_norm(high_res_event_fused)
        
        # 4. Combine features with scaling
        backbone_contribution = self.backbone_scale * backbone_processed
        event_contribution = self.event_scale * event_processed
        high_res_contribution = self.high_res_scale * high_res_fused
        high_res_event_contribution = self.high_res_event_scale * high_res_event_fused
        
        combined_features = (backbone_contribution + event_contribution +
                             high_res_contribution + high_res_event_contribution)
        
        # ---- New: Cross-Scale Transformer Integration ----
        combined_features = self.cross_scale_transformer(combined_features)
        # --------------------------------------------------
        
        # 5. Attention and refinement
        features = self.channel_attention(combined_features)
        features = self.spatial_attention(features)
        
        # Region attention
        region_attention = torch.sigmoid(self.region_attention(features))
        
        # Dense embeddings
        dense_embeddings = self.dense_embedder(features)
        dense_embeddings = self.refinement(torch.cat([dense_embeddings, region_attention], dim=1))
        
        # Sparse embeddings
        print('features.shape', features.shape)
        sparse_embeddings = self.sparse_embedder(features)
        sparse_embeddings = sparse_embeddings.flatten(2).transpose(1, 2)
        
        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    
def initialize_embedding_generator(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:  # Check if bias exists
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:  # Check if bias exists
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        if module.bias is not None:  # Check if bias exists
            nn.init.constant_(module.bias, 0)
