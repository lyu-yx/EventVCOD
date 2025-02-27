import torch
import torch.nn as nn
import torch.nn.functional as F  # We'll use this for interpolation
from typing import List, Tuple, Dict, Any, Optional, Type

# -----------------------------
# Auxiliary Modules (Mostly Unchanged)
# -----------------------------
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
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        
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
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(bin_size) for bin_size in [1, 2, 3, 6]])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        h, w = x.shape[2:]
        for pool in self.pools:
            feat = pool(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            features.append(feat)
        return torch.cat(features, dim=1)

class PositionEmbeddingRandom(nn.Module):
    """
    A stub positional embedding layer.
    Replace with a deterministic variant (e.g., sinusoidal) as needed.
    """
    def __init__(self, num_pos_feats: int = 64):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, shape: Tuple[int, int]) -> torch.Tensor:
        h, w = shape
        return torch.rand(self.num_pos_feats * 2, h, w)

def initialize_embedding_generator(module: nn.Module) -> None:
    """Initialize module weights."""
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

# ---------------------------------------------------------
# Transformer-based Video Feature Aggregator Module (Memory-Optimized)
# ---------------------------------------------------------
class VideoFeatureTransformerAggregator(nn.Module):
    """
    Aggregates a sequence of future-frame features using a Transformer encoder.
    Expected input shape: [B, T, S, F] where S = H * W (spatial positions).
    The transformer processes the temporal sequence for each spatial location,
    and outputs a summary feature map of shape [B, F, H, W].

    Memory optimizations:
      - Reduced number of layers (default 1)
      - Reduced number of heads (default 4)
      - Smaller feed-forward dimension (default input_dim * 2)
      - Zero dropout in the encoder layer
      - AMP autocasting during the transformer forward pass
    """
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 1,       # Reduced from 2
        num_heads: int = 4,        # Reduced from 8
        hidden_dim: Optional[int] = None,  # Set to input_dim * 2 if not provided
        spatial_size: Tuple[int, int] = (64, 64),
        max_T: int = 10  # maximum expected number of frames
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.max_T = max_T
        if hidden_dim is None:
            hidden_dim = input_dim * 2  # Reduced feed-forward dimension
        # Learnable positional encoding for the temporal dimension.
        self.pos_encoding = nn.Parameter(torch.randn(1, max_T, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True,
            dropout=0.0  # Disable dropout to reduce memory usage
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, S, F] where S = H * W.
        Returns:
            Aggregated feature map of shape [B, F, H, W].
        """
        B, T, S, F = x.shape
        # Rearrange: treat T frames per spatial location as a sequence.
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, S, T, F]
        x = x.view(B * S, T, F)
        # Interpolate positional encoding if T != max_T.
        if T != self.max_T:
            pos_enc = torch.nn.functional.interpolate(
                self.pos_encoding.transpose(1, 2), size=T, mode='linear', align_corners=False
            ).transpose(1, 2)
        else:
            pos_enc = self.pos_encoding
        x = x + pos_enc.repeat(B * S, 1, 1)
        # Use AMP autocasting to lower memory usage during transformer computation.
        with torch.cuda.amp.autocast(enabled=True):
            out = self.transformer_encoder(x)  # [B*S, T, F]
        out = out.mean(dim=1)  # Aggregate over time -> [B*S, F]
        out = self.proj(out)
        out = out.view(B, S, F)
        H, W = self.spatial_size
        out = out.view(B, H, W, F).permute(0, 3, 1, 2).contiguous()  # [B, F, H, W]
        return out

class SimpleHighResFusion(nn.Module):
    def __init__(self, out_channels: int = 256):
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

    def forward(self, feat: torch.Tensor, fused_accumulator: torch.Tensor) -> torch.Tensor:
        in_channels = feat.shape[1]
        if in_channels == 32:
            feat = self.adapter_32(feat)
        elif in_channels == 64:
            feat = self.adapter_64(feat)
        else:
            raise ValueError(f"Unsupported input channels: {in_channels}")
        fused_accumulator += feat
        return fused_accumulator

# -----------------------------
# Revised Embedding Generator
# -----------------------------
class EmbeddingGenerator(nn.Module):
    """
    An enhanced embedding generator that:
      1. Processes backbone and event features.
      2. Fuses high-resolution features.
      3. Incorporates future-frame features using a Transformer-based aggregator.
      4. Produces both sparse and dense embeddings.
      
    For cur_video["vision_feats"], we now assume each entry is a list/tuple of 3 feature levels.
    We only use the lower-level feature (index 2) from each frame, which has shape [S, 1, C] or [B, S, C].
    We then stack these across time (T=3) to obtain a tensor of shape [B, T, S, C].
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

        # 1. Backbone and event feature processing
        self.backbone_block = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
        )
        self.event_block = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
        )

        # 2. High-resolution feature fusion
        self.highres_fusion_conv = SimpleHighResFusion(out_channels=256)

        # 3. Transformer-based aggregator for video features.
        self.video_feature_transformer = VideoFeatureTransformerAggregator(
            input_dim=256,
            num_layers=1,            # Reduced number of layers
            num_heads=4,             # Reduced number of heads
            hidden_dim=mask_in_chans * 2,  # Reduced feed-forward dimension
            spatial_size=image_embedding_size,
            max_T=10  # Adjust as needed based on expected max number of future frames.
        )

        # 4. Attention modules
        self.channel_attention = ChannelAttention(mask_in_chans)
        self.spatial_attention = SpatialAttention()
        self.region_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=1),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=3, padding=1)
        )

        # 5. Embedding branches: dense and sparse
        self.dense_embedder = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim + 1, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            self.activation
        )
        self.sparse_embedder = nn.Sequential(
            PyramidPooling(mask_in_chans),
            nn.Conv2d(mask_in_chans * 5, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )

        # 6. Positional embedding for dense features (currently random)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

    def _fuse_high_res_features(
        self,
        features_list: List[torch.Tensor],
        target_size: Tuple[int, int],
        fused_accumulator: torch.Tensor
    ) -> torch.Tensor:
        """Helper function to interpolate and fuse high-res features."""
        for feat in features_list:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            fused_accumulator = self.highres_fusion_conv(feat, fused_accumulator)
        return fused_accumulator

    def forward(
        self, 
        backbone_features: torch.Tensor, 
        event_features: torch.Tensor,
        high_res_features: List[torch.Tensor],
        high_res_event_features: List[torch.Tensor],
        cur_video: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            backbone_features: [B, mask_in_chans, H, W]
            event_features: [B, mask_in_chans, H, W]
            high_res_features: List of tensors (e.g., [B, 32, 256, 256], [B, 64, 128, 128])
            high_res_event_features: List of tensors with the same structure.
            cur_video: Optional dict with key "vision_feats" containing a list of frames.
                       Each frame is a list/tuple of 3 features; we use the lower-level one (index 2)
                       which is expected to have shape [S, 1, C] or [B, S, C].
        Returns:
            sparse_embeddings: [B, (H*W), embed_dim] after flattening.
            dense_embeddings:  [B, embed_dim, H, W]
        """
        B, C, H, W = backbone_features.shape
        assert C == event_features.shape[1], "Mismatch in channel dimensions."

        # Process backbone and event features.
        backbone_processed = self.backbone_block(backbone_features)
        event_processed = self.event_block(event_features)

        # Fuse high-res features.
        fused_highres = torch.zeros((B, 256, H, W), device=backbone_features.device)
        fused_highres = self._fuse_high_res_features(high_res_features, (H, W), fused_highres)
        fused_highres = self._fuse_high_res_features(high_res_event_features, (H, W), fused_highres)

        # Combine all features.
        combined_features = backbone_processed + event_processed + fused_highres

        # Incorporate video features using the Transformer aggregator.
        if cur_video is not None:
            vision_feats = cur_video.get("vision_feats", None)
            if vision_feats is not None and len(vision_feats) > 0:
                # For each frame, use the lower-level feature (index 2).
                # Expected shape: [S, 1, C] or [B, S, C]. We squeeze the singleton dim if needed.
                feats_list = []
                for feat in vision_feats:
                    lower_feat = feat[2]  # Use the lower-level feature
                    if lower_feat.dim() == 3 and lower_feat.shape[1] == 1:
                        lower_feat = lower_feat.squeeze(1)  # becomes [S, C]
                    if lower_feat.dim() == 2:
                        lower_feat = lower_feat.unsqueeze(0)  # becomes [1, S, C]
                    # Now lower_feat should be [B, S, C]. Add a temporal dimension.
                    lower_feat = lower_feat.unsqueeze(1)  # now [B, 1, S, C]
                    feats_list.append(lower_feat)
                # Stack along the temporal dimension; T should be 3.
                feats_stack = torch.cat(feats_list, dim=1)  # [B, T, S, C]
                future_summary = self.video_feature_transformer(feats_stack)
                combined_features += future_summary

        # Apply attention modules.
        features = self.channel_attention(combined_features)
        features = self.spatial_attention(features)

        # Compute region attention and generate embeddings.
        region_attention = torch.sigmoid(self.region_attention(features))
        dense_embeddings = self.dense_embedder(features)
        dense_embeddings = self.refinement(torch.cat([dense_embeddings, region_attention], dim=1))
        sparse_embeddings = self.sparse_embedder(features)
        sparse_embeddings = sparse_embeddings.flatten(2).transpose(1, 2)

        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self) -> torch.Tensor:
        """Return positional embedding for dense features."""
        pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)
        device = next(self.parameters()).device
        return pe.to(device)
