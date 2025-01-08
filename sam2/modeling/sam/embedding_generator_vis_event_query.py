import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Type

# --------------------------
# Position Encoding (2D)
# --------------------------
class PositionalEncoding2D(nn.Module):
    """
    Standard sine/cosine positional embeddings for 2D feature maps:
      Output shape: [1, channels, H, W].
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] (used only for shape reference)
        Returns: PE [1, self.channels, H, W]
        """
        B, C, H, W = x.shape
        if self.channels % 4 != 0:
            raise ValueError("Channels must be multiple of 4 for 2D sine/cosine PE.")
        pe = torch.zeros(self.channels, H, W, device=x.device)

        half_dim = self.channels // 2
        # half_dim // 2 => quarter for x_sin, quarter for x_cos, similarly for y.
        div_term = torch.exp(
            torch.arange(0, half_dim, 2, device=x.device).float()
            * (-torch.log(torch.tensor(10000.0)) / (half_dim // 2))
        )

        pos_y = torch.arange(0, H, device=x.device).unsqueeze(1)
        pos_x = torch.arange(0, W, device=x.device).unsqueeze(1)

        # For X direction
        pe_x = torch.zeros(half_dim, H, W, device=x.device)
        pe_x[0::2, :, :] = torch.sin(pos_x * div_term).permute(1, 0)[:, None, :]
        pe_x[1::2, :, :] = torch.cos(pos_x * div_term).permute(1, 0)[:, None, :]

        # For Y direction
        pe_y = torch.zeros(half_dim, H, W, device=x.device)
        pe_y[0::2, :, :] = torch.sin(pos_y * div_term).permute(1, 0)[:, :, None]
        pe_y[1::2, :, :] = torch.cos(pos_y * div_term).permute(1, 0)[:, :, None]

        pe[:half_dim] = pe_x
        pe[half_dim:] = pe_y
        return pe.unsqueeze(0)  # [1, channels, H, W]


# --------------------------
# Dense Transformer Refiner
# --------------------------
class DenseTransformerRefiner(nn.Module):
    """
    Refines a feature map [B, in_channels, H, W] via a small Transformer encoder
    for improved dense embeddings:
      1) Flatten -> add position encoding
      2) Transformer encoder
      3) Reshape -> final conv to out_channels
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        # Project input channels to Transformer dimension
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False  # We'll pass [S, B, C]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

        # 2D sine-cosine position encoding
        self.pos_encoding = PositionalEncoding2D(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, H, W]
        returns: refined: [B, out_channels, H, W]
        """
        B, C, H, W = x.shape

        # 1) Project to hidden_dim
        feats = self.input_proj(x)  # [B, hidden_dim, H, W]

        # 2) Flatten -> [S, B, hidden_dim], S=H*W
        seq = feats.flatten(2).permute(2, 0, 1)  # [H*W, B, hidden_dim]

        # 3) Add positional encoding
        pe = self.pos_encoding(feats)           # [1, hidden_dim, H, W]
        pe_seq = pe.flatten(2).permute(2, 0, 1) # [H*W, 1, hidden_dim]
        seq = seq + pe_seq.expand(-1, B, -1)

        # 4) Transformer encoder
        encoded = self.transformer_encoder(seq)  # [H*W, B, hidden_dim]

        # 5) Reshape back -> [B, hidden_dim, H, W]
        encoded_map = encoded.permute(1, 2, 0).view(B, self.hidden_dim, H, W)

        # 6) Final projection
        refined = self.output_proj(encoded_map)  # [B, out_channels, H, W]
        return refined


# --------------------------
# Mini Transformer Decoder
# --------------------------
class MiniTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    A thin wrapper around PyTorch's TransformerDecoderLayer 
    (so it's easy to adjust hyperparameters).
    """
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward, dropout=dropout)


class MiniTransformerDecoder(nn.Module):
    """
    Simple Transformer Decoder that uses a set of learnable queries 
    to attend to flattened feature maps.
    """
    def __init__(self, d_model=256, nhead=8, num_layers=2, num_queries=4):
        super().__init__()
        decoder_layer = MiniTransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Learnable queries
        self.query_embed = nn.Embedding(num_queries, d_model)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        memory: [S, B, d_model], where S = H * W (or sum of multi-scale).
        Returns: [B, num_queries, d_model]
        """
        B = memory.shape[1]
        
        # Expand queries -> [num_queries, B, d_model]
        Q = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)
        
        # decode -> [num_queries, B, d_model]
        decoded = self.transformer_decoder(Q, memory)
        return decoded.permute(1, 0, 2)  # [B, num_queries, d_model]


# --------------------------
# Main Query + Transformer-based Generator
# --------------------------
class EfficientQueryEmbeddingGenerator(nn.Module):
    """
    Incorporates:
      - Multi-Resolution Fusion
      - Cross-Feature Gating
      - Channel & Spatial Attention
      - Transformer-based Dense Refinement (for more precise dense embeddings)
      - Mini Transformer Decoder (for sparse embeddings)

    Input / Output shapes remain consistent with the original.
    """
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        # Dense transformer refinement params
        dense_hidden_dim: int = 256,
        dense_num_layers: int = 2,
        dense_nhead: int = 8,
        dense_ffn_dim: int = 1024,
        # Sparse query-related params
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 2,
        num_queries: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.activation_fn = activation()

        # --------------------------------------
        # 1) Multi-Resolution Fusion (Backbone/Event)
        # --------------------------------------
        self.backbone_fusion = MultiResolutionFusion(mask_in_chans)
        self.event_fusion = MultiResolutionFusion(mask_in_chans)

        # --------------------------------------
        # 2) Cross-Feature Gating
        # --------------------------------------
        self.cross_gating = CrossFeatureGating(
            in_channels=mask_in_chans, 
            reduction=8, 
            backbone_weight=0.8, 
            event_weight=0.4
        )

        # --------------------------------------
        # 3) Channel & Spatial Attention
        # --------------------------------------
        self.channel_attention = ChannelAttention(mask_in_chans)
        self.spatial_attention = SpatialAttention()

        # --------------------------------------
        # 4) Transformer-based Dense Refinement
        # --------------------------------------
        self.dense_refiner = DenseTransformerRefiner(
            in_channels=mask_in_chans,
            hidden_dim=dense_hidden_dim,
            out_channels=embed_dim,   # final dimension of the dense embedding
            num_layers=dense_num_layers,
            nhead=dense_nhead,
            dim_feedforward=dense_ffn_dim,
        )

        # Region attention (optional)
        self.region_attention = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            self.activation_fn,
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
        )

        # --------------------------------------
        # 5) Sparse Embedder (Mini Transformer Decoder)
        # --------------------------------------
        self.mini_decoder = MiniTransformerDecoder(
            d_model=d_model, 
            nhead=nhead,
            num_layers=num_decoder_layers,
            num_queries=num_queries
        )
        # Project decoder output to embed_dim
        self.query_proj = nn.Linear(d_model, embed_dim)

        # --------------------------------------
        # Position encoding for dense output (if needed)
        # (We also have position encoding inside the DenseTransformerRefiner.)
        # If you want to replicate get_dense_pe(...) from your original,
        # you can still use a random or separate encoding layer.
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # --------------------------------------
        # Learnable scales for backbone/event
        # --------------------------------------
        self.backbone_scale = nn.Parameter(torch.ones(1))
        self.event_scale = nn.Parameter(torch.zeros(1))

    def forward(self,
                backbone_features: torch.Tensor, 
                event_features: torch.Tensor,
                high_res_features: List[torch.Tensor],
                high_res_event_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            - backbone_features: [B, mask_in_chans, H, W]
            - event_features:    [B, mask_in_chans, H, W]
            - high_res_features:        list of [B, c, Hx, Wx] (2 levels)
            - high_res_event_features:  list of [B, c, Hy, Wy] (2 levels)

        Output:
            - sparse_embeddings: [B, S, embed_dim]
            - dense_embeddings:  [B, embed_dim, H, W]
        """

        # 1) Fuse backbone & event features
        combined_backbone = self.backbone_fusion([*high_res_features, backbone_features])
        combined_event = self.event_fusion([*high_res_event_features, event_features])

        # 2) Cross gating
        gated_backbone, gated_event = self.cross_gating(combined_backbone, combined_event)

        # 3) Learnable scaling + sum
        scaled_backbone = self.backbone_scale * gated_backbone
        scaled_event = self.event_scale * gated_event
        combined_features = scaled_backbone + scaled_event

        # 4) Channel + Spatial Attention
        attn_feats = self.channel_attention(combined_features)
        attn_feats = self.spatial_attention(attn_feats)

        # 5) Transformer-based dense refinement
        #    This will yield [B, embed_dim, H, W]
        dense_embeddings = self.dense_refiner(attn_feats)

        # (Optional) region attention on refined embeddings
        region_mask = torch.sigmoid(self.region_attention(dense_embeddings))
        # If you want to incorporate region_mask back into the embeddings, you can do so:
        # e.g., dense_embeddings = dense_embeddings * (1 + region_mask)

        # 6) Sparse embeddings (query-based)
        #    Flatten *attn_feats or *dense_embeddings for the cross-attention memory
        #    Typically you'd use the same features that you want your queries to attend to:
        #    We'll use the *attention features* (before refinement) for variety
        B, C, H, W = attn_feats.shape
        memory = attn_feats.flatten(2).permute(2, 0, 1)  # [H*W, B, C]

        decoded_queries = self.mini_decoder(memory)      # [B, num_queries, d_model]
        sparse_embeddings = self.query_proj(decoded_queries)  # [B, num_queries, embed_dim]

        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self) -> torch.Tensor:
        """
        Return a random position encoding of shape [1, embed_dim, H, W],
        consistent with image_embedding_size from your original code.
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


# ------------------------------------------------------------------
# REUSE YOUR EXISTING MODULES BELOW (unchanged):
# ------------------------------------------------------------------

class MultiResolutionFusion(nn.Module):
    """
    Adapts features to a target channel dimension and fuses them.
    """
    def __init__(self, target_channels: int):
        super().__init__()
        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, target_channels, kernel_size=1),
                nn.BatchNorm2d(target_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in [32, 64, 256]  # Adjust if needed
        ])
        
        self.fusion_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(target_channels * 3, target_channels, kernel_size=1),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        normalized_weights = F.softmax(self.fusion_weights, dim=0)
        processed_features = []
        
        base_size = features_list[-1].shape[-2:]  # often largest resolution
        for adapter, feature, w in zip(self.feature_adapters, features_list, normalized_weights):
            x = adapter(feature)
            if x.shape[-2:] != base_size:
                x = F.interpolate(x, size=base_size, mode='bilinear', align_corners=False)
            processed_features.append(w * x)
        
        fused = torch.cat(processed_features, dim=1)
        return self.fusion_conv(fused)


class CrossFeatureGating(nn.Module):
    """
    Gating mechanism for combined_backbone vs. combined_event.
    """
    def __init__(self, in_channels=256, reduction=8, backbone_weight=0.8, event_weight=0.4):
        super().__init__()
        self.gate_event = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate_backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.backbone_weight = backbone_weight
        self.event_weight = event_weight

    def forward(self, combined_backbone, combined_event):
        gate_event = self.gate_event(combined_backbone)
        gated_combined_event = combined_event * gate_event
        
        gate_backbone = self.gate_backbone(combined_event)
        gated_combined_backbone = combined_backbone * gate_backbone
        
        # Weighted residual
        gated_combined_backbone = self.backbone_weight * gated_combined_backbone + \
                                  (1 - self.backbone_weight) * combined_backbone
        gated_combined_event = self.event_weight * gated_combined_event + \
                               (1 - self.event_weight) * combined_event
        return gated_combined_backbone, gated_combined_event


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    """
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
    """
    A simple spatial attention using max+avg pooling across channels.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


# Minimal residual block (unchanged)
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


# A random position embedding, same as your original reference
from prompt_gen.backbone.position_encoding import PositionEmbeddingRandom

# Initialization Utility
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
