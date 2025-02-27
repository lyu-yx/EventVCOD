import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Type

# Reuse (or adapt) your auxiliary modules:
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
    """
    Stub for your position embedding layer.
    Modify or replace with your actual implementation.
    """
    def __init__(self, num_pos_feats: int = 64):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, shape: Tuple[int, int]) -> torch.Tensor:
        """
        shape: (height, width)
        Return a random embedding of size [C, H, W].
        """
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
    """
    Simple GRU-based aggregator that processes a sequence of future-frame features
    and returns a single summary vector (the final hidden state).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            input_dim:  Feature dimension of each time step (e.g., 256 below).
            hidden_dim: Internal GRU hidden dimension.
            output_dim: The dimension we want in the final summary (e.g., mask_in_chans).
        """
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, 4096, 256]  (N = number of future frames, seq_len=4096, input_dim=256)
           or possibly [N, 4096, 1, 256] that you reshape to [N, 4096, 256].
        
        Returns:
            summary: [1, output_dim, 1, 1]
                - The final hidden-state vector (per batch=1 in this snippet)
                  re-shaped to [1, out_dim, 1, 1] so you can broadcast.
        """
        # GRU expects [batch_size, seq_len, input_dim].
        # Here, "batch_size" = N (# of future frames),
        # "seq_len" = 4096, and "input_dim" = 256 in this example.

        # x shape might be [N, 4096, 1, 256], so first squeeze the middle "1":
        # e.g. x = x.squeeze(2) -> [N, 4096, 256] if that's your storage format.
        if x.dim() == 4 and x.shape[2] == 1:
            x = x.squeeze(2)  # -> [N, 4096, 256]

        # Pass entire sequence through GRU
        out, hidden = self.gru(x)  
        # out:    [N, 4096, hidden_dim]  (every time step's hidden state)
        # hidden: [1, N, hidden_dim]     (final hidden state)

        # If you just want the final hidden state:
        final_state = hidden[-1]  # shape [N, hidden_dim]

        # Project to the desired output dimension
        summary_vector = self.out_fc(final_state)  # [N, output_dim]

        # Example: Suppose you have only 1 "batch" of future frames,
        # then N=some_frames_count. Typically you'd want B*N if you
        # process across a real batch dimension. Adapt as needed.

        # Reshape so we can broadcast into [B, output_dim, H, W] later.
        # For demonstration: returning [1, output_dim, 1, 1]
        summary_vector = summary_vector.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # shape -> [1, N, output_dim, 1, 1], or if N=1, [1, output_dim, 1, 1]
        # Adjust to your batch size as necessary.

        return summary_vector
    
class SpatialTemporalFeatureAggregator(nn.Module):
    """
    GRU-based aggregator that processes a sequence of future-frame features with spatial dimensions.
    
    Expected input shape: [S, T, B, F], where S = H*W (e.g. 4096 for 64x64) and F is feature dimension.
    For each spatial location, the GRU aggregates its T time steps.
    
    Returns:
        Aggregated features of shape [B, output_dim, H, W].
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, spatial_size: Tuple[int, int]):
        super().__init__()
        self.spatial_size = spatial_size  # e.g., (64, 64)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [S, T, B, F], where S should be H*W.
        # print("in rnn, x shape:", x.shape)
        S, T, B, F = x.shape
        # print("x shape:", x.shape)
        # Permute to bring spatial dimension (S) forward:
        # New shape: [B, S, T, F] so that each spatial location has its own temporal sequence.
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # Merge B and S dimensions: [B * S, T, F]
        x = x.view(B * S, T, F)
        
        # Run GRU: we only need the final hidden state for each sequence.
        # hidden shape: [num_layers, B*S, hidden_dim]. We take the last layer.
        _, hidden = self.gru(x)
        final_hidden = hidden[-1]  # shape: [B * S, hidden_dim]
        # print("final_hidden shape:", final_hidden.shape)
        # Project to the desired output channels.
        proj = self.out_fc(final_hidden)  # shape: [B * S, output_dim]
        # print("proj shape:", proj.shape)
        # Reshape back to [B, S, output_dim]
        proj = proj.view(B, S, -1)
        # print("proj shape:", proj.shape)
        # Reshape S back to spatial dimensions: [B, H, W, output_dim]
        H, W = self.spatial_size
        # print("H, W, B:", H, W, B)
        proj = proj.view(B, H, W, -1)
        # print("proj shape:", proj.shape)
        # Permute to [B, output_dim, H, W]
        proj = proj.permute(0, 3, 1, 2).contiguous()
        return proj

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

class SimpleHighResFusion(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        # Two separate "adapters"
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
        # If you also have 256-ch features, you could just pass them through an identity or
        # create an adapter_256.

    def forward(self, feat, fused_accumulator):
        """
        feat: [B, C, H, W], C could be 32 or 64 (or something else)
        fused_accumulator: the running sum or fused result, shape [B, 256, H, W]
        """
        in_channels = feat.shape[1]
        if in_channels == 32:
            feat = self.adapter_32(feat)  # -> [B, 256, H, W]
        elif in_channels == 64:
            feat = self.adapter_64(feat)  # -> [B, 256, H, W]
        else:
            raise ValueError(f"Unsupported input channels: {in_channels}")

        # Then add it to the accumulator
        fused_accumulator += feat
        return fused_accumulator

class HighResAlignment(nn.Module):
    """
    Learnable alignment block to downsample high-res features 
    to the target resolution and adjust channel dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 input_size: Tuple[int, int], target_size: Tuple[int, int],
                 norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels),):
        super().__init__()
        # Calculate the necessary stride based on input and target size.
        stride_h = input_size[0] // target_size[0]
        stride_w = input_size[1] // target_size[1]
        # Use a single convolution if the stride is consistent; otherwise, chain multiple layers.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=(stride_h, stride_w), padding=1)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))
    

class EmbeddingGenerator(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],  # e.g., (64, 64)
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.activation = activation()

        # Backbone and event minimal processing.
        self.backbone_block = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
        )
        self.event_block = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
        )

        # Alignment modules for high-res features.
        # Assume high_res_features: [B, 32, 256, 256] and [B, 64, 128, 128]
        self.align_backbone_32 = HighResAlignment(
            in_channels=32, out_channels=256,
            input_size=(256, 256), target_size=image_embedding_size,
            norm_layer=norm_layer,
        )
        self.align_backbone_64 = HighResAlignment(
            in_channels=64, out_channels=256,
            input_size=(128, 128), target_size=image_embedding_size,
            norm_layer=norm_layer,
        )
        self.align_event_32 = HighResAlignment(
            in_channels=32, out_channels=256,
            input_size=(256, 256), target_size=image_embedding_size,
            norm_layer=norm_layer,
        )
        self.align_event_64 = HighResAlignment(
            in_channels=64, out_channels=256,
            input_size=(128, 128), target_size=image_embedding_size,
            norm_layer=norm_layer,
        )

        # Fusion layer for high-res features.
        self.highres_fusion_fuse = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            norm_layer(256),
            nn.ReLU(inplace=True)
        )

        # Future feature fusion module (learnable fusion instead of addition).
        self.future_fusion = nn.Sequential(
            nn.Conv2d(mask_in_chans * 2, mask_in_chans, kernel_size=1),
            norm_layer(mask_in_chans),
            nn.ReLU(inplace=True)
        )

        # Future feature aggregator (GRU-based).
        self.video_feature_aggregator = nn.Sequential(
            nn.Conv1d(4096, mask_in_chans, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mask_in_chans, mask_in_chans, kernel_size=1),
            nn.Sigmoid()
        )
        self.video_feature_rnn = SpatialTemporalFeatureAggregator(
            input_dim=256,
            hidden_dim=mask_in_chans,
            output_dim=mask_in_chans,
            spatial_size=image_embedding_size
        )

        # Attention modules.
        self.channel_attention = ChannelAttention(mask_in_chans)
        self.spatial_attention = SpatialAttention()
        self.region_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=1),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=3, padding=1)
        )

        # Enhanced dense embedder with extra residual and self-attention blocks.
        self.dense_embedder = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            SelfAttentionBlock(mask_in_chans),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        )

        self.sparse_embedder = nn.Sequential(
            PyramidPooling(mask_in_chans),
            nn.Conv2d(mask_in_chans * 5, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim + 1, embed_dim, kernel_size=3, padding=1),
            norm_layer(embed_dim),
            self.activation
        )
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.dense_residual = nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)

    def forward(
        self, 
        backbone_features: torch.Tensor, 
        event_features: torch.Tensor,
        high_res_features: List[torch.Tensor],
        high_res_event_features: List[torch.Tensor],
        cur_video: Dict[str, List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = backbone_features.shape

        # Process backbone and event features.
        backbone_processed = self.backbone_block(backbone_features)
        event_processed = self.event_block(event_features)

        # Process and align high-res backbone features.
        aligned_backbone = []
        for feat in high_res_features:
            if feat.shape[1] == 32:
                aligned_backbone.append(self.align_backbone_32(feat))
            elif feat.shape[1] == 64:
                aligned_backbone.append(self.align_backbone_64(feat))
            else:
                raise ValueError(f"Unsupported backbone high-res channels: {feat.shape[1]}")
        fused_backbone_highres = sum(aligned_backbone)

        # Process and align high-res event features.
        aligned_event = []
        for feat in high_res_event_features:
            if feat.shape[1] == 32:
                aligned_event.append(self.align_event_32(feat))
            elif feat.shape[1] == 64:
                aligned_event.append(self.align_event_64(feat))
            else:
                raise ValueError(f"Unsupported event high-res channels: {feat.shape[1]}")
        fused_event_highres = sum(aligned_event)

        # Fuse high-res features via concatenation and fusion layer.
        fused_highres = self.highres_fusion_fuse(torch.cat([fused_backbone_highres, fused_event_highres], dim=1))

        # Combine backbone, event, and high-res features.
        combined_features = backbone_processed + 0.3 * event_processed + fused_highres

        # Incorporate future features with improved fusion strategy.
        if cur_video is not None:
            vision_feats = cur_video.get("vision_feats", None)
            if vision_feats is not None and len(vision_feats) > 0:
                feats_stack = torch.stack(
                    [feat[2].unsqueeze(1) if feat[2].dim() == 2 else feat[2] for feat in vision_feats],
                    dim=1
                )
                future_summary = self.video_feature_rnn(feats_stack)  # [B, mask_in_chans, 1, 1]
                # Expand future summary spatially.
                future_summary_expanded = future_summary.expand_as(combined_features)
                # Fuse via concatenation and learnable fusion.
                combined_features = self.future_fusion(torch.cat([combined_features, future_summary_expanded], dim=1))

        # Apply attention modules.
        features = self.channel_attention(combined_features)
        features = self.spatial_attention(features)

        # Generate dense embeddings with enhanced embedder and refinement.
        region_attention = torch.sigmoid(self.region_attention(features))
        dense_intermediate = self.dense_embedder(features)
        dense_refined = self.refinement(torch.cat([dense_intermediate, region_attention], dim=1))
        residual_dense = self.dense_residual(features)
        dense_embeddings = dense_refined + residual_dense

        # Generate sparse embeddings.
        sparse_embeddings = self.sparse_embedder(features)
        sparse_embeddings = sparse_embeddings.flatten(2).transpose(1, 2)

        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self) -> torch.Tensor:
        pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)
        device = next(self.parameters()).device
        return pe.to(device)