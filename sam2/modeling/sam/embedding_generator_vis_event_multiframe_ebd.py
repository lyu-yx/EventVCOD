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

class EmbeddingGenerator(nn.Module):
    """
    A simplified embedding generator that:
    1) Processes backbone and event features with minimal layers.
    2) Fuses high-resolution features (both backbone & event) in a single step.
    3) Optionally incorporates future-frame features from cur_video.
    4) Produces sparse and dense embeddings.
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

        # -----------------------
        # 1. Simple backbone path
        # -----------------------
        self.backbone_block = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
        )

        # ----------------------
        # 2. Simple event path
        # ----------------------
        self.event_block = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
        )

        # -----------------------------------------------
        # 3. Single-step high-resolution feature fusion
        #    (both backbone high_res and event high_res)
        # -----------------------------------------------
        self.highres_fusion_conv = SimpleHighResFusion(out_channels=256)

        # -------------------------------------------------
        # 4. Optional aggregator for future-frame features
        #    from cur_video["vision_feats"] & ["vision_feats_event"]
        # -------------------------------------------------
        # We'll pool them with a small MLP to incorporate into the 2D feature map.
        self.video_feature_aggregator = nn.Sequential(
            nn.Conv1d(4096, mask_in_chans, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mask_in_chans, mask_in_chans, kernel_size=1),
            nn.Sigmoid()  # you can choose an activation that best fits your usage
        )

        self.video_feature_rnn = SpatialTemporalFeatureAggregator(
            input_dim=256,         # feature dimension per spatial location at each time step
            hidden_dim=mask_in_chans,  # internal GRU hidden dimension
            output_dim=mask_in_chans,  # desired output channels (1024)
            spatial_size=image_embedding_size  # e.g., (64, 64)
        )

        # -------------------------------------------
        # 5. Attention modules + region attention
        # -------------------------------------------
        self.channel_attention = ChannelAttention(mask_in_chans)
        self.spatial_attention = SpatialAttention()
        self.region_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=1),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=3, padding=1)
        )

        # --------------------------------
        # 6. Dense & Sparse embeddings
        # --------------------------------
        self.dense_embedder = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        
        self.sparse_embedder = nn.Sequential(
            PyramidPooling(mask_in_chans),
            nn.Conv2d(mask_in_chans * 5, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )

        # Simple refinement that combines region attention with dense embeddings
        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim + 1, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            self.activation
        )

        # Position embedding
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

    def forward(
        self, 
        backbone_features: torch.Tensor, 
        event_features: torch.Tensor,
        high_res_features: List[torch.Tensor],
        high_res_event_features: List[torch.Tensor],
        cur_video: Dict[str, List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            backbone_features: [B, mask_in_chans, H, W]
            event_features: [B, mask_in_chans, H, W]
            high_res_features: List of Tensors, e.g. 2 Tensors
                in shapes [B, 32, 256, 256] and [B, 64, 128, 128]
            high_res_event_features: List of Tensors, same structure as above
            cur_video: dict with keys ["vision_feats", "vision_pos_embeds", 
                                       "vision_feats_event", "vision_pos_embeds_event"] 
                - each is a list (or tensor) of shape [4096, 1, 256], 
                  or zero for out-of-range frames.
        Returns:
            sparse_embeddings: [B, (H*W), embed_dim]  # after flatten+transpose
            dense_embeddings:  [B, embed_dim, H, W]
        """
        B, C, H, W = backbone_features.shape

        # 1. Minimal backbone processing
        backbone_processed = self.backbone_block(backbone_features)

        # 2. Minimal event processing
        event_processed = self.event_block(event_features)

        # 3. High-res fusion
        fused_highres = torch.zeros((B, 256, H, W), device=backbone_features.device)
        for feat in high_res_features:
            # Interpolate to match output H,W
            if feat.shape[-2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)

            fused_highres = self.highres_fusion_conv(feat, fused_highres)

        for feat in high_res_event_features:
            if feat.shape[-2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)

            fused_highres = self.highres_fusion_conv(feat, fused_highres)

        combined_features = backbone_processed + event_processed + fused_highres
        # print("combined_features shape:", combined_features.shape)
        # 4. Use the RNN aggregator if cur_video is provided
        if cur_video is not None:
            vision_feats = cur_video.get("vision_feats", None)
            if vision_feats is not None and len(vision_feats) > 0:
                # Assume that each vision_feats entry is a tuple where the third element is a tensor
                # of shape [B, 4096, 256]. We stack along a new time dimension.
                feats_stack = torch.stack(
                    [feat[2].unsqueeze(1) if feat[2].dim() == 2 else feat[2] for feat in vision_feats], dim=1
                )
                # Pass through the GRU aggregator.
                # The aggregator will first average over the spatial dimension (4096) to obtain [B, T, 256]
                # and then output a summary of shape [B, mask_in_chans, 1, 1] (mask_in_chans should be 256).
                future_summary = self.video_feature_rnn(feats_stack)
                
                # Combine the future summary with the combined_features
                combined_features += future_summary

        # 5. Attention
        features = self.channel_attention(combined_features)
        features = self.spatial_attention(features)

        # 6. Region attention + dense/sparse embeddings
        region_attention = torch.sigmoid(self.region_attention(features))
        dense_embeddings = self.dense_embedder(features)
        dense_embeddings = self.refinement(torch.cat([dense_embeddings, region_attention], dim=1))

        sparse_embeddings = self.sparse_embedder(features)
        sparse_embeddings = sparse_embeddings.flatten(2).transpose(1, 2)

        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self) -> torch.Tensor:
        """Reproduce the SAM-style positional embedding for dense features."""
        # Create the positional embedding on CPU
        pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)

        # Move it to the device of the current module's parameters
        device = next(self.parameters()).device
        pe = pe.to(device)

        return pe
