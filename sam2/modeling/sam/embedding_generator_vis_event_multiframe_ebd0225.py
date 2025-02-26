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
    def __init__(self, embed_dim, image_embedding_size, temporal_dim=256):
        super(EmbeddingGenerator, self).__init__()
        H, W = image_embedding_size  # spatial size of image embedding (e.g., 64x64 for 1024 input with stride 16)
        C = embed_dim                # number of channels in image embedding

        # Depthwise separable convolution to process backbone features (reduces params and comp.)
        self.backbone_dw_conv = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)  # depthwise conv
        self.backbone_pw_conv = nn.Conv2d(C, C, kernel_size=1, padding=0, bias=True)             # pointwise conv

        # Depthwise separable convolution to process event features (if event features have same channel dim as C)
        self.event_dw_conv = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        self.event_pw_conv = nn.Conv2d(C, C, kernel_size=1, padding=0, bias=True)

        # Feature fusion conv: combine backbone + event (use depthwise separable conv to reduce redundancy)
        self.fusion_dw_conv = nn.Conv2d(2*C, 2*C, kernel_size=3, padding=1, groups=2*C, bias=False)
        self.fusion_pw_conv = nn.Conv2d(2*C, C, kernel_size=1, padding=0, bias=True)

        # GRU-based temporal aggregator (ConvGRU for spatial consistency)
        self.temporal_gru = nn.GRU(
            input_size=C, hidden_size=C, batch_first=True, bidirectional=False
        )
        # The GRU will be applied per-pixel across time by flattening spatial dims. We use ConvGRU conceptually:
        # For efficiency, we use 1x1 conv to update hidden states as a proxy to ConvGRU gating.
        self.conv_update = nn.Conv2d(C * 2, C, kernel_size=1)   # update gate (using concatenated input and prev state)
        self.conv_reset  = nn.Conv2d(C * 2, C, kernel_size=1)   # reset gate
        self.conv_out    = nn.Conv2d(C * 2, C, kernel_size=1)   # output gate

        # Positional encoding for image (pre-computed sine pe or learned pe)
        self.dense_pos_enc = nn.Parameter(torch.randn(1, C, H, W), requires_grad=False)  # fixed positional encoding

    def get_dense_pe(self):
        """Return dense positional encoding for the image."""
        return self.dense_pos_enc

    def forward(self, backbone_features, event_features, high_res_features=None, high_res_event_features=None, state=None):
        """
        Generate dense embeddings by fusing image backbone features and event features.
        - backbone_features: Tensor of shape [B, C, H, W] from image encoder.
        - event_features: Tensor of shape [B, C, H, W] representing multi-frame or event data aligned with image.
        - high_res_features: (Optional) high-resolution image features [B, C_high, H*2, W*2] for refinement.
        - high_res_event_features: (Optional) high-resolution event features [B, C_high, H*2, W*2].
        - state: (Optional) previous hidden state for temporal GRU (to propagate info across frames).
        """
        # Depthwise separable conv on backbone features (efficient spatial processing)
        x_img = self.backbone_pw_conv(self.backbone_dw_conv(backbone_features))
        # Depthwise separable conv on event features
        x_event = self.event_pw_conv(self.event_dw_conv(event_features))

        # Fuse image and event features (concatenate along channel and apply depthwise separable conv)
        x_fused = torch.cat([x_img, x_event], dim=1)                # [B, 2C, H, W]
        x_fused = self.fusion_pw_conv(self.fusion_dw_conv(x_fused))  # [B, C, H, W]

        # If high-res features are provided, fuse with low-res using an efficient strategy:
        if high_res_features is not None:
            # Reduce high-res spatial processing by downsampling to match low-res, or using pointwise conv to reduce channels
            high_res_down = F.avg_pool2d(high_res_features, kernel_size=2)  # downsample high-res to [B, C_high, H, W]
            # Project high_res_down to C channels via 1x1 conv (pointwise) to avoid heavy high-res convolution
            proj_high_res = F.conv2d(high_res_down, weight=torch.eye(x_fused.shape[1]).view(x_fused.shape[1], x_fused.shape[1], 1, 1).to(x_fused.device))
            # Add or concatenate the projected high-res to fused features
            x_fused = x_fused + proj_high_res  # element-wise add for refinement detail

        # Temporal aggregation with ConvGRU concept for sequential frames (spatial consistency)
        # Flatten spatial dims to sequence length for GRU or apply pixel-wise GRU.
        B, C, H, W = x_fused.shape
        x_seq = x_fused.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C] treat each pixel as sequence element
        if state is None:
            h0 = torch.zeros(1, B * H * W, C, device=x_fused.device)  # initial hidden state (zero)
        else:
            # Use provided state or flatten it
            h0 = state.view(1, B * H * W, C)
        # Run GRU on the flattened spatial sequence (maintaining temporal continuity per pixel)
        out_seq, hN = self.temporal_gru(x_seq, h0)  # out_seq: [B, H*W, C]
        x_temporal = out_seq.permute(0, 2, 1).view(B, C, H, W)  # reshape back to [B, C, H, W]

        # Final embedding is the temporally aggregated feature plus positional encoding
        dense_embedding = x_temporal + self.dense_pos_enc  # ensure positional info is added

        # To optimize memory: free large intermediate tensors (especially high-res) if not needed further
        del x_img, x_event, x_fused, x_seq, out_seq
        if high_res_features is not None: del high_res_features, high_res_down, proj_high_res
        torch.cuda.empty_cache()  # hint to free GPU memory

        return dense_embedding, dense_embedding  # returning (sparse_embed_dummy, dense_embedding) to match expected output


    def get_dense_pe(self) -> torch.Tensor:
        """Reproduce the SAM-style positional embedding for dense features."""
        # Create the positional embedding on CPU
        pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)

        # Move it to the device of the current module's parameters
        device = next(self.parameters()).device
        pe = pe.to(device)

        return pe
