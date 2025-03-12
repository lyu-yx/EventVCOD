import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Type

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
        input_image_size: Tuple[int, int],  # e.g., (1024, 1024)
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels),
    ) -> None:
        super().__init__()
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.activation = activation()
        

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Reuse the backbone and event feature processing blocks from the original
        self.backbone_block = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
        )
        
        self.event_block = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1, groups=4),
            norm_layer(mask_in_chans),
            self.activation,
        )

        # Motion-guided attention for event features
        self.motion_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans//2, kernel_size=1),
            norm_layer(mask_in_chans//2),
            self.activation,
            nn.Conv2d(mask_in_chans//2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Keep the same alignment modules for high-res features
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

        # Improved feature fusion with channel-wise attention
        self.highres_fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            ChannelAttention(256, reduction=8),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )
        
        # Reuse temporal feature aggregators
        self.vision_feature_rnn = SpatialTemporalFeatureAggregator(
            input_dim=256,
            hidden_dim=512,
            output_dim=mask_in_chans,
            spatial_size=image_embedding_size
        )
        
        self.vision_event_feature_rnn = SpatialTemporalFeatureAggregator(
            input_dim=256,
            hidden_dim=512,
            output_dim=mask_in_chans,
            spatial_size=image_embedding_size
        )
        
        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=1),
            norm_layer(mask_in_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_in_chans, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Process future frame event features
        self.event_future_processor = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            nn.ReLU(inplace=True),
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
        )
        
        # Fusion for regular and event future features
        self.future_event_fusion = nn.Sequential(
            nn.Conv2d(mask_in_chans * 2, mask_in_chans, kernel_size=1),
            norm_layer(mask_in_chans),
            nn.ReLU(inplace=True),
        )
        
        # Temporal consistency module
        self.temporal_consistency = nn.Sequential(
            nn.Conv2d(mask_in_chans * 2, mask_in_chans, kernel_size=1),
            norm_layer(mask_in_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            nn.ReLU(inplace=True),
        )

        # Attention modules
        self.channel_attention = ChannelAttention(mask_in_chans)
        self.spatial_attention = SpatialAttention()
        
        # Multi-scale region attention for better boundary modeling
        self.region_attention = nn.ModuleDict({
            'scale1': nn.Sequential(
                nn.Conv2d(mask_in_chans, mask_in_chans // 4, kernel_size=1),
                self.activation,
                nn.Conv2d(mask_in_chans // 4, 1, kernel_size=3, padding=1)
            ),
            'scale2': nn.Sequential(
                nn.Conv2d(mask_in_chans, mask_in_chans // 4, kernel_size=1),
                self.activation,
                nn.Conv2d(mask_in_chans // 4, 1, kernel_size=5, padding=2)
            ),
            'scale3': nn.Sequential(
                nn.Conv2d(mask_in_chans, mask_in_chans // 4, kernel_size=1),
                self.activation,
                nn.Conv2d(mask_in_chans // 4, 1, kernel_size=7, padding=3)
            )
        })
        
        # Boundary-aware attention
        self.boundary_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=1),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, mask_in_chans // 2, kernel_size=3, padding=1, groups=4),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Position embedding processor for future frames
        self.pos_embed_processor = nn.Sequential(
            nn.Linear(256, 128),  # Assuming pos embeds are 256-dim
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        
        # Fusion module for vision features and position embeddings
        self.vision_pos_fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),  # Combine vision feat + processed pos embed
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )
        
        # New hierarchical mask prediction components
        # First level - low resolution mask (64x64)
        self.mask_predictor_level1 = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, mask_in_chans//2, kernel_size=3, padding=1),
            norm_layer(mask_in_chans//2),
            self.activation,
            nn.Conv2d(mask_in_chans//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Second level - medium resolution mask (256x256)
        self.mask_refiner_level2 = nn.Sequential(
            nn.Conv2d(mask_in_chans + 1, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, mask_in_chans//2, kernel_size=3, padding=1),
            norm_layer(mask_in_chans//2),
            self.activation,
            nn.Conv2d(mask_in_chans//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Third level - high resolution mask (1024x1024)
        self.mask_refiner_level3 = nn.Sequential(
            nn.Conv2d(mask_in_chans//2 + 1, mask_in_chans//2, kernel_size=3, padding=1),
            norm_layer(mask_in_chans//2),
            self.activation,
            nn.Conv2d(mask_in_chans//2, mask_in_chans//4, kernel_size=3, padding=1),
            norm_layer(mask_in_chans//4),
            self.activation,
            nn.Conv2d(mask_in_chans//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature extractors for different resolutions
        self.feature_extractor_level2 = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation)
        )
        
        self.feature_extractor_level3 = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans//2, kernel_size=3, padding=1),
            norm_layer(mask_in_chans//2),
            self.activation,
            nn.Conv2d(mask_in_chans//2, mask_in_chans//2, kernel_size=3, padding=1),
            norm_layer(mask_in_chans//2),
            self.activation
        )
        
        # Edge enhancement module for refinement
        self.edge_enhancement = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _safe_process_future_features(self, features_list, rnn_module):
        """Process future frame features safely with proper error handling"""
        if not features_list or len(features_list) == 0:
            return None
            
        feats_stack = []
        for feat in features_list:
            if isinstance(feat, tuple) and len(feat) > 2:
                if feat[2].dim() == 2:
                    # Reshape if needed
                    feats_stack.append(feat[2].unsqueeze(1))
                else:
                    feats_stack.append(feat[2])
            
        if not feats_stack:
            return None
            
        # Check shapes and stack if consistent
        if all(f.shape == feats_stack[0].shape for f in feats_stack):
            try:
                feats_stack = torch.stack(feats_stack, dim=0)
                return rnn_module(feats_stack)
            except Exception as e:
                # Safely handle any errors during processing
                print(f"Error processing future features: {e}")
                return None
        return None

    def _process_vision_with_pos_embeds(self, vision_feats, pos_embeds):
        """Process vision features with their position embeddings"""
        if not vision_feats or not pos_embeds or len(vision_feats) != len(pos_embeds):
            return None
            
        processed_features = []
        
        for i, (feat, pos) in enumerate(zip(vision_feats, pos_embeds)):
            if isinstance(feat, tuple) and len(feat) > 2 and isinstance(pos, tuple) and len(pos) > 2:
                # Process each feature with its positional embedding
                feat_tensor = feat[2]
                pos_tensor = pos[2]
                
                # Ensure shapes are compatible
                if feat_tensor.dim() >= 2 and pos_tensor.dim() >= 2:
                    # Process positional embedding
                    processed_pos = self.pos_embed_processor(pos_tensor)
                    
                    # Handle different dimensions (e.g., if feat is 2D and pos is 2D)
                    if feat_tensor.dim() == 2 and pos_tensor.dim() == 2:
                        # Combine feature and position embedding
                        combined = torch.cat([feat_tensor, processed_pos], dim=-1)
                        fused = self.vision_pos_fusion(combined)
                        processed_features.append(fused)
                    
        if not processed_features:
            return None
            
        # Stack processed features if they have the same shape
        if all(f.shape == processed_features[0].shape for f in processed_features):
            try:
                return torch.stack(processed_features, dim=0)
            except Exception as e:
                print(f"Error stacking processed features: {e}")
                return None
        return None
    
    def _enhace_edges(self, mask):
        """Enhance edges in a mask using gradient information"""
        # Calculate sobel gradients
        def sobel_edges(x):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            
            grad_x = F.conv2d(x, sobel_x, padding=1)
            grad_y = F.conv2d(x, sobel_y, padding=1)
            
            return torch.sqrt(grad_x**2 + grad_y**2)
        
        # Get edge information
        edges = sobel_edges(mask)
        
        # Process edges with edge enhancement module
        enhanced_edges = self.edge_enhancement(edges)
        
        # Combine with original mask using edge-aware blending
        return 0.8 * mask + 0.2 * enhanced_edges

    def forward(
        self, 
        backbone_features: torch.Tensor, 
        event_features: torch.Tensor,
        high_res_features: List[torch.Tensor],
        high_res_event_features: List[torch.Tensor],
        cur_video: Optional[Dict[str, List[torch.Tensor]]] = None
    ) -> torch.Tensor:
        B, C, H, W = backbone_features.shape

        # Process backbone features
        backbone_processed = self.backbone_block(backbone_features)
        
        # Process event features with motion-guided attention
        event_processed = self.event_block(event_features)
        motion_map = self.motion_attention(event_features)
        event_enhanced = event_processed * motion_map
        
        # Process and align high-res backbone features
        aligned_backbone = []
        for feat in high_res_features:
            if feat.shape[1] == 32:
                aligned_backbone.append(self.align_backbone_32(feat))
            elif feat.shape[1] == 64:
                aligned_backbone.append(self.align_backbone_64(feat))
            else:
                continue  # Skip incompatible features
        
        if aligned_backbone:
            fused_backbone_highres = sum(aligned_backbone)
        else:
            fused_backbone_highres = torch.zeros_like(backbone_processed)

        # Process and align high-res event features
        aligned_event = []
        for feat in high_res_event_features:
            if feat.shape[1] == 32:
                aligned_event.append(self.align_event_32(feat))
            elif feat.shape[1] == 64:
                aligned_event.append(self.align_event_64(feat))
            else:
                continue  # Skip incompatible features
        
        if aligned_event:
            fused_event_highres = sum(aligned_event)
        else:
            fused_event_highres = torch.zeros_like(backbone_processed)

        # Fuse high-res features with improved fusion module
        highres_combined = torch.cat([fused_backbone_highres, fused_event_highres], dim=1)
        fused_highres = self.highres_fusion(highres_combined)

        # Initialize future feature variables
        future_vision_features = None
        future_event_features = None
        
        # Extract and process all available future frame information
        if cur_video is not None:
            # Process standard vision features
            if "vision_feats" in cur_video and len(cur_video["vision_feats"]) > 0:
                future_vision_features = self._safe_process_future_features(
                    cur_video["vision_feats"], 
                    self.vision_feature_rnn
                )
                
            # Process event vision features
            if "vision_feats_event" in cur_video and len(cur_video["vision_feats_event"]) > 0:
                future_event_features = self._safe_process_future_features(
                    cur_video["vision_feats_event"], 
                    self.vision_event_feature_rnn
                )
                
            # Process vision features with their position embeddings (if not already processed)
            if future_vision_features is None and "vision_feats" in cur_video and "vision_pos_embeds" in cur_video:
                processed_with_pos = self._process_vision_with_pos_embeds(
                    cur_video["vision_feats"],
                    cur_video["vision_pos_embeds"]
                )
                
                if processed_with_pos is not None:
                    # Further process these features if needed
                    try:
                        future_vision_features = self.vision_feature_rnn(processed_with_pos)
                    except Exception:
                        future_vision_features = None
                        
            # Similarly process event features with their position embeddings as backup
            if future_event_features is None and "vision_feats_event" in cur_video and "vision_pos_embeds_event" in cur_video:
                processed_event_with_pos = self._process_vision_with_pos_embeds(
                    cur_video["vision_feats_event"],
                    cur_video["vision_pos_embeds_event"]
                )
                
                if processed_event_with_pos is not None:
                    try:
                        future_event_features = self.vision_event_feature_rnn(processed_event_with_pos)
                    except Exception:
                        future_event_features = None

        # Fuse future vision and event features if both are available
        fused_future_features = None
        if future_vision_features is not None and future_event_features is not None:
            # Process future event features
            future_event_processed = self.event_future_processor(future_event_features)
            
            # Apply temporal attention to both feature types
            vision_temp_attn = self.temporal_attention(future_vision_features)
            event_temp_attn = self.temporal_attention(future_event_processed)
            
            # Weight features by attention
            future_vision_features = future_vision_features * vision_temp_attn
            future_event_processed = future_event_processed * event_temp_attn
            
            # Fuse the two feature types
            fused_future_features = self.future_event_fusion(
                torch.cat([future_vision_features, future_event_processed], dim=1)
            )
        elif future_vision_features is not None:
            # Apply temporal attention to vision features
            vision_temp_attn = self.temporal_attention(future_vision_features)
            fused_future_features = future_vision_features * vision_temp_attn
        elif future_event_features is not None:
            # Process and apply attention to event features
            future_event_processed = self.event_future_processor(future_event_features)
            event_temp_attn = self.temporal_attention(future_event_processed)
            fused_future_features = future_event_processed * event_temp_attn

        # Combine features with future information if available
        if fused_future_features is not None:
            # Concatenate and use learnable fusion
            combined_features = self.temporal_consistency(
                torch.cat([backbone_processed + 0.3 * event_enhanced + fused_highres, 
                          fused_future_features], dim=1)
            )
        else:
            combined_features = backbone_processed + 0.3 * event_enhanced + fused_highres

        # Apply attention modules
        features = self.channel_attention(combined_features)
        features = self.spatial_attention(features)

        # Multi-scale region attention for better object localization
        region_attention_1 = torch.sigmoid(self.region_attention['scale1'](features))
        region_attention_2 = torch.sigmoid(self.region_attention['scale2'](features))
        region_attention_3 = torch.sigmoid(self.region_attention['scale3'](features))
        
        # Fuse region attention maps - differently weighted based on scale relevance
        region_attention = (0.5 * region_attention_1 + 0.3 * region_attention_2 + 0.2 * region_attention_3)
        
        # Boundary-aware attention
        boundary_map = self.boundary_attention(features)
        
        # Motion consistency map derived from event features
        motion_consistency = torch.sigmoid(motion_map)
        
        region_enhanced_features = features * region_attention

        # Apply boundary attention to help with precise edge detection
        boundary_enhanced_features = region_enhanced_features * boundary_map

        # Apply motion consistency from event data for temporal coherence
        motion_enhanced_features = boundary_enhanced_features * motion_consistency
                
        # Hierarchical mask prediction - Level 1 (64x64)
        mask_level1 = self.mask_predictor_level1(motion_enhanced_features)
        
        # Prepare features for the second level
        features_level2 = self.feature_extractor_level2(motion_enhanced_features)
        
        # Upsample mask and features to 256x256
        mask_level1_upsampled = F.interpolate(mask_level1, size=(128, 128), mode='bilinear', align_corners=False)
        features_level2_upsampled = F.interpolate(features_level2, size=(128, 128), mode='bilinear', align_corners=False)
        
        # Concatenate upsampled mask with features
        combined_level2 = torch.cat([features_level2_upsampled, mask_level1_upsampled], dim=1)
        
        # Predict mask at second level (256x256)
        mask_level2 = self.mask_refiner_level2(combined_level2)
        
        # Prepare features for the third level
        features_level3 = self.feature_extractor_level3(features_level2_upsampled)
        
        # Upsample mask and features to final size (1024x1024)
        mask_level2_upsampled = F.interpolate(mask_level2, size=(256, 256), mode='bilinear', align_corners=False)
        features_level3_upsampled = F.interpolate(features_level3, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Concatenate upsampled mask with features for final refinement
        combined_level3 = torch.cat([features_level3_upsampled, mask_level2_upsampled], dim=1)
        
        # Predict final high-resolution mask (1024x1024)
        mask_level3 = self.mask_refiner_level3(combined_level3)
        
        # Enhance mask edges for better boundary definition
        final_mask = self._enhace_edges(mask_level3)
        
        return final_mask
    

    def get_dense_pe(self) -> torch.Tensor:
        pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)
        device = next(self.parameters()).device
        return pe.to(device)