import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Type, Optional

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

class HighResAlignment(nn.Module):
    """
    Learnable alignment block to downsample high-res features 
    to the target resolution and adjust channel dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 input_size: Tuple[int, int], target_size: Tuple[int, int],
                 norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels)):
        super().__init__()
        stride_h = input_size[0] // target_size[0]
        stride_w = input_size[1] // target_size[1]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=(stride_h, stride_w), padding=1)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))

# ----------------------------
# Revised Mask Generator
# ----------------------------


class EmbeddingGenerator(nn.Module):
    def __init__(
        self,
        embed_dim: int, 
        image_embedding_size: Tuple[int, int],  # e.g., (64, 64) for low-res feature maps
        input_image_size: Tuple[int, int],      # e.g., (256, 256) final mask size
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels),
    ) -> None:
        super().__init__()
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.activation = activation()
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        
        # Backbone and event processing blocks
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
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=1),
            norm_layer(mask_in_chans // 2),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Alignment modules for high-res features with more flexible design
        self.align_highres = nn.ModuleDict({
            'backbone_32': nn.Sequential(
                nn.Conv2d(32, mask_in_chans, kernel_size=3, padding=1),
                norm_layer(mask_in_chans),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(image_embedding_size)
            ),
            'backbone_64': nn.Sequential(
                nn.Conv2d(64, mask_in_chans, kernel_size=3, padding=1),
                norm_layer(mask_in_chans),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(image_embedding_size)
            ),
            'event_32': nn.Sequential(
                nn.Conv2d(32, mask_in_chans, kernel_size=3, padding=1),
                norm_layer(mask_in_chans),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(image_embedding_size)
            ),
            'event_64': nn.Sequential(
                nn.Conv2d(64, mask_in_chans, kernel_size=3, padding=1),
                norm_layer(mask_in_chans),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(image_embedding_size)
            )
        })

        # High-res feature fusion (combining backbone and event cues)
        self.highres_fusion = nn.Sequential(
            nn.Conv2d(mask_in_chans * 2, mask_in_chans, kernel_size=1),
            norm_layer(mask_in_chans),
            nn.ReLU(inplace=True),
            ChannelAttention(mask_in_chans, reduction=8),
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            nn.ReLU(inplace=True),
        )
        
        # Attention modules on the combined features
        self.channel_attention = ChannelAttention(mask_in_chans)
        self.spatial_attention = SpatialAttention()
        
        # Multi-scale region attention for improved localization
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
        
        # Boundary-aware attention to refine edges
        self.boundary_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=1),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, mask_in_chans // 2, kernel_size=3, padding=1, groups=4),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ----------------------------
        # Hierarchical Mask Prediction
        # ----------------------------
        # Level 1: Low resolution mask
        self.mask_predictor_level1 = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=3, padding=1),
            norm_layer(mask_in_chans // 2),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Level 2: Feature extraction after upsampling
        self.feature_extractor_level2 = nn.Sequential(
            nn.Conv2d(mask_in_chans + 1, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation)
        )
        
        # Level 2: Final mask refinement at high resolution
        self.mask_refiner_level2 = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=3, padding=1),
            norm_layer(mask_in_chans // 2),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Optional edge enhancement for crisper boundaries
        self.edge_enhancement = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(initialize_embedding_generator)

    def _enhance_edges(self, mask: torch.Tensor) -> torch.Tensor:
        """Enhance edges in the mask using Sobel gradients and a learnable module."""
        def sobel_edges(x):
            sobel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            grad_x = F.conv2d(x, sobel_x, padding=1)
            grad_y = F.conv2d(x, sobel_y, padding=1)
            return torch.sqrt(grad_x**2 + grad_y**2)
        
        edges = sobel_edges(mask)
        enhanced_edges = self.edge_enhancement(edges)
        return 0.8 * mask + 0.2 * enhanced_edges

    def _process_highres_features(self, features: List[torch.Tensor], feature_type: str) -> torch.Tensor:
        """Process high-resolution features with proper error handling"""
        if not features:
            # Return an empty tensor of the right shape if no features provided
            return torch.zeros(1, 256, *self.image_embedding_size, device=self._get_device())
            
        aligned_features = []
        for feat in features:
            if feat.shape[1] == 32:
                key = f"{feature_type}_32"
                if key in self.align_highres:
                    aligned_features.append(self.align_highres[key](feat))
            elif feat.shape[1] == 64:
                key = f"{feature_type}_64"
                if key in self.align_highres:
                    aligned_features.append(self.align_highres[key](feat))
        
        if not aligned_features:
            return torch.zeros(1, 256, *self.image_embedding_size, device=self._get_device())
            
        return sum(aligned_features)

    def _get_device(self):
        """Helper to get the device of the model parameters"""
        return next(self.parameters()).device

    def forward(
        self, 
        backbone_features: torch.Tensor, 
        event_features: Optional[torch.Tensor] = None,
        high_res_features: Optional[List[torch.Tensor]] = None,
        high_res_event_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass of the embedding generator
        
        Args:
            backbone_features: Base features from backbone (B, C, H, W) where C=mask_in_chans
            event_features: Optional event-based features (B, C, H, W)
            high_res_features: Optional list of high-resolution backbone features
            high_res_event_features: Optional list of high-resolution event features
            
        Returns:
            torch.Tensor: Final mask of shape (B, 1, 256, 256)
        """
        # Process backbone features (always required)
        backbone_processed = self.backbone_block(backbone_features)
        
        # Process event features (or use zeros if not provided)
        if event_features is not None:
            event_processed = self.event_block(event_features)
        else:
            event_processed = torch.zeros_like(backbone_processed)
        
        # Process high-resolution features with proper error handling
        high_res_features = high_res_features or []
        high_res_event_features = high_res_event_features or []
        
        # Process and align high-res backbone features
        fused_backbone_highres = self._process_highres_features(high_res_features, "backbone")
        
        # Process and align high-res event features
        fused_event_highres = self._process_highres_features(high_res_event_features, "event")
        
        # Fuse high-res features if they exist
        if torch.sum(fused_backbone_highres) > 0 and torch.sum(fused_event_highres) > 0:
            highres_combined = torch.cat([fused_backbone_highres, fused_event_highres], dim=1)
            fused_highres = self.highres_fusion(highres_combined)
        else:
            # Create empty tensor if high-res features are missing
            fused_highres = torch.zeros_like(backbone_processed)
        
        # Combine all features (with a slight weighting for event cues)
        combined_features = backbone_processed + 0.3 * event_processed
        
        # Only add high-res features if they're valid
        if torch.sum(fused_highres) > 0:
            combined_features = combined_features + fused_highres
        
        # Apply channel and spatial attention
        features = self.channel_attention(combined_features)
        features = self.spatial_attention(features)
        
        # Multi-scale region attention for improved localization
        region_attention_1 = torch.sigmoid(self.region_attention['scale1'](features))
        region_attention_2 = torch.sigmoid(self.region_attention['scale2'](features))
        region_attention_3 = torch.sigmoid(self.region_attention['scale3'](features))
        region_attention = (0.5 * region_attention_1 + 0.3 * region_attention_2 + 0.2 * region_attention_3)
        
        # Apply boundary-aware attention
        boundary_map = self.boundary_attention(features)
        region_enhanced_features = features * region_attention
        boundary_enhanced_features = region_enhanced_features * boundary_map
        
        # ----------------------------
        # Hierarchical Mask Prediction
        # ----------------------------
        # Level 1: Predict a coarse mask
        mask_level1 = self.mask_predictor_level1(boundary_enhanced_features)
        
        # Concatenate the coarse mask with the original features
        combined_input_level2 = torch.cat([boundary_enhanced_features, mask_level1], dim=1)
        
        # First upsample the features to the final size, then process them
        combined_input_level2_upsampled = F.interpolate(
            combined_input_level2, 
            size=(256, 256), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Extract refined features at the higher resolution
        features_level2 = self.feature_extractor_level2(combined_input_level2_upsampled)
        
        # Produce the final refined mask
        mask_level2 = self.mask_refiner_level2(features_level2)
        
        # Enhance edges for crisper final mask (optional)
        final_mask = self._enhance_edges(mask_level2)
        
        return final_mask  # Shape: [B, 1, 256, 256]
    
    def get_dense_pe(self) -> torch.Tensor:
        """Get the positional encoding"""
        pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)
        return pe.to(self._get_device())