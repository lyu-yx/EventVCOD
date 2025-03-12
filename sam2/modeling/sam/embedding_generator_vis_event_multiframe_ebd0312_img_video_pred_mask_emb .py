import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Type

# ----------------------------
# Auxiliary Modules
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
        input_image_size: Tuple[int, int],        # e.g., (256, 256) final mask size
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
        
        # Alignment modules for high-res features
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

        # High-res feature fusion (combining backbone and event cues)
        self.highres_fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            ChannelAttention(256, reduction=8),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            norm_layer(256),
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
        # Level 1: Low resolution mask (e.g. 64x64)
        self.mask_predictor_level1 = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=3, padding=1),
            norm_layer(mask_in_chans // 2),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Level 2: Refine mask and upsample to final 256x256 resolution
        self.feature_extractor_level2 = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            ResidualBlock(mask_in_chans, mask_in_chans, activation)
        )
        self.mask_refiner_level2 = nn.Sequential(
            nn.Conv2d(mask_in_chans + 1, mask_in_chans, kernel_size=3, padding=1),
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

    def _enhace_edges(self, mask: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self, 
        backbone_features: torch.Tensor, 
        event_features: torch.Tensor,
        high_res_features: List[torch.Tensor],
        high_res_event_features: List[torch.Tensor]
    ) -> torch.Tensor:
        # Process backbone and event features
        backbone_processed = self.backbone_block(backbone_features)
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
        fused_backbone_highres = sum(aligned_backbone) if aligned_backbone else torch.zeros_like(backbone_processed)

        # Process and align high-res event features
        aligned_event = []
        for feat in high_res_event_features:
            if feat.shape[1] == 32:
                aligned_event.append(self.align_event_32(feat))
            elif feat.shape[1] == 64:
                aligned_event.append(self.align_event_64(feat))
        fused_event_highres = sum(aligned_event) if aligned_event else torch.zeros_like(backbone_processed)

        # Fuse high-res features
        highres_combined = torch.cat([fused_backbone_highres, fused_event_highres], dim=1)
        fused_highres = self.highres_fusion(highres_combined)

        # Combine all features (with a slight weighting for event cues)
        combined_features = backbone_processed + 0.3 * event_enhanced + fused_highres

        # Apply channel and spatial attention
        features = self.channel_attention(combined_features)
        features = self.spatial_attention(features)

        # Multi-scale region attention for improved localization
        region_attention_1 = torch.sigmoid(self.region_attention['scale1'](features))
        region_attention_2 = torch.sigmoid(self.region_attention['scale2'](features))
        region_attention_3 = torch.sigmoid(self.region_attention['scale3'](features))
        region_attention = (0.5 * region_attention_1 + 0.3 * region_attention_2 + 0.2 * region_attention_3)
        
        # Boundary-aware attention and motion consistency
        boundary_map = self.boundary_attention(features)
        motion_consistency = torch.sigmoid(motion_map)
        region_enhanced_features = features * region_attention
        boundary_enhanced_features = region_enhanced_features * boundary_map
        motion_enhanced_features = boundary_enhanced_features * motion_consistency
                
        # ----------------------------
        # Hierarchical Mask Prediction
        # ----------------------------
        # Level 1: Predict a low-resolution mask (e.g., 64x64)
        mask_level1 = self.mask_predictor_level1(motion_enhanced_features)
        
        # Level 2: Upsample and refine mask to 256x256
        mask_level1_upsampled = F.interpolate(mask_level1, size=self.input_image_size, mode='bilinear', align_corners=False)
        features_level2 = self.feature_extractor_level2(motion_enhanced_features)
        features_level2_upsampled = F.interpolate(features_level2, size=self.input_image_size, mode='bilinear', align_corners=False)
        combined_level2 = torch.cat([features_level2_upsampled, mask_level1_upsampled], dim=1)
        mask_level2 = self.mask_refiner_level2(combined_level2)
        
        # Enhance edges for a crisper final mask
        final_mask = self._enhace_edges(mask_level2)
        return final_mask  # Expected shape: [B, 1, 256, 256]
    def get_dense_pe(self) -> torch.Tensor:
        pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)
        device = next(self.parameters()).device
        return pe.to(device)