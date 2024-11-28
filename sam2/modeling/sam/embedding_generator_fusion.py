import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Type
from prompt_gen.backbone.position_encoding import PositionEmbeddingRandom

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


class MultiResolutionFusion(nn.Module):
    def __init__(self, target_channels: int):
        super().__init__()
        # Adaptive feature adjustment modules
        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, target_channels, kernel_size=1),
                nn.BatchNorm2d(target_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in [32, 64, 256]  # Adjust based on your specific feature channels
        ])
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        
        # Final fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(target_channels * 3, target_channels, kernel_size=1),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        # Normalize fusion weights
        normalized_weights = F.softmax(self.fusion_weights, dim=0)
        
        # Process and resize features
        processed_features = []
        for adapter, feature in zip(self.feature_adapters, features_list):
            # Adaptive channel adjustment
            feature_adapted = adapter(feature)
            
            # Resize to the resolution of the first (highest resolution) feature
            if feature_adapted.shape[-2:] != features_list[2].shape[-2:]:
                feature_adapted = F.interpolate(
                    feature_adapted, 
                    size=features_list[2].shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            processed_features.append(feature_adapted)
        
        # Weighted fusion
        weighted_features = [w * feat for w, feat in zip(normalized_weights, processed_features)]
        
        # Concatenate and fuse
        fused_features = torch.cat(weighted_features, dim=1)
        return self.fusion_conv(fused_features)



class CrossFeatureGating(nn.Module):
    def __init__(self, in_channels=256, reduction=8, backbone_weight=0.8, event_weight=0.4):
        """
        Cross-feature gating module with backbone dominance.

        Args:
            in_channels: Number of input channels for both features.
            reduction: Reduction ratio for the gating mechanism.
            backbone_weight: Weight for retaining original backbone information.
            event_weight: Weight for retaining original event information.
        """
        super(CrossFeatureGating, self).__init__()
        
        # Gating for combined_event using combined_backbone
        self.gate_event = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Gating for combined_backbone using combined_event
        self.gate_backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Weights for retaining information
        self.backbone_weight = backbone_weight
        self.event_weight = event_weight

    def forward(self, combined_backbone, combined_event):
        """
        Forward pass of the cross-feature gating module.

        Args:
            combined_backbone: Dominant feature map of shape [B, C, H, W].
            combined_event: Auxiliary feature map of shape [B, C, H, W].

        Returns:
            gated_combined_backbone: Enhanced dominant feature map.
            gated_combined_event: Modulated auxiliary feature map.
        """
        # Gating for combined_event using combined_backbone
        gate_event = self.gate_event(combined_backbone)  # Shape: [B, C, 1, 1]
        gated_combined_event = combined_event * gate_event  # Modulate event features

        # Gating for combined_backbone using combined_event
        gate_backbone = self.gate_backbone(combined_event)  # Shape: [B, C, 1, 1]
        gated_combined_backbone = combined_backbone * gate_backbone  # Modulate backbone features

        # Retain more information from the backbone
        gated_combined_backbone = self.backbone_weight * gated_combined_backbone + \
                                  (1 - self.backbone_weight) * combined_backbone
        
        # Optionally retain some information in combined_event
        gated_combined_event = self.event_weight * gated_combined_event + \
                               (1 - self.event_weight) * combined_event

        return gated_combined_backbone, gated_combined_event




class EmbeddingGenerator(nn.Module):
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
        
        # Multi-scale feature refinement for backbone and event features
        self.backbone_processor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=rate, dilation=rate),
                nn.BatchNorm2d(mask_in_chans),
                activation()
            ) for rate in [1, 2, 4, 8]
        ])
        
        self.event_processor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=rate, dilation=rate),
                nn.BatchNorm2d(mask_in_chans),
                activation()
            ) for rate in [1, 2, 4, 8]
        ])
        
        # Gated mechanism for controlling event feature contribution
        # self.gate = nn.Sequential(
        #     nn.Conv2d(mask_in_chans * 2, mask_in_chans, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )
        self.gate = CrossFeatureGating(mask_in_chans)

        # Channel and Spatial attention
        self.channel_attention = ChannelAttention(mask_in_chans)
        self.spatial_attention = SpatialAttention()
        
        # Dense and Sparse Embedding Generators
        self.dense_embedder = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        )
        
        self.sparse_embedder = nn.Sequential(
            PyramidPooling(mask_in_chans),
            nn.Conv2d(mask_in_chans * 5, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Edge detection and refinement
        self.edge_detector = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=1),
            activation(),
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=3, padding=1)
        )
        
        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim + 1, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            activation()
        )

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.fpn_backbone_fusion = MultiResolutionFusion(mask_in_chans)
        self.fpn_event_fusion = MultiResolutionFusion(mask_in_chans)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, 
            backbone_features: torch.Tensor, 
            event_features: torch.Tensor,
            high_res_features: List[torch.Tensor],
            high_res_event_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-scale processing for backbone and event features
        backbone_multiscale = torch.sum(torch.stack([layer(backbone_features) for layer in self.backbone_processor]), dim=0)
        event_multiscale = torch.sum(torch.stack([layer(event_features) for layer in self.event_processor]), dim=0)
        
        # Learnable fusion of FPN features
        high_res_features_fpn = high_res_features + [backbone_features]
        high_res_event_features_fpn = high_res_event_features + [event_features] 

        fpn_fusion_features = self.fpn_backbone_fusion(high_res_features_fpn)
        fpn_fusion_event_features = self.fpn_event_fusion(high_res_event_features_fpn)
        
        # Combine original and FPN features
        combined_backbone = backbone_multiscale + fpn_fusion_features
        combined_event = event_multiscale + fpn_fusion_event_features
        
        # Gated fusion: Control the contribution of event features
        # combined_features = torch.cat([combined_backbone, combined_event], dim=1)
        
        gated_combined_backbone, _ = self.gate(combined_backbone, combined_event)
        dominant_features = combined_backbone + gated_combined_backbone

        # Channel and spatial attention to enhance features
        features = self.channel_attention(dominant_features)
        features = self.spatial_attention(features)
        
        # Edge attention for boundary enhancement
        edge_attention = torch.sigmoid(self.edge_detector(features))
        
        # Generate dense embeddings with edge information
        dense_embeddings = self.dense_embedder(features)
        dense_embeddings = self.refinement(torch.cat([dense_embeddings, edge_attention], dim=1))
        
        # Generate sparse embeddings
        sparse_embeddings = self.sparse_embedder(features)
        sparse_embeddings = sparse_embeddings.flatten(2).transpose(1, 2)
        
        return sparse_embeddings, dense_embeddings
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