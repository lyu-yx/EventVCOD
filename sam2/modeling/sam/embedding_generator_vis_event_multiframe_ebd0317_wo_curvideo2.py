import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Type


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    def __init__(self, in_channels_list, out_channels, norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels)):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                norm_layer(out_channels),
                nn.ReLU(inplace=True)
            ))
        
    def forward(self, features_list):
        results = []
        laterals = [lateral_conv(feature) for lateral_conv, feature in zip(self.lateral_convs, features_list)]
        
        # Top-down path
        prev_features = laterals[-1]
        results.insert(0, self.output_convs[-1](prev_features))
        
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample and add
            upsample_size = laterals[i].shape[-2:]
            upsampled_features = F.interpolate(prev_features, size=upsample_size, mode='bilinear', align_corners=False)
            prev_features = laterals[i] + upsampled_features
            results.insert(0, self.output_convs[i](prev_features))
            
        return results


class CBAM(nn.Module):
    """Convolutional Block Attention Module - improved version of the original CBAM"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        chan_att = self.channel_attention(x)
        x = x * chan_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        
        return x * spatial_att


class TransformerEncoderLayer(nn.Module):
    """Lightweight transformer encoder layer for feature refinement"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, src):
        B, C, H, W = src.shape
        src_flat = src.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Self-attention
        src2 = self.self_attn(src_flat, src_flat, src_flat)[0]
        src_flat = src_flat + self.dropout1(src2)
        src_flat = self.norm1(src_flat)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_flat))))
        src_flat = src_flat + self.dropout2(src2)
        src_flat = self.norm2(src_flat)
        
        # Reshape back to spatial dimensions
        output = src_flat.transpose(1, 2).reshape(B, C, H, W)
        return output


class DeformableConvolution(nn.Module):
    """Approximation of deformable convolution using spatial transformer network"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.kernel_size = kernel_size
    
    def forward(self, x):
        # Generate offset field
        offset = self.offset_conv(x)
        
        # Apply standard convolution as approximation (in a real implementation, you would use
        # a proper deformable convolution here, but this is a simplified version)
        # For a full implementation, consider using external libraries with deformable conv support
        return self.conv(x)


class EventVisionFusion(nn.Module):
    """Advanced fusion module for event and vision features"""
    def __init__(self, channels, norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels)):
        super().__init__()
        self.event_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            norm_layer(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.vision_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            norm_layer(channels),
            nn.GELU()
        )
        
        self.event_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            norm_layer(channels),
            nn.GELU()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            norm_layer(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            norm_layer(channels),
            nn.GELU()
        )
        
        # Cross-modal attention
        self.cross_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            norm_layer(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, vision_feat, event_feat):
        # Enhance features
        vision_enhanced = self.vision_enhance(vision_feat)
        event_enhanced = self.event_enhance(event_feat)
        
        # Event gating
        event_gate = self.event_gate(event_enhanced)
        
        # Cross-modal attention
        combined = torch.cat([vision_enhanced, event_enhanced], dim=1)
        cross_attn = self.cross_attention(combined)
        
        # Apply attention and fuse
        vision_attended = vision_enhanced * cross_attn
        event_attended = event_enhanced * event_gate
        
        fused = self.fusion(torch.cat([vision_attended, event_attended], dim=1))
        return fused


class BiFPNLayer(nn.Module):
    """Bidirectional Feature Pyramid Network layer for efficient multi-scale feature fusion"""
    def __init__(self, feature_dims, out_dim, norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels)):
        super().__init__()
        self.num_levels = len(feature_dims)
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, out_dim, kernel_size=1) for dim in feature_dims
        ])
        
        # Top-down path
        self.top_down_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                norm_layer(out_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_levels - 1)
        ])
        
        # Bottom-up path
        self.bottom_up_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
                norm_layer(out_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_levels - 1)
        ])
        
        # Output convs
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                norm_layer(out_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(self.num_levels)
        ])
        
        # Fusion weights (learned)
        self.fusion_weights_td = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(self.num_levels - 1)
        ])
        self.fusion_weights_bu = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(self.num_levels - 1)
        ])
        
        # Apply weight initialization
        for modules in [self.lateral_convs, self.top_down_convs, self.bottom_up_convs, self.output_convs]:
            for m in modules:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def _weighted_sum(self, features, weights):
        weight_sum = sum(w for w in weights)
        normalized_weights = [w / weight_sum for w in weights]
        return sum(w * f for w, f in zip(normalized_weights, features))
    
    def forward(self, features):
        # Lateral connections
        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        top_down_features = [laterals[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample and add
            upsample_size = laterals[i].shape[-2:]
            upsampled = F.interpolate(top_down_features[0], size=upsample_size, mode='bilinear', align_corners=False)
            
            # Apply weighted fusion
            w = F.softmax(self.fusion_weights_td[i], dim=0)
            weighted = self._weighted_sum([laterals[i], upsampled], w)
            top_down = self.top_down_convs[i](weighted)
            top_down_features.insert(0, top_down)
        
        # Bottom-up pathway
        bottom_up_features = [top_down_features[0]]
        for i in range(1, self.num_levels):
            # Apply weighted fusion for previous bottom-up result (if not the first) and top-down
            if i == 1:
                combined = bottom_up_features[-1]
            else:
                w = F.softmax(self.fusion_weights_bu[i-2], dim=0)
                combined = self._weighted_sum([top_down_features[i], bottom_up_features[-1]], w)
            
            # Apply convolution and possibly downsampling
            if i < self.num_levels - 1:
                bottom_up = self.bottom_up_convs[i-1](combined)
                bottom_up_features.append(bottom_up)
            else:
                bottom_up_features.append(combined)
        
        # Final output connections
        outputs = [conv(feature) for conv, feature in zip(self.output_convs, bottom_up_features)]
        
        return outputs


class MaskDecoder(nn.Module):
    """Improved mask decoder with cascaded refinement"""
    def __init__(self, in_channels, mid_channels, norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels)):
        super().__init__()
        
        # Initial prediction branch
        self.initial_pred = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            norm_layer(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            norm_layer(mid_channels // 2),
            nn.GELU(),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # First refinement stage
        self.refine1 = nn.Sequential(
            nn.Conv2d(in_channels + 1, mid_channels, kernel_size=3, padding=1),
            norm_layer(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            norm_layer(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Second refinement stage (with residual connection)
        self.refine2 = nn.Sequential(
            nn.Conv2d(in_channels + 2, mid_channels, kernel_size=3, padding=1),
            norm_layer(mid_channels),
            nn.GELU(),
            DeformableConvolution(mid_channels, mid_channels),
            norm_layer(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features, original_size=None):
        # Initial coarse prediction
        initial_mask = self.initial_pred(features)
        
        # First refinement
        refine_input1 = torch.cat([features, initial_mask], dim=1)
        refined_mask1 = self.refine1(refine_input1)
        
        # Second refinement with residual from first stage
        refine_input2 = torch.cat([features, initial_mask, refined_mask1], dim=1)
        refined_mask2 = self.refine2(refine_input2)
        
        # Final mask adds residual connection
        final_mask = refined_mask2 + 0.1 * refined_mask1
        
        # Optionally upsample to original size
        if original_size is not None and (original_size != final_mask.shape[-2:]):
            final_mask = F.interpolate(final_mask, size=original_size, mode='bilinear', align_corners=False)
            
        return final_mask


class EmbeddingGenerator(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        norm_layer=lambda num_channels: nn.GroupNorm(8, num_channels),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        
        # Feature Processing with BiFPN for multi-resolution features
        self.bifpn = BiFPNLayer([32, 64, mask_in_chans], 256, norm_layer)
        
        # Event-Vision Advanced Fusion
        self.event_vision_fusion = EventVisionFusion(mask_in_chans, norm_layer)
        
        # High-Resolution Feature Processing
        self.highres_event_fpn = FPN([32, 64], 256, norm_layer)
        self.highres_vision_fpn = FPN([32, 64], 256, norm_layer)
        
        # Fusion of high-resolution features
        self.highres_fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            norm_layer(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            norm_layer(256),
            nn.GELU()
        )
        
        # Advanced attention mechanism (CBAM)
        self.cbam = CBAM(mask_in_chans)
        
        # Feature enhancement with transformer
        self.transformer = TransformerEncoderLayer(
            d_model=mask_in_chans, 
            nhead=8,
            dim_feedforward=mask_in_chans*2
        )
        
        # Backbone feature processing
        self.backbone_process = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            nn.GELU(),
            DeformableConvolution(mask_in_chans, mask_in_chans),
            norm_layer(mask_in_chans),
            nn.GELU()
        )
        
        # Event feature processing
        self.event_process = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            nn.GELU(),
            DeformableConvolution(mask_in_chans, mask_in_chans),
            norm_layer(mask_in_chans),
            nn.GELU()
        )
        
        # Final feature integration
        self.feature_integration = nn.Sequential(
            nn.Conv2d(mask_in_chans*2 + 256, mask_in_chans, kernel_size=1),
            norm_layer(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            nn.GELU()
        )
        
        # Position encoding
        self.position_encoding = nn.Parameter(torch.zeros(1, mask_in_chans, *image_embedding_size))
        nn.init.trunc_normal_(self.position_encoding)
        
        # Improved mask decoder with cascaded refinement
        self.mask_decoder = MaskDecoder(mask_in_chans, mask_in_chans, norm_layer)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(
        self, 
        backbone_features: torch.Tensor, 
        event_features: torch.Tensor,
        high_res_features: List[torch.Tensor],
        high_res_event_features: List[torch.Tensor],
    ) -> torch.Tensor:
        # Process backbone and event features with advanced modules
        backbone_processed = self.backbone_process(backbone_features)
        event_processed = self.event_process(event_features)
        
        # Process high-resolution features with FPN
        highres_vision_features = self.highres_vision_fpn(high_res_features)
        highres_event_features = self.highres_event_fpn(high_res_event_features)
        
        # Fuse high-resolution features at each scale and select the appropriate one
        highres_fused = self.highres_fusion(torch.cat([
            highres_vision_features[0], highres_event_features[0]
        ], dim=1))
        
        # Event-Vision fusion with cross-modal attention
        event_vision_fused = self.event_vision_fusion(backbone_processed, event_processed)
        
        # Apply CBAM attention
        attended_features = self.cbam(event_vision_fused)
        
        # Apply transformer for global context modeling
        transformed_features = self.transformer(attended_features)
        
        # Add positional encoding
        transformed_features = transformed_features + self.position_encoding
        
        highres_fused = F.interpolate(highres_fused, size=backbone_processed.shape[-2:], mode='bilinear', align_corners=False)
        
        # Integrate all features
        # print("backbone_processed", backbone_processed.shape)
        # print("event_processed", event_processed.shape)
        # print("highres_fused", highres_fused.shape)
        
        integrated_features = self.feature_integration(torch.cat([
            backbone_processed, event_processed, highres_fused
        ], dim=1))
        
        

        # Generate final mask with cascaded refinement
        pred_mask = self.mask_decoder(integrated_features, self.input_image_size)
        
        return pred_mask
    
    def get_dense_pe(self) -> torch.Tensor:
        pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)
        device = next(self.parameters()).device
        return pe.to(device)

