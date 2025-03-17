import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Type

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
    def __init__(self, num_pos_feats: int = 64):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, shape: Tuple[int, int]) -> torch.Tensor:
        h, w = shape
        return torch.rand(self.num_pos_feats * 2, h, w)

class TemporalFeatureAggregator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[2] == 1:
            x = x.squeeze(2)
        out, hidden = self.gru(x)
        final_state = hidden[-1]
        summary_vector = self.out_fc(final_state)
        summary_vector = summary_vector.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return summary_vector


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

    def forward(self, feat, fused_accumulator):
        in_channels = feat.shape[1]
        if in_channels == 32:
            feat = self.adapter_32(feat)
        elif in_channels == 64:
            feat = self.adapter_64(feat)
        else:
            raise ValueError(f"Unsupported input channels: {in_channels}")
        fused_accumulator += feat
        return fused_accumulator

class HighResAlignment(nn.Module):
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

        # Introduce an event gating module to derive an attention map from the event flow.
        self.event_gating = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=1),
            nn.Sigmoid()
        )

        # Alignment modules for high-res features.
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

        # Future feature fusion module.
        self.future_fusion = nn.Sequential(
            nn.Conv2d(mask_in_chans * 2, mask_in_chans, kernel_size=1),
            norm_layer(mask_in_chans),
            nn.ReLU(inplace=True)
        )

        # Future feature aggregator.
        self.video_feature_aggregator = nn.Sequential(
            nn.Conv1d(4096, mask_in_chans, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mask_in_chans, mask_in_chans, kernel_size=1),
            nn.Sigmoid()
        )

        # Attention modules.
        self.channel_attention = ChannelAttention(mask_in_chans)
        self.spatial_attention = SpatialAttention()
        self.region_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans // 2, kernel_size=1),
            self.activation,
            nn.Conv2d(mask_in_chans // 2, 1, kernel_size=3, padding=1)
        )

        # Enhanced dense embedder.
        self.dense_embedder = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            SelfAttentionBlock(mask_in_chans),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        )

        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim + 1, embed_dim, kernel_size=3, padding=1),
            norm_layer(embed_dim),
            self.activation
        )
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.dense_residual = nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=1)
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)

        # Final prediction branch (initial mask prediction).
        self.final_pred = nn.Sequential(
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
        
        # Skip adapter to incorporate early features.
        self.skip_adapter = nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=1)
        
        # Iterative mask refinement module.
        self.mask_refinement = nn.Sequential(
            nn.Conv2d(1 + mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            norm_layer(mask_in_chans),
            self.activation,
            nn.Conv2d(mask_in_chans, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(
        self, 
        backbone_features: torch.Tensor, 
        event_features: torch.Tensor,
        high_res_features: List[torch.Tensor],
        high_res_event_features: List[torch.Tensor],
    ) -> torch.Tensor:
        B, C, H, W = backbone_features.shape

        # Process backbone and event features.
        backbone_processed = self.backbone_block(backbone_features)
        event_processed = self.event_block(event_features)
        
        # Compute an event gating map from the event dataflow.
        event_gate = self.event_gating(event_processed)

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

        # Fuse high-res features.
        fused_highres = self.highres_fusion_fuse(torch.cat([fused_backbone_highres, fused_event_highres], dim=1))

        # Combine backbone, event, and high-res features.
        # Here we use the event gating map to modulate the combined features.
        combined_features = backbone_processed + 0.3 * event_processed + fused_highres
        combined_features = combined_features * event_gate

        # Apply attention modules.
        features = self.channel_attention(combined_features)
        features = self.spatial_attention(features)

        # Generate dense embeddings.
        region_attn = torch.sigmoid(self.region_attention(features))
        dense_intermediate = self.dense_embedder(features)
        dense_refined = self.refinement(torch.cat([dense_intermediate, region_attn], dim=1))
        residual_dense = self.dense_residual(features)
        dense_embeddings = dense_refined + residual_dense

        # Incorporate skip connection from the original backbone features.
        skip_feature = self.skip_adapter(backbone_features)
        enhanced_features = dense_embeddings + skip_feature

        # Initial mask prediction.
        initial_mask = self.final_pred(enhanced_features)

        # Iterative mask refinement.
        refinement_input = torch.cat([initial_mask, enhanced_features], dim=1)
        pred_mask = self.mask_refinement(refinement_input)

        # Upsample to original image size.
        pred_mask = self.upsample(pred_mask)

        return pred_mask

    def get_dense_pe(self) -> torch.Tensor:
        pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)
        device = next(self.parameters()).device
        return pe.to(device)
