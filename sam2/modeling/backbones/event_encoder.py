import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x, y):
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos

class EventFlowEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, d_model: int = 32, num_heads: int = 1, num_layers: int = 1, num_pos_feats: int = 64):
        super(EventFlowEncoder, self).__init__()
        
        # Apply more aggressive initial downsampling to reduce spatial size early
        self.initial_downsample = nn.Sequential(
            nn.Conv2d(in_channels, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(d_model // 2, d_model // 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Additional pooling to further reduce spatial size
        self.additional_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        # Convolutional layers for upsampling
        self.conv_up_1 = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=2, stride=2, padding=0),  # 2x upsample
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, d_model // 2, kernel_size=3, stride=1, padding=1),  # Intermediate layer
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model // 2, d_model // 4, kernel_size=2, stride=2, padding=0),  # Another 2x upsample
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, d_model // 4, kernel_size=3, stride=1, padding=1),  # Another intermediate layer
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model // 4, d_model, kernel_size=2, stride=2, padding=0)  # Final 2x upsample to reach 8x total
        )

        # Second upsampling (4x equivalent) with similar structure
        self.conv_up_2 = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=2, stride=2, padding=0),  # 2x upsample
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, d_model // 2, kernel_size=3, stride=1, padding=1),  # Intermediate layer
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d_model // 2, d_model, kernel_size=2, stride=2, padding=0)  # Final 2x upsample for 4x total
        )
        
        # Positional encoding
        self.position_encoding = PositionEmbeddingSine(num_pos_feats)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2, dropout=0.1, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, sample: torch.Tensor):
        # Initial downsampling to reduce spatial size
        x = self.initial_downsample(sample)  # [B, d_model, 64, 64]
        
        # Further downsampling to reduce spatial size to [32, 32]
        x = self.additional_pool(x)  # [B, d_model, 32, 32]
        
        # Positional encoding
        pos_enc = self.position_encoding(x)  # [B, d_model, 32, 32]
        
        # Flatten the spatial dimensions for the transformer input
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [H*W, B, d_model]
        pos_enc = pos_enc.flatten(2).permute(2, 0, 1)  # [H*W, B, d_model]
        
        # Apply transformer encoder with positional encoding
        x = self.transformer_encoder(x + pos_enc)  # [H*W, B, d_model]
        
        # Reshape back to the original spatial dimensions
        x = x.permute(1, 2, 0).view(B, C, H, W)  # [B, d_model, 32, 32]
        pos_enc = pos_enc.permute(1, 2, 0).view(B, C, H, W)  # [B, d_model, 32, 32]
        
        # Upsample using convolutional layers
        feature_map_1 = self.conv_up_1(x)  # [B, d_model, 256, 256]
        pos_enc_1 = F.interpolate(pos_enc, size=feature_map_1.shape[-2:], mode='bilinear', align_corners=False)
        
        feature_map_2 = self.conv_up_2(x)  # [B, d_model, 128, 128]
        pos_enc_2 = F.interpolate(pos_enc, size=feature_map_2.shape[-2:], mode='bilinear', align_corners=False)
        
        feature_map_3 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # [B, d_model, 64, 64]
        pos_enc_3 = F.interpolate(pos_enc, size=feature_map_3.shape[-2:], mode='bilinear', align_corners=False)
        
        # Output dictionary to match the original image encoder structure
        output = {
            "vision_features": feature_map_3,
            "vision_pos_enc": [pos_enc_1, pos_enc_2, pos_enc_3],
            "backbone_fpn": [feature_map_1, feature_map_2, feature_map_3],
        }
        return output

# Example usage
if __name__ == "__main__":
    encoder = EventFlowEncoder()
    dummy_input = torch.randn(1, 3, 1024, 1024)  # Batch size of 1, 3 channels (event flow), 1024x1024 resolution
    output = encoder(dummy_input)
    print(output)  # To verify the output structure
