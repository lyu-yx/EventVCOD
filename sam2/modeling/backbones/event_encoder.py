import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

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
        # The positions are expected to be normalized
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
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

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
    def __init__(self, in_channels: int = 3, d_model: int = 128, num_heads: int = 4, num_layers: int = 3, num_pos_feats: int = 128):
        """
        Lightweight Event Flow Encoder using Transformer
        Args:
            in_channels (int): Number of input channels, typically 3 for event flow (x, y differences, and time).
            d_model (int): Dimension of the model features.
            num_heads (int): Number of attention heads in the transformer.
            num_layers (int): Number of transformer encoder layers.
            num_pos_feats (int): Number of positional features for encoding.
        """
        super(EventFlowEncoder, self).__init__()
        
        # Initial downsampling to reduce the input size to 256x256
        self.initial_downsample = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, padding=0)
        
        # Linear projection to match the input dimension to the model dimension
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
        # Positional encoding
        self.position_encoding = PositionEmbeddingSine(num_pos_feats)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2, dropout=0.1, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection to reduce the spatial dimensions to 64x64
        self.final_downsample = nn.Conv2d(d_model, d_model, kernel_size=4, stride=4, padding=0)
        
    def forward(self, sample: torch.Tensor):
        # Initial downsampling to reduce spatial size
        sample = self.initial_downsample(sample)  # [B, in_channels, 256, 256]
        
        # Project input to model dimension
        x = self.input_proj(sample)  # [B, d_model, 256, 256]
        
        # Positional encoding
        pos_enc = self.position_encoding(x)  # [B, d_model, 256, 256]
        
        # Flatten the spatial dimensions for the transformer input
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [H*W, B, d_model]
        pos_enc = pos_enc.flatten(2).permute(2, 0, 1)  # [H*W, B, d_model]
        
        # Apply transformer encoder with positional encoding
        x = self.transformer_encoder(x + pos_enc)  # [H*W, B, d_model]
        
        # Reshape back to the original spatial dimensions
        x = x.permute(1, 2, 0).view(B, C, H, W)  # [B, d_model, 256, 256]
        
        # Final downsampling to 64x64 spatial dimensions
        x = self.final_downsample(x)  # [B, d_model, 64, 64]
        pos_enc = self.final_downsample(pos_enc.permute(1, 2, 0).view(B, C, H, W))  # [B, d_model, 64, 64]
        
        # Output dictionary to match the original image encoder structure
        output = {
            "vision_features": x,
            "vision_pos_enc": None,  # Positional encoding after downsampling
            "backbone_fpn": [x],
        }
        return output

# Example usage
if __name__ == "__main__":
    encoder = EventFlowEncoder()
    dummy_input = torch.randn(1, 3, 1024, 1024)  # Batch size of 1, 3 channels (event flow), 1024x1024 resolution
    output = encoder(dummy_input)
    print(output)  # To verify the output structure
