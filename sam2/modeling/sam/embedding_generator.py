import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Type
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


class EmbeddingGenerator(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Generates embeddings for SAM's mask decoder, specialized for camouflage detection.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (tuple(int, int)): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.activation = activation()
        
        # Multi-scale feature refinement
        self.multiscale_processor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=rate, dilation=rate),
                nn.BatchNorm2d(mask_in_chans),
                activation()
            ) for rate in [1, 2, 4, 8]
        ])
        
        # Channel attention module
        self.channel_attention = ChannelAttention(mask_in_chans)
        
        # Spatial attention module
        self.spatial_attention = SpatialAttention()
        
        # Dense embedding generator with residual blocks
        self.dense_embedder = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        )
        
        # Sparse embedding generator with global context
        self.sparse_embedder = nn.Sequential(
            PyramidPooling(mask_in_chans),
            nn.Conv2d(mask_in_chans * 5, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Edge attention for boundary enhancement
        self.edge_detector = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans//2, kernel_size=1),
            activation(),
            nn.Conv2d(mask_in_chans//2, 1, kernel_size=3, padding=1)
        )
        
        # Final refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim + 1, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            activation()
        )

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)


    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    

    def forward(self, backbone_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sparse and dense embeddings from backbone features.
        
        Args:
            backbone_features (torch.Tensor): Features from backbone [B, C, H, W]
            
        Returns:
            tuple: (sparse_embeddings, dense_embeddings)
                - sparse_embeddings: [B, 1, embed_dim]
                - dense_embeddings: [B, embed_dim, H, W]
        """
        # Multi-scale feature processing
        multiscale_features = []
        for processor in self.multiscale_processor:
            multiscale_features.append(processor(backbone_features))
        multiscale_output = torch.sum(torch.stack(multiscale_features), dim=0)
        
        # Apply attention mechanisms
        features = self.channel_attention(multiscale_output)
        features = self.spatial_attention(features)
        
        # Edge detection branch
        edge_attention = torch.sigmoid(self.edge_detector(features))
        
        # Generate dense embeddings
        dense_embeddings = self.dense_embedder(features)
        
        # Combine with edge information
        dense_embeddings = self.refinement(
            torch.cat([dense_embeddings, edge_attention], dim=1)
        )
        
        # Generate sparse embeddings
        sparse_embeddings = self.sparse_embedder(features)
        sparse_embeddings = sparse_embeddings.flatten(2)  # [B, embed_dim, 1]
        sparse_embeddings = sparse_embeddings.transpose(1, 2)  # [B, 1, embed_dim]
        
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


class EmbeddingGeneratorRes(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Generates embeddings for promptless SAM mask generation, specialized for camouflage detection.
        
        Arguments:
          embed_dim (int): Embedding dimension for mask generation.
          image_embedding_size (tuple(int, int)): The spatial size of the image embedding, as (H, W).
          mask_in_chans (int): The number of hidden channels used for encoding input masks.
          activation (nn.Module): The activation function for encoding layers.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.activation = activation()
        
        # Multi-scale feature refinement with FPN (CBR layers)
        self.fpn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=rate, dilation=rate),
                nn.BatchNorm2d(mask_in_chans),
                activation()
            ) for rate in [1, 2, 4, 8]
        ])
        
        # Self-Attention Module for refining feature map
        self.self_attention = nn.MultiheadAttention(embed_dim=mask_in_chans, num_heads=4, batch_first=True)

        # Gated Mechanism for Spatial Relevance
        self.gated_attention = nn.Sequential(
            nn.Conv2d(mask_in_chans, mask_in_chans, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Global Context Block for sparse embedding
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
            activation()
        )
        
        # Dense embedder with residual blocks and self-attention
        self.dense_embedder = nn.Sequential(
            ResidualBlock(mask_in_chans, mask_in_chans, activation),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        )
        
        # Final refinement layer for dense embedding
        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            activation()
        )

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)


    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    
    def forward(self, backbone_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sparse and dense embeddings from backbone features.
        
        Args:
            backbone_features (torch.Tensor): Features from backbone [B, C, H, W]
            
        Returns:
            tuple: (sparse_embeddings, dense_embeddings)
                - sparse_embeddings: [B, 1, embed_dim]
                - dense_embeddings: [B, embed_dim, H, W]
        """
        # Multi-scale feature processing using FPN
        multiscale_features = []
        for layer in self.fpn:
            # Apply the layer and add the input (residual connection)
            scale_output = layer(backbone_features)
            multiscale_features.append(scale_output + backbone_features)  # Residual connection
        multiscale_output = torch.sum(torch.stack(multiscale_features), dim=0)

        # Self-Attention Mechanism
        b, c, h, w = multiscale_output.size()
        features_flat = multiscale_output.flatten(2).permute(0, 2, 1)  # Flatten and permute for attention
        features_attention, _ = self.self_attention(features_flat, features_flat, features_flat)
        features_attention = features_attention.permute(0, 2, 1).view(b, c, h, w)  # Reshape back

        # Gated attention mechanism
        gated_features = self.gated_attention(features_attention) * multiscale_output

        # Generate dense embeddings
        dense_embeddings = self.dense_embedder(gated_features)
        dense_embeddings = self.refinement(dense_embeddings)  # [B, embed_dim, H, W]

        # Generate sparse embeddings using global context
        sparse_embeddings = self.global_context(gated_features)  # [B, embed_dim, 1, 1]
        sparse_embeddings = sparse_embeddings.flatten(2).transpose(1, 2)  # Reshape to [B, 1, embed_dim]

        return sparse_embeddings, dense_embeddings


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