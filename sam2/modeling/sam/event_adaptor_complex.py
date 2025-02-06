import torch
import torch.nn as nn
import torch.nn.functional as F

class EventAdaptor(nn.Module):
    def __init__(self, feature_channels, use_residual=True):
        """
        Simplified adaptor for FPN feature adaptation.
        
        Args:
            feature_channels (int): Number of input/output channels
            use_residual (bool): Whether to use residual connections
        """
        super().__init__()
        self.use_residual = use_residual
        
        # Main adaptation block
        self.adapt_block = nn.Sequential(
            # Spatial mixing
            nn.Conv2d(
                feature_channels, feature_channels,
                kernel_size=3, padding=1, groups=feature_channels,
                bias=False
            ),
            nn.BatchNorm2d(feature_channels),
            nn.GELU(),
            
            # Channel mixing
            nn.Conv2d(
                feature_channels, feature_channels * 2,
                kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(feature_channels * 2),
            nn.GELU(),
            nn.Conv2d(
                feature_channels * 2, feature_channels,
                kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(feature_channels)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize adaptor weights using modern techniques.
        Uses He initialization for conv layers and handles batchnorm specially.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Standard initialization for batchnorm
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass maintaining input dimensions.
        
        Args:
            x (torch.Tensor): Input feature tensor [B, C, H, W]
        
        Returns:
            torch.Tensor: Adapted feature tensor [B, C, H, W]
        """
        adapted = self.adapt_block(x)
        if self.use_residual:
            adapted = adapted + x
        return adapted


class EventAdaptorAttention(nn.Module):
    def __init__(self, feature_channels, use_residual=True, use_attention=True, dropout_rate=0.1):
        """
        Enhanced adaptor for FPN feature adaptation with attention and dropout.
        
        Args:
            feature_channels (int): Number of input/output channels.
            use_residual (bool): Whether to use residual connections.
            use_attention (bool): Whether to apply attention mechanisms.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Spatial Mixing Block
        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                feature_channels, feature_channels,
                kernel_size=3, padding=1, groups=feature_channels,
                bias=False
            ),
            nn.BatchNorm2d(feature_channels),
            nn.GELU()
        )
        
        # Channel Mixing Block
        self.channel_block = nn.Sequential(
            nn.Conv2d(
                feature_channels, feature_channels * 2,
                kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(feature_channels * 2),
            nn.GELU(),
            nn.Conv2d(
                feature_channels * 2, feature_channels,
                kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(feature_channels)
        )
        
        # Attention Mechanism (Squeeze-and-Excitation Block)
        if self.use_attention:
            self.attention_block = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # Global Avg Pooling
                nn.Conv2d(feature_channels, feature_channels // 4, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(feature_channels // 4, feature_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass maintaining input dimensions.
        
        Args:
            x (torch.Tensor): Input feature tensor [B, C, H, W]
        
        Returns:
            torch.Tensor: Adapted feature tensor [B, C, H, W]
        """
        residual = x
        
        # Spatial Mixing
        x = self.spatial_block(x)
        
        # Channel Mixing
        x = self.channel_block(x)
        
        # Attention Mechanism
        if self.use_attention:
            attention = self.attention_block(x)
            x = x * (1 + attention)  # Apply channel attention
        
        
        
        # Residual Connection
        if self.use_residual:
            x = x + residual
        
        # Dropout
        x = self.dropout(x)

        return x


class MultiLevelEventAdaptor(nn.Module):
    def __init__(self, in_channels_list, use_residual=True):
        """
        Adaptor for multiple FPN levels.
        
        Args:
            in_channels_list (list): List of channel counts for each level
            use_residual (bool): Whether to use residual connections
        """
        super().__init__()
        
        # Create separate adaptors for each level
        self.adaptors = nn.ModuleList([
            EventAdaptor(channels, use_residual)
            for channels in in_channels_list
        ])
    
    def forward(self, features):
        """
        Adapt features at each level.
        
        Args:
            features (list): List of feature tensors from FPN levels
        
        Returns:
            list: Adapted features maintaining original dimensions
        """
        return [
            adaptor(feature)
            for adaptor, feature in zip(self.adaptors, features)
        ]

class MultiLevelTinyEventAdaptor(nn.Module):
    def __init__(self, in_channels_list, use_residual=True):
        """
        Adaptor for multiple FPN levels.
        
        Args:
            in_channels_list (list): List of channel counts for each level
            use_residual (bool): Whether to use residual connections
        """
        super().__init__()
        
        # Create separate adaptors for each level
        self.adaptors = nn.ModuleList([
            TinyEventAdaptor(channels, use_residual)
            for channels in in_channels_list
        ])
    
    def forward(self, features):
        """
        Adapt features at each level.
        
        Args:
            features (list): List of feature tensors from FPN levels
        
        Returns:
            list: Adapted features maintaining original dimensions
        """
        return [
            adaptor(feature)
            for adaptor, feature in zip(self.adaptors, features)
        ]

class MultiLevelEventAdaptorAttention(nn.Module):
    def __init__(self, in_channels_list, use_residual=True, use_attention=True, dropout_rate=0.1):
        """
        Adaptor for multiple FPN levels.
        
        Args:
            in_channels_list (list): List of channel counts for each level
            use_residual (bool): Whether to use residual connections
        """
        super().__init__()
        
        # Create separate adaptors for each level
        self.adaptors = nn.ModuleList([
            EventAdaptorAttention(channels, use_residual, use_attention, dropout_rate)
            for channels in in_channels_list
        ])
    
    def forward(self, features):
        """
        Adapt features at each level.
        
        Args:
            features (list): List of feature tensors from FPN levels
        
        Returns:
            list: Adapted features maintaining original dimensions
        """
        return [
            adaptor(feature)
            for adaptor, feature in zip(self.adaptors, features)
        ]
def event_adaptor(event_data):
    """
    Wrapper function for feature adaptation.
    
    Args:
        event_data (Union[torch.Tensor, list]): Either a single tensor or list of FPN features
        
    Returns:
        Union[torch.Tensor, list]: Adapted features maintaining input dimensions
    """
    # Handle both single tensor and list of features
    if isinstance(event_data, list):
        in_channels_list = [feat.shape[1] for feat in event_data]
        adaptor = MultiLevelEventAdaptor(
            in_channels_list=in_channels_list,
            use_residual=True
        ).to(event_data[0].device)
        return adaptor(event_data)
    else:
        adaptor = EventAdaptor(
            feature_channels=event_data.shape[1],
            use_residual=True
        ).to(event_data.device)
        return adaptor(event_data)
    

def vis_adaptor(vis_data):
    """
    Wrapper function for feature adaptation.
    
    Args:
        event_data (Union[torch.Tensor, list]): Either a single tensor or list of FPN features
        
    Returns:
        Union[torch.Tensor, list]: Adapted features maintaining input dimensions
    """
    # Handle both single tensor and list of features
    if isinstance(vis_data, list):
        in_channels_list = [feat.shape[1] for feat in vis_data]
        adaptor = MultiLevelEventAdaptor(
            in_channels_list=in_channels_list,
            use_residual=True
        ).to(vis_data[0].device)
        return adaptor(vis_data)
    else:
        adaptor = EventAdaptor(
            feature_channels=vis_data.shape[1],
            use_residual=True
        ).to(vis_data.device)
        return adaptor(vis_data)
    

class TinyEventAdaptor(nn.Module):
    def __init__(self, feature_channels, use_residual=True):
        """
        A tiny yet efficient adaptor that mixes spatial and channel information.
        
        Args:
            feature_channels (int): Number of input (and output) channels.
            use_residual (bool): Whether to add a residual connection.
        """
        super().__init__()
        self.use_residual = use_residual
        
        # Spatial mixing: depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            feature_channels, feature_channels,
            kernel_size=3, padding=1, groups=feature_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(feature_channels)
        
        # Channel mixing: pointwise convolution
        self.pointwise_conv = nn.Conv2d(
            feature_channels, feature_channels,
            kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(feature_channels)
        
        self.activation = nn.ReLU(inplace=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights with He initialization for conv layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        identity = x
        
        # Spatial mixing
        out = self.depthwise_conv(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        # Channel mixing
        out = self.pointwise_conv(out)
        out = self.bn2(out)
        
        # Residual connection
        if self.use_residual:
            out = out + identity
        
        return self.activation(out)