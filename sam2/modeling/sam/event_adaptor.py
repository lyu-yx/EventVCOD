import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNFeatureAdaptor(nn.Module):
    def __init__(self, in_channels_list, use_residual=True):
        """
        Adaptor for transforming FPN features to event-like flow.
        
        Args:
            in_channels_list (list): List of input channel counts for each FPN level.
            use_residual (bool): Whether to use residual connections.
        """
        super().__init__()
        self.use_residual = use_residual

        # Create separate adaptation modules for each FPN level
        self.adapt_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, 
                    kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.GELU(),
                nn.Conv2d(
                    in_channels, in_channels, 
                    kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_features=in_channels)
            ) for in_channels in in_channels_list
        ])
        
        # Initialize weights (optional for fine control)
        for module in self.adapt_modules:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, high_res_event_features):
        """
        Adapt FPN features while maintaining input dimensions.
        
        Args:
            high_res_event_features (list): Input feature tensors from FPN levels.
        
        Returns:
            list: Adapted event-like features for each FPN level.
        """
        adapted_features = []

        for i, feature in enumerate(high_res_event_features):
            # Apply level-specific adaptation
            adapted_feature = self.adapt_modules[i](feature)
            
            # Optional residual connection
            if self.use_residual:
                adapted_feature = adapted_feature + feature
            
            adapted_features.append(adapted_feature)
        
        return adapted_features
