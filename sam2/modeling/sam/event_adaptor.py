import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNFeatureAdaptor(nn.Module):
    def __init__(self, in_channels_list):
        """
        Adaptor for transforming FPN features to event-like flow.
        
        Args:
            in_channels_list (list): List of input channel counts for each FPN level
        """
        super().__init__()
        
        # Create separate adaptation modules for each FPN level
        self.adapt_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(num_features=in_channels),
            ) for in_channels in in_channels_list
        ])
        
    def forward(self, high_res_event_features):
        """
        Adapt FPN features while maintaining input dimensions.
        
        Args:
            high_res_event_features (list): Input feature tensors 
            from FPN levels
        
        Returns:
            list: Adapted event-like features for each FPN level
        """
        adapted_features = []
        
        for i, feature in enumerate(high_res_event_features):
            # Apply level-specific adaptation
            adapted_feature = self.adapt_modules[i](feature)
            
            # Residual connection to preserve original feature information
            adapted_feature = adapted_feature + feature
            
            adapted_features.append(adapted_feature)
        
        return adapted_features
