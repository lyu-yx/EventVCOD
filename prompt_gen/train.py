import torch
import torch.nn as nn

# Simple FPN block for demonstration purposes (could be replaced with a more complex version)
class FPN(nn.Module):
    def __init__(self, in_channels):
        super(FPN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, 256, kernel_size=1)

    def forward(self, features):
        p1 = self.conv1(features[0])
        p2 = self.conv2(features[1])
        p3 = self.conv3(features[2])
        p4 = self.conv4(features[3])
        return [p1, p2, p3, p4]

# Kolmogorov-Arnold Network (KAN) for bounding box prediction
class KANBBoxPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(KANBBoxPredictor, self).__init__()
        
        # Decomposing bounding box prediction into sub-functions for each of the 4 coordinates
        self.fc_x1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc_y1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc_x2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc_y2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, fpn_features):
        # Average pooling across feature maps for simplicity
        pooled_features = [torch.mean(fmap, dim=(2, 3)) for fmap in fpn_features]
        
        # Concatenate features from all scales
        concatenated_features = torch.cat(pooled_features, dim=1)

        # Apply KAN decomposition for bbox prediction (x1, y1, x2, y2)
        x1 = self.fc_x1(concatenated_features)
        y1 = self.fc_y1(concatenated_features)
        x2 = self.fc_x2(concatenated_features)
        y2 = self.fc_y2(concatenated_features)
        
        # Concatenate predictions into a bounding box: [x1, y1, x2, y2]
        bbox = torch.cat([x1, y1, x2, y2], dim=1)
        return bbox

# Full model combining FPN and KAN for bbox prediction
class KANFPNModel(nn.Module):
    def __init__(self, in_channels):
        super(KANFPNModel, self).__init__()
        self.fpn = FPN(in_channels)
        self.kan_head = KANBBoxPredictor(input_dim=1024)  # Assuming 4 FPN levels each with 256-dim pooled features
    
    def forward(self, features):
        # Pass through FPN to get multi-scale features
        fpn_features = self.fpn(features)
        
        # Pass FPN features to KAN head for bbox prediction
        bbox = self.kan_head(fpn_features)
        return bbox

# Example usage
model = KANFPNModel(in_channels=512)
dummy_features = [torch.randn(1, 512, 32, 32), torch.randn(1, 512, 16, 16), torch.randn(1, 512, 8, 8), torch.randn(1, 512, 4, 4)]
bbox_prediction = model(dummy_features)
print("Predicted bounding box:", bbox_prediction)
