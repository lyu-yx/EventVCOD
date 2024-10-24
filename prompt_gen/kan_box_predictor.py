import torch
import torch.nn as nn
import torch.nn.functional as F

class KANBBoxPredictorVisionFeat(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_components=4):
        super(KANBBoxPredictorVisionFeat, self).__init__()
        
        # Kolmogorov-Arnold decomposition layers (one for each component)
        self.decomposition_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_components)
        ])
        
        # Aggregation layer to combine outputs from the decomposed components
        self.aggregation_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Predicting x_min, y_min, x_max, y_max
        )
    
    def forward(self, vision_features):
        # Global average pooling to reduce spatial dimensions
        pooled_features = F.adaptive_avg_pool2d(vision_features, (1, 1)).view(vision_features.size(0), -1)  # [batch, 256]

        # Apply Kolmogorov-Arnold decomposition to generate intermediate features
        decomposed_features = [layer(pooled_features) for layer in self.decomposition_layers]
        
        # Concatenate the decomposed features
        combined_features = torch.cat(decomposed_features, dim=1)  # [batch, hidden_dim * num_components]
        
        # Aggregation to predict final bounding box coordinates
        bbox = self.aggregation_layer(combined_features)  # [batch, 4]
        
        return bbox


class KANBBoxPredictorFPN(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_components=4):
        super(KANBBoxPredictorFPN, self).__init__()
        
        # Kolmogorov-Arnold decomposition layers (one for each component)
        self.decomposition_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * 3, hidden_dim),  # input_dim * 3 to handle concatenation of three scales
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_components)
        ])
        
        # Aggregation layer to combine outputs from the decomposed components
        self.aggregation_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Predicting x_min, y_min, x_max, y_max
        )
    
    def forward(self, fpn_features):
        # Extract the three FPN feature maps
        fpn_feature_1, fpn_feature_2, fpn_feature_3 = fpn_features

        # Global average pooling for each FPN feature map to reduce spatial dimensions
        pooled_feature_1 = F.adaptive_avg_pool2d(fpn_feature_1, (1, 1)).view(fpn_feature_1.size(0), -1)  # [batch, 256]
        pooled_feature_2 = F.adaptive_avg_pool2d(fpn_feature_2, (1, 1)).view(fpn_feature_2.size(0), -1)  # [batch, 256]
        pooled_feature_3 = F.adaptive_avg_pool2d(fpn_feature_3, (1, 1)).view(fpn_feature_3.size(0), -1)  # [batch, 256]

        # Concatenate pooled features from different scales
        combined_features = torch.cat([pooled_feature_1, pooled_feature_2, pooled_feature_3], dim=1)  # [batch, 256 * 3]

        # Apply Kolmogorov-Arnold decomposition to generate intermediate features
        decomposed_features = [layer(combined_features) for layer in self.decomposition_layers]
        
        # Concatenate the decomposed features
        combined_decomposed_features = torch.cat(decomposed_features, dim=1)  # [batch, hidden_dim * num_components]
        
        # Aggregation to predict final bounding box coordinates
        bbox = self.aggregation_layer(combined_decomposed_features)  # [batch, 4]
        
        return bbox


# Example usage of the KANBBoxPredictor
if __name__ == "__main__":
    # Create the KAN-based bounding box predictor model
    kan_model = KANBBoxPredictorVisionFeat(input_dim=256, hidden_dim=128, num_components=4)

    # Example vision features with size [batch, 256, 64, 64]
    vision_features = torch.randn(8, 256, 64, 64)  # Batch size of 8

    # Predict bounding boxes using the model
    bbox_predictions = kan_model(vision_features)

    # Output shape should be [batch, 4] representing [x_min, y_min, x_max, y_max] for each sample in the batch
    print("Predicted bounding boxes:", bbox_predictions.shape)  # Expected: [8, 4]

        # Create the KAN-based bounding box predictor model for FPN features
    kan_model_fpn = KANBBoxPredictorFPN(input_dim=256, hidden_dim=128, num_components=4)

    # Example FPN features with sizes [b, 256, 256, 256], [b, 256, 128, 128], [b, 256, 64, 64]
    fpn_feature_1 = torch.randn(8, 256, 256, 256)  # Batch size of 8
    fpn_feature_2 = torch.randn(8, 256, 128, 128)
    fpn_feature_3 = torch.randn(8, 256, 64, 64)

    # Predict bounding boxes using the model
    bbox_predictions = kan_model_fpn([fpn_feature_1, fpn_feature_2, fpn_feature_3])

    # Output shape should be [batch, 4] representing [x_min, y_min, x_max, y_max] for each sample in the batch
    print("Predicted bounding boxes:", bbox_predictions.shape)  # Expected: [8, 4]

