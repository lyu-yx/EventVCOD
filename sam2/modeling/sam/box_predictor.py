import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, embedding_dim, num_keypoints):
        super(AttentionModule, self).__init__()
        self.attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=8)
        self.keypoint_layer = nn.Linear(embedding_dim, num_keypoints)
        self.relu = nn.ReLU()

    def forward(self, image_embeddings, image_pe):
        # Generate keypoints (extract keypoint embeddings)
        keypoints = self.keypoint_layer(image_embeddings)  # (batch_size, num_keypoints, embedding_dim)
        keypoints = self.relu(keypoints)

        # Apply attention mechanism over keypoints
        attended_keypoints, _ = self.attention_layer(keypoints, keypoints, keypoints)
        
        return attended_keypoints

class BoxPredictorKAN(nn.Module):
    def __init__(self, embedding_dim, num_keypoints, hidden_dim=256, num_boxes=1):
        super(BoxPredictorKAN, self).__init__()
        
        # Attention module to attend over keypoint features
        self.attention_module = AttentionModule(embedding_dim, num_keypoints)
        
        # Fully connected layers for box prediction
        self.fc1 = nn.Linear(embedding_dim + embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer (4 values for each box: x_min, y_min, x_max, y_max)
        self.box_predictor = nn.Linear(hidden_dim, num_boxes * 4)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, image_embeddings, image_pe):
        # Apply attention over keypoints
        attended_keypoints = self.attention_module(image_embeddings, image_pe)
        
        # Concatenate image embeddings and positional encodings
        x = torch.cat([attended_keypoints.mean(dim=1), image_pe], dim=-1)
        
        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Predict bounding boxes
        box_output = self.box_predictor(x)
        
        # Reshape the output to (num_boxes, 4) where each box has [x_min, y_min, x_max, y_max]
        box_output = box_output.view(-1, 4)
        
        return box_output
