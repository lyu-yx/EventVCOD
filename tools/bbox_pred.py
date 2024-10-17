import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Set paths to your dataset
image_folder = "path/to/image_folder"  # .jpg images
mask_folder = "path/to/mask_folder"  # .png binary segmentation masks

# Load data
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
mask_paths = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith('.png')]

# Preprocess data
def load_data(image_paths, mask_paths):
    images, boxes = [], []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Find bounding box from the binary mask
        y_indices, x_indices = np.where(mask == 255)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        images.append(img)
        boxes.append([x_min, y_min, x_max, y_max])
    return np.array(images), np.array(boxes)

images, boxes = load_data(image_paths, mask_paths)

# Split data into training and validation
train_images, val_images, train_boxes, val_boxes = train_test_split(images, boxes, test_size=0.2, random_state=42)

# Custom Dataset class
class CamouflageDataset(Dataset):
    def __init__(self, images, boxes, transform=None):
        self.images = images
        self.boxes = boxes
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.resize(self.images[idx], (224, 224)) / 255.0
        box = self.boxes[idx]
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        if self.transform:
            img = self.transform(img)
        box = torch.tensor(box, dtype=torch.float32)
        return img, box

# Data Augmentation and Transformations
transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(10),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
])

# Data Loaders
train_dataset = CamouflageDataset(train_images, train_boxes, transform=transform)
val_dataset = CamouflageDataset(val_images, val_boxes)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model Definition
class BoundingBoxModel(nn.Module):
    def __init__(self):
        super(BoundingBoxModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

model = BoundingBoxModel()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, boxes in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, boxes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, boxes in val_loader:
                outputs = model(images)
                loss = criterion(outputs, boxes)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

# Plot training history
plt.plot(train_losses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save model
torch.save(model.state_dict(), 'camouflaged_bbox_model.pth')
