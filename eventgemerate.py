from glob import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def load_frames(image_folder):
    """
    Load and convert all image frames from a folder into PyTorch tensors.

    Parameters:
    - image_folder (str): Path to the folder containing the image files.

    Returns:
    - list of torch.Tensor: A list of tensors corresponding to the frames.
    """
    # Create a transformation pipeline to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor
    ])

    # Use glob to find all JPEG images in the folder, sorted by filename
    image_files = sorted(glob.glob(f"{image_folder}/*.jpg"))

    frames = []
    for image_path in image_files:
        # Open the image
        image = Image.open(image_path).convert('RGB')  # Convert to RGB if needed

        # Apply the transformation pipeline
        tensor = transform(image)
        
        frames.append(tensor)

    return frames

def rgb_to_grayscale(tensor):
    """
    Convert an RGB image tensor to grayscale using the luminosity method.

    Parameters:
    - tensor (torch.Tensor): RGB image tensor of shape [3, H, W]

    Returns:
    - torch.Tensor: Grayscale image tensor of shape [H, W]
    """
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    grayscale = torch.sum(tensor * weights, dim=1)
    return grayscale

def generate_event_data(frames, threshold=None):
    """
    Generate event data from a sequence of RGB video frames, optionally applying a threshold.

    Parameters:
    - frames (list of torch.Tensor): The RGB video frames as a list of tensors.
    - threshold (int, optional): Activation threshold for pixel changes.

    Returns:
    - torch.Tensor: A tensor representing the event data.
    """
    event_frames = []
    
    # Convert all frames to grayscale first
    grayscale_frames = [rgb_to_grayscale(frame) for frame in frames]
    
    for i in range(1, len(grayscale_frames)):
        # Calculate the absolute difference between consecutive grayscale frames
        frame_diff = torch.abs(grayscale_frames[i] - grayscale_frames[i - 1])
        
        # Apply threshold if specified
        if threshold is not None:
            frame_diff = (frame_diff > threshold).int()
        
        event_frames.append(frame_diff)
    
    return torch.stack(event_frames)

def visualize_event_data(event_data, frame_interval=1, figsize_per_image=(6, 4)):
    """
    Visualize event data frames with adjustable figure size.

    Parameters:
    - event_data (torch.Tensor): The event data tensor.
    - frame_interval (int): Interval between frames to visualize. Default is 1, showing every frame.
    - figsize_per_image (tuple): Size of each image in inches (width, height).
    """
    num_frames = event_data.shape[0]
    # Calculate total figure size based on number of images to display
    total_width = figsize_per_image[0] * min(5, num_frames)
    total_height = figsize_per_image[1]
    
    fig, axes = plt.subplots(nrows=1, ncols=min(5, num_frames), figsize=(total_width, total_height))

    if num_frames == 1:
        axes = [axes]  # Make it iterable

    for i, ax in enumerate(axes):
        idx = i * frame_interval
        if idx < num_frames:
            # Ensure the data is in the correct shape (H, W) for grayscale images
            frame = event_data[idx].squeeze()  # This removes any single-dimensional entries from the dimensions
            if frame.dim() != 2:
                raise ValueError(f"Expected 2D tensor, got {frame.dim()}D tensor instead.")
            ax.imshow(frame.numpy(), cmap='gray')
            ax.set_title(f'Frame {idx}')
            ax.axis('off')
    plt.show()

def save_event_frames_as_png(event_data, save_path, threshold=None):
    """
    Save each frame in the event data tensor as a PNG file, applying a threshold if specified.

    Parameters:
    - event_data (torch.Tensor): The event data tensor of shape [N, 1, H, W].
    - save_path (str): Directory path where the PNG files will be saved.
    - threshold (float, optional): Threshold value to convert the frames into binary images. If None, saves as grayscale.
    """
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Loop through each frame in the tensor
    for i in tqdm(range(event_data.shape[0])):
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size if necessary
        # Squeeze to remove the channel dimension and convert tensor to numpy array
        frame = event_data[i].squeeze()

        # Apply threshold if provided
        if threshold is not None:
            frame = (frame > threshold).float()  # Convert to binary image

        ax.imshow(frame.numpy(), cmap='gray', aspect='auto')  # 'aspect' can be adjusted if needed
        ax.axis('off')  # Turn off axis
        # Save each frame as PNG
        fig.savefig(os.path.join(save_path, f'Frame_{i:05d}.png'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free up memory

image_folder = 'D:\Dateset\MoCA-Mask-Pseudo\MoCA-Video-Train\crab_1\Frame'
save_path = './figs/MoCA-crab_10-psudo'  # Define your save path


frames = load_frames(image_folder)
print(f"Loaded {len(frames)} frames.")


# Generate event data with a threshold of 15
# event_data = generate_event_data(frames, threshold=0.2)
event_data = generate_event_data(frames)
print(event_data.shape)  # Should print (num_frame
save_event_frames_as_png(event_data*255, save_path)