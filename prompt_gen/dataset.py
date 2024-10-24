import os
import random
import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm


def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1
    )
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


# Dataset for training, with bounding box extraction
class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        # Get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        # Sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # Filter matching pairs
        self.filter_files()
        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        # Dataset size
        self.size = len(self.images)
        print('>>> training/validating with {} samples'.format(self.size))

    def __getitem__(self, index):
        # Read assets/gts
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # Data augmentation
        image, gt = cv_random_flip(image, gt)
        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)
        image = colorEnhance(image)
        gt = randomPeper(gt)

        # Transform to tensor
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        # Derive bounding box from the binary mask
        bbox = self.get_bounding_box(gt)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float)

        return image, gt, bbox_tensor

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def get_bounding_box(self, mask):
        mask_np = mask.squeeze().numpy()  # Convert mask to numpy array
        rows, cols = np.where(mask_np > 0)
        if len(rows) == 0 or len(cols) == 0:
            return [0, 0, 0, 0]  # No object detected, return empty bbox

        # Bounding box coordinates (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = np.min(cols), np.min(rows), np.max(cols), np.max(rows)
        return [x_min, y_min, x_max, y_max]

    def __len__(self):
        return self.size


# DataLoader for training, integrating the new dataset with bbox
def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    # Wrapping the data loader with tqdm for progress bar
    # data_loader = tqdm(data_loader, desc="Training", total=len(data_loader), ncols=100)
    
    return data_loader

def get_test_loader(image_root, gt_root, batchsize, testsize, shuffle=False, num_workers=4, pin_memory=True):
    dataset = TestDataset(image_root=image_root, gt_root=gt_root, testsize=testsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    # Wrapping the data loader with tqdm for progress bar
    # data_loader = tqdm(data_loader, desc="Testing", total=len(data_loader), ncols=100)
    
    return data_loader

# test dataset and loader
class TestDataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        # Get the file paths for images and ground truth masks
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

        # Sort images and ground truths to ensure they match
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        
        # Transformation for images and masks
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        # Load image and mask
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])

        # Apply transformations
        image = self.transform(image).unsqueeze(0)  # Add batch dimension for the image
        gt = self.gt_transform(gt)

        # Extract bounding box from ground truth mask
        bbox = self.get_bounding_box(gt)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float)

        # Get image name for saving or logging results
        name = os.path.basename(self.images[self.index])
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        # Increment index for the next image, wrap around if needed
        self.index += 1
        self.index = self.index % self.size

        # Load image for post-processing (used for visualization or saving purposes)
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        return image, gt, bbox_tensor, name, np.array(image_for_post)

    def get_bounding_box(self, mask):
        # Convert mask tensor to numpy
        mask_np = mask.squeeze().numpy()  # Remove channel dimension if it exists
        rows, cols = np.where(mask_np > 0)
        if len(rows) == 0 or len(cols) == 0:
            return [0, 0, 0, 0]  # No object detected, return an empty bbox

        # Bounding box coordinates (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = np.min(cols), np.min(rows), np.max(cols), np.max(rows)
        return [x_min, y_min, x_max, y_max]

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size