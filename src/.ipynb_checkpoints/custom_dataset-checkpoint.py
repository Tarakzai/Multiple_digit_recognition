import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, json_file, resize=None ,transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            json_file (str): Path to the JSON file containing labels.
            resize    (int,int) : Resize the image.
            transform (callable, optional): Transforms to apply to images.
        """
        self.root_dir  = root_dir
        self.augment   = albumentations.Compose([albumentations.Normalize(always_apply=True)])
        self.transform = transform
        with open(json_file, 'r') as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list(self.labels.keys())[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Convert to grayscale if needed

        if self.resize is not None:
            image = image.resize((self.resize[1],self.resize[0]),resample=Image.BILINEAR)
        
        if self.transform: # Not working with this at the moment
            image = self.transform(image)

        image = np.array(image)
        augmented_image = self.augment(image)
        image = augmented_image["image"]

        image= np.transpose(image,(2, 0, 1)).astype(np.float32)
        label = self.labels[img_name]
        return {
            "images": torch.tensor(image,dtype=torch.float),
            "labels": torch.tensor(label,dtype=torch.long)
        }

# Define your transform, including resizing
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize to desired W,H
    transforms.ToTensor()
])