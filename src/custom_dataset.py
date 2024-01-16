import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, json_file, resize=None, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            json_file (str): Path to the JSON file containing labels.
            resize (Tuple[int, int]): Resize the image.
            transform (callable, optional): Transforms to apply to images.
        """
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        # Remove the normalization step
        # self.normalize = transforms.Normalize(mean=[0.485], std=[0.229])
        with open(json_file, 'r') as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list(self.labels.keys())[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Convert to grayscale if needed

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        if self.transform:
            image = self.transform(image)

        #image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        image = np.transpose(image, (0, 1 , 2))

        # Remove the normalization using transforms.Normalize
        # image = self.normalize(torch.tensor(image))

        label = self.labels[img_name]
        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.long)
        }

