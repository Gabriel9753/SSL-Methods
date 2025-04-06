import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_downloader.dataset_utils import get_dataset_dfs, load_images
from rich import print
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, preload_images=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preload_images = preload_images
        self.images = []
        if self.preload_images:
            self.images = load_images(self.image_paths)

        self.image_labels = labels if labels is not None else [None] * len(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.preload_images:
            image = self.images[index]
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
        else:
            image = Image.open(self.image_paths[index]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.image_labels[index]
        return image, label, self.image_paths[index]

