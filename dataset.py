import torch
import numpy as np
from torchvision import datasets, transforms
import os
from typing import *

"""Custom Dataset for loading numpy .npz MRI data"""
class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, train=True, transform=None, num_samples=None):
        data = np.load(npz_path)
        
        # Load data using 'train' and 'validate' keys based on reference
        if train:
            self.images = data['train']
        else:
            self.images = data['validate']

        if num_samples is not None:
            self.images = self.images[:num_samples]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        
        # Ensure shape is (H, W, 1) for torchvision.transforms.ToTensor()
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        # Convert to float32 as required by PyTorch
        img = img.astype(np.float32)

        if self.transform:
            img = self.transform(img)

        # Return dummy label (0) to match the (images, labels) unpacking downstream
        return img, 0

"""Dataloader function return train and test loaders"""
def get_dataloader(dataset_name: str, data_dir: str, img_size: int, batch_size: int, num_train: int, num_test: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if dataset_name not in ["mnist", "mri"]:
        print(f"[Dataset Error] Input dataset name error, please choose only mnist or mri")
        raise Exception("Input dataset name error")
    
    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
    ])
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
        # Use only a subset to make the experiment faster
        train_dataset.data = train_dataset.data[: num_train]
        train_dataset.targets = train_dataset.targets[:num_train]
        test_dataset.data = test_dataset.data[:num_test]
        test_dataset.targets = test_dataset.targets[:num_test]
    else:
        # MRI Dataset
        data_path = f"{data_dir}/tutorial_data.npz"
        if not os.path.exists(data_path):
            print(f"[MRI Data Error] Please download data from https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/tutorial_data.tar.gz")
            raise Exception("MRI data path not found")
        train_dataset = MRIDataset(
            npz_path=data_path,
            train=True,
            transform=transform,
            num_samples=num_train
        )
        test_dataset = MRIDataset(
            npz_path=data_path,
            train=False,
            transform=transform,
            num_samples=num_test
        )

    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader