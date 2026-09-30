"""
PyTorch implementation for voxelmorph written by WK Ong at UiO 30.09.2026
Reference: https://colab.research.google.com/drive/1WiqyF7dCdnNBIANEY80Pxw_mVz4fyV-S?usp=sharing#scrollTo=G-8J-ebBlxVz
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import utils
import dataset
import models

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "mnist"
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_TRAIN = 10000
NUM_TEST = 1000
SEED = 42
CKPT_DIR = "./checkpoint"
DATA_DIR = "./data"
RESULT_DIR = "./results"

if __name__ == "__main__":
    # Set seed for reproducibility
    utils.set_seed(seed= SEED)
    print("[Device] Running Device:", DEVICE)
    # Dataloader
    train_loader, test_loader = dataset.get_dataloader(
        dataset_name= DATASET_NAME, data_dir= DATA_DIR, img_size= IMAGE_SIZE, batch_size= BATCH_SIZE, num_train= NUM_TRAIN, num_test= NUM_TEST
    )
    # Create model
    model = models.VoxelMorph().to(DEVICE)
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    # Run training
    model = utils.train_voxelmorph(
        model= model, optimizer= optimizer, num_epochs= EPOCHS, device= DEVICE, train_loader= train_loader, results_dir= RESULT_DIR, dataset_name= DATASET_NAME
    )
    # Save trained model
    os.makedirs(CKPT_DIR, exist_ok= True)
    ckpt_path = f"{CKPT_DIR}/model_{DATASET_NAME}.pth"
    torch.save(model.state_dict(),ckpt_path)
    print(f"[Save Model] Model saved to {ckpt_path}")
    # Run inference on the trained model
    utils.run_inference(
        model= model, test_loader= test_loader, device= DEVICE, results_dir= RESULT_DIR, dataset_name= DATASET_NAME
    )