import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

"""Set seed for reproducebility"""
def set_seed(seed: int= 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""
Create a random smooth deformation field.
images:
    [B, 1, H, W]
returns:
    deformation:
    [B, 2, H, W]
deformation[:, 0] = dx
deformation[:, 1] = dy
"""
def create_random_deformation(images,max_displacement=5.0):
    B, C, H, W = images.shape

    # Random displacement
    displacement = torch.randn(B,2, H, W,device=images.device,)

    # Smooth the displacement
    displacement = F.avg_pool2d( displacement,kernel_size=15,stride=1,padding=7,)

    # Normalize
    displacement = displacement / (
        displacement.abs().amax(dim=(1, 2, 3),keepdim=True) + 1e-8
    )

    displacement = displacement * max_displacement

    return displacement

"""
Warp an image using a displacement field.
image:
    [B, C, H, W]
displacement:
    [B, 2, H, W]
returns:
    warped image
    [B, C, H, W]
"""
def warp_image(image, displacement):
    B, C, H, W = image.shape
    # Create identity grid
    y, x = torch.meshgrid(
        torch.arange(H, device=image.device),
        torch.arange(W, device=image.device),
        indexing="ij",
    )
    grid = torch.stack((x, y),dim=-1,).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    # Add displacement
    grid = grid + displacement.permute(0, 2, 3, 1 )
    # Convert pixel coordinates to [-1, 1]
    grid_x = 2.0 * grid[..., 0] / (W - 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / (H - 1) - 1.0
    normalized_grid = torch.stack((grid_x, grid_y),dim=-1,)
    # Sample image
    warped = F.grid_sample(image,normalized_grid,mode="bilinear",padding_mode="border",align_corners=True)

    return warped

"""Mean squared error"""
def similarity_loss(fixed,warped,):
    return torch.mean((fixed - warped) ** 2)

"""Encourage the deformation field to be smooth"""
def smoothness_loss( displacement):
    dx = displacement[:, :, :, 1:] - displacement[:, :, :, :-1]
    dy = displacement[:, :, 1:, :] - displacement[:, :, :-1, :]
    loss = (dx.pow(2).mean() +dy.pow(2).mean())
    return loss

"""Main training loop for voxelmorph"""
def train_voxelmorph(model: nn.Module, optimizer: torch.optim.Optimizer, num_epochs: int, device: torch.device, train_loader: torch.utils.data.DataLoader, results_dir: str, dataset_name: str) -> nn.Module:
    # Training
    smooth_lambda = 0.05
    loss_history = []
    for epoch in tqdm(range(1, num_epochs+1), desc= "Voxelmorph mnist training"):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            # Fixed image
            fixed = images
            # Create deformation
            true_displacement = create_random_deformation(fixed,max_displacement=5.0,)
            # Create moving image
            moving = warp_image(fixed,true_displacement,)

            # Predict deformation
            warped, predicted_displacement = model(moving,fixed,)
            # Loss
            loss_image = similarity_loss(fixed,warped)
            loss_smooth = smoothness_loss(predicted_displacement)
            loss = loss_image+ smooth_lambda * loss_smooth
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        loss_history.append(epoch_loss)

        tqdm.write(f"Epoch [{epoch}/{num_epochs}] Loss: {epoch_loss:.6f}")

    # Plot training loss
    os.makedirs(results_dir, exist_ok= True)
    fig= plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"VoxelMorph {dataset_name} Training Loss")
    plt.tight_layout()
    save_path = f"{results_dir}/{dataset_name}_train_loss.png"
    plt.savefig(save_path,dpi=200)
    plt.close(fig)
    print(f"[Save Train] Training results saved in {save_path}")

    return model

"""Run voxelmorph inference on the trained model"""
def run_inference(
    model: nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    results_dir: str, 
    dataset_name: str,
    grid_step: int = 4
):
    os.makedirs(results_dir, exist_ok=True)
    save_path = f"{results_dir}/{dataset_name}_registration_results.png"

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        fixed = images
        
        # Create synthetic deformation field and warp image
        true_displacement = create_random_deformation(fixed, max_displacement=5.0)
        moving = warp_image(fixed, true_displacement)
        
        # Model prediction
        warped, predicted_displacement = model(moving, fixed)

    num_examples = min(3, images.shape[0])
    
    # Created 5 columns: Moving, Fixed, Deformation Field, Warped, Difference
    fig, axes = plt.subplots(num_examples, 5, figsize=(15, 3 * num_examples))

    # Spatial coordinates for quiver overlay
    H, W = fixed.shape[2], fixed.shape[3]
    Y, X = np.mgrid[0:H:grid_step, 0:W:grid_step]

    for i in tqdm(range(num_examples), desc="Inferencing"):
        # Moving Image
        axes[i, 0].imshow(moving[i, 0].cpu(), cmap="gray")
        if i == 0:
            axes[i, 0].set_title("Moving", fontsize=11, fontweight="bold")

        # Fixed Image
        axes[i, 1].imshow(fixed[i, 0].cpu(), cmap="gray")
        if i == 0:
            axes[i, 1].set_title("Fixed", fontsize=11, fontweight="bold")

        # Predicted Deformation Field (Quiver Plot)
        flow = predicted_displacement[i].cpu().numpy()
        U = flow[0, ::grid_step, ::grid_step]
        V = flow[1, ::grid_step, ::grid_step]
        
        axes[i, 2].imshow(moving[i, 0].cpu(), cmap="gray")
        axes[i, 2].quiver(
            X, Y, U, V, 
            color='red', 
            angles='xy', 
            scale_units='xy', 
            scale=0.5,       # Lower scale = larger arrow lengths 
            width=0.007,     # Increased arrow shaft thickness
            headwidth=4.5,   # Wider arrowhead
            headlength=5.0,  # Longer arrowhead
            alpha=0.9
        )
        if i == 0:
            axes[i, 2].set_title("Deformation Field", fontsize=11, fontweight="bold")

        # Warped Image
        axes[i, 3].imshow(warped[i, 0].cpu(), cmap="gray")
        if i == 0:
            axes[i, 3].set_title("Warped", fontsize=11, fontweight="bold")

        # Difference Image
        difference = torch.abs(fixed[i, 0] - warped[i, 0])
        axes[i, 4].imshow(difference.cpu(), cmap="hot")
        if i == 0:
            axes[i, 4].set_title("Difference", fontsize=11, fontweight="bold")

        # Hide axes frames for clean display
        for j in range(5):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)  # Prevents memory accumulation and pop-up displays
    print(f"[Save Result] Image registration results saved to {save_path}")