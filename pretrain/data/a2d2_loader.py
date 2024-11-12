"""Module to build a DataLoader for the A2D2 dataset with custom collation.

While images can be stacked to tensors in the usual way, we process point clouds as lists of tensors
due to their varying shapes. This module provides a custom collation function to prepare batches of
images and point clouds from individual samples.

Functions:
    collate_fn: Custom collation function to prepare batches of images and point clouds.
    build_loader: Function to construct the DataLoader for the A2D2 dataset with custom collation.
"""
from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tensor]]:
    """Collate function to create batches of images and point clouds from individual samples.

    Args:
        batch (list): List of samples where each sample is a tuple (image, point_cloud) of tensors.
    
    Returns:
        tuple: A tuple containing:
            - images (torch.Tensor): A tensor of batched images of shape (batch_size, C, H, W).
            - point_clouds (list): A list with (length == batch_size) of point cloud tensors,
            each of varying shape depending on the number of points in the point cloud.
    """
    images = [item[0] for item in batch]
    point_clouds = [item[1] for item in batch]

    images = torch.stack(images)
    return images, point_clouds


def build_loader(
    dataset: Dataset, 
    batch_size: int = 32, 
    num_workers: int = 16, 
    shuffle: bool = True
) -> DataLoader:
    """Build a DataLoader for the A2D2 dataset with custom batching.

    Args:
        dataset (torch.utils.data.Dataset): The dataset object, yielding (image, point_cloud) pairs.
        batch_size (int): Number of samples per batch (default is 32).
        num_workers (int): Number of subprocesses to use for data loading (default is 16).
        shuffle (bool): Whether to shuffle the dataset at every epoch (default is True).
    
    Returns:
        DataLoader: A DataLoader instance with the custom collate function for handling 
        (image, point_cloud) pairs.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return dataloader


if __name__ == "__main__":
    batch_size = 8
    num_workers = 4

    # Initialize the DataLoader
    dataloader = build_loader(batch_size=batch_size, num_workers=num_workers, shuffle=True)
    print(f"DataLoader initialized with batch size {batch_size} and {num_workers} workers.")

    # Fetch a few batches and print their shapes
    for i, (images, point_clouds) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Number of point clouds: {len(point_clouds)}")
        print(
            f"  First point cloud shape: {point_clouds[0].shape if len(point_clouds) > 0 else 'N/A'}"
        )
        if i == 2:  # Stop after 3 batches
            break
