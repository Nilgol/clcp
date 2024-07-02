import torch
import numpy as np
from torch.utils.data import DataLoader
from a2d2_dataset import A2D2Dataset
import matplotlib.pyplot as plt

def collate_fn(batch):
    images, point_clouds = zip(*batch)
    
    # Find the maximum number of points in a point cloud
    max_points = max(pc.shape[0] for pc in point_clouds)
    
    # Pad point clouds to have the same number of points
    padded_point_clouds = []
    for pc in point_clouds:
        if pc.shape[0] < max_points:
            # Pad with zeros
            padding = np.zeros((max_points - pc.shape[0], 3))
            pc = np.vstack([pc, padding])
        padded_point_clouds.append(pc)
    
    # Convert to numpy array before converting to tensor
    padded_point_clouds = np.array(padded_point_clouds)
    
    # Convert to tensors
    images = torch.stack(images)
    padded_point_clouds = torch.tensor(padded_point_clouds, dtype=torch.float32)
    
    return images, padded_point_clouds

root_path = '/homes/math/golombiewski/workspace/data/A2D2'
config_path = '/homes/math/golombiewski/workspace/data/A2D2_general/cams_lidars.json'

# Instantiate the dataset
dataset = A2D2Dataset(root_path, config_path)

# Creating a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Fetch a single batch
batch = next(iter(dataloader))
images, point_clouds = batch

# Check shapes
print("Image batch shape:", images.shape)  # Expected: torch.Size([4, 3, 1208, 1920])
print("Point cloud batch shape:", point_clouds.shape)  # Expected: torch.Size([4, 14062, 3])

# Check individual images and point clouds
for i in range(len(images)):
    print(f"Image {i} shape: {images[i].shape}")
    print(f"Point Cloud {i} shape: {point_clouds[i].shape}")

# Verify padding in the point clouds
for i in range(len(point_clouds)):
    pc = point_clouds[i].numpy()
    num_points = (pc[:, 0] != 0).sum()  # Assuming valid points have non-zero x-coordinate
    print(f"Point Cloud {i} has {num_points} valid points out of {pc.shape[0]}")

