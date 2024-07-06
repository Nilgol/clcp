import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from a2d2_dataset import A2D2Dataset


def test_a2d2_dataset():
    # Initialize dataset
    root_path = "/path/to/a2d2/dataset"
    config_path = "/path/to/cams_lidars.json"
    dataset = A2D2Dataset(root_path, config_path, crop_size=(224, 224))

    # Check data pairs
    print(f"Total data pairs: {len(dataset)}")
    for i in range(5):  # Check 5 random pairs
        idx = np.random.randint(0, len(dataset))
        lidar_path, image_path = dataset.data_pairs[idx]
        print(f"Pair {i}:")
        print(f"  Lidar path: {lidar_path}")
        print(f"  Image path: {image_path}")
        print(f"  Image size: {cv2.imread(image_path).shape}")
        print(f"  Number of points: {np.load(lidar_path)['points'].shape[0]}")

    # Test __getitem__ method
    for i in range(5):  # Check 5 random preprocessed pairs
        idx = np.random.randint(0, len(dataset))
        image_tensor, points_tensor = dataset[idx]
        print(f"Preprocessed pair {i}:")
        print(f"  Image tensor shape: {image_tensor.shape}")
        print(f"  Points tensor shape: {points_tensor.shape}")

        # Visualize point cloud on image
        visualize_point_cloud_on_image(image_tensor, points_tensor)

def visualize_point_cloud_on_image(image_tensor, points_tensor):
    image = image_tensor.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
    points = points_tensor.numpy()
    plt.imshow(image)
    plt.scatter(points[:, 1], points[:, 0], s=1, c=points[:, 2], cmap='viridis')
    plt.show()

if __name__ == "__main__":
    test_a2d2_dataset()
