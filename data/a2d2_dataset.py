import glob
import os
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from .a2d2_utils import load_config, undistort_image, random_crop

class A2D2Dataset(Dataset):
    def __init__(self,
                root_path = '/homes/math/golombiewski/workspace/data/A2D2',
                config_path = '/homes/math/golombiewski/workspace/data/A2D2_general/cams_lidars.json',
                missing_keys_file='/homes/math/golombiewski/workspace/clcl/data/missing_keys_files.pkl',
                missing_point_clouds_file='/homes/math/golombiewski/workspace/clcl/data/empty_point_clouds.pkl',
                image_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]),
                crop_size=(896, 896)):
        self.root_path = root_path
        self.image_transform = image_transform
        self.crop_size = crop_size
        self.config = load_config(config_path)
        self.missing_keys_files = self._load_missing_keys_file(missing_keys_file) 
        self.missing_point_clouds = self._load_missing_point_clouds_file(missing_point_clouds_file) 
        self.data_pairs = self._create_data_pairs()

    def _load_missing_keys_file(self, missing_keys_file):
        with open(missing_keys_file, 'rb') as f:
            missing_keys_files = pickle.load(f)
        return missing_keys_files
    
    def _load_missing_point_clouds_file(self, missing_point_clouds_file):
        with open(missing_point_clouds_file, 'rb') as f:
            missing_point_clouds = pickle.load(f)
        return missing_point_clouds
    
    def _create_data_pairs(self):
        lidar_paths = sorted(glob.glob(os.path.join(self.root_path, '*/lidar/cam_front_center/*.npz')))
        data_pairs = []

        for lidar_path in lidar_paths:
            if lidar_path in self.missing_keys_files or lidar_path in self.missing_point_clouds:                # print(f"Skipping file with missing keys: {lidar_path}")
                continue
            seq_name = lidar_path.split('/')[-4]  # Correct index to get the sequence name
            image_file_name = lidar_path.split('/')[-1].replace('lidar', 'camera').replace('.npz', '.png')
            image_path = os.path.join(self.root_path, seq_name, 'camera/cam_front_center', image_file_name)
            data_pairs.append((lidar_path, image_path))

        return data_pairs


    def __len__(self):
        return len(self.data_pairs)

    def _getitem_unsafe(self, idx):
        lidar_path, image_path = self.data_pairs[idx]

        lidar_data = np.load(lidar_path)

        if 'reflectance' not in lidar_data:
            print(f"Missing 'reflectance' in file: {lidar_path}")
            # Handle the missing key (e.g., assign default values or skip the file)
            reflectance = np.zeros((lidar_data['points'].shape[0], 1))  # Example default value
        else:
            reflectance = lidar_data['reflectance'].reshape(-1, 1)

        if 'row' not in lidar_data:
            print(f"Missing 'row' in file: {lidar_path}")
            row = np.zeros((lidar_data['points'].shape[0], 1))  # Default value
        else:
            row = lidar_data['row'].reshape(-1, 1)
            
        if 'col' not in lidar_data:
            print(f"Missing 'col' in file: {lidar_path}")
            col = np.zeros((lidar_data['points'].shape[0], 1))  # Default value
        else:
            col = lidar_data['col'].reshape(-1, 1)

        point_cloud = np.hstack((
            lidar_data['points'],
            reflectance,
            lidar_data['row'].reshape(-1, 1),
            lidar_data['col'].reshape(-1, 1)
        ))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        undistorted_image = undistort_image(image, 'front_center', self.config)

        if self.crop_size:
            undistorted_image, point_cloud = random_crop(undistorted_image, point_cloud, self.crop_size)

        undistorted_image = cv2.resize(undistorted_image, (224, 224))

        if self.image_transform:
            undistorted_image = self.image_transform(undistorted_image)

        points_tensor = torch.tensor(point_cloud[:, :4], dtype=torch.float32)

        return undistorted_image, points_tensor
    
    def __getitem__(self, idx):
        max_attempts = 100
        attempt = 0
        
        while attempt < max_attempts:
            image, point_cloud = self._getitem_unsafe(idx)
            if point_cloud.size(0) > 0:  # Check if the point cloud is not empty
                return image, point_cloud
            else:
                attempt += 1
                idx = (idx + 1) % len(self.data_pairs)  # Try the next index
        
        # If no valid pair is found after max_attempts
        raise RuntimeError(f"Could not find a valid sample after {max_attempts} attempts starting from index {idx}")



if __name__ == "__main__":
    # Basic functionality tests
    crop = 896
    dataset = A2D2Dataset(crop_size=(crop, crop))
    num_samples = 10

    # Check dataset length
    print(f"Total data pairs: {len(dataset)}")

    # ratios = collect_point_retention_ratios(dataset, num_samples)

    # Analyze the ratios
    # print(f"Mean points retained ratio: {ratios.mean()}")
    # print(f"Median points retained ratio: {np.median(ratios)}")
    # print(f"Min points retained ratio: {ratios.min()}")
    # print(f"Max points retained ratio: {ratios.max()}")
    # print(f"Standard deviation: {ratios.std()}")

    # Check random data pairs
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))

        lidar_path, image_path = dataset.data_pairs[idx]
        image_tensor, points_tensor = dataset[idx]
        print(f"Pair {i+1}:")
        # print(f"  Image size: {cv2.imread(image_path).shape}")
        # print(f"  Number of points: {np.load(lidar_path)['points'].shape[0]}")
        # print(f"  Image tensor shape: {image_tensor.shape}")
        # print(f"  Points tensor shape: {points_tensor.shape}")
        # print('Number of points after cropping:', points_tensor.shape[0])
        # print('Ratio of points after cropping:', points_tensor.shape[0] / np.load(lidar_path)['points'].shape[0])