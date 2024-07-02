import glob
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from a2d2_utils import load_config, undistort_image, transform_point_cloud_to_cam_view, random_crop

class A2D2Dataset(Dataset):
    def __init__(self, root_path, config_path, transform=None, crop_size=None):
        self.root_path = root_path
        self.transform = transform
        self.crop_size = crop_size

        self.config = load_config(config_path)
        self.data_pairs = self._create_data_pairs()

    def _create_data_pairs(self):
        lidar_paths = sorted(glob.glob(os.path.join(self.root_path, '*/lidar/cam_front_center/*.npz')))
        data_pairs = []
        for lidar_path in lidar_paths:
            seq_name = lidar_path.split('/')[-4]  # Correct index to get the sequence name
            image_file_name = lidar_path.split('/')[-1].replace('lidar', 'camera').replace('.npz', '.png')
            image_path = os.path.join(self.root_path, seq_name, 'camera/cam_front_center', image_file_name)
            
            # Check if files exist
            if not os.path.exists(lidar_path):
                print(f"Lidar file does not exist: {lidar_path}")
            if not os.path.exists(image_path):
                print(f"Image file does not exist: {image_path}")
            
            data_pairs.append((lidar_path, image_path))
        return data_pairs


    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        lidar_path, image_path = self.data_pairs[idx]

        # Load and process Lidar data
        lidar_data = np.load(lidar_path)
        points = lidar_data['points']
        lidar_view = self.config['cameras']['front_center']['view']
        vehicle_view = self.config['vehicle']['view']
        points_transformed = transform_point_cloud_to_cam_view(points, lidar_view, vehicle_view)

        # Add reflectance to point cloud dictionary
        point_cloud = {
            'x': points_transformed[:, 0],
            'y': points_transformed[:, 1],
            'z': points_transformed[:, 2],
            'reflectance': points_transformed[:, 3],  # Assuming the 4th column is reflectance
            'row': points_transformed[:, 4],  # Assuming you have these values
            'col': points_transformed[:, 5]
        }

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        undistorted_image = undistort_image(image, 'front_center', self.config)

        if self.crop_size:
            undistorted_image, point_cloud = random_crop(undistorted_image, point_cloud, self.crop_size)

        if self.transform:
            undistorted_image = self.transform(undistorted_image)

        # Convert image to tensor
        image_tensor = transforms.ToTensor()(undistorted_image)

        # Convert point cloud to tensor
        points_tensor = torch.tensor(
            np.vstack((
                point_cloud['x'], 
                point_cloud['y'], 
                point_cloud['z'], 
                point_cloud['reflectance']
            )).T
        )

        return image_tensor, points_tensor