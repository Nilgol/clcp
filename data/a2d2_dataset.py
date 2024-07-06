import glob
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from a2d2_utils import load_config, undistort_image, transform_point_cloud_to_cam_view, random_crop

class A2D2Dataset(Dataset):
    def __init__(self,
                 root_path = '/homes/math/golombiewski/workspace/data/A2D2',
                 config_path = '/homes/math/golombiewski/workspace/data/A2D2_general/cams_lidars.json',
                 transform = transforms.ToTensor(),
                 crop_size=None):
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
        print('lidar_data type:', type(lidar_data))
        print('lidar_data keys:', list(lidar_data.keys()))
        print('lidar_data points type:', type(lidar_data['points']))
        print('lidar_data points shape:', lidar_data['points'].shape)
        points = lidar_data['points']
        lidar_view = self.config['lidars']['front_center']['view']
        camera_view = self.config['cameras']['front_center']['view']
        points_transformed = transform_point_cloud_to_cam_view(points, lidar_view, camera_view)
        print('points_transformed shape:', points_transformed.shape)
        # Add reflectance to point cloud dictionary
        point_cloud = {
            'x': points_transformed[:, 0],
            'y': points_transformed[:, 1],
            'z': points_transformed[:, 2],
            'reflectance': lidar_data['reflectance'],
            'row': lidar_data['row'],
            'col': lidar_data['col']
        }

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        undistorted_image = undistort_image(image, 'front_center', self.config)

        if self.crop_size:
            undistorted_image, point_cloud = random_crop(undistorted_image, point_cloud, self.crop_size)

        if self.transform:
            undistorted_image = self.transform(undistorted_image)

        # Convert image to tensor
        # undistorted_image = transforms.ToTensor(undistorted_image)

        # Convert point cloud to tensor
        points_tensor = torch.tensor(
            np.vstack((
                point_cloud['x'], 
                point_cloud['y'], 
                point_cloud['z'], 
                point_cloud['reflectance']
            )).T
        )

        return undistorted_image, points_tensor
    
if __name__ == "__main__":
    # Basic functionality tests
    dataset = A2D2Dataset()
    
    # Check dataset length
    print(f"Total data pairs: {len(dataset)}")

    dataset.__getitem__(0)
    
    # # Check random data pairs
    # for i in range(5):
    #     idx = np.random.randint(0, len(dataset))
    #     lidar_path, image_path = dataset.data_pairs[idx]
    #     print(f"Pair {i}:")
    #     print(f"  Lidar path: {lidar_path}")
    #     print(f"  Image path: {image_path}")
    #     print(f"  Image size: {cv2.imread(image_path).shape}")
    #     print(f"  Number of points: {np.load(lidar_path)['points'].shape[0]}")
    
    # # Test __getitem__
    # for i in range(5):
    #     idx = np.random.randint(0, len(dataset))
    #     image_tensor, points_tensor = dataset[idx]
    #     print(f"Preprocessed pair {i}:")
    #     print(f"  Image tensor shape: {image_tensor.shape}")
    #     print(f"  Points tensor shape: {points_tensor.shape}")