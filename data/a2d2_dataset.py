import glob
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from a2d2_utils import load_config, undistort_image, random_crop

class A2D2Dataset(Dataset):
    def __init__(self,
                 root_path = '/homes/math/golombiewski/workspace/data/A2D2',
                 config_path = '/homes/math/golombiewski/workspace/data/A2D2_general/cams_lidars.json',
                 image_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]),
                 crop_size=None):
        self.root_path = root_path
        self.image_transform = image_transform
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

        lidar_data = np.load(lidar_path)

        if 'reflectance' not in lidar_data:
            print(f"Missing 'reflectance' in file: {lidar_path}")
            # Handle the missing key (e.g., assign default values or skip the file)
            reflectance = np.zeros((lidar_data['points'].shape[0], 1))  # Example default value
        else:
            reflectance = lidar_data['reflectance'].reshape(-1, 1)

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

        points_tensor = torch.tensor(point_cloud[:, :4])

        return undistorted_image, points_tensor


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
        print(f"  Image size: {cv2.imread(image_path).shape}")
        print(f"  Number of points: {np.load(lidar_path)['points'].shape[0]}")
        print(f"  Image tensor shape: {image_tensor.shape}")
        print(f"  Points tensor shape: {points_tensor.shape}")
        print('Number of points after cropping:', points_tensor.shape[0])
        print('Ratio of points after cropping:', points_tensor.shape[0] / np.load(lidar_path)['points'].shape[0])