"""A module to load corresponding image and point cloud pairs from A2D2 dataset for self-supervised
pretraining, implemented as a PyTorch Dataset. The class only uses unlabeled data.

Note that dataset-related filepaths are not currently set via config and should be set here."""
import glob
import os
import pickle
from typing import List, Tuple
from pathlib import Path

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset

from data.a2d2_utils import load_config, undistort_image, random_crop

# Default paths for A2D2 dataset and its configuration file
A2D2_ROOT_PATH = "/homes/math/<me>/workspace/data/A2D2"
A2D2_CONFIG_PATH = "/homes/math/<me>/workspace/data/A2D2_general/cams_lidars.json"

# Paths to files listing samples with missing keys or point clouds
MISSING_KEYS_FILE = str(Path(__file__).parent / "missing_keys_files.pkl")
MISSING_POINT_CLOUDS_FILE = str(Path(__file__).parent / "empty_point_clouds.pkl")

class A2D2Dataset(Dataset):
    """A PyTorch Dataset for loading and processing the A2D2 dataset.

    This dataset class supports loading pairs of unlabeled camera images and lidar point clouds
    from A2D2 dataset for multimodal pretraining. Samples are taken from the front center camera
    split (28,637 sample pairs) and only unlabeled data are used.
    
    Default transforms are joint cropping according go 'crop_size' of images and point clouds.
    Images are then resized to 224x224 pixels and normalized. More aggressive image augmentations
    to simulate adverse conditions can be applied using the 'augment' flag.
    Due to the sparsity of lidar data, random cropping might discard all points a the point cloud
    sample. In such cases, the dataset will try the next sample up to 100 times before raising an
    error (which might indicate too aggressive cropping). 

    The class supports a deterministic train/val split with a validation ratio of 'val_ratio'. The
    split is applied across scenes so that the last val_ratio*100 percent of samples from each scene
    are reserved for validation. The remaining samples are used for training.

    Since some of the datapoints are faulty (missing files or keys), we load a list of these
    datapoints from pickle files and skip them during dataset creation.

    Args:
        root_path (str): Root directory of the A2D2 semenatic segmentation dataset.
        config_path (str): Path to the A2D2 configuration file.
        missing_keys_file (str): Path to a file listing data samples with missing metadata.
        missing_point_clouds_file (str): Path to a file listing samples with missing point cloud data.
        crop_size (tuple): Size of the crop applied to images and corresponding point clouds.
        val_ratio (float): Proportion of the dataset reserved for validation (0 <= val_ratio <= 0.5).
        split (str): Specifies the dataset split ('train' or 'val').
        augment (bool): If True, applies additional image augmentations to simulate adverse conditions.

    Attributes:
        image_transform (albumentations.Compose): Image transformation pipeline, consisting of
            default transforms with or without additional augmentations. 
        data_pairs (list of tuples): Pairs of filepaths for lidar and camera data chosen according
        to 'split'.

    Raises:
        ValueError: If `split` is not 'train' or 'val'.
    """
    def __init__(
        self,
        root_path=A2D2_ROOT_PATH,
        config_path=A2D2_CONFIG_PATH,
        missing_keys_file=MISSING_KEYS_FILE,
        missing_point_clouds_file=MISSING_POINT_CLOUDS_FILE,
        crop_size=(896, 896),
        val_ratio=0.2,
        split="train",
        augment=False,
    ):
        self.root_path = root_path
        self.crop_size = crop_size
        self.config = load_config(config_path)

        # Some A2D2 sequences are missing keys or have empty point clouds which we stored in pickle
        # files and load here in order to skip them during dataset creation.
        self.missing_keys_files = self._load_missing_keys_file(missing_keys_file)
        self.missing_point_clouds = self._load_missing_point_clouds_file(
            missing_point_clouds_file
        )

        self.augment = augment
        self.image_transform = self._get_image_transform()

        self.val_ratio = self._validate_val_ratio(val_ratio)
        self.train_data_pairs, self.val_data_pairs = self._create_data_pairs()
        self.data_pairs = self._select_data_pairs(split)

    def _load_missing_keys_file(self, missing_keys_file):
        with open(missing_keys_file, "rb") as f:
            missing_keys_files = pickle.load(f)
        return missing_keys_files

    def _load_missing_point_clouds_file(self, missing_point_clouds_file):
        with open(missing_point_clouds_file, "rb") as f:
            missing_point_clouds = pickle.load(f)
        return missing_point_clouds

    def _get_default_transforms(self) -> A.Compose:
        """Return the default image transformations (normalize and convert to tensor)."""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _get_augmentations(self) -> A.Compose:
        """Return more aggressive image augmentations to simulate adverse conditions."""
        return A.Compose([
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=(-0.35, -0.2), contrast_limit=(-0.35, -0.2), p=0.2),
                A.ToGray(p=0.1),
            ], p=1.0),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 9), sigma_limit=0, p=1.0),
                A.GlassBlur(sigma=0.7, max_delta=1, iterations=2, mode="fast", p=1.0),
                A.GaussNoise(var_limit=(10.0, 500.0), mean=0, per_channel=True, p=1.0),
            ], p=0.5),
        ], p=0.9)

    def _get_image_transform(self) -> A.Compose:
        """Combine default transforms and augmentations based on the augment flag."""
        if self.augment:
            return A.Compose([self._get_augmentations(), self._get_default_transforms()])
        return self._get_default_transforms()

    def _validate_val_ratio(self, val_ratio: float) -> float:
        """Set val_ratio between 0 and 0.5 if it is not within that range."""
        if not 0 <= val_ratio <= 0.5:
            adjusted_val_ratio = max(0, min(val_ratio, 0.5))
            print(f"Warning: val_ratio should be between 0 and 0.5. Setting it to {adjusted_val_ratio}.")
            return adjusted_val_ratio
        return val_ratio

    def _select_data_pairs(self, split: str) -> List[Tuple[str, str]]:
        """Select either train or val datapairs based on provided split argument."""
        if split == "train":
            return self.train_data_pairs
        if split == "val":
            return self.val_data_pairs
        raise ValueError(f"Invalid split value: {split}. Use 'train' or 'val'.")

    def _create_data_pairs(self):
        """Create training and validation data pairs from available point cloud paths.
        
        Filters out paths with missing keys or point clouds and splits remaining paths across
        scenes into training and validation pairs based on `val_ratio`.

        Returns:
            Tuple: Lists of tuples containing (lidar_path, image_path) for training and validation.
        """
        lidar_paths = self._filter_valid_paths()
        scene_dict = self._organize_paths_by_scene(lidar_paths)

        return self._split_data_pairs(scene_dict)

    def _filter_valid_paths(self) -> List[str]:
        """Filters out paths with missing keys or empty point clouds.

        Returns:
            List[str]: List of valid lidar paths.
        """
        all_paths = glob.glob(os.path.join(self.root_path, "*/lidar/cam_front_center/*.npz"))
        return [
            path for path in all_paths
            if path not in self.missing_keys_files and path not in self.missing_point_clouds
        ]

    def _organize_paths_by_scene(self, lidar_paths: List[str]) -> dict:
        """Organizes lidar paths by sequence name.

        Args:
            lidar_paths (List[str]): List of valid lidar paths.

        Returns:
            dict: Dictionary where keys are sequence names and values are lists of lidar paths.
        """
        scene_dict = {}
        for path in lidar_paths:
            seq_name = path.split("/")[-4]
            if seq_name not in scene_dict:
                scene_dict[seq_name] = []
            scene_dict[seq_name].append(path)
        return scene_dict

    def _split_data_pairs(self, scene_dict: dict) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Splits lidar paths by scene into training and validation pairs.

        Args:
            scene_dict (dict): Dictionary where keys are sequence names and values are lists of lidar paths.

        Returns:
            Tuple: Two lists (one for train, one for val split), each contains pairs of filepaths
                to point cloud and corresponding image files.
        """
        train_data_pairs = []
        val_data_pairs = []

        for seq_name, lidar_paths in scene_dict.items():
            num_samples = len(lidar_paths)
            num_val_samples = int(num_samples * self.val_ratio)
            train_lidar_paths, val_lidar_paths = lidar_paths[:-num_val_samples], lidar_paths[-num_val_samples:]

            train_data_pairs.extend((lidar, self._generate_image_path(seq_name, lidar)) for lidar in train_lidar_paths)
            val_data_pairs.extend((lidar, self._generate_image_path(seq_name, lidar)) for lidar in val_lidar_paths)

        return train_data_pairs, val_data_pairs

    def _generate_image_path(self, seq_name: str, lidar_path: str) -> str:
        """Generates the corresponding image path for a given lidar path.

        Args:
            seq_name (str): Sequence name derived from lidar path.
            lidar_path (str): Original lidar path.

        Returns:
            str: Path to the corresponding image file.
        """
        image_file_name = lidar_path.split("/")[-1].replace("lidar", "camera").replace(".npz", ".png")
        return os.path.join(self.root_path, seq_name, "camera/cam_front_center", image_file_name)

    def __len__(self):
        return len(self.data_pairs)

    def _getitem_unsafe(self, idx):
        lidar_path, image_path = self.data_pairs[idx]
        lidar_data = np.load(lidar_path)

        point_cloud = np.hstack(
            (
                lidar_data["points"],
                lidar_data["reflectance"].reshape(-1, 1),
                lidar_data["row"].reshape(-1, 1),
                lidar_data["col"].reshape(-1, 1),
            )
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        undistorted_image = undistort_image(image, "front_center", self.config)

        if self.crop_size:
            undistorted_image, point_cloud = random_crop(
                undistorted_image, point_cloud, self.crop_size
            )

        undistorted_image = cv2.resize(undistorted_image, (224, 224))

        if self.image_transform:
            undistorted_image = self.image_transform(image=undistorted_image)["image"]

        points_tensor = torch.tensor(point_cloud[:, :4], dtype=torch.float32)

        return undistorted_image, points_tensor

    def __getitem__(self, idx):
        """Random crop might discard ALL points from the point cloud.
        If the point cloud is empty, try the next sample."""
        max_attempts = 100
        attempt = 0

        while attempt < max_attempts:
            image, point_cloud = self._getitem_unsafe(idx)
            if point_cloud.size(0) > 0:  # Point cloud non-empty?
                return image, point_cloud
            attempt += 1
            idx = (idx + 1) % len(self.data_pairs)  # Try next index

        raise RuntimeError(
            f"Could not find a valid sample after {max_attempts} attempts starting from index {idx}"
        )


# The following contains some code used for manual functionality testing
if __name__ == "__main__":
    crop = 896
    train_set = A2D2Dataset(crop_size=(crop, crop), val_ratio=0.1, split="train", augment=False)
    val_set = A2D2Dataset(crop_size=(crop, crop), val_ratio=0.1, split="val", augment=False)
    num_samples = 200
    output_dir = "/homes/math/<me>/workspace/fast/crop_resized_images"
    os.makedirs(output_dir, exist_ok=True)

    def denormalize_image(tensor, mean, std):
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        tensor = tensor * std + mean
        return tensor

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    import random

    indices = list(range(len(train_set)))
    random.shuffle(indices)

    # Iterate over the dataset and save the images
    for i in indices[:num_samples]:
        image, _ = train_set[i]

        # Denormalize the image
        image = denormalize_image(image.unsqueeze(0), mean, std).squeeze(0)

        # Convert tensor to a NumPy array for saving
        image_np = image.permute(1, 2, 0).numpy()  # CHW to HWC
        image_np = (image_np * 255).astype(np.uint8)

        # Save the image
        cv2.imwrite(
            os.path.join(output_dir, f"image_{i}.png"),
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
        )
        print(f"Saved image_{i}.png")

    print(f"{num_samples} cropped and resized images saved in '{output_dir}' directory.")

    # Check dataset length
    # print(f"Total train pairs: {len(train_set)}")
    # print(f"Total val pairs: {len(val_set)}")

    # ratios = collect_point_retention_ratios(dataset, num_samples)

    # Check random data pairs
    # for i in range(num_samples):
    #     idx = np.random.randint(0, len(dataset))

    #     lidar_path, image_path = dataset.data_pairs[idx]
    #     image_tensor, points_tensor = dataset[idx]
    #     print(f"Pair {i+1}:")
    # print(f"  Image size: {cv2.imread(image_path).shape}")
    # print(f"  Number of points: {np.load(lidar_path)['points'].shape[0]}")
    # print(f"  Image tensor shape: {image_tensor.shape}")
    # print(f"  Points tensor shape: {points_tensor.shape}")
    # print('Number of points after cropping:', points_tensor.shape[0])
    # print('Ratio of points after cropping:', points_tensor.shape[0] / np.load(lidar_path)['points'].shape[0])

    # Analyze the ratios
    # print(f"Mean points retained ratio: {ratios.mean()}")
    # print(f"Median points retained ratio: {np.median(ratios)}")
    # print(f"Min points retained ratio: {ratios.min()}")
    # print(f"Max points retained ratio: {ratios.max()}")
    # print(f"Standard deviation: {ratios.std()}")
