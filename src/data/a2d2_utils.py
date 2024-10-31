"""Some utility functions used in the A2D2 dataset class, most are extracted from the official
A2D2 dataset tutorial notebook."""
from typing import Tuple, Any
import json
import numpy as np
import cv2

def random_crop(image: np.ndarray, combined_points: np.ndarray, crop_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly crops an image and filters its associated point cloud to match the cropped area.

    Args:
        image (np.ndarray): The input image to be cropped, with shape (H, W, 3).
        combined_points (np.ndarray): An array representing point cloud data associated with the image.
            Shape is (num_points, 6), where columns represent x, y, z coordinates, reflectance, and
            pixel row and column indices within the original image.
        crop_size (Tuple[int, int]): The dimensions (width, height) of the crop.

    Raises:
        ValueError: If `crop_size` is larger than the `image` dimensions.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - cropped_image (np.ndarray): The randomly cropped image with shape (crop_height, crop_width, 3).
            - new_points (np.ndarray): Filtered point cloud data within the cropped region, with adjusted row
              and column indices.
    """
    height, width, _ = image.shape
    crop_width, crop_height = crop_size

    if crop_width > width or crop_height > height:
        raise ValueError("Crop size must be smaller than image size")

    left = np.random.randint(0, width - crop_width)
    upper = np.random.randint(0, height - crop_height)

    cropped_image = image[upper : upper + crop_height, left : left + crop_width, :]

    mask = (
        (combined_points[:, 4] >= upper)
        & (combined_points[:, 4] < upper + crop_height)
        & (combined_points[:, 5] >= left)
        & (combined_points[:, 5] < left + crop_width)
    )

    new_points = combined_points[mask]
    new_points[:, 4] -= upper
    new_points[:, 5] -= left

    return cropped_image, new_points

# From the official A2D2 dataset tutorial notebook
def load_config(config_path):
    """Load the A2D2 dataset configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)

# From the official A2D2 dataset tutorial notebook
def undistort_image(image, cam_name, config):
    """Undistort an image using the camera parameters from the A2D2 dataset configuration file."""
    if cam_name in [
        "front_left",
        "front_center",
        "front_right",
        "side_left",
        "side_right",
        "rear_center",
    ]:
        # get parameters from config file
        intr_mat_undist = np.asarray(config["cameras"][cam_name]["CamMatrix"])
        intr_mat_dist = np.asarray(config["cameras"][cam_name]["CamMatrixOriginal"])
        dist_parms = np.asarray(config["cameras"][cam_name]["Distortion"])
        lens = config["cameras"][cam_name]["Lens"]

        if lens == "Fisheye":
            return cv2.fisheye.undistortImage(
                image, intr_mat_dist, D=dist_parms, Knew=intr_mat_undist
            )
        if lens == "Telecam":
            return cv2.undistort(
                image,
                intr_mat_dist,
                distCoeffs=dist_parms,
                newCameraMatrix=intr_mat_undist,
            )
        else:
            return image
    else:
        return image


### The following funtions are for testing purposes only

def collect_point_retention_ratios(dataset: Any, num_samples: int) -> np.ndarray:
    """Collects the ratio of points retained in each point cloud after random cropping.

    This function randomly samples point clouds from the provided dataset, applies random cropping, 
    and calculates the ratio of points retained in the cropped region to the original number of points.

    Args:
        dataset (A2D2Dataset): A dataset object that supports indexing and returns an image and point
            cloud tensor. Each entry in `dataset.data_pairs` contains the file path to the original
            point cloud.
        num_samples (int): The number of samples to collect for calculating retention ratios.

    Returns:
        np.ndarray: An array of shape (num_samples,) containing the retention ratio of points for each sample.
    """
    ratios = np.zeros(num_samples)
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        _, points_tensor = dataset[idx]
        original_num_points = np.load(dataset.data_pairs[idx][0])["points"].shape[0]
        points_retained_ratio = points_tensor.shape[0] / original_num_points
        ratios[i] = points_retained_ratio
        print(i + 1)
    return ratios

# From the official A2D2 dataset tutorial notebook
def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    """Project lidar points to the corresponding camera image."""
    image = np.copy(image_orig)

    # get rows and cols
    rows = (lidar["row"] + 0.5).astype(np.intc)
    cols = (lidar["col"] + 0.5).astype(np.intc)

    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar["distance"])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar["distance"])

    # get distances
    distances = lidar["distance"]
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray(
        [np.asarray(hsv_to_rgb(0.75 * c, np.sqrt(pixel_opacity), 1.0)) for c in colours]
    )
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = (1.0 - pixel_opacity) * np.multiply(
            image[pixel_rows, pixel_cols, :], colours[i]
        ) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)

# From the official A2D2 dataset tutorial notebook
def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB."""
    if s == 0.0:
        return v, v, v

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
