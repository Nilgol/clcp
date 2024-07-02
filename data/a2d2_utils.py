import json
import random
from PIL import Image
import numpy as np
import numpy.linalg as la
import cv2

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def undistort_image(image, cam_name, config):
    intr_mat_undist = np.asarray(config['cameras'][cam_name]['CamMatrix'])
    intr_mat_dist = np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
    dist_parms = np.asarray(config['cameras'][cam_name]['Distortion'])
    lens = config['cameras'][cam_name]['Lens']
    
    if lens == 'Fisheye':
        return cv2.fisheye.undistortImage(image, intr_mat_dist, D=dist_parms, Knew=intr_mat_undist)
    elif lens == 'Telecam':
        return cv2.undistort(image, intr_mat_dist, distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
    return image

def transform_point_cloud_to_cam_view(points, src_view, target_view):
    trans = get_transform_from_to(src_view, target_view)
    points_hom = np.ones((points.shape[0], 4))
    points_hom[:, 0:3] = points
    points_trans = (np.dot(trans, points_hom.T)).T 
    return points_trans[:, 0:3]

def get_transform_from_to(src_view, target_view):
    trans_to_global = get_transform_to_global(src_view)
    trans_from_global = get_transform_to_global(target_view)
    return np.dot(np.linalg.inv(trans_from_global), trans_to_global)

def get_transform_to_global(view):
    # Get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)
    
    # Get origin 
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)
    
    # Rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis
    
    # Origin
    transform_to_global[0:3, 3] = origin
    
    return transform_to_global


def get_axes_of_a_view(view):
    x_axis = np.array(view['x-axis'])
    y_axis = np.array(view['y-axis'])
     
    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)

    EPSILON = 1.0e-10  # Small constant to avoid division by zero
    
    if x_axis_norm < EPSILON or y_axis_norm < EPSILON:
        raise ValueError("Norm of input vector(s) too small.")
        
    # Normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm
    
    # Make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)
 
    # Create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)
    
    # Calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)
    
    if y_axis_norm < EPSILON or z_axis_norm < EPSILON:
        raise ValueError("Norm of view axis vector(s) too small.")
        
    # Make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm
    
    return x_axis, y_axis, z_axis

def get_origin_of_a_view(view):
    return np.array(view['origin'])

def random_crop(image, point_cloud, crop_size):
    """Randomly crop the image and adjust the point cloud."""
    width, height = image.size
    crop_width, crop_height = crop_size

    if crop_width > width or crop_height > height:
        raise ValueError("Crop size must be smaller than image size")

    left = random.randint(0, width - crop_width)
    upper = random.randint(0, height - crop_height)

    cropped_image = image.crop((left, upper, left + crop_width, upper + crop_height))

    # Adjust point cloud
    new_point_cloud = {k: [] for k in point_cloud}
    for i in range(len(point_cloud['x'])):
        if (left <= point_cloud['col'][i] < left + crop_width) and (upper <= point_cloud['row'][i] < upper + crop_height):
            new_point_cloud['x'].append(point_cloud['x'][i])
            new_point_cloud['y'].append(point_cloud['y'][i])
            new_point_cloud['z'].append(point_cloud['z'][i])
            new_point_cloud['reflectance'].append(point_cloud['reflectance'][i])
            new_point_cloud['row'].append(point_cloud['row'][i] - upper)
            new_point_cloud['col'].append(point_cloud['col'][i] - left)

    return cropped_image, new_point_cloud