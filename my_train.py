import json
import pprint
from os.path import join
import glob

import numpy as np
import numpy.linalg as la
import open3d as o3
import cv2
import matplotlib.pylab as pt
import matplotlib.pyplot as plt

root_path = '/homes/math/golombiewski/workspace/data/A2D2'
config_path = '/homes/math/golombiewski/workspace/data/A2D2_general/cams_lidars.json'
# print(f'Root path set to:\n {root_path}\nConfig path set to:\n {config_path}')

with open (config_path, 'r') as f:
    config = json.load(f)

file_names = sorted(glob.glob(join(root_path, '*/lidar/cam_front_center/*.npz')))

print(f'Number of lidar point clouds for cam_front_center angle:\n {len(file_names)}')

# select the lidar point cloud
file_name_lidar = file_names[0]

# read the lidar data
lidar_front_center = np.load(file_name_lidar)

print(list(lidar_front_center.keys()))

points = lidar_front_center['points']
reflectance = lidar_front_center['reflectance']
timestamps = lidar_front_center['timestamp']
rows = lidar_front_center['row']
cols = lidar_front_center['col']
distance = lidar_front_center['distance']
depth = lidar_front_center['depth']
lidar_ids = lidar_front_center['lidar_id']

# Create array of RGB colour values from the given array of reflectance values
def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)

def create_open3d_pc(lidar, cam_image=None):
    # create open3d point cloud
    pcd = o3.geometry.PointCloud()
    
    # assign point coordinates
    pcd.points = o3.utility.Vector3dVector(lidar['points'])
    
    # assign colours
    if cam_image is None:
        median_reflectance = np.median(lidar['reflectance'])
        colours = colours_from_reflectances(lidar['reflectance']) / (median_reflectance * 5)
        
        # clip colours for visualisation on a white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar['row'] + 0.5).astype(int)
        cols = (lidar['col'] + 0.5).astype(int)
        colours = cam_image[rows, cols, :] / 255.0
        
    pcd.colors = o3.utility.Vector3dVector(colours)
    
    return pcd

pcd_front_center = create_open3d_pc(lidar_front_center)