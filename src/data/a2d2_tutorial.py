"""The official A2D2 dataset tutorial notebook converted to a .py script."""
#!/usr/bin/env python
# coding: utf-8

# Welcome to the tutorial on the A2D2 Dataset. In this tutorial, we will learn how to work with the configuration file, the view item, the LiDAR data, camera images, and 3D bounding boxes.
#

# # Working with the configuration file

# The configuration JSON file plays an important role in processing the A2D2 dataset.

# In[1]:


import json
import pprint


# Open the configuration file

# In[2]:


with open("cams_lidars.json", "r") as f:
    config = json.load(f)


# Show config file

# In[3]:


pprint.pprint(config)


# The configuration file contains three main items:

# In[4]:


config.keys()


# Each sensor in 'lidars', 'cameras', and 'vehicle' has an associated 'view'. A view is a sensor coordinate system, defined by an origin, an x-axis, and a y-axis. These are specified in cartesian coordinates (in m) relative to an external coordinate system. Unless otherwise stated the external coordinate system is the car's frame of reference.
#
# 'vehicle' contains a 'view' object specifying the frame of reference of the car. It also contains an 'ego-dimensions' object, which specifies the extension of the vehicle in the frame of reference of the car.
#
# The 'lidars' object contains objects specifying the extrinsic calibration parameters for each LiDAR sensor. Our car has five LiDAR sensors: 'front_left', 'front_center', 'front_right', 'rear_right', and 'rear_left'. Each LiDAR has a 'view' defining its pose in the frame of reference of the car.
#
# The 'cameras' object contains camera objects which specify their calibration parameters. The car has six cameras: 'front_left', 'front_center', 'front_right', 'side_right', 'rear_center' and 'side_left'. Each camera object contains:
# - 'view'- pose of the camera relative to the external coordinate system (frame of reference of the car)
# - 'Lens'- type of lens used. It can take two values: 'Fisheye' or 'Telecam'
# - 'CamMatrix' - the intrinsic camera matrix of undistorted camera images
# - 'CamMatrixOriginal' - the intrinsic camera matrix of original (distorted) camera images
# - 'Distortion' - distortion parameters of original (distorted) camera images
# - 'Resolution' - resolution (columns, rows) of camera images (same for original and undistorted images)
# - 'tstamp_delay'- specifies a known delay in microseconds between actual camera frame times (default: 0)
#

# Display the contents of 'vehicle':

# In[5]:


config["vehicle"].keys()


# Likewise for LiDAR sensors:

# In[6]:


config["lidars"].keys()


# Here we see the names of the LiDAR sensors mounted on the car. For example, the configuration parameters for the front_left LiDAR sensor can be accessed using

# In[7]:


config["lidars"]["front_left"]


# The camera sensors mounted on the car can be obtained using

# In[8]:


config["cameras"].keys()


# Configuration parameters for a particular camera can be accessed using e.g.

# In[10]:


config["cameras"]["front_left"]


# ## Working with view objects

# We have seen that the vehicle and each sensor in the configuration file have a 'view' object. A view specifies the pose of a sensor relative to an external coordinate system, here the frame of reference of the car. In the following we use the term 'global' interchangeably with 'frame of reference of the car'.
#
# A view associated with a sensor can be accessed as follows:

# In[11]:


view = config["cameras"]["front_left"]["view"]


# In[12]:


import numpy as np
import numpy.linalg as la


# Define a small constant to avoid errors due to small vectors.

# In[13]:


EPSILON = 1.0e-10  # norm should not be small


# The following functions get the axes and origin of a view.

# In[14]:


def get_axes_of_a_view(view):
    x_axis = view["x-axis"]
    y_axis = view["y-axis"]

    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)

    if x_axis_norm < EPSILON or y_axis_norm < EPSILON:
        raise ValueError("Norm of input vector(s) too small.")

    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm

    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)

    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)

    # calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)

    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")

    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm

    return x_axis, y_axis, z_axis


# In[18]:


def get_origin_of_a_view(view):
    return view["origin"]


# A homogeneous transformation matrix from view point to global coordinates (inverse "extrinsic" matrix) can be obtained as follows. Note that this matrix contains the axes and the origin in its columns.

# In[19]:


def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)

    # get origin
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)

    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis

    # origin
    transform_to_global[0:3, 3] = origin

    return transform_to_global


# For the view defined above

# In[20]:


transform_to_global = get_transform_to_global(view)
print(transform_to_global)


# Homogeneous transformation matrix from global coordinates to view point coordinates ("extrinsic" matrix)

# In[21]:


def get_transform_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    trans = np.eye(4)
    rot = np.transpose(transform_to_global[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])

    return trans


# For the view defined above

# In[22]:


transform_from_global = get_transform_from_global(view)
print(transform_from_global)


# The transform_to_global and transform_from_global matrices should be the inverse of one another.
# Check that muliplying them results in an identity matrix (subject to numerical precision):

# In[23]:


print(np.matmul(transform_from_global, transform_to_global))


# The global-to-view rotation matrix can be obtained using

# In[24]:


def get_rot_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    # get rotation
    rot = np.transpose(transform_to_global[0:3, 0:3])

    return rot


# For the view defined above

# In[25]:


rot_from_global = get_rot_from_global(view)
print(rot_from_global)


# The rotation matrix from this view point to the global coordinate system can be obtained using

# In[26]:


def get_rot_to_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    # get rotation
    rot = transform_to_global[0:3, 0:3]

    return rot


# For the view defined above

# In[27]:


rot_to_global = get_rot_to_global(view)
print(rot_to_global)


# Let us see how we can calculate a rotation matrix from a source view to a target view

# In[28]:


def rot_from_to(src, target):
    rot = np.dot(get_rot_from_global(target), get_rot_to_global(src))

    return rot


# A rotation matrix from front left camera to front right camera

# In[25]:


src_view = config["cameras"]["front_left"]["view"]
target_view = config["cameras"]["front_right"]["view"]
rot = rot_from_to(src_view, target_view)
print(rot)


# A rotation matrix in the opposite direction (front right camera -> front left camera)

# In[26]:


rot = rot_from_to(target_view, src_view)
print(rot)


# In the same manner, we can also calculate a transformation matrix from a source view to a target view. This will give us a 4x4 homogeneous transformation matrix describing the total transformation (rotation and shift) from the source view coordinate system into the target view coordinate system.

# In[27]:


def transform_from_to(src, target):
    transform = np.dot(get_transform_from_global(target), get_transform_to_global(src))

    return transform


# A transformation matrix from front left camera to front right camera

# In[28]:


src_view = config["cameras"]["front_left"]["view"]
target_view = config["cameras"]["front_right"]["view"]
trans = transform_from_to(src_view, target_view)
print(trans)


# A transformation matrix in the opposite direction (front right camera -> front left camera)

# In[29]:


transt = transform_from_to(target_view, src_view)
print(transt)


# Check if the product of the two opposite transformations results in a near identity matrix.

# In[30]:


print(np.matmul(trans, transt))


# We have seen that by using views we can transform coordinates from one sensor to another, or from a sensor to global global coordinates (and vice versa). In the following section, we read point clouds corresponding to all cameras. The point clouds are in camera view coordinates. In order to get a coherent view of the point clouds, we need to transform them into global coordinates.

# # Working with LiDAR data
# First, read a LiDAR point cloud corresponding to the front center camera. The LiDAR data is saved in compressed numpy format, which can be read as follows:

# In[32]:


from os.path import join
import glob

# root_path = './camera_lidar_semantic_bboxes/'
root_path = "/homes/math/golombiewski/workspace/data/A2D2"
# get the list of files in lidar directory
file_names = sorted(glob.glob(join(root_path, "*/lidar/cam_front_center/*.npz")))

# select the lidar point cloud
file_name_lidar = file_names[7]

# read the lidar data
lidar_front_center = np.load(file_name_lidar)


# Let us explore the LiDAR data using the LiDAR points within the field of view of the front center camera.
# List keys:

# In[33]:


print(list(lidar_front_center.keys()))


# Get 3D point measurements

# In[37]:


points = lidar_front_center["points"]


# Get reflectance measurements

# In[38]:


reflectance = lidar_front_center["reflectance"]


# Get timestamps

# In[40]:


timestamps = lidar_front_center["timestamp"]


# Get coordinates of LiDAR points in image space

# In[41]:


rows = lidar_front_center["row"]
cols = lidar_front_center["col"]


# Get distance and depth values

# In[42]:


distance = lidar_front_center["distance"]
depth = lidar_front_center["depth"]


# Since the car is equipped with five LiDAR sensors, you can get the LiDAR sensor ID of each point using

# In[43]:


lidar_ids = lidar_front_center["lidar_id"]


# One way of visualizing point clouds is to use the Open3D library. The library supports beyond visualization other functionalities useful for point cloud processing. For more information on the library please refer to http://www.open3d.org/docs/release/.

# In[44]:


import open3d as o3


# To visualize the LiDAR point clouds, we need to create an Open3D point cloud from the 3D points and reflectance values. The following function generates colors based on the reflectance values.

# In[46]:


# Create array of RGB colour values from the given array of reflectance values
def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)


# Now we can create Open3D point clouds for visualization

# In[92]:


def create_open3d_pc(lidar, cam_image=None):
    # create open3d point cloud
    pcd = o3.geometry.PointCloud()

    # assign point coordinates
    pcd.points = o3.utility.Vector3dVector(lidar["points"])

    # assign colours
    if cam_image is None:
        median_reflectance = np.median(lidar["reflectance"])
        colours = colours_from_reflectances(lidar["reflectance"]) / (median_reflectance * 5)

        # clip colours for visualisation on a white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar["row"] + 0.5).astype(np.intc)
        cols = (lidar["col"] + 0.5).astype(np.intc)
        colours = cam_image[rows, cols, :] / 255.0

    pcd.colors = o3.utility.Vector3dVector(colours)

    return pcd


# Generate Open3D point cloud for the LiDAR data associated with the front center camera

# In[93]:


pcd_front_center = create_open3d_pc(lidar_front_center)


# Visualize the point cloud

# In[94]:


o3.visualization.draw_geometries([pcd_front_center])


# Let us transform LiDAR points from the camera view to the global view.
#
# First, read the view for the front center camera from the configuration file:

# In[95]:


src_view_front_center = config["cameras"]["front_center"]["view"]


# The vehicle view is the global view

# In[96]:


vehicle_view = target_view = config["vehicle"]["view"]


# The following function maps LiDAR data from one view to another. Note the use of the function 'transform_from_to'. LiDAR data is provided in a camera reference frame.

# In[97]:


def project_lidar_from_to(lidar, src_view, target_view):
    lidar = dict(lidar)
    trans = transform_from_to(src_view, target_view)
    points = lidar["points"]
    points_hom = np.ones((points.shape[0], 4))
    points_hom[:, 0:3] = points
    points_trans = (np.dot(trans, points_hom.T)).T
    lidar["points"] = points_trans[:, 0:3]

    return lidar


# Now project the LiDAR points to the global frame (the vehicle frame of reference)

# In[98]:


lidar_front_center = project_lidar_from_to(
    lidar_front_center, src_view_front_center, vehicle_view
)


# Create open3d point cloud for visualizing the transformed points

# In[99]:


pcd_front_center = create_open3d_pc(lidar_front_center)


# Visualise:

# In[100]:


o3.visualization.draw_geometries([pcd_front_center])


# For a more visible transformation:

# In[101]:


target_view = config["lidars"]["rear_right"]["view"]
lidar_front_center = project_lidar_from_to(
    lidar_front_center, src_view_front_center, target_view
)
pcd_front_center = create_open3d_pc(lidar_front_center)
o3.visualization.draw_geometries([pcd_front_center])


# # Working with images

# Import the necessary packages for reading, saving and showing images.

# In[102]:


import cv2

get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pylab as pt


# Let us load the image corresponding to the above point cloud.

# In[103]:


def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split("/")
    file_name_image = file_name_image[-1].split(".")[0]
    file_name_image = file_name_image.split("_")
    file_name_image = (
        file_name_image[0]
        + "_"
        + "camera_"
        + file_name_image[2]
        + "_"
        + file_name_image[3]
        + ".png"
    )

    return file_name_image


# In[104]:


seq_name = file_name_lidar.split("/")[7]
file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)
file_name_image = join(root_path, seq_name, "camera/cam_front_center/", file_name_image)
image_front_center = cv2.imread(file_name_image)

# print(f"file_name_lidar: {file_name_lidar}")
# print(f"seq_name: {seq_name}")
# print(f"file_name_image: {file_name_image}")


# Display image

# In[105]:


image_front_center = cv2.cvtColor(image_front_center, cv2.COLOR_BGR2RGB)


# In[106]:


pt.fig = pt.figure(figsize=(15, 15))

# display image from front center camera
pt.imshow(image_front_center)
pt.axis("off")
pt.title("front center")


# In order to map point clouds onto images, or in order to color point clouds using colors drived from images, we need to perform distortion correction.

# In[107]:


def undistort_image(image, cam_name):
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
        elif lens == "Telecam":
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


# In[108]:


undist_image_front_center = undistort_image(image_front_center, "front_center")


# In[109]:


pt.fig = pt.figure(figsize=(15, 15))
pt.imshow(undist_image_front_center)
pt.axis("off")
pt.title("front center")


# Each image has a timestamp and a LiDAR point cloud associated with it. The timestamp information is saved for each image in JSON format. Let us open the file for the front center camera.

# In[110]:


file_name_image_info = file_name_image.replace(".png", ".json")


def read_image_info(file_name):
    with open(file_name, "r") as f:
        image_info = json.load(f)

    return image_info


image_info_front_center = read_image_info(file_name_image_info)


# Display the information for the front center camera

# In[111]:


pprint.pprint(image_info_front_center)


# We can see that the camera info contains the camera name, the time stamp in TAI (international atomic time) and a dictionary associating the LiDAR IDs with names of the LiDARs.

# The LiDAR points are already mapped onto the undistorted images. The rows and columns of the corresponding pixels are saved in the lidar data.

# Let us list the keys once again:

# In[112]:


lidar_front_center.keys()


# Print the timestamp of each point in the LiDAR measurement.

# In[113]:


pprint.pprint(lidar_front_center["timestamp"])


# Here we also see the timestamps of each measurement point in TAI. The camera is lagging behind the LiDAR points, i.e. the LiDAR measurements are taken before the corresponding image is captured. (timestamp_lidar-timestamp_camera)/(1000000) gives us the time difference between the measurement times of lidar data and the corresponding camera frame in seconds.

# In[114]:


def plot_lidar_id_vs_delat_t(image_info, lidar):
    timestamps_lidar = lidar["timestamp"]
    timestamp_camera = image_info["cam_tstamp"]
    time_diff_in_sec = (timestamps_lidar - timestamp_camera) / (1e6)
    lidar_ids = lidar["lidar_id"]
    pt.fig = pt.figure(figsize=(15, 5))
    pt.plot(time_diff_in_sec, lidar_ids, "go", ms=2)
    pt.grid(True)
    ticks = np.arange(len(image_info["lidar_ids"].keys()))
    ticks_name = []
    for key in ticks:
        ticks_name.append(image_info["lidar_ids"][str(key)])
    pt.yticks(ticks, tuple(ticks_name))
    pt.ylabel("LiDAR sensor")
    pt.xlabel("delta t in sec")
    pt.title(image_info["cam_name"])
    pt.show()


# If we plot the lidar_ids versus the time difference  for the front center camera we obtain

# In[115]:


plot_lidar_id_vs_delat_t(image_info_front_center, lidar_front_center)


# Now we use col and row to map the LiDAR data onto images. The first function we use converts HSV to RGB. Please refere to the wikipedia article https://en.wikipedia.org/wiki/HSL_and_HSV for more information.

# In[116]:


def hsv_to_rgb(h, s, v):
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


# The following function visualizes the mapping

# In[118]:


def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
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


# Visualise the mapping of the LiDAR point clouds onto the front center image

# In[119]:


image = map_lidar_points_onto_image(undist_image_front_center, lidar_front_center)


# Show the image

# In[120]:


pt.fig = pt.figure(figsize=(20, 20))
pt.imshow(image)
pt.axis("off")


# The same result can be obtained by mapping the LiDAR point clouds using intrinsic camera parameters.

# Let us finally open the semantic segmentation label corresponding to the above image

# In[121]:


def extract_semantic_file_name_from_image_file_name(file_name_image):
    file_name_semantic_label = file_name_image.split("/")
    file_name_semantic_label = file_name_semantic_label[-1].split(".")[0]
    file_name_semantic_label = file_name_semantic_label.split("_")
    file_name_semantic_label = (
        file_name_semantic_label[0]
        + "_"
        + "label_"
        + file_name_semantic_label[2]
        + "_"
        + file_name_semantic_label[3]
        + ".png"
    )

    return file_name_semantic_label


# In[122]:


seq_name = file_name_lidar.split("/")[7]
file_name_semantic_label = extract_semantic_file_name_from_image_file_name(file_name_image)
file_name_semantic_label = join(
    root_path, seq_name, "label/cam_front_center/", file_name_semantic_label
)
semantic_image_front_center = cv2.imread(file_name_semantic_label)


# Display the semantic segmentation label

# In[123]:


semantic_image_front_center = cv2.cvtColor(semantic_image_front_center, cv2.COLOR_BGR2RGB)


# In[124]:


pt.fig = pt.figure(figsize=(15, 15))
pt.imshow(semantic_image_front_center)
pt.axis("off")
pt.title("label front center")


# We can use the semantic segmentation label to colour lidar points. This creates a 3D semantic label for a given frame.

# First we need to undistort the semantic segmentation label.

# In[129]:


semantic_image_front_center_undistorted = undistort_image(
    semantic_image_front_center, "front_center"
)
pt.fig = pt.figure(figsize=(15, 15))
pt.imshow(semantic_image_front_center_undistorted)
pt.axis("off")
pt.title("label front center")


# In[127]:


pcd_lidar_colored = create_open3d_pc(
    lidar_front_center, semantic_image_front_center_undistorted
)


# Visualize the coloured lidar points

# In[128]:


o3.visualization.draw_geometries([pcd_lidar_colored])
