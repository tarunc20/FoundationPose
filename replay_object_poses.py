from os.path import join
import h5py
import yaml
import trimesh
import pyrender
import os
import genesis as gs
from tqdm import tqdm
import numpy as np
import cv2
import scipy 
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     OffscreenRenderer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import argparse 

def load_data(extrinsic_file, video_file, serials):
    with open(extrinsic_file, "r") as f:
        extrinsics = yaml.load(f, Loader=yaml.FullLoader)

    serial_to_extrinsic_tf = {}
    serial_to_intrinsic_tf = {}
    for cam_info in extrinsics:
        serial = cam_info["serial_number"]
        extrinsic = cam_info["transformation"]
        intrinsic = cam_info["color_intrinsic_matrix"]
        serial_to_extrinsic_tf[serial] = extrinsic
        serial_to_intrinsic_tf[serial] = intrinsic

    # load data from h5
    session_h5 = h5py.File(video_file, "r")

    # read camera intrinsic directly from h5
    camera_intrinsics = [serial_to_intrinsic_tf[serial] for serial in serials]
    camera_extrinsics = [serial_to_extrinsic_tf[serial] for serial in serials]
    num_frames = len(session_h5["imgs"])
    rgbs = session_h5["imgs"]
    depths = session_h5["depths"]

    return camera_extrinsics, camera_intrinsics, num_frames, rgbs, depths

def project_points_to_image(camera_intrinsics, camera_extrinsics, points_3d):
    """
    Project a group of 3D points back to 2D image points using the camera intrinsics and extrinsics.
    
    Args:
    - camera_intrinsics (np.ndarray): 3x3 intrinsic matrix of the camera.
    - camera_extrinsics (np.ndarray): 4x4 extrinsic matrix of the camera (rotation and translation).
    - points_3d (np.ndarray): Nx3 array of 3D points in world coordinates.
    
    Returns:
    - image_points (np.ndarray): Nx2 array of 2D points in image coordinates (u, v).
    """
    # Convert the 3D points to homogeneous coordinates (Nx4)
    points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # (778, 4)
    
    # Apply the inverse of the extrinsics to convert points to the camera frame
    extrinsics_inv = np.linalg.inv(camera_extrinsics)  # (4, 4)
    points_camera_frame = (extrinsics_inv @ points_3d_homogeneous.T).T  # (778, 4)
    
    # Perspective projection: project onto the image plane using the intrinsics
    points_projected_homogeneous = (camera_intrinsics @ points_camera_frame[:, :3].T).T  # (778, 3)
    
    # Convert from homogeneous coordinates to 2D
    u = points_projected_homogeneous[:, 0] / points_projected_homogeneous[:, 2]
    v = points_projected_homogeneous[:, 1] / points_projected_homogeneous[:, 2]
    
    # Stack into Nx2 array
    image_points = np.vstack([u, v]).T  # (778, 2)
    
    return image_points

def list_of_lists(a):
    return list(map(lambda x: list(x), list(a)))
NUM_FRAMES = 400
# Get extrinsic and intrinsics
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str)
    parser.add_argument("--camera_file", type=str)
    parser.add_argument("--camera_idx", type=int)
    parser.add_argument("--mesh_path", type=str)
    parser.add_argument("--out_dir", type=str) 
    #parser.add_argument("--object_poses", type=str) # directory containing object poses
    args = parser.parse_args()

    h5_file = args.h5_file
    extrinsic_file = args.camera_file
    serials = ['244222072252', '250122071059', '125322060645', '246422071990', '246422071818', '246422070730', '250222072777', '204222063088']

    object_mesh_path = args.mesh_path
    object_poses_dir = f"{args.out_dir}/scene_center_poses_{args.camera_idx}_tool_True_avg_False_tex.npy"
    object_center_transforms = np.load(object_poses_dir)
    object_trimesh = trimesh.load(args.mesh_path)
    object_trimesh.vertices *= 0.001

    to_origin, extents = trimesh.bounds.oriented_bounds(object_trimesh)
    camera_extrinsics, camera_intrinsics, num_frames, rgbs, depths = load_data(extrinsic_file, h5_file, serials)
    object_trimesh.apply_transform(to_origin)
    orig_vertices = object_trimesh.vertices.copy()
    all_ims = []
    registration_idxs = np.load(f"{args.out_dir}/registration_idxs_tool_True_avg_False_tex.npy")
    for i in tqdm(range(NUM_FRAMES)):
        # compute transform
        object_trimesh.vertices = orig_vertices.copy()
        object_trimesh.apply_transform(object_center_transforms[i])
        object_points_2d = project_points_to_image(camera_intrinsics[args.camera_idx], np.eye(4), object_trimesh.vertices)
        image_width, image_height = 640, 480
        background_image = np.flip(rgbs[i, args.camera_idx], axis=-1)

        points_2d = object_points_2d[::200, :]
        points_2d = points_2d[points_2d[:, 0] >= 0]
        points_2d = points_2d[points_2d[:, 1] >= 0]
        points_2d = points_2d[points_2d[:, 1] < 480]
        points_2d = points_2d[points_2d[:, 0] < 640]
        points_2d = points_2d.astype(np.uint64)
        color = None 
        if i in registration_idxs:
            color = np.array([[255, 0, 0]])
        else:
            color = np.array([[0, 0, 255]])
        background_image[points_2d[:, 1], points_2d[:, 0]] =color 
        all_ims.append(background_image)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{args.out_dir}/visualization_{args.camera_idx}_True_avg_False_tex.mp4", fourcc, 30, (640, 480))
    for i in range(NUM_FRAMES):
        frame = all_ims[i]
        video.write(frame[:, :, ::-1])
    video.release()