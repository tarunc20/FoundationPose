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
from PIL import Image, ImageDraw, ImageFont
import argparse 

import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import transformations

import numpy as np

ARIAL_FONT = ImageFont.truetype("arial.ttf", 18)
def add_caption(image_array, caption, font_path="arial.ttf", font_size=25, position=(10, 10), color=(255, 255, 255)):
    """
    Add a caption to an image stored as a NumPy array.
    
    Args:
    image_array (numpy.ndarray): Input image as a NumPy array.
    caption (str): Text to be added as caption.
    font_path (str): Path to the font file. Default is "arial.ttf".
    font_size (int): Font size. Default is 20.
    position (tuple): Position of the caption (x, y). Default is (10, 10).
    color (tuple): RGB color of the text. Default is white (255, 255, 255).
    
    Returns:
    numpy.ndarray: Image array with the caption added.
    """
    # Convert NumPy array to PIL Image
    image = Image.fromarray(image_array)
    
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    draw.text(position, caption, font=ARIAL_FONT, fill=color)
    return np.array(image)

def pixel_iou(pix1, pix2):
  intersection = [p for p in pix1 if p in pix2]
  union = intersection + [p for p in pix1 if p not in pix2] + [p for p in pix2 if p not in pix1]
  return np.round(len(intersection) / len(union), 3)

def slerp(q0, q1, t):
    # Ensure unit quaternions
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    
    # Calculate the dot product
    dot = np.dot(q0, q1)
    
    # If the dot product is negative, negate one quaternion to ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    # Threshold for linear interpolation for very close quaternions
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    
    # Calculate the angle between the quaternions
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    # Perform slerp
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q0) + (s1 * q1)

def slerp_4x4(R0, R1, t):
    # Extract 3x3 rotation matrices from 4x4 inputs
    q0 = transformations.quaternion_from_matrix(R0)[[3, 0, 1, 2]]
    q1 = transformations.quaternion_from_matrix(R1)[[3, 0, 1, 2]]
    t_res = t * R0[:3, 3] + (1 - t) * R1[:3, 3]
    q_res = slerp(q0, q1, t)[[1, 2, 3, 0]]
    M = transformations.quaternion_matrix(q_res)
    M[:3, 3] = t_res
    return M

def matrix_distance(T1, T2):
    # Extract rotation matrices (upper 3x3)
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    
    # Extract translation vectors
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    
    # Rotation distance (in radians)
    R_diff = np.arccos(np.clip((np.trace(R1.T @ R2) - 1)/2, -1, 1))
    #print(f"Rotation difference: {R_diff}")
    # Translation distance (in meters)
    t_diff = np.linalg.norm(t1 - t2)
    #print(f"Translation distance: {t_diff}")
    # Weight to make 1cm = 1degree
    weight = np.pi / (180 * 0.01)  # ≈ 1.745
    
    # Combined distance where 1cm error = 1degree error
    return R_diff + weight * t_diff, R_diff

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

def get_mask_indices(mask):
    return list(map(lambda x: list(x), list(zip(*np.where(mask != 0)))))

NUM_FRAMES = 350
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str)
    parser.add_argument("--camera_file", type=str)
    parser.add_argument("--camera_idx", type=int)
    parser.add_argument("--mesh_path", type=str)
    parser.add_argument("--out_dir", type=str) 
    parser.add_argument('--use_ransac', action="store_true")
    parser.add_argument('--use_pcd', action="store_true")
    #parser.add_argument("--object_poses", type=str) # directory containing object poses
    args = parser.parse_args()

    h5_file = args.h5_file
    extrinsic_file = args.camera_file
    serials = ['244222072252', '250122071059', '125322060645', '246422071990', '246422071818', '246422070730', '250222072777', '204222063088']

    object_mesh_path = args.mesh_path
    object_poses_dir = f"{args.out_dir}/scene_center_poses_{args.camera_idx}_tool_{args.use_ransac}_avg_{args.use_pcd}_t.npy"
    object_center_transforms = np.load(object_poses_dir)
    object_trimesh = trimesh.load(args.mesh_path)
    object_trimesh.vertices *= 0.001

    to_origin, extents = trimesh.bounds.oriented_bounds(object_trimesh)
    camera_extrinsics, camera_intrinsics, num_frames, rgbs, depths = load_data(extrinsic_file, h5_file, serials)
    object_trimesh.apply_transform(to_origin)
    orig_vertices = object_trimesh.vertices.copy()
    all_ims = []
    registration_idxs = np.load(f"{args.out_dir}/registration_idxs_tool_{args.use_ransac}_avg_{args.use_pcd}_t.npy")
    # load point clouds from out dir 
    object_sam_pcds = h5py.File(f"{args.out_dir.split('fp_tool')[0]}/sam2_pcds_tool.h5")
    # compute outlier frames
    all_sam_frames =[]
    video = cv2.VideoCapture(f"{args.out_dir.split('fp_tool')[0]}/sam_tool/test_{args.camera_idx}_tool.mp4")
    for frame in range(NUM_FRAMES):
      ret, frame = video.read()
      all_sam_frames.append(frame)
    sam_masks = h5py.File(f"{args.out_dir.split('fp_tool')[0]}/sam2_masks_tool.h5", "r")
    # load trajectory frames
    all_traj_frames = []
    video = cv2.VideoCapture(f"{args.out_dir}/traj_vid_{args.camera_idx}_tool_{args.use_ransac}_avg_{args.use_pcd}_t.mp4")
    for frame in range(NUM_FRAMES):
      ret, frame = video.read()
      all_traj_frames.append(frame)
    outlier_idxs = []
    for i in tqdm(range(NUM_FRAMES)):
        full_diff, r_diff = matrix_distance(object_center_transforms[i], object_center_transforms[i-1])
        if i > 0 and i - 1 in outlier_idxs:
            if r_diff < 0.5 and full_diff - r_diff < 0.1:
                outlier_idxs.append(i)
        elif i > 0 and (r_diff > 0.9 or full_diff - r_diff > 0.1):
            outlier_idxs.append(i)
    corrected_poses = []
    for i in tqdm(range(NUM_FRAMES)):
        if i in outlier_idxs:
            less, greater = i - 1, i + 1
            while less in outlier_idxs:
                less -= 1
            while greater in outlier_idxs:
                greater += 1
            frac = (i - less) / (greater - less)
            correct_rotation = slerp_4x4(object_center_transforms[less], object_center_transforms[greater], frac)
            corrected_poses.append(correct_rotation)
        else:
            corrected_poses.append(object_center_transforms[i].copy())


    for i in tqdm(range(NUM_FRAMES)):
        # compute transform
        frame_idxs = get_mask_indices(np.asarray(sam_masks[f"masks_{args.camera_idx}"][i]))
        object_trimesh.vertices = orig_vertices.copy()
        object_trimesh.apply_transform(object_center_transforms[i])
        object_points_2d = project_points_to_image(camera_intrinsics[args.camera_idx], np.eye(4), object_trimesh.vertices)
        outlier_frame = np.flip(rgbs[i, args.camera_idx], axis=-1)
        points_2d = object_points_2d[::200, :]
        points_2d = points_2d[points_2d[:, 0] >= 0]
        points_2d = points_2d[points_2d[:, 1] >= 0]
        points_2d = points_2d[points_2d[:, 1] < 480]
        points_2d = points_2d[points_2d[:, 0] < 640]
        points_2d = np.unique(points_2d.astype(np.uint64), axis=0)
        color = None 
        if i in registration_idxs or i in outlier_idxs:
            color = np.array([[255, 0, 0]])
        else:
            color = np.array([[0, 0, 255]])
        outlier_frame[points_2d[:, 1], points_2d[:, 0]] = color
        #outlier_frame = add_caption(outlier_frame, f"IOU: {pixel_iou(list_of_lists(points_2d[:, [1, 0]]), frame_idxs)}")
        object_trimesh.vertices = orig_vertices.copy()
        dist = 0
        # if i > 0 and matrix_distance(object_center_transforms[i], object_center_transforms[i-1])[1] > 0.75:
        #     dist = 1.5
        if i > 0:
            corrected_poses[i] = slerp_4x4(corrected_poses[i-1], corrected_poses[i], 0.15)
        object_trimesh.apply_transform(corrected_poses[i])
        object_points_2d = project_points_to_image(camera_intrinsics[args.camera_idx], np.eye(4), object_trimesh.vertices)
        image_width, image_height = 640, 480
        corr_outlier_frame = np.flip(rgbs[i, args.camera_idx], axis=-1)

        points_2d = object_points_2d[::200, :]
        points_2d = points_2d[points_2d[:, 0] >= 0]
        points_2d = points_2d[points_2d[:, 1] >= 0]
        points_2d = points_2d[points_2d[:, 1] < 480]
        points_2d = points_2d[points_2d[:, 0] < 640]
        points_2d = np.unique(points_2d.astype(np.uint64), axis=0)
        color = None 
        if i in registration_idxs or dist > 1:
            color = np.array([[255, 0, 0]])
        else:
            color = np.array([[0, 0, 255]])

        corr_outlier_frame[points_2d[:, 1], points_2d[:, 0]] = color
        #corr_outlier_frame = add_caption(corr_outlier_frame, f"IOU: {pixel_iou(list_of_lists(points_2d[:, [1,0]]), frame_idxs)}")
        # add object points into image 
        top_frame = np.concatenate((all_sam_frames[i], all_traj_frames[i]), axis=1)
        bottom_frame = np.concatenate((outlier_frame[:, :, ::-1], corr_outlier_frame[:, :, ::-1]), axis=1)
        #background_image[observed_points[:, 1], observed_points[:, 0]] = np.array([[0, 255, 0]]) 
        all_ims.append(np.concatenate((top_frame, bottom_frame), axis=0))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{args.out_dir}/better_tracking_2.mp4", fourcc, 30, (640 * 2, 480 * 2))
    for i in range(NUM_FRAMES):
        frame = all_ims[i]
        video.write(frame)
    video.release()