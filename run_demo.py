# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import datetime
from estimater import *
from datareader import *
import fpsample 
from chamferdist import ChamferDistance
import open3d as o3d
import argparse
import torch 
from scipy.spatial import cKDTree
import h5py 
import copy 
import yaml 
from tqdm import tqdm
from trimesh.visual import TextureVisuals
import pickle 
import transformations

def icp_pose(object_pcd, prev_pose, mesh):
  object_pcd_o3d = o3d.geometry.PointCloud()
  object_pcd_o3d.points = o3d.utility.Vector3dVector(object_pcd)

  mesh_transformed = mesh.copy()
  mesh_transformed.apply_transform(prev_pose)
  mesh_points = fpsample.bucket_fps_kdline_sampling(mesh_transformed.vertices, 100, h=3)
  mesh_pcd = o3d.geometry.PointCloud()
  mesh_pcd.points = o3d.utility.Vector3dVector(mesh_transformed.vertices[mesh_points])

  # Estimate normals for both point clouds
  object_pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  mesh_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

  # Perform ICP registration
  threshold = 0.005  # distance threshold
  reg_p2p = o3d.pipelines.registration.registration_icp(
      object_pcd_o3d, 
      mesh_pcd, 
      threshold, 
      np.eye(4),  # identity matrix as initial transformation
      o3d.pipelines.registration.TransformationEstimationPointToPlane(),
      o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
  )
  # Compute the refined pose
  refined_pose = np.dot(reg_p2p.transformation, prev_pose)

  return refined_pose

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

#NUM_CAMS = 2
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return np.round(iou, 3)

def construct_dataset(center_poses, yaml_file_path, args):
  yaml_file = open(yaml_file_path)
  yaml_data = yaml.safe_load(yaml_file)
  serials = [yaml_data[i]['serial_number'] for i in range(len(yaml_data))]
  all_camera_intrinsics = np.array([np.array(yaml_data[i]['color_intrinsic_matrix']) for i in range(8)])
  all_camera_extrinsics = np.array([yaml_data[i]['transformation'] for i in range(8)])
  if args.partial_dataset is None:
    dataset = {
      "video_name": args.test_scene_dir.split('/')[-1],
      "object_names": {
        "target_object": None,
        "tool_object": None,
      },
      "camera_seq": serials,
      "camera_intrinsics": all_camera_intrinsics,
      "camera_extrinsics": all_camera_extrinsics, 
      "target_object_pose": [], 
      "tool_object_pose": [],  
      "start_frame": 0,
      "end_frame": NUM_FRAMES,    
    } 
  else:
    with open(args.test_scene_dir, "rb") as f:
      dataset = pickle.load(f)
  if args.t == "tool":
    dataset["object_names"]["tool_object"] = args.mesh_file.split('/')[-1]
    idx = CAMERA_IDXS[0]
    tool_object_pose = np.einsum('njk,ij->nik', np.array(center_poses[idx]), all_camera_extrinsics[idx])
    dataset["tool_object_pose"] = tool_object_pose.copy()
  if args.t == "target":
    dataset["object_names"]["target_object"] = args.mesh_file.split('/')[-1]
    idx = CAMERA_IDXS[0]
    target_object_pose = np.einsum('njk,ij->nik', np.array(center_poses[idx]), all_camera_extrinsics[idx])
    dataset["target_object_pose"] = target_object_pose.copy()
  if args.partial_dataset is None:
    pickle.dump(dataset, open(f"{args.test_scene_dir}/dataset_obj_{args.test_scene_dir.split('/')[-1]}.pkl", "wb"))


def optimize_poses(poses, cam_extrinsics, to_origin):
  """
  poses: list of poses predicted by each FoundationPose estimator
  cam_extrinsics: list of camera extrinsics for each pose
  """
  world_poses = {cam: cam_extrinsics[cam] @ (poses[cam]) for cam in cam_extrinsics.keys()} # removed to_origin 
  best_error = np.inf 
  best_estim = np.zeros((4, 4))
  for i in cam_extrinsics.keys():
    inlier_idxs = [j for j in cam_extrinsics.keys() if matrix_distance(world_poses[i], world_poses[j])[0] < 0.25 and j != i]
    if len(inlier_idxs) == 0:
      continue
    inlier_idxs += [i]
    mean_mat = np.mean([world_poses[idx][:3, :3] for idx in inlier_idxs], axis=0)
    mean_pos = np.mean([world_poses[idx][:3, 3] for idx in inlier_idxs], axis=0)
    u, _, v_t = np.linalg.svd(mean_mat)
    model = np.zeros((4, 4))
    model[3, 3] = 1
    model[:3, :3] = u @ v_t 
    model[:3, 3] = mean_pos 
    error = sum([matrix_distance(cam_extrinsics[j], model)[0] for j in inlier_idxs]) / len(inlier_idxs)

    if best_error > error:
      best_error = error 
      best_estim = model.copy()
  return best_estim 

def optimize_poses_pcd(poses, object_mesh, object_pcds, to_origin, all_camera_extrinsics, pcd_filter_thresh):
  object_mesh.apply_transform(np.linalg.inv(to_origin))
  dists = {i : None for i in CAMERA_IDXS}
  chamferDist = ChamferDistance()
  for i in CAMERA_IDXS:
    object_mesh.apply_transform(poses[i])
    if len(object_pcds[i]) == 0:
      dists[i]= np.inf
    else:
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(object_pcds[i])
      cl, ind = pcd.remove_statistical_outlier(nb_neighbors=min(100, len(object_pcds[i])), std_ratio=1.5)
      object_pcds[i] = np.asarray(cl.points)
      if len(object_pcds[i]) == 0:
        dists[i] = np.inf
        continue 
      sampled_vertices = object_pcds[i].copy()
      dists[i] = np.linalg.norm(sampled_vertices.mean(axis=0) - object_mesh.vertices.mean(axis=0))#distances.mean())
    object_mesh.apply_transform(np.linalg.inv(poses[i]))
  object_mesh.apply_transform(to_origin)
  good_idxs = [i for i in CAMERA_IDXS if dists[i] < pcd_filter_thresh]
  print(f"good_idxs: {good_idxs, dists}")
  if len(good_idxs) == 0:
    good_idxs = [min(dists, key=dists.get), min(dists, key=dists.get)]
    return all_camera_extrinsics[good_idxs[0]] @ poses[good_idxs[0]]
  else:
    return optimize_poses({cam: poses[cam] for cam in good_idxs}, {cam: all_camera_extrinsics[cam] for cam in good_idxs}, to_origin)

def compute_pcds_dist(poses, object_mesh, object_pcds, to_origin, all_camera_extrinsics):
  object_mesh.apply_transform(np.linalg.inv(to_origin))
  dists = []
  for i in CAMERA_IDXS:
    object_mesh.apply_transform(poses[i])
    dists.append(np.linalg.norm(object_mesh.vertices.mean(axis=0) - object_pcds[i].mean(axis=0)))
    object_mesh.apply_transform(np.linalg.inv(poses[i]))
  object_mesh.apply_transform(to_origin)
  return dists 

def get_name(path):
    return os.path.basename(os.path.normpath(path))

def depth2xyzmap(depth, K, uvs=None):
  invalid_mask = (depth<0.001)
  H,W = depth.shape[:2]
  if uvs is None:
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:,0]
    vs = uvs[:,1]
  zs = depth[vs,us]
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = np.zeros((H,W,3), dtype=np.float32)
  xyz_map[vs,us] = pts
  xyz_map[invalid_mask] = 0
  return xyz_map


USE_TEXTURE = True
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--config_file', type=str)
  parser.add_argument('--data_dir', type=str)
  parser.add_argument('--uuid', type=str)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--t', type=str)
  parser.add_argument('--extrinsic_file', type=str, default='')
  args = parser.parse_args()
  set_logging_format()
  set_seed(0)
  config = yaml.safe_load(open(args.config_file, "r"))
  START_FRAME = config['foundation_pose']['start_frame']
  NUM_FRAMES = config['foundation_pose']['num_frames']
  RUN_UUID = args.uuid
  CAMERA_IDXS =  config['foundation_pose']['camera_idxs']
  print(f"trajectory: {get_name(args.data_dir)} {args.t} object: {args.mesh_file}")
  init_points = []
  # sam_pts = h5py.File(f"{args.data_dir}/sam2_pcds_{args.t}.h5", "r")
  # for i in range(len(sam_pts['cam_0'].keys())):
  #     init_pts = [np.array(sam_pts[f'cam_{cam}'][f'pcd_{i}']) * 1e-3 for cam in range(8)]
  #     init_points.append(init_pts)
  other_mesh = trimesh.load(args.mesh_file, process=True)
  # load in texture information
  print(f"Register: {config['foundation_pose']['register']}")
  if USE_TEXTURE:
    try:
      texture_file = args.mesh_file.replace('decim_mesh_files', 'textures').replace('.obj', '.jpg')
      texture = cv2.imread(texture_file)
      from PIL import Image
      im = Image.open(texture_file)
      uv = other_mesh.visual.uv
      material = trimesh.visual.texture.SimpleMaterial(image=im)
      color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
      other_mesh.visual = color_visuals
    except:
      print(f"Error loading texture file: {texture_file}")
  mesh=None 
  if len(other_mesh.vertices) > config['foundation_pose']['decim_verts_num']: # fix
    mesh = other_mesh.simplify_quadric_decimation(config['foundation_pose']['decim_verts_num']) #trimesh.Trimesh(vertices=samples, process=True)
    del other_mesh
  else:
    mesh = copy.deepcopy(other_mesh)

  mesh.vertices *= 0.001
  mesh_copy = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy())
  debug = config['foundation_pose']['debug']
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  mesh_copy.apply_transform(to_origin)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  estimaters = {}
  for cam in CAMERA_IDXS:
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    estimaters[cam] = est
  logging.info("estimator initialization done")
  data = h5py.File(f"{args.data_dir}/data00000000.h5", "r")
  yaml_file = open(args.extrinsic_file)
  yaml_data = yaml.safe_load(yaml_file)
  #serials = [yaml_data[i]['serial_number'] for i in range(len(yaml_data))]
  serials = ['244222072252', '250122071059', '125322060645', '246422071990', '246422071818', '246422070730', '250222072777', '204222063088']
  #all_camera_intrinsics = [np.array(yaml_data[i]['color_intrinsic_matrix']) for i in range(NUM_CAMS)]
  all_camera_intrinsics = {
    i: np.array(yaml_data[i]['color_intrinsic_matrix'])
    for i in CAMERA_IDXS
  }
  # each extrinsic is T^{world}_{cam} -> takes points in camera frame and turns them into world frame
  all_camera_extrinsics = {
    i: np.array(yaml_data[i]['transformation'])
    for i in CAMERA_IDXS
  }
  registration_idxs = []

  imgs = {i: [] for i in CAMERA_IDXS}
  all_poses = {i: [] for i in CAMERA_IDXS}
  all_center_poses = {i: [] for i in CAMERA_IDXS}
  #sam_masks = h5py.File(f"{args.data_dir}/sam2_masks_{args.t}.h5", "r")
  sam_masks = np.load(f"{args.data_dir}/masks.npy") * 255
  for i in tqdm(range(START_FRAME, START_FRAME + NUM_FRAMES)):
    all_colors = {cam: None for cam in CAMERA_IDXS}
    all_depths = {cam: None for cam in CAMERA_IDXS}
    register_now = config['foundation_pose']['register']
    for cam in CAMERA_IDXS:
      color = data['imgs'][i, cam, :, :, :].astype(np.uint8)[...,:3]
      color = cv2.resize(color, (640, 480), interpolation=cv2.INTER_NEAREST)
      depth = data['depths'][i, cam, :, :].astype(np.float64)
      depth /= 1e3
      depth = cv2.resize(depth, (640,480), interpolation=cv2.INTER_NEAREST)
      depth[(depth<0.001) | (depth>=np.inf)] = 0
      all_colors[cam] = color
      all_depths[cam] = depth

    if i != START_FRAME and not register_now:
      # all previous poses should be same 
      prev_poses = {cam: estimaters[cam].pose_last.cpu().detach().numpy() for cam in CAMERA_IDXS}
      all_est_poses = {cam: None for cam in CAMERA_IDXS}
      prev_dists = []
      for cam in CAMERA_IDXS:
        pose = estimaters[cam].track_one(
          rgb=all_colors[cam], 
          depth=all_depths[cam], 
          K=all_camera_intrinsics[cam], 
          iteration=config['foundation_pose']['track_refine_iter'], 
          ob_mask=np.asarray(sam_masks[i, cam]), 
        )
        all_est_poses[cam] = pose.copy()
        if prev_poses[cam].shape == (1, 4, 4):
          prev_poses[cam] = prev_poses[cam][0]
        prev_dists.append(matrix_distance(all_est_poses[cam], (prev_poses[cam] @ estimaters[cam].get_tf_to_centered_mesh().cpu().numpy()))[0])
      # pcd_dists = compute_pcds_dist(all_est_poses, mesh_copy, init_points[i], to_origin, all_camera_extrinsics)
      # if min(pcd_dists) > 0.03 or min(prev_dists) > 0.3:
      #   print(f"Registering: {min(pcd_dists), min(prev_dists)}")
      #   register_now = True 
    if i==START_FRAME or register_now:
      prev_poses = {cam: None for cam in CAMERA_IDXS}
      all_est_poses = {cam: None for cam in CAMERA_IDXS}
      for cam in CAMERA_IDXS:
        mask = np.asarray(sam_masks[i, cam])
        pose = None 
        if i == START_FRAME:
          pose = estimaters[cam].register(
            K=all_camera_intrinsics[cam], 
            rgb=all_colors[cam], 
            depth=all_depths[cam], 
            ob_mask=mask, 
            iteration=config['foundation_pose']['est_refine_iter'])
          prev_poses[cam] = pose.copy()
        else:
          prev_poses[cam] = estimaters[cam].pose_last.cpu().detach().numpy()
          pose = estimaters[cam].register(
            K=all_camera_intrinsics[cam], 
            rgb=all_colors[cam], 
            depth=all_depths[cam], 
            ob_mask=mask, 
            iteration=config['foundation_pose']['est_refine_iter'])
        all_est_poses[cam] = pose
        if prev_poses[cam].shape == (1, 4, 4):
          prev_poses[cam] = prev_poses[cam][0]
    # compute all pairwise distances
    if config['foundation_pose']['use_ransac']:
      all_dists = [matrix_distance(all_est_poses[CAMERA_IDXS[cam1]], all_est_poses[CAMERA_IDXS[cam2]])[0]
        for cam1 in range(len(CAMERA_IDXS)) for cam2 in range(cam1 + 1, len(CAMERA_IDXS))]
      # if min(all_dists) > 3:
      #   all_est_poses = {cam: (prev_poses[cam].copy() @ estimaters[cam].get_tf_to_centered_mesh().cpu().numpy()) for cam in CAMERA_IDXS}
      #   for cam in CAMERA_IDXS:
      #     estimaters[cam].pose_last = torch.tensor(prev_poses[cam]).to('cuda:0')  #@ torch.linalg.inv(estimaters[cam].get_tf_to_centered_mesh()) #@ estimaters[cam].get_tf_to_centered_mesh()# .copy()
      #     if estimaters[cam].pose_last.shape == (1, 4, 4):
      #       estimaters[cam].pose_last = estimaters[cam].pose_last[0]
      # else:
      if config['foundation_pose']['use_pcd']:
        pcds = {cam: depth2xyzmap(all_depths[cam], all_camera_intrinsics[cam])[sam_masks[i, cam] > 0] for cam in CAMERA_IDXS}
        opt_pose = optimize_poses_pcd(all_est_poses, mesh_copy, pcds, to_origin, all_camera_extrinsics, config['foundation_pose']['pcd_filter_thresh'])
      else:
        opt_pose = optimize_poses(all_est_poses, all_camera_extrinsics, to_origin)
      if np.linalg.norm(opt_pose) < 1e-7:
        all_est_poses = {cam: (prev_poses[cam].copy() @ estimaters[cam].get_tf_to_centered_mesh().cpu().numpy()) for cam in CAMERA_IDXS}
        for cam in CAMERA_IDXS:
          estimaters[cam].pose_last = torch.tensor(prev_poses[cam]).to('cuda:0') #@ torch.linalg.inv(estimaters[cam].get_tf_to_centered_mesh())# .double()# .copy()
          if estimaters[cam].pose_last.shape == (1, 4, 4):
            estimaters[cam].pose_last = estimaters[cam].pose_last[0]
      else:
        for cam in CAMERA_IDXS:
          all_est_poses[cam] = (np.linalg.inv(all_camera_extrinsics[cam]) @ opt_pose) # @ to_origin
          estimaters[cam].pose_last = torch.tensor(all_est_poses[cam]).to('cuda:0') @ torch.linalg.inv(estimaters[cam].get_tf_to_centered_mesh()).double()
          if estimaters[cam].pose_last.shape == (1, 4, 4):
            estimaters[cam].pose_last = estimaters[cam].pose_last[0]

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    for cam in CAMERA_IDXS:
      if debug>=1:
        if all_est_poses[cam].shape == (1, 4, 4):
          all_est_poses[cam] = all_est_poses[cam][0]
        center_pose = all_est_poses[cam] @ np.linalg.inv(to_origin)
        assert center_pose.shape == (4, 4), f"center pose shape: {center_pose.shape, all_est_poses[cam].shape}"
        vis = draw_posed_3d_box(all_camera_intrinsics[cam], img=all_colors[cam], ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(all_colors[cam], ob_in_cam=center_pose, scale=0.1, K=all_camera_intrinsics[cam], thickness=3, transparency=0, is_input_rgb=True)
        mesh.apply_transform(np.linalg.inv(all_est_poses[cam]))

      if debug>=3:
        os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
        imageio.imwrite(f'{debug_dir}/track_vis/img_{i}.png', vis[:, :, ::-1])
      imgs[cam].append(vis)
      all_poses[cam].append(pose)
      all_center_poses[cam].append(center_pose)
  all_pre_optim_poses = [
    all_camera_extrinsics[CAMERA_IDXS[0]] @ all_center_poses[CAMERA_IDXS[0]][step]
    for step in range(len(all_center_poses[CAMERA_IDXS[0]]))
  ]

  object_trimesh = trimesh.load(args.mesh_file)
  object_trimesh.vertices *= 0.001
  to_origin, extents = trimesh.bounds.oriented_bounds(object_trimesh)
  object_trimesh.apply_transform(to_origin)
  orig_vertices = object_trimesh.vertices.copy()

  # 4x4 image - top left is foundationpose view top right is sam bottom left is 
  # outlier detected bottom right is outlier optimized
  all_frames = []
  outlier_idxs = []
  all_post_optim_poses = []
  for i in tqdm(range(START_FRAME, START_FRAME + NUM_FRAMES)):
    if i > START_FRAME and i - 1 in outlier_idxs:
        if matrix_distance(all_pre_optim_poses[i-START_FRAME], all_pre_optim_poses[i-1-START_FRAME])[1] < 0.5:
            outlier_idxs.append(i)
    elif i > START_FRAME and matrix_distance(all_pre_optim_poses[i-START_FRAME], all_pre_optim_poses[i-1-START_FRAME])[1] > 5: # 0.9:
        outlier_idxs.append(i)
  # perform linear interpolation between poses
  for i in tqdm(range(START_FRAME, START_FRAME + NUM_FRAMES)):
    if i in outlier_idxs:
      less, greater = i - 1, i + 1 
      while less in outlier_idxs:
          less -= 1
      while greater in outlier_idxs:
          greater += 1
      frac = (i - less) / (greater - less)
      correct_rotation = slerp_4x4(all_pre_optim_poses[less-START_FRAME], all_pre_optim_poses[greater-START_FRAME], frac)
      all_post_optim_poses.append(correct_rotation)
    else:
      all_post_optim_poses.append(all_pre_optim_poses[i-START_FRAME])
  for cam in CAMERA_IDXS:
    # generate all frames 
    all_sam_frames = []
    for frame in range(config['foundation_pose']['num_frames']):
      # Create a copy of the original image
      img = np.asarray(data['imgs'][frame, cam, :, :, :]).copy()
      
      # Get the SAM mask for this frame and camera
      sam_mask = sam_masks[frame, cam]
      
      # Set pixels to red where the SAM mask is nonzero
      img[sam_mask > 0] = [0, 0, 255]  # RGB format: red
      all_sam_frames.append(img)
    for i in tqdm(range(START_FRAME, START_FRAME + NUM_FRAMES)):
      # outlier detected frame 
      sam_frame = all_sam_frames[i]
      # outlier frame 
      outlier_frame = np.asarray(data['imgs'][i, cam, :, :, :]).copy()
      object_trimesh.vertices = orig_vertices.copy()
      object_trimesh.apply_transform(all_pre_optim_poses[i-START_FRAME])
      object_points_2d = project_points_to_image(all_camera_intrinsics[cam], all_camera_extrinsics[cam], object_trimesh.vertices)
      points_2d = object_points_2d[::200, :]
      points_2d = points_2d[points_2d[:, 0] >= 0]
      points_2d = points_2d[points_2d[:, 1] >= 0]
      points_2d = points_2d[points_2d[:, 1] < 480]
      points_2d = points_2d[points_2d[:, 0] < 640]
      points_2d = points_2d.astype(np.uint64)
      color = None 
      if i in outlier_idxs:
          color = np.array([[0, 0, 255]])
      else:
          color = np.array([[255, 0, 0]])
      outlier_frame[points_2d[:, 1], points_2d[:, 0]] = color
      # corrected outlier frame
      corr_outlier_frame = np.asarray(data['imgs'][i, cam, :, :, :]).copy()
      object_trimesh.vertices = orig_vertices.copy()
      object_trimesh.apply_transform(all_post_optim_poses[i-START_FRAME])
      object_points_2d = project_points_to_image(all_camera_intrinsics[cam], all_camera_extrinsics[cam], object_trimesh.vertices)
      points_2d = object_points_2d[::200, :]
      points_2d = points_2d[points_2d[:, 0] >= 0]
      points_2d = points_2d[points_2d[:, 1] >= 0]
      points_2d = points_2d[points_2d[:, 1] < 480]
      points_2d = points_2d[points_2d[:, 0] < 640]
      points_2d = points_2d.astype(np.uint64)
      color = None 
      if i in outlier_idxs:
          color = np.array([[0, 0, 255]])
      else:
          color = np.array([[255, 0, 0]])
      corr_outlier_frame[points_2d[:, 1], points_2d[:, 0]] = color
      frame = np.concatenate(
        (np.concatenate((imgs[cam][i - START_FRAME], sam_frame), axis=1),
         np.concatenate((outlier_frame, corr_outlier_frame), axis=1)),
         axis=0
      )
      all_frames.append(frame)
  os.makedirs(f"{args.data_dir}/fp_{args.t}/results_{RUN_UUID}", exist_ok=True)
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(f"{args.data_dir}/fp_{args.t}/results_{RUN_UUID}/traj_vid.mp4", fourcc, 30, (vis.shape[1] * 2, vis.shape[0] * 2))
  for img in all_frames:
    out.write(img)
  out.release()
  np.save(f"{args.data_dir}/fp_{args.t}/results_{RUN_UUID}/all_fp_poses.npy", np.array(all_pre_optim_poses))
  np.save(f"{args.data_dir}/fp_{args.t}/results_{RUN_UUID}/all_outlier_poses.npy", np.array(all_post_optim_poses))
  config_file = {
    'mesh_file': args.mesh_file,
    'est_refine_iter': config['foundation_pose']['est_refine_iter'],
    'track_refine_iter': config['foundation_pose']['track_refine_iter'],
    'trajectory': get_name(args.data_dir),
    'use_pcd': config['foundation_pose']['use_pcd'],
    'use_ransac': config['foundation_pose']['use_ransac'], 
    'camera_idxs': config['foundation_pose']['camera_idxs'],
    'decim_verts_num': config['foundation_pose']['decim_verts_num'],
    'pcd_filter_thresh': config['foundation_pose']['pcd_filter_thresh'],
    'start_frame': config['foundation_pose']['start_frame'],
    'num_frames': config['foundation_pose']['num_frames'],
    'extrinsic_file': args.extrinsic_file,
    'register': config['foundation_pose']['register'],
  }
  pickle.dump(config_file, open(f"{args.data_dir}/fp_{args.t}/results_{RUN_UUID}/config.pkl", "wb"))
  