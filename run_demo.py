# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


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
    return R_diff + weight * t_diff

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

CAMERA_IDXS = [1, 5, 7]
def optimize_poses(poses, cam_extrinsics, to_origin):
  """
  poses: list of poses predicted by each FoundationPose estimator
  cam_extrinsics: list of camera extrinsics for each pose
  """
  world_poses = {cam: cam_extrinsics[cam] @ (poses[cam]) for cam in CAMERA_IDXS} # removed to_origin 

  best_error = np.inf 
  best_estim = np.zeros((4, 4))
  for i in CAMERA_IDXS:
    inlier_idxs = [j for j in CAMERA_IDXS if matrix_distance(world_poses[i], world_poses[j]) < 0.25 and j != i]
    if len(inlier_idxs) == 0:
      continue
    #print(f"Inlier idxs: {inlier_idxs}")
    inlier_idxs += [i]
    mean_mat = np.mean([world_poses[idx][:3, :3] for idx in inlier_idxs], axis=0)
    mean_pos = np.mean([world_poses[idx][:3, 3] for idx in inlier_idxs], axis=0)
    #print(f"mean_pos: {mean_pos}")
    u, s, v_t = np.linalg.svd(mean_mat)
    #print(f"mean_mat: {u @ v_t}")
    model = np.zeros((4, 4))
    model[3, 3] = 1
    model[:3, :3] = u @ v_t 
    model[:3, 3] = mean_pos 
    error = sum([matrix_distance(cam_extrinsics[j], model) for j in inlier_idxs]) / len(inlier_idxs)
    # sum(all values in inliers - mean of inliers)
    #print(f"error: {error}")
    #print(f"Inlier idxs: {inlier_idxs}")
    if best_error > error:
      best_error = error 
      best_estim = model.copy()
  return best_estim 

def optimize_poses_pcd(poses, object_mesh, object_pcds, to_origin, all_camera_extrinsics):
  object_mesh.apply_transform(np.linalg.inv(to_origin))
  dists = []
  chamferDist = ChamferDistance()
  for i in range(len(poses)):
    object_mesh.apply_transform(poses[i])
    if len(object_pcds[i]) == 0:
      dists.append(np.inf)
    else:
      pcd = o3d.geometry.PointCloud()
      #print(f"mean before: {object_pcds[i].mean(axis=0)}")
      pcd.points = o3d.utility.Vector3dVector(object_pcds[i])
      cl, ind = pcd.remove_statistical_outlier(nb_neighbors=min(150, len(object_pcds[i])), std_ratio=1.5)
      object_pcds[i] = np.asarray(cl.points)
      #print(f"mean after: {object_pcds[i].mean(axis=0)}")
      if len(object_pcds[i]) == 0:
        dists.append(np.inf)
        continue 
      sampled_vertices = object_pcds[i].copy()#[fpsample.bucket_fps_kdline_sampling(object_pcds[i], min(100, len(object_pcds[i])), h=3)]
      #tree = cKDTree(sampled_vertices)
      #distances, _ = tree.query(object_mesh.vertices, k=1)
      #dist = chamferDist(torch.tensor(object_mesh.vertices).unsqueeze(0).float(), torch.tensor(sampled_vertices).unsqueeze(0).float())
      dists.append(np.linalg.norm(sampled_vertices.mean(axis=0) - object_mesh.vertices.mean(axis=0)))#distances.mean())
    object_mesh.apply_transform(np.linalg.inv(poses[i]))
  object_mesh.apply_transform(to_origin)
  good_idxs = [i for i in range(len(poses)) if dists[i] < 0.03]
  print(f"good idxs: {dists}")
  #return all_camera_extrinsics[np.argmin(dists)] @ poses[np.argmin(dists)]
  return optimize_poses([poses[i] for i in good_idxs], [all_camera_extrinsics[i] for i in good_idxs], to_origin)

NUM_FRAMES = 10
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--robotool_datafile', type=str, default=None)
  parser.add_argument('--camera_idx', type=int, default=0)
  parser.add_argument('--registration_frequency', type=int, default=10)
  parser.add_argument('--t', type=str)
  parser.add_argument('--use_ransac', action="store_true")
  parser.add_argument('--use_pcd', action="store_true")
  args = parser.parse_args()
  set_logging_format()
  set_seed(0)

  init_points = []
  sam_pts = h5py.File(f"{args.test_scene_dir}/sam2_pcds_{args.t}_updated.h5", "r")
  for i in range(len(sam_pts['cam_0'].keys())):
      init_pts = [np.array(sam_pts[f'cam_{cam}'][f'pcd_{i}']) * 1e-3 for cam in range(8)]
      init_points.append(init_pts)
  other_mesh = trimesh.load(args.mesh_file, process=True)
  # load in texture information
  # samples, _ = trimesh.sample.sample_surface(orig_mesh, count=200000)
  mesh=None 
  if len(other_mesh.vertices) > 2000000: # fix
    mesh = other_mesh.simplify_quadric_decimation(200000) #trimesh.Trimesh(vertices=samples, process=True)
    del other_mesh
  else:
    mesh = copy.deepcopy(other_mesh)

  # from PIL import Image
  # texture_img = Image.open("/svl/u/tarunc/blue_scooper.jpg")
  # material = trimesh.visual.texture.SimpleMaterial(image=texture_img)
  # visuals = TextureVisuals(uv=mesh.visual.uv, image=texture_img, material=material)
  # mesh.visual = visuals
  #transform = np.eye(4)
  #transform[:3, :3] *= 1e-3
  #mesh.apply_transform(transform)
  mesh.vertices *= 0.001
  mesh_copy = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy())
  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  mesh_copy.apply_transform(to_origin)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  #glctx = dr.RasterizeGLContext()
  glctx = dr.RasterizeCudaContext()
  #estimaters = []
  estimaters = {}
  for cam in CAMERA_IDXS:
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    estimaters[cam] = est
  #est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  # for cam in range(NUM_CAMS):
  #   est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  #   estimaters.append(est)
  logging.info("estimator initialization done")
  data = h5py.File(args.robotool_datafile, "r")
  yaml_file = open("/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml")
  yaml_data = yaml.safe_load(yaml_file)
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
  #all_camera_extrinsics = [np.array(yaml_data[i]['transformation']) for i in range(NUM_CAMS)]
  registration_idxs = []

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf, K=all_camera_intrinsics[CAMERA_IDXS[0]])
  imgs = {i: [] for i in CAMERA_IDXS}
  all_poses = {i: [] for i in CAMERA_IDXS}
  all_center_poses = {i: [] for i in CAMERA_IDXS}
  #imgs = [[] for _ in range(NUM_CAMS)]
  #all_poses = [[] for _ in range(NUM_CAMS)]
  #all_center_poses = [[] for _ in range(NUM_CAMS)]
  sam_masks = None 
  if os.path.exists(f"{args.test_scene_dir}/sam2_masks_{args.t}_updated.h5"):
    sam_masks = h5py.File(f"{args.test_scene_dir}/sam2_masks_{args.t}_updated.h5", "r")
  for i in tqdm(range(NUM_FRAMES)):#len(data['imgs'])):#reader.color_files)):
    #logging.info(f'i:{i}')
    #color1 = reader.get_color(i)
    #depth1 = reader.get_depth(i)
    all_colors = {cam: None}
    all_depths = {cam: None}
    register_now = True
    for cam in CAMERA_IDXS:
      color = data['imgs'][i, cam, :, :, :].astype(np.uint8)[...,:3]
      color = cv2.resize(color, (640, 480), interpolation=cv2.INTER_NEAREST)
      depth = data['depths'][i, cam, :, :].astype(np.float64)
      depth /= 1e3
      depth = cv2.resize(depth, (640,480), interpolation=cv2.INTER_NEAREST)
      # if register_now:
      #   color[0, 0] = np.array([[255, 255, 255]])
      depth[(depth<0.001) | (depth>=np.inf)] = 0
      all_colors[cam] = color
      all_depths[cam] = depth

    if i != 0:
      # all previous poses should be same 
      prev_poses = {cam: estimaters[cam].pose_last.cpu().detach().numpy() for cam in CAMERA_IDXS}
      all_est_poses = {cam: None for cam in CAMERA_IDXS}
      prev_dists = []
      for cam in CAMERA_IDXS:
        pose = estimaters[cam].track_one(rgb=all_colors[cam], depth=all_depths[cam], K=all_camera_intrinsics[cam], iteration=args.track_refine_iter, ob_mask=np.asarray(sam_masks[f"masks_{cam}"][i]))
        all_est_poses[cam] = pose.copy()
        if prev_poses[cam].shape == (1, 4, 4):
          prev_poses[cam] = prev_poses[cam][0]
        prev_dists.append(matrix_distance(all_est_poses[cam], (prev_poses[cam] @ estimaters[cam].get_tf_to_centered_mesh().cpu().numpy())))
      # if min(prev_dists) > 0.3:
      #   register_now = True 
      #   registration_idxs.append(i)
      #   print(f"DISTS: {prev_dists}")
    if i==0 or register_now:
      prev_poses = {cam: None for cam in CAMERA_IDXS}
      all_est_poses = {cam: None for cam in CAMERA_IDXS}
      for cam in CAMERA_IDXS:
        if sam_masks is not None:
          mask = np.asarray(sam_masks[f"masks_{cam}"][i])
        else:
          mask = reader.get_mask(cam, os.path.join(args.test_scene_dir, "masks"))
        pose = None 
        if i == 0:
          pose = estimaters[cam].register(K=all_camera_intrinsics[cam], rgb=all_colors[cam], depth=all_depths[cam], ob_mask=mask, iteration=args.est_refine_iter)
          prev_poses[cam] = pose.copy()
        else:
          prev_poses[cam] = estimaters[cam].pose_last.cpu().detach().numpy()
          pose = estimaters[cam].register(K=all_camera_intrinsics[cam], rgb=all_colors[cam], depth=all_depths[cam], ob_mask=mask, iteration=args.est_refine_iter)
        all_est_poses[cam] = pose
        if prev_poses[cam].shape == (1, 4, 4):
          prev_poses[cam] = prev_poses[cam][0]
      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, intrinsic_mat)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    # compute all pairwise distances
    if args.use_ransac:
      all_dists = [matrix_distance(all_est_poses[CAMERA_IDXS[cam1]], all_est_poses[CAMERA_IDXS[cam2]]) 
        for cam1 in range(len(CAMERA_IDXS)) for cam2 in range(cam1 + 1, len(CAMERA_IDXS))]
      if min(all_dists) > 3:
        all_est_poses = {cam: (prev_poses[cam].copy() @ estimaters[cam].get_tf_to_centered_mesh().cpu().numpy()) for cam in CAMERA_IDXS}
        for cam in CAMERA_IDXS:
          estimaters[cam].pose_last = torch.tensor(prev_poses[cam]).to('cuda:0')  #@ torch.linalg.inv(estimaters[cam].get_tf_to_centered_mesh()) #@ estimaters[cam].get_tf_to_centered_mesh()# .copy()
          if estimaters[cam].pose_last.shape == (1, 4, 4):
            estimaters[cam].pose_last = estimaters[cam].pose_last[0]
      else:
        if args.use_pcd:
          opt_pose = optimize_poses_pcd(all_est_poses, mesh_copy, init_points[i], to_origin, all_camera_extrinsics)
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
    #np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
    for cam in CAMERA_IDXS:
      if debug>=1:
        #assert all_est_poses[cam].shape == (4, 4), f"shape: {all_est_poses[cam].shape}"
        if all_est_poses[cam].shape == (1, 4, 4):
          all_est_poses[cam] = all_est_poses[cam][0]
        center_pose = all_est_poses[cam] @ np.linalg.inv(to_origin)
        assert center_pose.shape == (4, 4), f"center pose shape: {center_pose.shape, all_est_poses[cam].shape}"
        # mesh_copy.apply_transform(center_pose)
        # points_2d = project_points_to_image(all_camera_intrinsics[cam], np.eye(4), mesh_copy.vertices).astype(np.uint8) #+ np.array([[240, 320]])
        # points_2d = points_2d[(points_2d[:, 0] > 0) & (points_2d[:, 0] < 480) & (points_2d[:, 1] > 0) & (points_2d[:, 1] < 640)]
        vis = draw_posed_3d_box(all_camera_intrinsics[cam], img=all_colors[cam], ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(all_colors[cam], ob_in_cam=center_pose, scale=0.1, K=all_camera_intrinsics[cam], thickness=3, transparency=0, is_input_rgb=True)
        #vis[points_2d[:, 0], points_2d[:, 1]] = np.array([[0, 0, 255]])
        #cv2.imshow('1', vis[...,::-1])
        #cv2.waitKey(1)
        mesh.apply_transform(np.linalg.inv(all_est_poses[cam]))

      if debug>=2:
        os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
        imageio.imwrite(f'{debug_dir}/track_vis/img_{i}.png', vis[:, :, ::-1])
      imgs[cam].append(vis)
      all_poses[cam].append(pose)
      all_center_poses[cam].append(center_pose)
      #print(f"pose: {pose}, center_pose: {center_pose}")
  all_poses = np.array(all_poses)
  all_center_poses = np.array(all_center_poses)
  os.makedirs(f"{args.test_scene_dir}/fp_{args.t}", exist_ok=True)
  breakpoint()
  for cam in CAMERA_IDXS:
    np.save(f"{args.test_scene_dir}/fp_{args.t}/scene_poses_{cam}_{args.t}_{args.use_ransac}_avg_{args.use_pcd}_tex.npy", all_poses[cam])
    np.save(f"{args.test_scene_dir}/fp_{args.t}/scene_center_poses_{cam}_{args.t}_{args.use_ransac}_avg_{args.use_pcd}_tex.npy", all_center_poses[cam])
  np.save(f"{args.test_scene_dir}/fp_{args.t}/registration_idxs_{args.t}_{args.use_ransac}_avg_{args.use_pcd}_tex.npy", np.array(registration_idxs))
  if debug >= 2:
    for cam in CAMERA_IDXS:
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      out = cv2.VideoWriter(f"{args.test_scene_dir}/fp_{args.t}/traj_vid_{cam}_{args.t}_{args.use_ransac}_avg_{args.use_pcd}_tex.mp4", fourcc, 30, (vis.shape[1], vis.shape[0]))
      for img in imgs[cam]:
        out.write(img)
      out.release()


# render scene 
  