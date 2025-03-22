import open3d as o3d
import trimesh 
import numpy as np
from PIL import Image  
import pyrender
import yaml
import h5py
import os
from pyrender import OffscreenRenderer, Scene, Mesh, DirectionalLight, SpotLight, Node, IntrinsicsCamera

def load_data(extrinsic_file, video_file, serials):
    with open(extrinsic_file, "r") as f:
        extrinsics = yaml.load(f, Loader=yaml.FullLoader)

    serial_to_extrinsic_tf = {}
    serial_to_intrinsic_tf = {}
    for i, cam_info in enumerate(extrinsics):
        extrinsic = cam_info["transformation"]
        intrinsic = cam_info["color_intrinsic_matrix"]
        serial_to_extrinsic_tf[serials[i]] = extrinsic
        serial_to_intrinsic_tf[serials[i]] = intrinsic

    # load data from h5
    session_h5 = h5py.File(video_file, "r")

    # read camera intrinsic directly from h5
    camera_intrinsics = [serial_to_intrinsic_tf[serial] for serial in serials]
    camera_extrinsics = [serial_to_extrinsic_tf[serial] for serial in serials]
    num_frames = len(session_h5["imgs"])
    rgbs = session_h5["imgs"]
    depths = session_h5["depths"]

    return camera_extrinsics, camera_intrinsics, num_frames, rgbs, depths

def create_normal_visualization(mesh, normal_length=0.005, color=None):
    """
    Create a line set visualization of vertex normals
    
    Args:
        mesh: Trimesh object
        normal_length: Length of normal visualization lines
        color: RGB color for the normal lines
        
    Returns:
        pyrender.Mesh object with normal visualization
    """
    # Sample a subset of vertices for cleaner visualization 
    indices = np.random.choice(len(mesh.vertices), size=min(1000, len(mesh.vertices)), replace=False)
    vertices = mesh.vertices[indices]
    normals = mesh.vertex_normals[indices]
    
    # Create start and end points for each normal line
    starts = vertices
    ends = vertices + normals * normal_length
    
    # Interleave start/end points to create line segments
    points = np.vstack([starts, ends]).reshape(-1, 3)
    
    # Create connectivity for lines (pairs of consecutive vertices)
    indices = np.arange(len(points)).reshape(-1, 2)
    
    # Set color (default red for hand, blue for object)
    if color is None:
        color = [1.0, 0.0, 0.0, 1.0]  # Red with alpha=1
    colors = np.tile(color, (len(points), 1))
    
    # Create line set primitive in pyrender
    normal_vis = pyrender.Mesh.from_points(points, colors=colors)
    return normal_vis

if __name__ == "__main__":
    # Load and prepare the mesh with texture
    mesh_path = "/svl/u/tarunc/black_cup.obj"
    texture_path = "/svl/u/tarunc/black_cup.jpg"
    output_path = "/svl/u/tarunc/black_cup_visualization.jpg"
    decim_mesh_path = "/svl/u/tarunc/black_cup-decimated_to_1000_vertices.obj"
    # Load the mesh and apply texture
    mesh = trimesh.load(decim_mesh_path)
    texture = Image.open(texture_path)
    material = trimesh.visual.texture.SimpleMaterial(image=texture)
    color_visuals = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=texture, material=material)
    mesh.visual = color_visuals
    breakpoint()
    mesh.vertices *= 1e-3
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh.vertices)
    mesh.apply_transform(to_origin)
    
    # Set up the scene with lighting
    scene = Scene(ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
    
    # Add lights to the scene
    direc_l = DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    spot_l = SpotLight(color=[1.0, 1.0, 1.0], intensity=5.0,
                innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
    
    # Create a camera
    camera = IntrinsicsCamera(fx=600, fy=600, cx=320, cy=240)
    
    # Create a rotation that looks at the object from a good angle
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.7071, -0.7071, -0.2],
        [0.0, 0.7071, 0.7071, 0.2],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Add mesh to the scene
    object_node = Node(mesh=Mesh.from_trimesh(mesh, smooth=True))
    scene.add_node(object_node)
    
    # Add normal visualization to help understand the mesh structure
    normal_vis = create_normal_visualization(
        mesh, 
        normal_length=0.01,
        color=[0.0, 0.0, 1.0, 1.0]  # Blue for normals
    )
    normal_node = Node(mesh=normal_vis)
    scene.add_node(normal_node)
    
    # Add camera and lights to the scene
    scene.add(camera, pose=camera_pose)
    scene.add(direc_l, pose=camera_pose)
    scene.add(spot_l, pose=camera_pose)
    
    # Create an offscreen renderer
    r = OffscreenRenderer(viewport_width=640, viewport_height=480)
    
    # Render the scene
    color, depth = r.render(scene)
    
    # Save the rendering as a JPEG file
    Image.fromarray((color).astype(np.uint8)).save(output_path)
    
    # Clean up the renderer
    r.delete()
    
    print(f"Mesh visualization saved to {output_path}")
    

