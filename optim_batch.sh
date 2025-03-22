#!/bin/bash
#SBATCH --account viscam 
#SBATCH --job-name batch-reconstruct 
#SBATCH --partition=svl 
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G

cd /svl/u/tarunc/tool_use_benchmark/HO-Cap/examples
source ~/.bashrc 
conda activate ho-cap

# # 9e8d2e67_scoop_sand6
# python meshsdf_loss_example.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/9e8d2e67_scoop_sand6 \
#     --mesh_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/old_data/0c78ea75/mesh/wooden_spoon_scan.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=tool

# python meshsdf_loss_example.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/9e8d2e67_scoop_sand6 \
#     --mesh_file=/svl/u/tarunc/pitcher.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=target

# # 16713eef_scoop_coffee3
# python meshsdf_loss_example.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/16713eef_scoop_coffee3 \
#     --mesh_file=/svl/u/tarunc/quarter_cup.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=tool

# python meshsdf_loss_example.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/16713eef_scoop_coffee3 \
#     --mesh_file=/svl/u/tarunc/pitcher.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=target

# # 242181cc_scoop_icecream4
# python meshsdf_loss_example.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/242181cc_scoop_icecream4 \
#     --mesh_file=/svl/u/tarunc/green_scooper.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=tool

# python meshsdf_loss_example.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/242181cc_scoop_icecream4 \
#     --mesh_file=/svl/u/tarunc/pitcher.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=target

# # 1aac3dc9_scoop_icecream3
# python meshsdf_loss_example.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/1aac3dc9_scoop_icecream3 \
#     --mesh_file=/svl/u/tarunc/blue_scooper.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=tool

# python meshsdf_loss_example.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/1aac3dc9_scoop_icecream3 \
#     --mesh_file=/svl/u/tarunc/pitcher.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=target

# cd /svl/u/tarunc/tool_use_benchmark/HandReconstruction
# conda activate reconstruct-hand
# # 9e8d2e67_scoop_sand6
# python sim_visualization.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/9e8d2e67_scoop_sand6 \
#     --mesh_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/old_data/0c78ea75/mesh/wooden_spoon_scan.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=tool

# python sim_visualization.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/9e8d2e67_scoop_sand6 \
#     --mesh_path=/svl/u/tarunc/pitcher.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=target

# # 16713eef_scoop_coffee3
# python sim_visualization.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/16713eef_scoop_coffee3 \
#     --mesh_path=/svl/u/tarunc/quarter_cup.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=tool

# python sim_visualization.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/16713eef_scoop_coffee3 \
#     --mesh_path=/svl/u/tarunc/pitcher.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=target

# # 242181cc_scoop_icecream4
# python sim_visualization.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/242181cc_scoop_icecream4 \
#     --mesh_path=/svl/u/tarunc/green_scooper.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=tool

# python sim_visualization.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/242181cc_scoop_icecream4 \
#     --mesh_path=/svl/u/tarunc/pitcher.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=target

# # 1aac3dc9_scoop_icecream3
# python sim_visualization.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/1aac3dc9_scoop_icecream3 \
#     --mesh_path=/svl/u/tarunc/blue_scooper.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=tool

# python sim_visualization.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/1aac3dc9_scoop_icecream3 \
#     --mesh_path=/svl/u/tarunc/pitcher.obj \
#     --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
#     --t=target


python meshsdf_loss_example.py \
    --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/cfbef9d1_scoop_coffee5/ \
    --mesh_file=/svl/u/tarunc/wooden_spoon_updated.obj \
    --camera_file=/viscam/projects/robotool/videos_0121/camera_ext_calibration_0121.yaml \
    --t=tool