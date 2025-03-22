#!/bin/bash
#SBATCH --account viscam 
#SBATCH --job-name batch-reconstruct 
#SBATCH --partition=svl 
#SBATCH --gres=gpu:1 
#SBATCH --mem-per-gpu=64G 

cd /svl/u/tarunc/tool_use_benchmark/HandReconstruction
source ~/.bashrc 
conda activate reconstruct-hand

python sim_visualization.py \
    --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/a22047ab/ \
    --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml \
    --mesh_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/a22047ab/mesh/spoon_mesh.obj

python sim_visualization.py \
    --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/78606156/ \
    --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml \
    --mesh_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/78606156/mesh/green_brush.obj

python sim_visualization.py \
    --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/21b2c1fa/ \
    --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml \
    --mesh_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/21b2c1fa/mesh/spoon_mesh.obj

python sim_visualization.py \
    --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/cf050037/ \
    --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml \
    --mesh_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/cf050037/mesh/spoon_mesh.obj

python sim_visualization.py \
    --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/7d87f11c/ \
    --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml \
    --mesh_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/7d87f11c/mesh/green_brush.obj

python sim_visualization.py \
    --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/832fe9a0/ \
    --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml \
    --mesh_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/832fe9a0/mesh/black_knife.obj

python sim_visualization.py \
    --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/5696fced/ \
    --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml \
    --mesh_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/5696fced/mesh/brush_scan.obj


python sim_visualization.py \
    --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/278f3099/ \
    --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml \
    --mesh_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/278f3099/mesh/hanger_mesh.obj

# python sim_visualization.py \
#     --out_dir=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/9ceaac04/ \
#     --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml \
#     --mesh_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/9ceaac04/mesh/spoon_mesh.obj