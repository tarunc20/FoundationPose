#!/bin/bash
#SBATCH --account viscam 
#SBATCH --job-name batch-reconstruct 
#SBATCH --partition=svl 
#SBATCH --gres=gpu:2 
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4
#SBATCH --mem-per-gpu=128G 

cd /svl/u/tarunc/tool_use_benchmark/HandReconstruction
source ~/.bashrc 
conda activate reconstruct-hand

# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/a22047ab/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/a22047ab --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml
# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/78606156/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/78606156 --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml
# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/21b2c1fa/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/21b2c1fa --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml
# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/cf050037/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/cf050037 --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml

# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/7d87f11c/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/7d87f11c --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml
# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/832fe9a0/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/832fe9a0 --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml

# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/5696fced/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/5696fced --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml
# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/9ceaac04/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/9ceaac04 --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml

# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/0c78ea75/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/0c78ea75/ --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml
# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/278f3099/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/278f3099 --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml
# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/afb6467e/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/afb6467e --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml

# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/bad428f7/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/bad428f7 --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml
# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/8dfeca2b/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/8dfeca2b --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml
# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/23c17b5c/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/23c17b5c --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml

# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/437c550b/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/437c550b --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml
# python reconstruct.py --data_file=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/2d41a7fc/data00000000.h5 --result_path=/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/2d41a7fc --camera_file=/svl/u/tarunc/camera_ext_calibration.yaml

python joint_losses.py 