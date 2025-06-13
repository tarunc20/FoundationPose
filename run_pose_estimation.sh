#!/bin/bash
#SBATCH --account viscam 
#SBATCH --job-name generate_viz
#SBATCH --partition=svl 
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4
#SBATCH --output=/svl/u/tarunc/tool_use_benchmark/FoundationPose/slurm_outs/%j.out
#SBATCH --error=/svl/u/tarunc/tool_use_benchmark/FoundationPose/slurm_outs/%j.err
# General purpose script for running FoundationPose estimation on both tool and target objects

# Default values
CONFIG_FILE="fp_config.yaml"
EXTRINSIC_FILE="/svl/u/tarunc/reconstructed_cameras_scaled.yaml"

# Help message
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --data_dir DIR       Path to the data directory (required)"
    echo "  --tool_mesh FILE     Path to the tool mesh file (required)"
    echo "  --target_mesh FILE   Path to the target mesh file (optional)"
    echo "  --config_file FILE   Path to the config file (default: fp_config.yaml)"
    echo "  --extrinsic_file FILE Path to the extrinsic file (default: /svl/u/tarunc/reconstructed_cameras_new.yaml)"
    echo "  --help               Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --tool_mesh)
            TOOL_MESH="$2"
            shift 2
            ;;
        --target_mesh)
            TARGET_MESH="$2"
            shift 2
            ;;
        --config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --extrinsic_file)
            EXTRINSIC_FILE="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check required arguments
# if [ -z "$DATA_DIR" ]  then
#     echo "Error: data_dir is required."
#     show_help
# fi

# Set up the environment
cd /svl/u/tarunc/tool_use_benchmark/FoundationPose
source ~/.bashrc 
conda activate /svl/u/tarunc/pytorch3d_testing

# Generate a unique ID for this run
uuid=$(date +%s)
#uuid="testing_cropped_cams"
echo "Run UUID: $uuid"

# Run tool pose estimation
echo "Running tool pose estimation..."
if [ ! -z "$TOOL_MESH" ]; then
    echo "Running tool pose estimation..."
    python run_demo.py \
        --mesh_file=$TOOL_MESH \
        --config_file=$CONFIG_FILE \
        --data_dir=$DATA_DIR \
        --uuid=$uuid \
        --extrinsic_file=$EXTRINSIC_FILE \
        --t=tool
else
    echo "No tool mesh provided, skipping tool pose estimation."
fi

# If target mesh is provided, run target pose estimation
if [ ! -z "$TARGET_MESH" ]; then
    echo "Running target pose estimation..."
    python run_demo.py \
        --mesh_file=$TARGET_MESH \
        --config_file=$CONFIG_FILE \
        --data_dir=$DATA_DIR \
        --uuid=$uuid \
        --extrinsic_file=$EXTRINSIC_FILE \
        --t=target \
        --crop_masks 
else
    echo "No target mesh provided, skipping target pose estimation."
fi

echo "Pose estimation completed with UUID: $uuid" 