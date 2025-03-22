#!/bin/bash

# TODO add ability to do all 8 views at once generate visualizations for all

# Function to display usage information
usage() {
    echo "Usage: $0 -f <h5_file> -x[1-8] <x_coordinate> -y[1-8] <y_coordinate> -m <mesh_file> -c <camera_file> -o <output_dir> -r <registration_frequency> -s <sam_version> "
    echo "  -f: Path to h5 file"
    echo "  -x[1-8]: List of pixel x coordinates"
    echo "  -y[1-8]: List of pixel y coordinates"
    echo "  -m: Path to mesh file for object being used"
    echo "  -c: Path to camera calibration file"
    echo "  -i: Index of camera (out of 8)" 
    echo "  -o: Output directory"
    echo "  -s: SAM version (1 or 2)"
    echo "  -r: Registration frequency for FoundationPose"
    exit 1
}

# Initialize variables
h5_file=""
x_coords=()
y_coords=()
mesh_file=""
camera_file=""
output_dir=""
sam_version=1
registration_frequency=10

while [[ $# -gt 0 ]]; do
    case $1 in
        -x)
            current_list="x"
            shift
            ;;
        -y)
            current_list="y"
            shift
            ;;
        -f)
            h5_file="$2"
            shift 2
            ;;
        -m)
            mesh_file="$2"
            shift 2
            ;;
        -c)
            camera_file="$2"
            shift 2
            ;;
        -o)
            output_dir="$2"
            shift 2
            ;;
        -s) 
            sam_version="$2"
            shift 2
            ;;
        -r)
            reconstruction_frequency="$2"
            shift 2
            ;;
        *)
            if [[ $current_list = "x" ]]; then
                x_coords+=("$1")
            elif [[ $current_list = "y" ]]; then
                y_coords+=("$1")
            else
                echo "Error: Argument $1 is not associated with -x or -y"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if required arguments are provided
# if [ -z "$h5_file" ] || [ -z "$x_coords" ] || [ -z "$y_coords" ] || [ -z "$mesh_file" ] || [ -z "$camera_file" ] || [ -z "$output_dir" ]; then
#     echo "Error: Missing required arguments"
#     usage
# fi

# # Validate inputs
# if [ ! -f "$h5_file" ]; then
#     echo "Error: H5 file does not exist"
#     exit 1
# fi

# if [ ! -f "$mesh_file" ]; then
#     echo "Error: Mesh file does not exist"
#     exit 1
# fi

# if [ ! -f "$camera_file" ]; then
#     echo "Error: Camera calibration file does not exist"
#     exit 1
# fi

# if ! [[ "$x_coord" =~ ^[0-9]+$ ]] || ! [[ "$y_coord" =~ ^[0-9]+$ ]]; then
#     echo "Error: X and Y coordinates must be integers"
#     exit 1
# fi

# # If SAM version is provided, validate it
# if [ ! -z "$sam_version" ]; then
#     if [ "$sam_version" != "sam" ] && [ "$sam_version" != "sam2" ]; then
#         echo "Error: Invalid SAM version. Use 'sam' or 'sam2'"
#         exit 1
#     fi
# fi

# Display the inputs
echo "H5 File: $h5_file"
echo "X Coordinate: ${x_coords[@]}"
echo "Y Coordinate: ${y_coords[@]}"
echo "Mesh File: $mesh_file"
echo "Camera Calibration File: $camera_file"
echo "Output Directory: $output_dir"


source ~/.bashrc 
rm -rf $output_dir
mkdir $output_dir 

echo "-------- RUNNING SAM --------"

if [ "$sam_version" = "1" ]; then 
    mkdir $output_dir/masks
    for i in {0..7}; do
        conda activate sam
        cd ../segment-anything 
        python segment_img.py --file="$h5_file" --x="${x_coords[i]}" --y="${y_coords[i]}" --cam="$i" --out="$output_dir/masks"
    done
fi 
if [ "$sam_version" = "2" ]; then
    # do stuff
    conda activate sam2
    cd ../sam2 
    python video_segmentation.py --h5_file="$h5_file" --out_dir="$output_dir" -x "${x_coords[@]}" -y "${y_coords[@]}"
fi 

# run foundation pose 
echo "-------- RUNNING FOUNDATION POSE --------"
conda activate /svl/u/tarunc/pytorch3d_testing
cd ../FoundationPose
for i in {0..7}; do
    python run_demo.py \
    --mesh_file="$mesh_file" \
    --test_scene_dir="$output_dir" \
    --debug=2 \
    --robotool_datafile="$h5_file" \
    --est_refine_iter=20 \
    --track_refine_iter=20 \
    --camera_idx=$i \
    --registration_frequency=$reconstruction_frequency
done 

# # run hand reconstruction
echo "-------- RUNNING HAND RECONSTRUCTION --------"
conda activate reconstruct-hand
cd ../HandReconstruction
python reconstruct.py \
--data_file="$h5_file" \
--result_path="$output_dir" \
--camera_file="$camera_file"

# run visualization
echo "-------- RUNNING VISUALIZATION --------"
for i in {0..7}; do
    python visualization.py \
    --h5_file="$h5_file" \
    --camera_file="$camera_file" \
    --camera_idx=$i \
    --mesh_path="$mesh_file" \
    --out_dir="$output_dir" 
done 