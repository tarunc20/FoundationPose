#!/bin/bash
#SBATCH --account viscam 
#SBATCH --job-name generate_viz
#SBATCH --partition=svl 
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G

# Default values
data_dir=""
config_dir=""
object="" # should be tool or target 
mesh_file=""
run_id=$(date +%s)

# Parse named arguments
while getopts ":d:c:o:" opt; do
  case $opt in
    d) data_dir="$OPTARG" ;;
    c) config_dir="$OPTARG" ;;
    o) object="$OPTARG" ;;
    m) mesh_file="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument" >&2; exit 1 ;;
  esac
done

# Check if required arguments are provided
if [ -z "$data_dir" ] || [ -z "$config_dir" ] || [ -z "$object" ] || [ -z "$mesh_file" ]; then
  echo "Usage: $0 -d <data_dir> -c <config_dir> -o <object>"
  exit 1
fi

# Your script logic here
echo "Data Directory: $data_dir"
echo "Config Directory: $config_dir"
echo "Object: $object"
echo "Run id: $run_id"
echo "Mesh file: $mesh_file" 


