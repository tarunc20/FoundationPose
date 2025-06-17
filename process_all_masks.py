import os
import numpy as np
from glob import glob
from collections import defaultdict
from tqdm import tqdm

# Base directory where all experiments are located
BASE_DIR = "/viscam/projects/robotool/data/videos_0527/"

# EXPERIMENT_DIRS = [
#     "c40b65ca_blue_scooper_0",
#     "d2133627_plastic_scoop_0",
#     "fea56da7_measuring_cup_0",
#     "fea56da7_measuring_cup_1",
# ]

EXPERIMENT_DIRS = [
    #"2379b837_coffee_0",
    #"28dfb756_pestie_1",
    #"2d2f0621_milk_0",
    #"4a0042e8_knife_paper_0",
    #"2379b837_coffee_1",
    #"28dfb756_pestie_2",
    #"2d2f0621_milk_1"
    "97ca6950_plastic_scoop_retest_0",
]

def process_experiment_directory(experiment_dir):
    """
    Process all backward*.npy files in the masks/i.mp4 directories for a given experiment.
    
    Args:
        experiment_dir (str): Path to the experiment directory
        
    Returns:
        list: A list of 8 lists, where each inner list contains the loaded numpy arrays
              for one mp4 directory (masks/0.mp4 through masks/7.mp4)
    """
    full_path = os.path.join(BASE_DIR, experiment_dir)
    mask_lists = [[] for _ in range(8)]
    
    # Process each masks/i.mp4 directory
    for i in range(8):
        mp4_dir = os.path.join(full_path, f"masks/cam0{i}.mp4")
        
        # Check if directory exists
        if not os.path.exists(mp4_dir):
            print(f"Warning: Directory {mp4_dir} does not exist")
            continue
        
        # Find all backward*.npy files
        all_npy_files = glob(os.path.join(mp4_dir, "*.npy"))
        # Sort by the frame number that comes after "backward_"
        all_npy_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        # Load each numpy file
        for npy_file in tqdm(all_npy_files, desc=f"Loading masks for {os.path.basename(mp4_dir)}"):
            try:
                data = np.load(npy_file)
                if len(data.shape) == 3:
                    mask_lists[i].append(data[0, :, :])
                else:
                    mask_lists[i].append(data)
            except Exception as e:
                assert False, f"Error loading {npy_file}: {e}"
                #print(f"Error loading {npy_file}: {e}")   
    min_mask_len = min([len(mask_lists[i]) for i in range(8) if len(mask_lists[i]) > 0])
    for i in range(8):
        if len(mask_lists[i]) > 0:
            mask_lists[i] = np.asarray(mask_lists[i])[:min_mask_len][:, None, :, :]
        else:
            mask_lists[i] = np.zeros((min_mask_len, 1, 480, 640))
    full_array = np.concatenate(mask_lists, axis=1)
    print(full_array.shape)
    print("Unique values in full_array:", np.unique(full_array))
    np.save(os.path.join(full_path, "masks_auxiliary.npy"), full_array * 255)
    return full_array

def process_all_experiments():
    """
    Process all experiment directories and return a dictionary mapping
    experiment directory names to their respective mask lists.
    
    Returns:
        dict: A dictionary where keys are experiment directory names and values
              are lists of 8 lists containing loaded numpy arrays
    """
    all_experiments = {}
    for exp_dir in EXPERIMENT_DIRS:
        print(f"Processing {exp_dir}...")
        mask_lists = process_experiment_directory(exp_dir)
        all_experiments[exp_dir] = mask_lists
        
        # Print summary for this experiment
        for i, masks in enumerate(mask_lists):
            print(f"  masks/cam0{i}.mp4: {len(masks)} backward files loaded")
    
    return all_experiments

if __name__ == "__main__":
    # Process all experiments and get the results
    all_experiment_masks = process_all_experiments()
    
    # Print overall summary
    print("\nSummary:")
    for exp_dir, mask_lists in all_experiment_masks.items():
        total_masks = sum(len(masks) for masks in mask_lists)
        print(f"{exp_dir}: {total_masks} total backward mask files loaded")
