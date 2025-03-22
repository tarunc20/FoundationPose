#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np
import h5py
from tqdm import tqdm
import glob

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


def process_video(video_path, crop_size=50):
    """
    Process a single video:
    1. Extract all frames
    2. Crop each frame (remove crop_size pixels from each edge)
    3. Convert to binary mask (255 if not white, 0 if white)
    
    Args:
        video_path: Path to the video file
        crop_size: Number of pixels to crop from each edge
        
    Returns:
        List of processed frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Crop the frame
        h, w = frame.shape[:2]
        if h <= 2*crop_size or w <= 2*crop_size:
            print(f"Warning: Frame too small to crop {crop_size} pixels from each edge")
            cropped = frame
        else:
            cropped = frame[crop_size:h-crop_size, crop_size:w-crop_size]
        
        # Convert to binary mask (255 if not white, 0 if white)
        # White is defined as RGB(255,255,255)
        is_white = np.all(cropped == 255, axis=2)
        binary_mask = np.where(is_white, 0, 255).astype(np.uint8)
        
        frames.append(binary_mask)
    
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser(description='Process videos to binary masks and save to H5')
    parser.add_argument('input_dir', type=str, help='Directory containing MP4 videos')
    parser.add_argument('output_file', type=str, help='Output H5 file path')
    parser.add_argument('--crop', type=int, default=50, help='Pixels to crop from each edge (default: 50)')
    parser.add_argument('--pattern', type=str, default='*.mp4', help='File pattern to match videos (default: *.mp4)')
    args = parser.parse_args()
    
    # Find all video files in the input directory
    video_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    
    if not video_files:
        print(f"Error: No video files found in {args.input_dir} matching pattern {args.pattern}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process the first video to determine dimensions and frame count
    first_video_frames = process_video(video_files[0], args.crop)
    
    if not first_video_frames:
        print("Error: Failed to process the first video")
        return
    
    expected_frame_count = len(first_video_frames)
    frame_height, frame_width = first_video_frames[0].shape
    
    print(f"Expected frame count: {expected_frame_count}")
    print(f"Frame dimensions after cropping: {frame_width}x{frame_height}")
    
    # Create H5 file
    with h5py.File(args.output_file, 'w') as h5f:
        # Create dataset for masks
        masks_dataset = h5f.create_dataset(
            'masks', 
            shape=(expected_frame_count, len(video_files), frame_height, frame_width), 
            dtype=np.uint8
        )
        
        for video_idx, video_path in enumerate(video_files):
            frames = process_video(video_path, args.crop) if video_idx > 0 else first_video_frames
            
            if len(frames) != expected_frame_count:
                print(f"Warning: Video {video_path} has {len(frames)} frames, expected {expected_frame_count}")
                continue
            
            # Check if any frames have unexpected dimensions
            valid_frames = [frame for frame in frames if frame.shape == (frame_height, frame_width)]
            invalid_frames = len(frames) - len(valid_frames)
            
            if invalid_frames > 0:
                print(f"Warning: {invalid_frames} frames in video {video_path} have unexpected dimensions")
            
            # Convert valid frames to numpy array and assign to dataset in one operation
            if valid_frames:
                valid_indices = np.arange(len(valid_frames))
                masks_dataset[valid_indices, video_idx, :, :] = np.array(valid_frames)
    
    print(f"Successfully saved processed masks to {args.output_file}")

if __name__ == "__main__":
    main() 