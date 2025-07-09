#!/usr/bin/env python3
"""
ManiSkill to LeRobot Dataset Converter

Converts ManiSkill dataset format (.npz files) to LeRobot format for VLA training.
Supports Panda robot demonstrations with RGB observations, actions, and joint states.
"""

import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List, Any
import argparse
from tqdm import tqdm


def create_directory_structure(output_dir: str) -> Path:
    """Create necessary directory structure for LeRobot dataset."""
    base_path = Path(output_dir)
    
    (base_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (base_path / "meta").mkdir(parents=True, exist_ok=True)
    (base_path / "videos" / "chunk-000" / "observation.images.main").mkdir(parents=True, exist_ok=True)
    
    return base_path


def resize_image_with_padding(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Resize image while maintaining aspect ratio and adding padding."""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result


def create_video_from_frames(frames: np.ndarray, output_path: str, fps: int = 30):
    """Create MP4 video from RGB frames."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    resized_frames = [resize_image_with_padding(frame, (224, 224)) for frame in frames]
    resized_frames = np.array(resized_frames)
    h, w = resized_frames.shape[1:3]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in resized_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def calculate_image_statistics_from_raw(all_rgb_data: List[np.ndarray]) -> Dict[str, Any]:
    """Computing image statistics from RGB data"""    
    valid_rgb_data = [rgb for rgb in all_rgb_data if rgb.size > 0]    
    all_pixels = []
    total_frames = 0
    
    for episode_rgb in valid_rgb_data:
        # episode_rgb shape: (num_frames, H, W, 3), values: [0, 255]
        normalized_rgb = episode_rgb.astype(np.float32) / 255.0
        
        pixels = normalized_rgb.reshape(-1, 3)  # (num_pixels, 3)
        if len(pixels) > 50000:  
            indices = np.random.choice(len(pixels), 50000, replace=False)
            pixels = pixels[indices]
        
        all_pixels.extend(pixels)
        total_frames += len(episode_rgb)
    
    all_pixels = np.array(all_pixels)  # (total_pixels, 3)
    
    stats = {
        'mean': [[[float(all_pixels[:, 0].mean())]], 
                 [[float(all_pixels[:, 1].mean())]], 
                 [[float(all_pixels[:, 2].mean())]]],
        'std': [[[float(all_pixels[:, 0].std())]], 
                [[float(all_pixels[:, 1].std())]], 
                [[float(all_pixels[:, 2].std())]]],
        'max': [[[float(all_pixels[:, 0].max())]], 
                [[float(all_pixels[:, 1].max())]], 
                [[float(all_pixels[:, 2].max())]]],
        'min': [[[float(all_pixels[:, 0].min())]], 
                [[float(all_pixels[:, 1].min())]], 
                [[float(all_pixels[:, 2].min())]]]
    }
    return stats


def extract_robot_state(joints: np.ndarray) -> np.ndarray:
    """Extract robot state from joint array (first 7 joints for Panda)."""
    return joints[:, :7]


def process_episode(episode_data: Dict[str, np.ndarray], episode_idx: int, task_index: int = 0) -> pd.DataFrame:
    """Process single episode and return DataFrame."""
    actions = episode_data['action']
    joints = episode_data['joints']
    robot_state = extract_robot_state(joints)
    
    episode_length = actions.shape[0]
    timestamps = np.arange(episode_length, dtype=np.float32) / 30.0
    frame_indices = np.arange(episode_length, dtype=np.int64)
    indices = np.arange(episode_length, dtype=np.int64)
    episode_indices = np.full(episode_length, episode_idx, dtype=np.int64)
    task_indices = np.full(episode_length, task_index, dtype=np.int64)
    
    df_data = {
        'action': list(actions),
        'observation.state': list(robot_state),
        'timestamp': timestamps,
        'frame_index': frame_indices,
        'episode_index': episode_indices,
        'index': indices,
        'task_index': task_indices
    }
    
    return pd.DataFrame(df_data)

def calculate_statistics(all_dataframes: List[pd.DataFrame], all_rgb_data: List[np.ndarray]) -> Dict[str, Any]:
    """Calculate statistics for all data."""
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    stats = {}
    
    # Action statistics
    actions = np.stack(combined_df['action'].values)
    stats['action'] = {
        'mean': actions.mean(axis=0).tolist(),
        'std': actions.std(axis=0).tolist(),
        'max': actions.max(axis=0).tolist(),
        'min': actions.min(axis=0).tolist()
    }
    
    # State statistics
    states = np.stack(combined_df['observation.state'].values)
    stats['observation.state'] = {
        'mean': states.mean(axis=0).tolist(),
        'std': states.std(axis=0).tolist(),
        'max': states.max(axis=0).tolist(),
        'min': states.min(axis=0).tolist()
    }
    
    stats['observation.images.main'] = calculate_image_statistics_from_raw(all_rgb_data)


    # Scalar field statistics
    for field in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
        values = combined_df[field].values
        stats[field] = {
            'mean': [float(values.mean())],
            'std': [float(values.std())],
            'max': [float(values.max())],
            'min': [float(values.min())]
        }
    
    return stats


def create_meta_files(base_path: Path, episode_lengths: List[int], total_frames: int):
    """Create metadata files for LeRobot dataset."""
    
    # episodes.jsonl
    episodes_data = []
    for i, length in enumerate(episode_lengths):
        episodes_data.append({
            "episode_index": i,
            "tasks": ["Pick the cube to the target position."],
            "length": length
        })
    
    with open(base_path / "meta" / "episodes.jsonl", 'w') as f:
        for episode in episodes_data:
            f.write(json.dumps(episode) + '\n')
    
    # tasks.jsonl
    with open(base_path / "meta" / "tasks.jsonl", 'w') as f:
        f.write(json.dumps({"task_index": 0, "task": "Pick the cube to the target position."}) + '\n')
    
    # info.json
    info_data = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": len(episode_lengths),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episode_lengths),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {
            "train": f"0:{len(episode_lengths)}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]
            },
            "observation.state": {
                "dtype": "float32", 
                "shape": [7],
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
            },
            "observation.images.main": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 224,
                    "video.width": 224,
                    "video.channels": 3,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None}
        }
    }
    
    with open(base_path / "meta" / "info.json", 'w') as f:
        json.dump(info_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Convert ManiSkill dataset to LeRobot format')
    parser.add_argument('input_dir', help='Directory containing .npz files')
    parser.add_argument('output_dir', help='Output directory for LeRobot dataset')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS (default: 30)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    base_path = create_directory_structure(args.output_dir)
    
    npz_files = sorted(list(input_path.glob("train_data_*.npz")))
    
    if not npz_files:
        print(f"No .npz files found in {input_path}")
        return
    
    print(f"Found {len(npz_files)} .npz files")
    
    all_dataframes = []
    all_rgb_data = []
    episode_lengths = []
    global_index = 0
    
    for episode_idx, npz_file in enumerate(tqdm(npz_files, desc="Processing episodes")):
        with np.load(npz_file) as data:
            episode_data = {key: data[key] for key in data.keys()}
        
        if 'rgb' in episode_data:
            all_rgb_data.append(episode_data['rgb'])
  
        df = process_episode(episode_data, episode_idx)
        
        episode_length = len(df)
        df['index'] = range(global_index, global_index + episode_length)
        global_index += episode_length
        
        parquet_path = base_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        if 'rgb' in episode_data:
            video_path = base_path / "videos" / "chunk-000" / "observation.images.main" / f"episode_{episode_idx:06d}.mp4"
            create_video_from_frames(episode_data['rgb'], video_path, args.fps)
        
        all_dataframes.append(df)
        episode_lengths.append(episode_length)
    
    stats = calculate_statistics(all_dataframes, all_rgb_data)
    
    with open(base_path / "meta" / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    total_frames = sum(episode_lengths)
    create_meta_files(base_path, episode_lengths, total_frames)
    
    print(f"\nConversion completed")
    print(f"Total episodes: {len(episode_lengths)}")
    print(f"Total frames: {total_frames}")
    print(f"Output directory: {base_path}")


if __name__ == "__main__":
    main()
