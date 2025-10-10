#!/usr/bin/env python3
"""
ManiSkill to LeRobot Dataset Converter
Converts HDF5 trajectories to LeRobot format.
"""

import json
import numpy as np
import pandas as pd
import cv2
import h5py
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import argparse
from tqdm import tqdm


def load_metadata(h5_file: Path) -> Dict:
    json_file = h5_file.with_suffix('.json')
    if json_file.exists():
        with open(json_file) as f:
            return json.load(f)
    return {}


def detect_rgb_cameras(obs_group: h5py.Group) -> List[str]:
    cameras = []
    if 'sensor_data' in obs_group:
        sensor_data = obs_group['sensor_data']
        for camera_name in sensor_data.keys():
            if 'rgb' in sensor_data[camera_name]:
                cameras.append(camera_name)
    return cameras


def load_trajectory_from_h5(h5_file: Path) -> Tuple[List[Dict[str, np.ndarray]], Dict]:
    episodes = []
    metadata = load_metadata(h5_file)
    
    with h5py.File(h5_file, 'r') as f:
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]
        
        if not traj_keys:
            raise ValueError("No trajectories found in HDF5 file")
        
        first_traj = f[traj_keys[0]]
        actions = first_traj['actions'][:]
        action_dim = actions.shape[1]
        
        rgb_cameras = detect_rgb_cameras(first_traj['obs']) if 'obs' in first_traj else []
        
        if 'obs' in first_traj and 'agent' in first_traj['obs'] and 'qpos' in first_traj['obs']['agent']:
            qpos = first_traj['obs']['agent']['qpos'][:]
            state_dim = qpos.shape[1]
        else:
            state_dim = None
        
        print(f"Detected: action_dim={action_dim}, state_dim={state_dim}, cameras={rgb_cameras}")
        
        for traj_key in traj_keys:
            traj = f[traj_key]
            actions = traj['actions'][:]
            episode_data = {'actions': actions}
            
            if rgb_cameras and 'obs' in traj:
                for camera_name in rgb_cameras:
                    rgb = traj['obs']['sensor_data'][camera_name]['rgb'][:]
                    episode_data[f'rgb_{camera_name}'] = rgb[:len(actions)]
            
            if state_dim and 'obs' in traj:
                qpos = traj['obs']['agent']['qpos'][:]
                episode_data['robot_state'] = qpos[:len(actions)]
            
            episodes.append(episode_data)
    
    info = {
        'action_dim': action_dim,
        'state_dim': state_dim,
        'rgb_cameras': rgb_cameras,
        'metadata': metadata
    }
    
    return episodes, info


def create_directory_structure(output_dir: str, rgb_cameras: List[str], num_episodes: int, chunks_size: int = 1000) -> Path:
    base_path = Path(output_dir)
    
    num_chunks = (num_episodes + chunks_size - 1) // chunks_size
    
    for chunk_idx in range(num_chunks):
        (base_path / "data" / f"chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
        
        for camera_name in rgb_cameras:
            (base_path / "videos" / f"chunk-{chunk_idx:03d}" / f"observation.images.{camera_name}").mkdir(parents=True, exist_ok=True)
    
    (base_path / "meta").mkdir(parents=True, exist_ok=True)
    return base_path


def resize_image_with_padding(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    resized_frames = [resize_image_with_padding(frame, (224, 224)) for frame in frames]
    h, w = 224, 224
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in resized_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def calculate_image_statistics(all_rgb_data: List[np.ndarray]) -> Dict[str, Any]:
    all_pixels = []
    
    for episode_rgb in all_rgb_data:
        normalized_rgb = episode_rgb.astype(np.float32) / 255.0
        pixels = normalized_rgb.reshape(-1, 3)
        if len(pixels) > 50000:
            indices = np.random.choice(len(pixels), 50000, replace=False)
            pixels = pixels[indices]
        all_pixels.extend(pixels)
    
    all_pixels = np.array(all_pixels)
    
    return {
        'mean': [[[float(all_pixels[:, i].mean())]] for i in range(3)],
        'std': [[[float(all_pixels[:, i].std())]] for i in range(3)],
        'max': [[[float(all_pixels[:, i].max())]] for i in range(3)],
        'min': [[[float(all_pixels[:, i].min())]] for i in range(3)]
    }


def process_episode(episode_data: Dict[str, np.ndarray], episode_idx: int, 
                   has_state: bool, fps: int, task_index: int = 0) -> pd.DataFrame:
    actions = episode_data['actions']
    episode_length = actions.shape[0]
    timestamps = np.arange(episode_length, dtype=np.float32) / fps
    
    df_data = {
        'action': list(actions),
        'timestamp': timestamps,
        'frame_index': np.arange(episode_length, dtype=np.int64),
        'episode_index': np.full(episode_length, episode_idx, dtype=np.int64),
        'index': np.arange(episode_length, dtype=np.int64),
        'task_index': np.full(episode_length, task_index, dtype=np.int64)
    }
    
    if has_state and 'robot_state' in episode_data:
        df_data['observation.state'] = list(episode_data['robot_state'])
    
    return pd.DataFrame(df_data)


def calculate_statistics(all_dataframes: List[pd.DataFrame], all_rgb_data_by_camera: Dict[str, List[np.ndarray]], 
                        has_state: bool) -> Dict[str, Any]:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    stats = {}
    
    actions = np.stack(combined_df['action'].values)
    stats['action'] = {
        'mean': actions.mean(axis=0).tolist(),
        'std': actions.std(axis=0).tolist(),
        'max': actions.max(axis=0).tolist(),
        'min': actions.min(axis=0).tolist()
    }
    
    if has_state and 'observation.state' in combined_df:
        states = np.stack(combined_df['observation.state'].values)
        stats['observation.state'] = {
            'mean': states.mean(axis=0).tolist(),
            'std': states.std(axis=0).tolist(),
            'max': states.max(axis=0).tolist(),
            'min': states.min(axis=0).tolist()
        }
    
    for camera_name, rgb_data in all_rgb_data_by_camera.items():
        if rgb_data:
            stats[f'observation.images.{camera_name}'] = calculate_image_statistics(rgb_data)
    
    for field in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
        values = combined_df[field].values
        stats[field] = {
            'mean': [float(values.mean())],
            'std': [float(values.std())],
            'max': [float(values.max())],
            'min': [float(values.min())]
        }
    
    return stats


def create_meta_files(base_path: Path, episode_lengths: List[int], total_frames: int,
                     action_dim: int, state_dim: Optional[int], rgb_cameras: List[str], 
                     metadata: Dict, task_name: str, chunks_size: int, fps: int):
    
    num_chunks = (len(episode_lengths) + chunks_size - 1) // chunks_size
    
    with open(base_path / "meta" / "episodes.jsonl", 'w') as f:
        for i, length in enumerate(episode_lengths):
            f.write(json.dumps({
                "episode_index": i,
                "tasks": [task_name],
                "length": length
            }) + '\n')
    
    with open(base_path / "meta" / "tasks.jsonl", 'w') as f:
        f.write(json.dumps({"task_index": 0, "task": task_name}) + '\n')
    
    features = {
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [f"action_{i}" for i in range(action_dim)]
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None}
    }
    
    if state_dim:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [f"joint_{i}" for i in range(state_dim)]
        }
    
    for camera_name in rgb_cameras:
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": [224, 224, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(fps),
                "video.height": 224,
                "video.width": 224,
                "video.channels": 3,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }
    
    robot_type = "unknown"
    if metadata and 'env_info' in metadata:
        env_id = metadata['env_info'].get('env_id', 'unknown')
        robot_type = env_id.split('-')[0].lower() if '-' in env_id else 'unknown'
    
    info_data = {
        "codebase_version": "v2.0",
        "robot_type": robot_type,
        "total_episodes": len(episode_lengths),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episode_lengths) * len(rgb_cameras),
        "total_chunks": num_chunks,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{len(episode_lengths)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features
    }
    
    if rgb_cameras:
        info_data["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    
    with open(base_path / "meta" / "info.json", 'w') as f:
        json.dump(info_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 dataset to LeRobot format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', help='Path to ManiSkill .h5 trajectory file')
    parser.add_argument('output_dir', help='Output directory for LeRobot dataset')
    parser.add_argument('--fps', type=int, default=30, metavar='N',
                       help='Video FPS (default: %(default)s)')
    parser.add_argument('--task-name', type=str, metavar='NAME',
                       help='Task description (default: auto-detected from metadata)')
    parser.add_argument('--chunks-size', type=int, default=1000, metavar='N',
                       help='Episodes per chunk (default: %(default)s)')
    
    args = parser.parse_args()
    
    if args.chunks_size <= 0:
        raise ValueError("chunks-size must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    print(f"Loading trajectories from {input_path}")
    episodes, info = load_trajectory_from_h5(input_path)
    print(f"Found {len(episodes)} episodes")
    
    task_name = args.task_name
    if not task_name and info['metadata'] and 'env_info' in info['metadata']:
        task_name = info['metadata']['env_info'].get('env_id', 'Unknown task')
    if not task_name:
        task_name = "Unknown task"
    
    base_path = create_directory_structure(args.output_dir, info['rgb_cameras'], len(episodes), args.chunks_size)
    
    all_dataframes = []
    all_rgb_data_by_camera = {camera: [] for camera in info['rgb_cameras']}
    episode_lengths = []
    global_index = 0
    
    for episode_idx, episode_data in enumerate(tqdm(episodes, desc="Processing episodes")):
        chunk_idx = episode_idx // args.chunks_size
        
        for camera_name in info['rgb_cameras']:
            rgb_key = f'rgb_{camera_name}'
            if rgb_key in episode_data:
                all_rgb_data_by_camera[camera_name].append(episode_data[rgb_key])
        
        df = process_episode(episode_data, episode_idx, info['state_dim'] is not None, args.fps)
        episode_length = len(df)
        df['index'] = range(global_index, global_index + episode_length)
        global_index += episode_length
        
        parquet_path = base_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        for camera_name in info['rgb_cameras']:
            rgb_key = f'rgb_{camera_name}'
            if rgb_key in episode_data:
                video_path = base_path / "videos" / f"chunk-{chunk_idx:03d}" / f"observation.images.{camera_name}" / f"episode_{episode_idx:06d}.mp4"
                create_video_from_frames(episode_data[rgb_key], video_path, args.fps)
        
        all_dataframes.append(df)
        episode_lengths.append(episode_length)
    
    stats = calculate_statistics(all_dataframes, all_rgb_data_by_camera, info['state_dim'] is not None)
    with open(base_path / "meta" / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    total_frames = sum(episode_lengths)
    num_chunks = (len(episode_lengths) + args.chunks_size - 1) // args.chunks_size
    create_meta_files(base_path, episode_lengths, total_frames,
                     info['action_dim'], info['state_dim'], info['rgb_cameras'],
                     info['metadata'], task_name, args.chunks_size, args.fps)
    
    print(f"\nConversion completed!")
    print(f"Episodes: {len(episode_lengths)}, Frames: {total_frames}, Chunks: {num_chunks}")
    print(f"Output directory: {base_path}")


if __name__ == "__main__":
    main()