#!/usr/bin/env python3
"""
Converts ManiSkill HDF5 trajectory files to LeRobot v3.0 format.

Usage:
    python convert_maniskill_to_lerobot.py input.h5 output_dir --task-name "Pick cube"

For more information: https://github.com/huggingface/lerobot
"""

import json
import logging
import numpy as np
import pandas as pd
import cv2
import h5py
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass
import tyro
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_FPS = 30
DEFAULT_IMAGE_SIZE = "640x480"
DEFAULT_CHUNKS_SIZE = 1000


@dataclass
class Args:
    traj_path: str
    """Path to ManiSkill .h5 trajectory file"""
    
    output_dir: str
    """Output directory for LeRobot dataset"""
    
    fps: int = DEFAULT_FPS
    """Video FPS (default: 30)"""
    
    task_name: Optional[str] = None
    """Task description (default: auto-detected from metadata)"""
    
    chunks_size: int = DEFAULT_CHUNKS_SIZE
    """Episodes per chunk (default: 1000)"""
    
    image_size: str = DEFAULT_IMAGE_SIZE
    """Output image size as WIDTHxHEIGHT or single value for square (default: 640x480)"""
    
    robot_type: Optional[str] = None
    """Robot type (default: auto-detected, e.g., "panda", "ur5")"""


def load_metadata(h5_file: Path) -> Dict[str, Any]:
    json_file = h5_file.with_suffix('.json')
    if json_file.exists():
        try:
            with open(json_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse metadata JSON: {e}")
    return {}


def detect_rgb_cameras(obs_group: h5py.Group) -> List[str]:
    cameras = []
    if 'sensor_data' in obs_group:
        sensor_data = obs_group['sensor_data']
        for camera_name in sensor_data.keys():
            if 'rgb' in sensor_data[camera_name]:
                cameras.append(camera_name)
    return cameras


def load_trajectory_from_h5(h5_file: Path) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    if not h5_file.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
        
    episodes = []
    metadata = load_metadata(h5_file)
    
    with h5py.File(h5_file, 'r') as f:
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]
        
        if not traj_keys:
            raise ValueError(f"No trajectories found in {h5_file}. Expected keys starting with 'traj_'")
        
        first_traj = f[traj_keys[0]]
        actions = first_traj['actions'][:]
        action_dim = actions.shape[1]
        
        rgb_cameras = detect_rgb_cameras(first_traj['obs']) if 'obs' in first_traj else []
        
        state_dim = None
        if 'obs' in first_traj and 'agent' in first_traj['obs'] and 'qpos' in first_traj['obs']['agent']:
            qpos = first_traj['obs']['agent']['qpos'][:]
            state_dim = qpos.shape[1]
        
        logger.info(f"Detected: action_dim={action_dim}, state_dim={state_dim}, cameras={rgb_cameras}")
        
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


def parse_image_size(size_str: str) -> Tuple[int, int]:
    if 'x' in size_str:
        parts = size_str.split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid image size format: {size_str}. Expected 'WIDTHxHEIGHT' or 'SIZE'")
        width, height = int(parts[0]), int(parts[1])
    else:
        width = height = int(size_str)
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got: {width}x{height}")
    
    return width, height


def create_directory_structure(
    output_dir: str, 
    rgb_cameras: List[str], 
    num_episodes: int, 
    chunks_size: int = DEFAULT_CHUNKS_SIZE
) -> Path:
    base_path = Path(output_dir)
    num_chunks = (num_episodes + chunks_size - 1) // chunks_size
    
    for chunk_idx in range(num_chunks):
        (base_path / "data" / f"chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
        
        for camera_name in rgb_cameras:
            camera_path = base_path / "videos" / f"observation.images.{camera_name}" / f"chunk-{chunk_idx:03d}"
            camera_path.mkdir(parents=True, exist_ok=True)

    (base_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    return base_path


def resize_image_with_padding(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return result


def create_video_from_frames(
    frames: np.ndarray, 
    output_path: Path, 
    fps: int, 
    image_width: int, 
    image_height: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    target_size = (image_width, image_height)
    resized_frames = [resize_image_with_padding(frame, target_size) for frame in frames]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (image_width, image_height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")
    
    for frame in resized_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def process_episode(
    episode_data: Dict[str, np.ndarray], 
    episode_idx: int, 
    has_state: bool, 
    fps: int, 
    task_index: int = 0, 
    task_name: str = "Unknown task"
) -> pd.DataFrame:
    actions = episode_data['actions']
    episode_length = actions.shape[0]
    timestamps = np.arange(episode_length, dtype=np.float32) / fps
    
    df_data = {
        'action': [row.tolist() for row in actions],
        'timestamp': timestamps,
        'frame_index': np.arange(episode_length, dtype=np.int64),
        'episode_index': np.full(episode_length, episode_idx, dtype=np.int64),
        'index': np.arange(episode_length, dtype=np.int64),
        'task_index': np.full(episode_length, task_index, dtype=np.int64),
        'task': [task_name] * episode_length
    }
    
    if has_state and 'robot_state' in episode_data:
        df_data['observation.state'] = [row.tolist() for row in episode_data['robot_state']]
    
    column_order = ['action', 'observation.state', 'timestamp', 'frame_index', 
                    'episode_index', 'index', 'task_index', 'task']
    
    df = pd.DataFrame(df_data)
    
    # Ensure task is stored as string
    if 'task' in df.columns:
        df['task'] = df['task'].astype(str)
    
    return df[[col for col in column_order if col in df.columns]]


def calculate_statistics(
    all_dataframes: List[pd.DataFrame], 
    all_rgb_data_by_camera: Dict[str, List[np.ndarray]], 
    has_state: bool
) -> Dict[str, Any]:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    stats = {}
    
    actions = np.stack(combined_df['action'].values)
    stats['action'] = {
        'mean': actions.mean(axis=0).tolist(),
        'std': actions.std(axis=0).tolist(),
        'max': actions.max(axis=0).tolist(),
        'min': actions.min(axis=0).tolist(),
        'count': [len(actions)]
    }
    
    if has_state and 'observation.state' in combined_df:
        states = np.stack(combined_df['observation.state'].values)
        stats['observation.state'] = {
            'mean': states.mean(axis=0).tolist(),
            'std': states.std(axis=0).tolist(),
            'max': states.max(axis=0).tolist(),
            'min': states.min(axis=0).tolist(),
            'count': [len(states)]
        }
    
    for camera_name, rgb_data in all_rgb_data_by_camera.items():
        if rgb_data:
            all_pixels = []
            total_frames = 0
            for episode_rgb in rgb_data:
                normalized_rgb = episode_rgb.astype(np.float32) / 255.0
                pixels = normalized_rgb.reshape(-1, 3)
                total_frames += len(episode_rgb)
                if len(pixels) > 50000:
                    indices = np.random.choice(len(pixels), 50000, replace=False)
                    pixels = pixels[indices]
                all_pixels.extend(pixels)
            
            all_pixels = np.array(all_pixels)
            stats[f'observation.images.{camera_name}'] = {
                'mean': [[[float(all_pixels[:, i].mean())]] for i in range(3)],
                'std': [[[float(all_pixels[:, i].std())]] for i in range(3)],
                'max': [[[float(all_pixels[:, i].max())]] for i in range(3)],
                'min': [[[float(all_pixels[:, i].min())]] for i in range(3)],
                'count': [[[total_frames]]] * 3
            }
    
    for field in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
        values = combined_df[field].values
        stats[field] = {
            'mean': [float(values.mean())],
            'std': [float(values.std())],
            'max': [int(values.max())] if field != 'timestamp' else [float(values.max())],
            'min': [int(values.min())] if field != 'timestamp' else [float(values.min())],
            'count': [len(values)]
        }
    
    return stats


def create_meta_files(
    base_path: Path, 
    episode_lengths: List[int], 
    total_frames: int,
    action_dim: int, 
    state_dim: Optional[int], 
    rgb_cameras: List[str], 
    metadata: Dict[str, Any], 
    task_name: str, 
    chunks_size: int, 
    fps: int, 
    image_width: int, 
    image_height: int, 
    all_dataframes: List[pd.DataFrame],
    all_rgb_data_by_camera: Dict[str, List[np.ndarray]],
    robot_type_override: Optional[str] = None
) -> None:
    num_chunks = (len(episode_lengths) + chunks_size - 1) // chunks_size
    
    episodes_data = []
    dataset_from_index = 0
    
    for ep_idx, (length, df) in enumerate(zip(episode_lengths, all_dataframes)):
        chunk_idx = ep_idx // chunks_size
        
        episode_meta = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": 0,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_from_index + length,
            "tasks": [task_name],
            "length": length,
        }
        
        for camera_name in rgb_cameras:
            prefix = f"videos/observation.images.{camera_name}"
            episode_meta[f"{prefix}/chunk_index"] = chunk_idx
            episode_meta[f"{prefix}/file_index"] = ep_idx
            episode_meta[f"{prefix}/from_timestamp"] = float(df['timestamp'].iloc[0])
            episode_meta[f"{prefix}/to_timestamp"] = float(df['timestamp'].iloc[-1])
        
        actions = np.stack(df['action'].values)
        episode_meta["stats/action/min"] = actions.min(axis=0).tolist()
        episode_meta["stats/action/max"] = actions.max(axis=0).tolist()
        episode_meta["stats/action/mean"] = actions.mean(axis=0).tolist()
        episode_meta["stats/action/std"] = actions.std(axis=0).tolist()
        episode_meta["stats/action/count"] = [length]
        
        if state_dim and 'observation.state' in df:
            states = np.stack(df['observation.state'].values)
            episode_meta["stats/observation.state/min"] = states.min(axis=0).tolist()
            episode_meta["stats/observation.state/max"] = states.max(axis=0).tolist()
            episode_meta["stats/observation.state/mean"] = states.mean(axis=0).tolist()
            episode_meta["stats/observation.state/std"] = states.std(axis=0).tolist()
            episode_meta["stats/observation.state/count"] = [length]
        
        for camera_name in rgb_cameras:
            if camera_name in all_rgb_data_by_camera and ep_idx < len(all_rgb_data_by_camera[camera_name]):
                rgb_data = all_rgb_data_by_camera[camera_name][ep_idx].astype(np.float32) / 255.0
                prefix = f"stats/observation.images.{camera_name}"
                episode_meta[f"{prefix}/min"] = [[[float(rgb_data[..., i].min())]] for i in range(3)]
                episode_meta[f"{prefix}/max"] = [[[float(rgb_data[..., i].max())]] for i in range(3)]
                episode_meta[f"{prefix}/mean"] = [[[float(rgb_data[..., i].mean())]] for i in range(3)]
                episode_meta[f"{prefix}/std"] = [[[float(rgb_data[..., i].std())]] for i in range(3)]
                episode_meta[f"{prefix}/count"] = [[[length]]]
        
        for field in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']:
            values = df[field].values
            episode_meta[f"stats/{field}/min"] = [int(values.min())] if field != 'timestamp' else [float(values.min())]
            episode_meta[f"stats/{field}/max"] = [int(values.max())] if field != 'timestamp' else [float(values.max())]
            episode_meta[f"stats/{field}/mean"] = [float(values.mean())]
            episode_meta[f"stats/{field}/std"] = [float(values.std())]
            episode_meta[f"stats/{field}/count"] = [length]
        
        episode_meta["meta/episodes/chunk_index"] = 0
        episode_meta["meta/episodes/file_index"] = 0
        
        episodes_data.append(episode_meta)
        dataset_from_index += length
    
    episodes_df = pd.DataFrame(episodes_data)
    episodes_df.to_parquet(base_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    
    tasks_df = pd.DataFrame({"task_index": [0]}, index=[task_name])
    tasks_df.index.name = None
    tasks_df.to_parquet(base_path / "meta" / "tasks.parquet", index=True)
    
    # Determine robot type: use override if provided, otherwise auto-detect
    robot_type = "unknown"
    if robot_type_override:
        robot_type = robot_type_override
    elif metadata and 'env_info' in metadata:
        env_id = metadata['env_info'].get('env_id', 'unknown')
        robot_type = env_id.split('-')[0].lower() if '-' in env_id else 'unknown'
    
    features = {
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [f"action_{i}" for i in range(action_dim)],
            "fps": float(fps)
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [f"joint_{i}" for i in range(state_dim)],
            "fps": float(fps)
        } if state_dim else {},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": float(fps)},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(fps)},
        "task": {"dtype": "string", "shape": [1], "names": None, "fps": float(fps)}
    }
    
    # Remove empty observation.state if not present
    if not state_dim:
        del features["observation.state"]
    
    for camera_name in rgb_cameras:
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": [image_height, image_width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(fps),
                "video.height": image_height,
                "video.width": image_width,
                "video.channels": 3,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }
    
    data_files_size = sum(f.stat().st_size for f in (base_path / "data").rglob("*.parquet"))
    data_files_size_mb = int(data_files_size / (1024 * 1024))
    
    info_data = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": len(episode_lengths),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episode_lengths) * len(rgb_cameras),
        "total_chunks": num_chunks,
        "chunks_size": chunks_size,
        "fps": fps,
        "data_files_size_in_mb": data_files_size_mb,
        "splits": {"train": f"0:{len(episode_lengths)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features
    }
    
    with open(base_path / "meta" / "info.json", 'w') as f:
        json.dump(info_data, f, indent=2)


def main(args: Args):
    if args.chunks_size <= 0:
        raise ValueError("--chunks-size must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    input_path = Path(args.traj_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        logger.info(f"Loading trajectories from {input_path}")
        episodes, info = load_trajectory_from_h5(input_path)
        logger.info(f"Found {len(episodes)} episodes")
        
        task_name = args.task_name
        if not task_name and info['metadata'] and 'env_info' in info['metadata']:
            task_name = info['metadata']['env_info'].get('env_id', 'Unknown task')
        if not task_name:
            task_name = "Unknown task"
            logger.warning("No task name provided and couldn't auto-detect. Using 'Unknown task'")
        
        base_path = create_directory_structure(args.output_dir, info['rgb_cameras'], len(episodes), args.chunks_size)
        image_width, image_height = parse_image_size(args.image_size)
        
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
            
            df = process_episode(episode_data, episode_idx, info['state_dim'] is not None, args.fps,
                               task_index=0, task_name=task_name)
            episode_length = len(df)
            df['index'] = range(global_index, global_index + episode_length)
            global_index += episode_length
                    
            for camera_name in info['rgb_cameras']:
                rgb_key = f'rgb_{camera_name}'
                if rgb_key in episode_data:
                    video_path = base_path / "videos" / f"observation.images.{camera_name}" / f"chunk-{chunk_idx:03d}" / f"file-{episode_idx:03d}.mp4"
                    create_video_from_frames(episode_data[rgb_key], video_path, args.fps, image_width, image_height)

            all_dataframes.append(df)
            episode_lengths.append(episode_length)
        
        num_chunks = (len(episodes) + args.chunks_size - 1) // args.chunks_size
        logger.info(f"Saving data to {num_chunks} chunk(s)")
        
        for chunk_idx in range(num_chunks):
            start_ep = chunk_idx * args.chunks_size
            end_ep = min((chunk_idx + 1) * args.chunks_size, len(all_dataframes))
            
            chunk_dfs = all_dataframes[start_ep:end_ep]
            combined_df = pd.concat(chunk_dfs, ignore_index=True)
            
            # Force task column to be string type for parquet
            if 'task' in combined_df.columns:
                combined_df['task'] = combined_df['task'].astype('string')

            parquet_path = base_path / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet"
            
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            schema_fields = []
            for col in combined_df.columns:
                if col == 'task':
                    schema_fields.append(pa.field('task', pa.string()))
                elif col in ['action', 'observation.state']:
                    schema_fields.append(pa.field(col, pa.list_(pa.float32())))
                elif col == 'timestamp':
                    schema_fields.append(pa.field(col, pa.float32()))
                elif col in ['frame_index', 'episode_index', 'index', 'task_index']:
                    schema_fields.append(pa.field(col, pa.int64()))
            
            schema = pa.schema(schema_fields)
            table = pa.Table.from_pandas(combined_df, schema=schema)
            pq.write_table(table, parquet_path)
        
        logger.info("Calculating statistics")
        stats = calculate_statistics(all_dataframes, all_rgb_data_by_camera, info['state_dim'] is not None)
        with open(base_path / "meta" / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Creating metadata files")
        total_frames = sum(episode_lengths)
        create_meta_files(
            base_path, episode_lengths, total_frames,
            info['action_dim'], info['state_dim'], info['rgb_cameras'],
            info['metadata'], task_name, args.chunks_size, args.fps, 
            image_width, image_height, all_dataframes, all_rgb_data_by_camera,
            robot_type_override=args.robot_type
        )
        
        logger.info(f"\n{'='*80}")
        logger.info("Conversion completed successfully!")
        logger.info(f"{'='*80}")
        logger.info(f"Episodes: {len(episode_lengths)}")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Chunks: {num_chunks}")
        logger.info(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    parsed_args = tyro.cli(Args)
    sys.exit(main(parsed_args))