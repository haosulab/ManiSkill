# ManiSkill to LeRobot Converter

Example implementation for converting ManiSkill datasets to LeRobot format. Built for training VLA pi0 models on robotic manipulation tasks.

**Example output**: [maniskill-panda-pickcube](https://huggingface.co/datasets/dancher00/maniskill-panda-pickcube)

## Usage

```bash
python convert_to_lerobot.py /path/to/maniskill/data /path/to/output --fps 30
```

**Input**: Directory with `train_data_*.npz` files  
**Output**: LeRobot-compatible dataset

## Requirements

```bash
pip install numpy pandas opencv-python pyarrow tqdm
```

## Output Structure

```
output_dir/
├── data/chunk-000/           # Parquet files with actions/states
├── videos/chunk-000/         # MP4 videos (224x224, RGB)
├── meta/                     # Dataset metadata and statistics
└── README.md                 # Dataset documentation
```

## Data Format

- **Task**: ManiSkill PickCube-v1 (Panda robot)
- **Actions**: 7-DOF (x,y,z,rx,ry,rz,gripper)
- **States**: First 7 joints from ManiSkill joint array
- **Videos**: RGB observations resized with padding
- **FPS**: 30 (configurable)

## Notes

This converter was used to train VLA pi0 models on ManiSkill PickCube-v1 task. Modify `extract_robot_state()` and metadata for different robots or tasks.

Not production code - adapt for your specific use case.