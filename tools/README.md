# ManiSkill to LeRobot Converter

Converts ManiSkill demonstration trajectories (HDF5 format) to LeRobot dataset format for VLA training. 

State-only trajectories (no RGB) are supported - converter auto-detects available data.

**Example output**: [maniskill-panda-pickcube](https://huggingface.co/datasets/dancher00/maniskill-panda-pickcube)

## Usage

```bash
python convert_to_lerobot.py \
  ~/.maniskill/demos/PickCube-v1/teleop/trajectory.rgbd.pd_joint_pos.physx_cpu.h5 \
  ./output
```

### Optional arguments
```bash
python convert_to_lerobot.py input.h5 ./output \
  --fps 60 \
  --task-name "Pick the red cube" \
  --chunks-size 500
```

## Requirements

```bash
pip install h5py numpy pandas opencv-python pyarrow tqdm
```

## Output Structure

```
output/
├── data/
│   └── chunk-000/           # Parquet files with actions/states
├── videos/
│   └── chunk-000/           # MP4 videos (224x224)
└── meta/
    ├── info.json            # Dataset metadata
    ├── stats.json           # Statistics
    ├── episodes.jsonl       # Episode info
    └── tasks.jsonl          # Task descriptions
```

## Tested with robots
- Panda (7 DoF)
- SO100 (6 DoF)
