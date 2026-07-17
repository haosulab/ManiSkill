"""Convert ManiSkill PickCube-v1 demos (replayed with rgb obs) into a LeRobot v3.0 dataset."""

import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

H5 = Path(
    "/home/kelin/dataset/maniskill/pickcube/PickCube-v1/motionplanning/"
    "20260717_135248.rgb.pd_joint_pos.physx_cpu.h5"
)
OUT = Path("/home/kelin/dataset/maniskill/pickcube/lerobot")
REPO_ID = "maniskill/pickcube"
FPS = 20  # ManiSkill PickCube-v1 control_freq
TASK = "Pick up the cube and move it to the goal position."

CAM = "base_camera"

features = {
    "observation.images.base_camera": {
        "dtype": "video",
        "shape": (128, 128, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (9,),
        "names": [f"qpos_{i}" for i in range(9)],
    },
    "observation.qvel": {
        "dtype": "float32",
        "shape": (9,),
        "names": [f"qvel_{i}" for i in range(9)],
    },
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": [f"action_{i}" for i in range(8)],
    },
}

if OUT.exists():
    shutil.rmtree(OUT)

ds = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=FPS,
    features=features,
    root=OUT,
    robot_type="panda",
    use_videos=True,
)

f = h5py.File(H5, "r")
traj_keys = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))

total = 0
for k in traj_keys:
    t = f[k]
    actions = np.asarray(t["actions"], dtype=np.float32)  # (N, 8)
    rgb = np.asarray(t[f"obs/sensor_data/{CAM}/rgb"])  # (N+1, 128,128,3)
    qpos = np.asarray(t["obs/agent/qpos"], dtype=np.float32)  # (N+1, 9)
    qvel = np.asarray(t["obs/agent/qvel"], dtype=np.float32)  # (N+1, 9)

    n = actions.shape[0]
    assert rgb.shape[0] == n + 1, f"{k}: rgb {rgb.shape[0]} != {n + 1}"

    # Pair obs[i] with action[i]; the terminal obs[N] has no action so it is dropped.
    for i in range(n):
        ds.add_frame(
            {
                "observation.images.base_camera": rgb[i],
                "observation.state": qpos[i],
                "observation.qvel": qvel[i],
                "action": actions[i],
                "task": TASK,
            }
        )
    ds.save_episode()
    total += n
    print(f"  {k}: {n} frames -> saved")

f.close()
print(f"\nDONE: {len(traj_keys)} episodes, {total} frames -> {OUT}")
