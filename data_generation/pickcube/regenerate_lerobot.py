"""Regenerate the PickCube-v1 LeRobot dataset with two changes vs. convert_to_lerobot.py:

1. Camera resolution 128x128 -> 480x640 (height x width).
2. The green goal target sphere is rendered *into* the base_camera video
   (ManiSkill normally hides ``goal_site`` from sensor observations).

Method: env-state replay. For every recorded frame we set the exact stored
``env_state`` and re-render the base_camera sensor at the new resolution with
``_hidden_objects`` cleared so the goal sphere is visible. No physics is
re-simulated, so the trajectory is identical to the source; only the RGB
observation changes. qpos / qvel / action are copied verbatim from the h5.
"""

import shutil
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401  (registers PickCube-v1)
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

H5 = Path(
    "/home/kelin/dataset/maniskill/pickcube/PickCube-v1/motionplanning/"
    "20260717_135248.rgb.pd_joint_pos.physx_cpu.h5"
)
OUT = Path("/home/kelin/dataset/maniskill/pickcube/lerobot")
REPO_ID = "maniskill/pickcube"
FPS = 20  # PickCube-v1 control_freq
TASK = "Pick up the cube and move it to the goal position."
CAM = "base_camera"
HEIGHT, WIDTH = 480, 640

features = {
    "observation.images.base_camera": {
        "dtype": "video",
        "shape": (HEIGHT, WIDTH, 3),
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


def build_env():
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        num_envs=1,
        sensor_configs=dict(width=WIDTH, height=HEIGHT),
    )
    return env


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)  # safe-destruct: regenerating our own derived dataset from the source h5

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=OUT,
        robot_type="panda",
        use_videos=True,
    )

    env = build_env()
    u = env.unwrapped
    device = u.device

    f = h5py.File(H5, "r")
    traj_keys = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))

    total = 0
    for k in traj_keys:
        t = f[k]
        actions = np.asarray(t["actions"], dtype=np.float32)  # (N, 8)
        qpos = np.asarray(t["obs/agent/qpos"], dtype=np.float32)  # (N+1, 9)
        qvel = np.asarray(t["obs/agent/qvel"], dtype=np.float32)  # (N+1, 9)

        actor_states = {name: np.asarray(t[f"env_states/actors/{name}"]) for name in t["env_states/actors"]}
        art_states = {
            name: np.asarray(t[f"env_states/articulations/{name}"]) for name in t["env_states/articulations"]
        }

        n = actions.shape[0]

        # Reset builds the scene; seed keeps the layout consistent with the source
        # (states below fully override poses regardless).
        env.reset(seed=int(k.split("_")[1]))
        # Un-hide the goal target so it appears in the sensor (base_camera) render.
        u._hidden_objects = []
        if hasattr(u, "goal_site"):
            u.goal_site.show_visual()

        for i in range(n):
            state = {
                "actors": {
                    name: torch.as_tensor(arr[i], dtype=torch.float32, device=device)[None]
                    for name, arr in actor_states.items()
                },
                "articulations": {
                    name: torch.as_tensor(arr[i], dtype=torch.float32, device=device)[None]
                    for name, arr in art_states.items()
                },
            }
            u.set_state_dict(state)
            obs = u.get_obs()
            rgb = obs["sensor_data"][CAM]["rgb"][0].cpu().numpy().astype(np.uint8)  # (480,640,3)

            ds.add_frame(
                {
                    "observation.images.base_camera": rgb,
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
    env.close()
    print(f"\nDONE: {len(traj_keys)} episodes, {total} frames -> {OUT}")


if __name__ == "__main__":
    main()
