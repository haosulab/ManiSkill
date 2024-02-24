from mani_skill2.envs.scenes.tasks import SequentialTaskEnv, PickSequentialTaskEnv
from mani_skill2.envs.scenes.tasks.planner import PickSubtask, plan_data_from_file
from mani_skill2.utils.scene_builder.ai2thor import (
    ProcTHORSceneBuilder, ArchitecTHORSceneBuilder,
    iTHORSceneBuilder, RoboTHORSceneBuilder,
)
from mani_skill2.envs.scenes.base_env import SceneManipulationEnv
from mani_skill2.utils.wrappers import RecordEpisode


import gymnasium as gym
import numpy as np
import torch

import os

render_mode = (
    "rgb_array" if (
        "SAPIEN_NO_DISPLAY" in os.environ
        and int(os.environ["SAPIEN_NO_DISPLAY"]) == 1
    ) else "human"
)
render_mode = "rgb_array"
print("RENDER_MODE", render_mode)

SCENE_IDX_TO_APPLE_PLAN = {
    0: [PickSubtask(obj_id="objects/Apple_5_111")],
    1: [PickSubtask(obj_id="objects/Apple_16_40")],
    2: [PickSubtask(obj_id="objects/Apple_12_64")],
    3: [PickSubtask(obj_id="objects/Apple_29_113")],
    4: [PickSubtask(obj_id="objects/Apple_28_35")],
    5: [PickSubtask(obj_id="objects/Apple_17_88")],
    6: [PickSubtask(obj_id="objects/Apple_1_35")],
    7: [PickSubtask(obj_id="objects/Apple_25_48")],
    8: [PickSubtask(obj_id="objects/Apple_9_46")],
    9: [PickSubtask(obj_id="objects/Apple_13_72")],
}

SCENE_IDX = 6
env: SequentialTaskEnv = gym.make(
    'SequentialTask-v0',
    obs_mode='state',
    render_mode=render_mode,
    control_mode='pd_joint_delta_pos',
    reward_mode='dense',
    robot_uids='fetch',
    scene_builder_cls=ArchitecTHORSceneBuilder,
    task_plans=[SCENE_IDX_TO_APPLE_PLAN[SCENE_IDX]],
    scene_idxs=SCENE_IDX,
    num_envs=2,
)

env = RecordEpisode(env, '.', save_trajectory=False)
env.reset(seed=0)

# print force on robot
robot_link_names = [link.name for link in env.agent.robot.links]
print(robot_link_names)
print(torch.norm(env.agent.robot.get_net_contact_forces(robot_link_names), dim=-1))

for step_num in range(30):
    action = np.zeros(env.action_space.shape)

    # torso up
    action[..., -4] = 1
    # head still
    action[..., -5] = 0
    # gripper open
    action[..., -7] = 1

    # move forward and left
    action[..., -3:-1] = 1

    obs, rew, term, trunc, info = env.step(action=action)
env.close()
