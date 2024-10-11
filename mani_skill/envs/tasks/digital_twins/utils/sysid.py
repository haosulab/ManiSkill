"""
Use Simulated annealing to optimize the stiffness and damping of the robot arm and minimize 6d pose errors when open-loop unrolling demonstration trajectories.
Adapted from simpler for MS3, currently only supports gpu sim
"""
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import sapien.physx as physx
import torch
import tyro

# Lerobot datasets related imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from read_lerobot_data import get_all_episodes
from simulated_annealing import sa
from tqdm import tqdm

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.agents.robots.koch.koch import Koch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig


@dataclass
class Args:
    robot_uids: str = "koch-v1.1"
    dataset_type: str = "Lerobot"
    """The dataset object type, current options: [Lerobot]"""
    repo_id: str = ""
    """repo_id for Lerobot dataset"""
    root: Optional[str] = None
    """root directory if dataset is stored locally"""
    dataset_qpos_label: str = "observation.state"
    """qpos observation label in given Lerobot dataset"""
    dataset_action_label: str = "action"
    """action observation label in given Lerobot dataset"""
    max_episode_steps: int = 100
    """maximum steps to replay per episode in given dataset"""
    control_mode: str = "pd_joint_pos"
    """the control mode of the robot"""
    sim_freq: int = 120
    """simulation steps per second"""
    control_freq: int = 30
    """control steps per second"""
    angle_units: str = "degree"
    """angle units in [degree, radians]"""
    evaluate_episode: int = -1
    """Whether to evaluate a single episode"""


@register_env("sysid_env", max_episode_steps=np.inf)
class empty_env(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(
        self,
        *args,
        robot_uids="koch-v1.1",
        robot_init_qpos_noise=0.0,
        sim_freq=120,
        control_freq=30,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.s_freq = sim_freq
        self.c_freq = control_freq
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=self.s_freq,
            control_freq=self.c_freq,
            scene_config=SceneConfig(contact_offset=0.01),
        )


def calc_err_entire_dataset(
    env,
    kinematic_env,
    episode_qpos,
    episode_actions,
    termination,
    stiffness=1e3,
    damping=1e2,
    objective="tcp_pose",
):
    assert objective in ["tcp_pose", "qpos_mse"]

    env.agent.controllers["pd_joint_pos"].config.stiffness = stiffness
    env.agent.controllers["pd_joint_pos"].config.damping = damping
    env.agent.controllers["pd_joint_pos"].set_drive_property()

    kinematic_env.reset()
    env.reset()
    env.agent.reset(torch.deg2rad(episode_qpos[0]))
    # have to apply changes to gpu
    if physx.is_gpu_enabled():
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()

    qpos_diff = []
    qpos_mse = []
    trans_errors = []
    rot_errors = []
    tcp_pose_errors = []
    for i in range(len(episode_actions) - 1):
        # less envs, more variance and chance to overfit
        valid_envs = i <= termination
        if valid_envs.sum() / episode_actions[i].shape[0] < 0.5:
            break

        env.step(torch.deg2rad(episode_actions[i]))

        # qpos_error = torch.abs((torch.deg2rad(episode_qpos[i+1][25]) - env.agent.robot.qpos[25]))
        qpos_error = torch.abs(
            (torch.deg2rad(episode_qpos[i + 1]) - env.agent.robot.qpos)
        )[valid_envs]
        qpos_diff.append(qpos_error.mean().item())
        qpos_mse.append(qpos_error.pow(2).mean().item())

        if objective == "tcp_pose":
            kinematic_env.agent.reset(torch.deg2rad(episode_qpos[i + 1]))
            if physx.is_gpu_enabled():
                env.scene._gpu_apply_all()
                env.scene.px.gpu_update_articulation_kinematics()
                env.scene._gpu_fetch_all()

            # view tcp pose in robot base frame
            sim_base_tcp_pose = env.agent.robot.pose.inv() * env.agent.tcp.pose
            kinematics_base_tcp_pose = (
                kinematic_env.agent.robot.pose.inv() * kinematic_env.agent.tcp.pose
            )

            # pos error is simple mean euclidean distance
            pos_error = torch.linalg.norm(
                sim_base_tcp_pose.p - kinematics_base_tcp_pose.p, dim=-1
            )
            pos_error = pos_error[valid_envs].mean().item()
            trans_errors.append(pos_error)

            # rotation error is mean arcsin(1/(2sqrt(2)) * frobenius norm of difference in matrices)
            # intuition is largest difference in 3D rotation is 180 - corresponding to frobenius norm of 2sqrt(2)
            R_sim_tcp = quaternion_to_matrix(sim_base_tcp_pose.q)  # shape (b, 9)
            R_kin_tcp = quaternion_to_matrix(kinematics_base_tcp_pose.q)  # shape (b, 9)
            frobenius_norm = (
                (R_sim_tcp - R_kin_tcp).pow(2).sum(dim=[-1, -2]).sqrt()
            )  # shape (b)
            rot_error = ((1 / (2 * np.sqrt(2))) * frobenius_norm).arcsin()
            rot_error = rot_error[valid_envs].mean().item()
            rot_errors.append(rot_error)

            tcp_pose_errors.append(pos_error + rot_error)

    print("mean_qpos_error_degrees", np.mean(np.rad2deg(qpos_diff)))
    if objective == "tcp_pose":
        print("mean tcp trans error", np.mean(trans_errors))
        print("mean tcp rot error", np.mean(rot_errors))
        return np.mean(tcp_pose_errors)
    return np.mean(qpos_mse)


def replay(env, episode_qpos, episode_actions, stiffness=1e3, damping=1e2):
    env.reset(seed=0)
    env.agent.reset(torch.deg2rad(episode_qpos[0]))

    env.agent.controllers["pd_joint_pos"].config.stiffness = stiffness
    env.agent.controllers["pd_joint_pos"].config.damping = damping
    env.agent.controllers["pd_joint_pos"].set_drive_property()

    diff = []
    print("ep_len", len(episode_actions))
    for i in range(len(episode_actions) - 1):
        action = episode_actions[i]
        env.step(torch.deg2rad(action))

        diff.append(
            np.abs(
                (np.deg2rad(episode_qpos[i + 1]) - env.agent.robot.qpos[0].numpy())
            ).mean()
        )

    print("mean_deg_diff", np.mean(np.rad2deg(diff)))
    import matplotlib.pyplot as plt

    plt.plot(np.rad2deg(diff))
    plt.ylabel("mean abs deg diff")
    plt.xlabel("step number")
    plt.show()


if __name__ == "__main__":
    args = tyro.cli(Args)

    assert args.repo_id != "", "Enter a repo_id"
    dataset = LeRobotDataset(args.repo_id, root=args.root)

    episode_qpos, episode_actions, termination = get_all_episodes(
        dataset,
        args.dataset_qpos_label,
        args.dataset_action_label,
        torch.device("cuda"),
        max_ep_len=args.max_episode_steps,
    )

    env = gym.make(
        "sysid_env",
        num_envs=episode_qpos.shape[1],
        robot_uids="koch-v1.1",
        control_mode="pd_joint_pos",
        sim_freq=120,
        control_freq=30,
    )
    kinematic_env = gym.make(
        "sysid_env",
        num_envs=episode_qpos.shape[1],
        robot_uids="koch-v1.1",
        control_mode="pd_joint_pos",
        sim_freq=120,
        control_freq=30,
    )

    # replay(env, episode_qpos, episode_actions)
    # thing = calc_err_entire_dataset(env, kinematic_env, episode_qpos, episode_actions, termination, stiffness=1e3, damping=1e2, objective="qpos_mse")
    # print(thing)

    # thing = calc_err_entire_dataset(env, kinematic_env, episode_qpos, episode_actions, termination, stiffness=1e3, damping=1e2, objective="tcp_pose")
    # thing = calc_err_entire_dataset(env, kinematic_env, episode_qpos, episode_actions, termination, stiffness=1e3, damping=1e2, objective="tcp_pose")
    # thing = calc_err_entire_dataset(env, kinematic_env, episode_qpos, episode_actions, termination, stiffness=1e3, damping=1e2, objective="tcp_pose")
    # thing = calc_err_entire_dataset(env, kinematic_env, episode_qpos, episode_actions, termination, stiffness=1e3, damping=1e2, objective="tcp_pose")
    # thing = calc_err_entire_dataset(env, kinematic_env, episode_qpos, episode_actions, termination, stiffness=1e3, damping=1e2, objective="tcp_pose")
    # thing = calc_err_entire_dataset(env, kinematic_env, episode_qpos, episode_actions, termination, stiffness=1e3, damping=1e2, objective="tcp_pose")
    # thing = calc_err_entire_dataset(env, kinematic_env, episode_qpos, episode_actions, termination, stiffness=1e3, damping=1e2, objective="tcp_pose")

    init_stiffness = np.array([1e3] * 6)
    init_damping = np.array([1e2] * 6)
    stiffness_low = np.array([1e3 * 0.5] * 6)
    stiffness_high = np.array([1e3 * 1.5] * 6)
    damping_low = np.array([1e2 * 0.5] * 6)
    damping_high = np.array([1e2 * 1.5] * 6)

    raw_action_to_stiffness = (
        lambda x: stiffness_low
        + (stiffness_high - stiffness_low) * x[: len(stiffness_high)]
    )
    raw_action_to_damping = (
        lambda x: damping_low
        + (damping_high - damping_low)
        * x[len(stiffness_high) : 2 * len(stiffness_high)]
    )

    init_action = np.concatenate(
        [
            (init_stiffness - stiffness_low) / (stiffness_high - stiffness_low),
            (init_damping - damping_low) / (damping_high - damping_low),
        ]
    )

    opt_fxn = lambda x: calc_err_entire_dataset(
        env,
        kinematic_env,
        episode_qpos,
        episode_actions,
        termination,
        raw_action_to_stiffness(x),
        raw_action_to_damping(x),
        objective="qpos_mse",
    )
    opt = sa.minimize(
        opt_fxn,
        init_action,
        opt_mode="continuous",
        step_max=200,
        t_max=1.5,
        t_min=0,
        bounds=[[0, 1]] * (len(init_stiffness) * 2),
    )
    opt.results()
    recovered_stiffness = raw_action_to_stiffness(opt.best_state)
    recovered_damping = raw_action_to_damping(opt.best_state)
    print("end_stiffness", recovered_stiffness)
    print("end_damping", recovered_damping)
