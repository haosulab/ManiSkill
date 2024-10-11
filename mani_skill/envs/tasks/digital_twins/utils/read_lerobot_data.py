"""
Functions for reading from lerobot datasets e.g. returning actions and qpos data from a trajectory or the entire dataset
"""

import numpy as np
import torch
import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def get_qpos_and_action(dataset, qpos_label, action_label, episode):
    has_state = qpos_label in dataset.hf_dataset.features
    has_action = action_label in dataset.hf_dataset.features
    assert has_state, (
        "Dataset doesn't contain state observations under label " + qpos_label
    )
    assert has_action, (
        "Dataset doesn't contain action observations under label " + action_label
    )

    from_idx = dataset.episode_data_index["from"][episode]
    to_idx = dataset.episode_data_index["to"][episode]

    data = dataset.hf_dataset.select_columns([qpos_label, action_label])
    obs = data[from_idx:to_idx][qpos_label]
    actions = data[from_idx:to_idx][action_label]

    return torch.stack(obs, dim=0), torch.stack(actions, dim=0)


def get_all_episodes(
    dataset,
    qpos_label,
    action_label,
    device,
    max_ep_len=50,
    control_mode="pd_joint_pos",
):
    """
    returns padded qpos and actions for all episodes in lerobot dataset along with episode termination indices
    qpos and actions in shape (#episode_steps, #dataset_episodes, #robot_dof)
    this function assumes adequate memory space for the actions and observations in device memory
    """
    assert control_mode in [
        "pd_joint_pos",
        "pd_joint_delta_pos",
    ], "control mode must be one of [pd_joint_pos, pd_joint_delta_pos]"
    # while lerobot stores features in single tensors already, padding is necessary for parallel compute
    # conversion to padding forces us to split the tensor into episodes regardless
    episodes = [
        get_qpos_and_action(dataset, qpos_label, action_label, i)
        for i in range(dataset.num_episodes)
    ]
    # set max ep length to minimum of predifined max length and true maximum length
    max_ep_len = min(max_ep_len, np.max([len(ep[0]) for ep in episodes]))
    termination_indices = [len(ep[0]) - 1 for ep in episodes]
    # if pd_joint_pos = last qpos or pd_joint_delta_pos = zero vector, then no movement occurs, function should take qpos
    get_action_pad_val = (
        lambda x: x[-1].clone()
        if control_mode == "pd_joint_pos"
        else lambda x: torch.zeros_like(x[-1])
    )
    padded_qpos = []
    padded_actions = []
    for (qpos, actions) in episodes:
        qpos = qpos[:max_ep_len]
        actions = actions[:max_ep_len]
        if (
            len(qpos) < max_ep_len
        ):  # lerobot stores same amount of qpos obs as actions - meaning last qpos obs missing
            qpos = torch.cat(
                [qpos, qpos[-1].view(1, -1).repeat(max_ep_len - len(qpos), 1)], dim=0
            )
            action_padding = (
                get_action_pad_val(actions)
                .view(1, -1)
                .repeat(max_ep_len - len(actions), 1)
            )
            actions = torch.cat([actions, action_padding], dim=0)
        padded_qpos.append(qpos)
        padded_actions.append(actions)
    padded_qpos = torch.stack(padded_qpos, dim=1).to(device)
    padded_actions = torch.stack(padded_actions, dim=1).to(device)
    termination_indices = torch.tensor(termination_indices).long().to(device)
    return padded_qpos, padded_actions, termination_indices
