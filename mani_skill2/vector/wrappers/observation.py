from copy import deepcopy

import torch

from ..vec_env import VecEnv, VecEnvObservationWrapper


def batch_isin(x: torch.Tensor, inds: torch.Tensor):
    """A batch version of `torch.isin`.

    Args:
        x (torch.Tensor): [B, ...], integer
        inds (torch.Tensor): [B, N], integer

    Returns:
        torch.Tensor: [B, ...], boolean
    """

    # # For-loop version
    # out = []
    # for x_i, inds_i in zip(x.unbind(0), inds.unbind(0)):
    #     out.append(torch.isin(x_i, inds_i))
    # out = torch.stack(out, dim=0)

    bs = x.size(0)
    max_ind, _ = torch.max(x.reshape(bs, -1), dim=-1)  # [B]
    # Acquire the maximum index of each sample in batch
    offset = torch.cumsum(max_ind + 1, dim=0)
    offset = torch.nn.functional.pad(offset[:-1], (1, 0), value=0)
    # Add offset to avoid indexing collision
    _shape = (bs,) + (1,) * (x.dim() - 1)
    # Remap indices
    _x = x + offset.view(_shape)
    _inds = inds + offset.view(bs, 1)
    return torch.isin(_x, _inds)


class VecRobotSegmentationObservationWrapper(VecEnvObservationWrapper):
    """Add a binary mask for robot links."""

    def __init__(self, venv: VecEnv, replace=True):
        super().__init__(venv)

        from mani_skill2.utils.wrappers.observation import (
            RobotSegmentationObservationWrapper,
        )

        self.observation_space = deepcopy(venv.observation_space)
        RobotSegmentationObservationWrapper.init_observation_space(
            self.observation_space, replace=replace
        )
        self.replace = replace

        # Cache robot link ids
        # NOTE(jigu): Assume robots are the same and thus can be batched
        robot_link_ids = self.get_attr("robot_link_ids")
        self.robot_link_ids = torch.tensor(
            robot_link_ids, dtype=torch.int32, device=self.device
        )

    @torch.no_grad()
    def update_robot_link_ids(self, indices=None):
        robot_link_ids = self.get_attr("robot_link_ids", indices=indices)
        robot_link_ids = torch.tensor(robot_link_ids, dtype=torch.int32)
        robot_link_ids = robot_link_ids.to(device=self.device, non_blocking=True)
        indices = self._get_indices(indices)
        self.robot_link_ids[indices] = robot_link_ids

    def observation_image(self, observation: dict):
        image_obs = observation["image"]
        for cam_images in image_obs.values():
            if "Segmentation" not in cam_images:
                continue
            seg = cam_images["Segmentation"]  # [B, H, W, 4]
            # [B, H, W, 1]
            robot_seg = batch_isin(seg[..., 1:2], self.robot_link_ids)
            if self.replace:
                cam_images.pop("Segmentation")
            cam_images["robot_seg"] = robot_seg
        return observation

    def observation_pointcloud(self, observation: dict):
        pointcloud_obs = observation["pointcloud"]
        if "Segmentation" not in pointcloud_obs:
            return observation
        seg = pointcloud_obs["Segmentation"]  # [N, 4]
        robot_seg = batch_isin(seg[..., 1:2], self.robot_link_ids)  # [N, 1]
        if self.replace:
            pointcloud_obs.pop("Segmentation")
        pointcloud_obs["robot_seg"] = robot_seg
        return observation

    @torch.no_grad()
    def observation(self, observation: dict):
        if "image" in observation:
            observation = self.observation_image(observation)
        if "pointcloud" in observation:
            observation = self.observation_pointcloud(observation)
        return observation

    def reset_wait(self, indices=None, **kwargs):
        obs = super().reset_wait(indices=indices, **kwargs)
        self.update_robot_link_ids(indices=indices)
        return self.observation(obs)
