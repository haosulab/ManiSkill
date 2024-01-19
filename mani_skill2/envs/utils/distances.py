import numpy as np
import torch

from mani_skill2.utils.structs.pose import Pose


def position_distance(pose1: Pose, pose2: Pose):
    if torch is not None and isinstance(pose1.raw_pose, torch.Tensor):
        return torch.linalg.norm(pose1.p - pose2.p)
    else:
        return np.linalg.norm(pose1.p - pose2.p)
