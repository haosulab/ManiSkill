from functools import partial

import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from mani_skill2.utils.common import flatten_dict_keys
from mani_skill2.vector import make as make_vec_env


def make_env(env_id, obs_mode):
    import mani_skill2.envs  # fmt: skip
    from mani_skill2.utils.wrappers.observation import FlattenObservationWrapper

    env = gym.make(env_id, obs_mode=obs_mode)
    return FlattenObservationWrapper(env)


# NOTE(jigu): SB3 and pytest do not work for C++ extensions
def test_obs_mode(obs_mode="image"):
    env_id = "PickCube-v0"
    n_envs = 2

    env_fns = [partial(make_env, env_id, obs_mode=obs_mode) for _ in range(n_envs)]
    sb3_env = SubprocVecEnv(env_fns)
    ms2_env = make_vec_env(env_id, n_envs, obs_mode=obs_mode)

    np.random.seed(2022)
    sb3_env.seed(2022)
    ms2_env.seed(2022)

    def check_fn(sb3_obs, ms2_obs):
        ms2_obs = flatten_dict_keys(ms2_obs)
        for k, v in sb3_obs.items():
            v2 = ms2_obs[k]
            if isinstance(v2, torch.Tensor):
                v2 = v2.cpu().numpy()
            if v.dtype == np.uint8:
                # https://github.com/numpy/numpy/issues/19183
                np.testing.assert_allclose(
                    np.float32(v), np.float32(v2), err_msg=k, atol=1
                )
            elif np.issubdtype(v.dtype, np.integer):
                np.testing.assert_equal(v, v2, err_msg=k)
            else:
                np.testing.assert_allclose(v, v2, err_msg=k, atol=1e-4)

    for i in range(2):
        print("Episode", i)
        sb3_obs = sb3_env.reset()
        ms2_obs = ms2_env.reset()
        check_fn(sb3_obs, ms2_obs)

        for t in range(5):
            actions = [ms2_env.action_space.sample() for _ in range(n_envs)]

            sb3_obs, sb3_rews, sb3_dones, sb3_infos = sb3_env.step(actions)
            ms2_obs, ms2_rews, ms2_dones, ms2_infos = ms2_env.step(actions)

            check_fn(sb3_obs, ms2_obs)
            np.testing.assert_allclose(sb3_rews, ms2_rews)

    sb3_env.close()
    ms2_env.close()


if __name__ == "__main__":
    for obs_mode in [
        "image",
        "rgbd",
        "pointcloud",
        "rgbd_robot_seg",
        "pointcloud_robot_seg",
    ]:
        print("Testing", obs_mode)
        test_obs_mode(obs_mode)
