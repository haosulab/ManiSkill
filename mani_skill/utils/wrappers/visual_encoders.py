from typing import Dict, Literal

import gymnasium as gym
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


class VisualEncoderWrapper(gym.ObservationWrapper):
    def __init__(self, env, encoder: Literal["r3m"], encoder_config=dict()):
        self.base_env: BaseEnv = env.unwrapped
        assert encoder == "r3m", "Only encoder='r3m' is supported at the moment"
        if encoder == "r3m":
            assert self.base_env.obs_mode in [
                "rgbd",
                "rgb",
            ], "r3m encoder requires obs_mode to be set to rgbd or rgb"
            import torchvision.transforms as T
            from r3m import load_r3m

            # self.num_images = len(venv.observation_space['image'])
            self.model = load_r3m("resnet18")  # resnet18, resnet34
            self.model.eval()
            self.model.to(self.base_env.device)
            self.transforms = T.Compose(
                [
                    T.Resize((224, 224), antialias=True),
                ]
            )  # HWC -> CHW
            self.single_image_embedding_size = 512  # for resnet18

        self.base_env.update_obs_space(
            common.to_numpy(
                self.observation(common.to_tensor(self.base_env._init_raw_obs))
            )
        )
        super().__init__(env)

    @torch.no_grad()
    def observation(self, obs: Dict):
        vec_img_embeddings_list = []
        image_obs = obs.pop("sensor_data")
        del obs["sensor_param"]
        for image in image_obs.values():
            vec_image = image["rgb"]  # (N, H, W, 3), [0, 255] torch.int16
            vec_image = self.transforms(
                vec_image.permute(0, 3, 1, 2)
            )  # (numenv, 3, 224, 224)
            vec_image = vec_image.to(self.base_env.device)
            vec_img_embedding = self.model(
                vec_image
            ).detach()  # (numenv, self.single_image_embedding_size)
            vec_img_embeddings_list.append(vec_img_embedding)
        vec_embedding = torch.cat(
            vec_img_embeddings_list, dim=-1
        )  # (numenv, self.image_embedding_size)
        obs["embedding"] = vec_embedding
        return obs
