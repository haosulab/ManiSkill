from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from mani_skill2.utils.common import flatten_state_dict, flatten_dict_space_keys
from gymnasium import spaces
import torch
import numpy as np

class VisualEncoder(VecEnvObservationWrapper):
    def __init__(self, venv, encoder):
        assert encoder == 'r3m', "Only encoder='r3m' is supported"
        from r3m import load_r3m
        import torchvision.transforms as T
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_images = len(venv.observation_space['image'])
        self.model = load_r3m("resnet18") # resnet18, resnet34
        self.model.eval()
        self.model.to(self.device)
        self.transforms = T.Compose([T.Resize((224, 224)), ]) # HWC -> CHW
        self.single_image_embedding_size = 512 # for resnet18
        self.image_embedding_size = self.single_image_embedding_size * self.num_images

        self.state_size = 0
        for k in ['agent', 'extra']:
            self.state_size += sum([v.shape[0] for v in flatten_dict_space_keys(venv.single_observation_space[k]).spaces.values()])

        new_single_space_dict = spaces.Dict({
            'state': spaces.Box(-float("inf"), float("inf"), shape=(self.state_size,), dtype=np.float32),
            'embedding': spaces.Box(-float("inf"), float("inf"), shape=(self.image_embedding_size,), dtype=np.float32),
        })
        self.embedding_size = self.image_embedding_size + self.state_size
        super().__init__(venv, new_single_space_dict)

    @torch.no_grad()
    def observation(self, obs):
        # assume a structure of obs['image']['base_camera']['rgb']
        # simplified
        vec_img_embeddings_list = []
        for camera in ['base_camera', 'hand_camera']:
            vec_image = torch.Tensor(obs['image'][camera]['rgb']) # (numenv, H, W, 3), [0, 255] uint8
            vec_image = self.transforms(vec_image.permute(0, 3, 1, 2)) # (numenv, 3, 224, 224)
            vec_image = vec_image.to(self.device)
            vec_img_embedding = self.model(vec_image).detach() # (numenv, self.single_image_embedding_size)
            vec_img_embeddings_list.append(vec_img_embedding)

        vec_embedding = torch.cat(vec_img_embeddings_list, dim=-1)  # (numenv, self.image_embedding_size)
        ret_dict = {}
        state = np.hstack([
            flatten_state_dict(obs["agent"]),
            flatten_state_dict(obs["extra"]),
        ])
        ret_dict['state'] = torch.Tensor(state).to(self.device)  # (numenv, self.state_size)
        ret_dict['embedding'] = vec_embedding
        return ret_dict # device is cuda