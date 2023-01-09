# Import required packages
import gym
import gym.spaces as spaces
from tqdm.notebook import tqdm
from tqdm import tqdm
import numpy as np
import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
import h5py
from mani_skill2.utils.io_utils import load_json
import matplotlib.pyplot as plt
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.data import Dataset, DataLoader
"""
SB3 won't be able to use ManiSkill observations out of the box so we can define a custom observation wrapper to 
make the ManiSkill environment conform with SB3. Here, we are simply going to take the two RGB images, 
two depth images from both cameras (base camera and hand camera) and the state data and create a workable observation for SB3.
"""
class ManiSkillRGBDWrapper(gym.ObservationWrapper):
    def __init__(self, env, image_size=(128, 128)) -> None:
        super().__init__(env)
        obs_space = env.observation_space
        self.image_size = image_size
        
        # We want the following states to be kept in the observations. 
        # obs_space is the original environment's observation space
        state_spaces = [
            obs_space["agent"]['base_pose'], # pose of the robot
            obs_space["agent"]['qpos'], # robot configuration position
            obs_space["agent"]['qvel'], # robot configuration velocity
        ]
        for k in obs_space["extra"]:
            # includes gripper pose and goal information depending on environment
            state_spaces.append(obs_space["extra"][k])
        # Define the new state space
        state_size = sum([space.shape[0] for space in state_spaces])
        state_space = spaces.Box(-float("inf"), float("inf"), shape=(state_size, ))
        
        # Get the image dimensions. Note that there is a base_camera and a hand_camera, both of which output the same shape
        h, w, _ = obs_space["image"]["base_camera"]["rgb"].shape
        if self.image_size is not None:
            h, w = self.image_size
        new_shape = (h, w, 8) # the shape is HxWx8, where 8 comes from combining two RGB images and two depth images
        low = np.full(new_shape, -float("inf"))
        high = np.full(new_shape, float("inf"))
        rgbd_space = spaces.Box(low, high, dtype=obs_space["image"]["base_camera"]["rgb"].dtype)
        
        # create the observation space
        self.observation_space = spaces.Dict({
            "rgbd": rgbd_space,
            "state": state_space
        })

    @staticmethod # make this static so both RL and IL tutorials can use this
    def convert_observation(observation, image_size):
        # This function replaces the original observations. We scale down images by 255 and 
        # flatten the states in the original observations
        image_obs = observation["image"]
        rgb = image_obs["base_camera"]["rgb"] / 255.0
        depth = image_obs["base_camera"]["depth"]
        rgb2 = image_obs["hand_camera"]["rgb"] / 255.0
        depth2 = image_obs["hand_camera"]["depth"]
        if image_size is not None and image_size != (rgb.shape[0], rgb.shape[1]):
            rgb = cv2.resize(rgb, image_size, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, image_size, interpolation=cv2.INTER_LINEAR)[:,:,None]
            rgb2 = cv2.resize(rgb2, image_size, interpolation=cv2.INTER_LINEAR)
            depth2 = cv2.resize(depth2, image_size, interpolation=cv2.INTER_LINEAR)[:,:,None]
        from mani_skill2.utils.common import flatten_state_dict
        state = np.hstack([
            flatten_state_dict(observation["agent"]),
            flatten_state_dict(observation["extra"])
        ])
        
        # combine the RGB and depth images
        rgbd = np.concatenate([rgb, depth, rgb2, depth2], axis=2)
        obs = dict(rgbd=rgbd, state=state)
        return obs
    def observation(self, observation):
        return ManiSkillRGBDWrapper.convert_observation(observation, image_size=self.image_size)

    def step(self, action):
        o, r, d, info = super().step(action)
        d = False
        if info["elapsed_steps"] >= 200:
            # trunate episodes after 100 steps rather than the default 200 steps for faster training
            # set TimeLimit.truncated to True to tell SB3 it's a truncation and not task success
            info["TimeLimit.truncated"] = True
            d = True
        return o,r,d,info


class IMPALA(nn.Module):
    def __init__(self, in_channel, num_pixels, out_feature_size=384, out_channel=None):
        super(IMPALA, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        in_channel = in_channel
        fcs = [64, 64, 64]

        self.stem = nn.Conv2d(in_channel, fcs[0], kernel_size=4, stride=4)
        in_channel = fcs[0]

        for num_ch in fcs:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            in_channel = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.img_feat_size = num_pixels // (4**len(fcs) * 16) * fcs[-1]

        self.fc = nn.Linear(self.img_feat_size, out_feature_size)
        self.final = nn.Linear(out_feature_size, self.out_channel) if out_channel else None

    def forward(self, rgbd, **kwargs):
        x = self.stem(rgbd)
        # x = feature
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.reshape(x.shape[0], self.img_feat_size)
        x = F.relu(self.fc(x))

        if self.final:
            x = self.final(x)

        return x


class Policy(nn.Module):
    def __init__(self, observation_space, action_space, hidden_units=[256, 128], activation=nn.ReLU):
        super().__init__()
        self.feature_extractor = IMPALA(in_channel=8, num_pixels=(128*128), out_feature_size=384)
        mlp_layers = []
        prev_units = 384 + observation_space['state'].shape[0]
        for h in hidden_units[:-1]:
            mlp_layers += [nn.Linear(prev_units, h), activation()]
            prev_units = h
        mlp_layers += [nn.Linear(prev_units, action_space.shape[0]), nn.Tanh()]
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, observations) -> th.Tensor:
        features = self.feature_extractor(observations['rgbd'].permute((0, 3, 1, 2)))
        features = th.cat([features, observations['state']], dim=1)
        return self.mlp(features)

"""
Define a Pytorch dataset that we can create a Dataloader from and iterate over for training
"""
class ManiSkillRGBDDataset(Dataset):
    def __init__(self, dataset_file: str, image_size=(128, 128), load_count: int = -1) -> None:
        self.dataset_file = dataset_file
        self.image_size = image_size
        import h5py
        from mani_skill2.utils.io_utils import load_json

        # Load the trajectory data from the .h5 file
        self.data = h5py.File(dataset_file, "r")

        # Load associated json
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)

        self.episodes = self.json_data["episodes"] # each episode/demonstration meta data
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        
        def load_h5_data(data):
            out = dict()
            for k in data.keys():
                if isinstance(data[k], h5py.Dataset):
                    out[k] = data[k][:]
                else:
                    out[k] = load_h5_data(data[k])
            return out
        self.trajectories = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            ep_len = eps["elapsed_steps"]
            self.total_frames += ep_len
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            
            trajectory = load_h5_data(trajectory)
            self.trajectories.append(trajectory)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # sample frame by uniformally sampling trajectory, then sampling a frame
        trajectory_id = idx % len(self.trajectories)
        trajectory = self.trajectories[trajectory_id]
        frame_id = idx % len(trajectory["actions"])
        action = th.from_numpy(trajectory["actions"][frame_id])
        def select_index_dict(data, index):
            out = dict()
            for k in data:
                if isinstance(data[k], dict):
                    out[k] = select_index_dict(data[k], index)
                else:
                    out[k] = data[k][index]
            return out
        obs = select_index_dict(trajectory["obs"], frame_id)
        # we can reuse the observation wrapper code here to convert the demonstration dataset observations into ones for the model
        obs = ManiSkillRGBDWrapper.convert_observation(obs, image_size=self.image_size)
        # for space conservation depth data is stored in uint16 and scaled by 2**10, we transform it back to the right scale here
        obs['rgbd'][:,:,3] = obs['rgbd'][:,:,3] / (2**10)
        obs['rgbd'][:,:,7] = obs['rgbd'][:,:,7] / (2**10)
        for k in obs:
            obs[k] = th.from_numpy(obs[k]).float()
        return obs, action

if __name__ == "__main__":
    env_id = "PickCube-v0"
    obs_mode = "rgbd"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "dense"
    
    num_cpu = 8
    batch_size = 256
   

    def make_env(env_id: str, rank: int, seed: int = 0, record_dir: str = None):
        def _init() -> gym.Env:
            import mani_skill2.envs
            env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode)
            env = ManiSkillRGBDWrapper(env)
            if record_dir is not None:
                env = RecordEpisode(env, record_dir, info_on_video=True, render_mode="cameras")
            env.seed(seed+rank)
            return env
        set_random_seed(seed)
        return _init

    # create one eval environment
    eval_env = make_env(env_id, 0)()
    eval_env.reset()

    # create our policy
    policy = Policy(eval_env.observation_space, eval_env.action_space, hidden_units=[256, 128], activation=nn.ReLU)
    device = "cuda" if th.cuda.is_available() else "cpu"
    policy = policy.to(device)

    # simple training loop. We leave out adding validation sets, regularization techniques, metric logging etc. to the user
    loss_fn = nn.MSELoss()
    optim = th.optim.Adam(policy.parameters(), lr=3e-4)
    iterations = 100000
    epochs = 10000
    

    evaluate=True

    if not evaluate:
        dataset = ManiSkillRGBDDataset(dataset_file="./demos/rigid_body/PickCube-v0/trajectory.rgbd.pd_ee_delta_pose.h5")
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True, shuffle=True)
        print(len(dataset))
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter("logs/bc_pickcube_xuanlin_impala")
        steps = 0
        best_epoch_loss = 9999
        pbar = tqdm(dataloader, total=iterations)
        for epoch in range(epochs):
            epoch_loss = 0
            
            for step, batch in enumerate(dataloader):
                steps += 1
                optim.zero_grad()
                observations, actions = batch
                for k in observations:
                    observations[k] = observations[k].to(device)
                actions = actions.to(device)
                pred_actions = policy(observations)
                
                loss = loss_fn(actions, pred_actions)
                loss.backward()
                loss_val = loss.item()
                optim.step()
                writer.add_scalar("train/mse_loss", loss_val, steps)
                epoch_loss += loss_val
                pbar.set_postfix(dict(loss=loss_val))
                pbar.update(1)
                if steps % 25000 == 0:
                    save_data = dict(
                        optim=optim.state_dict(),
                        policy=policy.state_dict(),
                        step=steps,
                        best_epoch_loss=best_epoch_loss,
                    )
                    print(f"save checkpoint {steps}")
                    th.save(save_data, f"ckpt_{steps}.pth")
                if steps >= iterations:
                    break
            
            epoch_loss = epoch_loss / len(dataloader)
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                save_data = dict(
                    optim=optim.state_dict(),
                    policy=policy.state_dict(),
                    step=steps,
                    best_epoch_loss=best_epoch_loss,
                )
                print(f"New best loss {best_epoch_loss}, saving to best.pth")
                th.save(save_data, "best.pth")
            
            writer.add_scalar("train/mse_loss_epoch", epoch_loss, epoch)
            if steps >= iterations:
                break
        save_data = dict(
            optim=optim.state_dict(),
            policy=policy.state_dict(),
            step=steps,
            best_epoch_loss=best_epoch_loss
        )
        th.save(save_data, "end.pth")
    else:
        # pass
        save_data=th.load("best.pth")
        policy.load_state_dict(save_data["policy"])
        print(save_data['step'], save_data['best_epoch_loss'])
    traj_id = 0
    obs=eval_env.reset(seed=traj_id)
    policy.eval()
    for i in range(10000):
        # gt_actions = dataset.data[f"traj_{traj_id}"]["actions"]
        for k in obs:
            obs[k] = th.from_numpy(obs[k]).float()[None].to(device)
        with th.no_grad():
            acts = policy(obs)
            acts = acts.cpu().numpy()[0]
        
        # if i < 39:
        #     gta=gt_actions[i]
        #     print(((gta-acts)**2).mean())
        #     acts=gta
        obs, r, d, info = eval_env.step(acts)

        eval_env.render()
        info = info
        d=d
        if info["success"]:
            print("success", traj_id)
            traj_id += 1
            obs=eval_env.reset(seed=traj_id)
            # obs = eval_env.reset(seed=dataset.episodes[traj_id]["episode_seed"])
        elif d:
            print("fail", traj_id)
            traj_id += 1
            obs = eval_env.reset(seed=traj_id)