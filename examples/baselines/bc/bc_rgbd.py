import datetime
import os
from dataclasses import asdict, dataclass

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from tqdm import tqdm


@dataclass
class Config:
    # wandb project name
    project: str = "ManiSkill"
    # wandb group name
    group: str = "BehaviorCloning"
    # training dataset and evaluation environment
    env: str = "PickCube-v1"
    # demo path
    demo_path: str = ""
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # training batch size
    batch_size: int = 128
    # evaluation frequency, will evaluate eval_freq training steps
    eval_freq: int = int(500)
    # record videos
    video: bool = False
    # video save path
    video_path = "./videos"
    # how often a video should be recorded
    video_freq: int = 1000
    # training random seed
    seed: int = 42
    # training device
    device: str = "cuda"
    # learning rate
    lr = 3e-4
    # number of parallel eval envs
    num_envs = 50
    # dataset base directory
    dir = "/root/"
    # log to wandb
    wandb: bool = True
    # number of trajectories to load, max is usually 1000
    load_count: int = 500
    # seed
    seed: int = 2024
    # where experiment outputs should be stored
    output_dir = "./output/"
    # save frequency
    save_freq: int = 10000
    # simulation backend
    sim_backend: str = "cpu"
    # max number of evaluation steps per eval episode
    max_episode_steps: int = 100


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=log_name.replace(os.path.sep, "__"),
        tags=["bc"],
    )
    wandb.run.save()


def flatten_state_dict_with_space(state_dict: dict) -> np.ndarray:
    states = []
    for key in state_dict.keys():
        value = state_dict[key]
        if isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
        elif isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)
    if len(states) == 0:
        return np.empty(0)
    else:
        if isinstance(states[0], torch.Tensor):
            try:
                return torch.hstack(states)
            except:
                return torch.column_stack(states)
        else:
            try:
                return np.hstack(states)
            except:  # dirty fix for concat trajectory of states
                return np.column_stack(states)


class ManiSkillDataset(Dataset):
    def __init__(self, dataset_file: str, device: torch.device, load_count) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.rgb = []
        self.depth = []
        self.actions = []
        self.dones = []
        self.states = []
        self.total_frames = 0
        self.device = device

        if load_count > len(self.episodes):
            print(
                f"Load count exceeds number of available episodes, loading {len(self.episodes)} which is the max number of episodes present"
            )
            load_count = len(self.episodes)

        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            agent = trajectory["obs"]["agent"]
            extra = trajectory["obs"]["extra"]

            state = np.hstack(
                [
                    flatten_state_dict_with_space(agent),
                    flatten_state_dict_with_space(extra),
                ]
            )
            self.states.append(state)

            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            self.rgb.append(trajectory["obs"]["sensor_data"]["base_camera"]["rgb"][:-1])
            self.depth.append(
                trajectory["obs"]["sensor_data"]["base_camera"]["depth"][:-1]
            )
            self.actions.append(trajectory["actions"])

            # print(trajectory.keys())
        self.rgb = np.vstack(self.rgb) / 255.0
        self.depth = np.vstack(self.depth) / 1024.0
        self.states = np.vstack(self.states)
        self.actions = np.vstack(self.actions)
        assert self.depth.shape[0] == self.actions.shape[0]
        assert self.rgb.shape[0] == self.actions.shape[0]

        # self.rewards = np.vstack(self.rewards

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        out = {}
        out["action"] = (
            torch.from_numpy(self.actions[idx]).float().to(device=self.device)
        )
        depth = torch.from_numpy(self.depth[idx]).float().to(device=self.device)
        rgb = torch.from_numpy(self.rgb[idx]).float().to(device=self.device)
        out["rgbd"] = torch.cat([rgb, depth], dim=-1)
        out["state"] = torch.from_numpy(self.states[idx]).float().to(device=self.device)
        return out


# taken from here
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


class PlainConv(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_dim=256,
        max_pooling=True,
        inactivated_output=False,  # False for ConvBody, True for CNN
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 64]
            nn.Conv2d(16, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=not inactivated_output)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.encoder = PlainConv(
            in_channels=4, out_dim=256, max_pooling=False, inactivated_output=False
        )
        self.final_mlp = make_mlp(
            256 + state_dim, [512, 256, action_dim], last_act=False
        )
        self.get_eval_action = self.get_action = self.forward

    def forward(self, rgbd, state):
        img = rgbd.permute(0, 3, 1, 2)  # (B, C, H, W)
        feature = self.encoder(img)
        x = torch.cat([feature, state], dim=1)
        return self.final_mlp(x)

    @torch.no_grad()
    def act(self, obs):

        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                obs[k] = torch.from_numpy(v).to(device=device)
            obs[k] = obs[k].float()
        obs["rgbd"][:, :, :, -1] = obs["rgbd"][:, :, :, -1] / 1024.0
        obs["rgbd"][:, :, :, :3] = obs["rgbd"][:, :, :, :3] / 255.0

        if gpu_sim:
            return self(obs["rgbd"], obs["state"])
        return self(obs["rgbd"], obs["state"]).cpu().data.numpy()


def build_env(env, video=False, seed=0, control_mode=""):
    def inner():

        x = gym.make(
            env,
            obs_mode="rgbd",
            control_mode=control_mode,
            render_mode="rgb_array" if video else "sensors",
            max_episode_steps=args.max_episode_steps,
            sim_backend="cpu",
        )
        if video:
            x = RecordEpisode(
                x,
                os.path.join(log_path, args.video_path),
                save_trajectory=False,
                info_on_video=True,
            )
        x = FlattenRGBDObservationWrapper(x)
        x = gym.wrappers.ClipAction(x)
        x = CPUGymWrapper(x)

        x.action_space.seed(seed)
        x.observation_space.seed(seed)
        return x

    return inner


def make_eval_envs(env, num_envs, control_mode, gpu_sim=False):
    if gpu_sim:
        env = gym.make(
            env,
            num_envs=num_envs,
            obs_mode="rgbd",
            control_mode=control_mode,
            render_mode="sensors",
            max_episode_steps=args.max_episode_steps,
        )
        if args.video:
            env = RecordEpisode(
                env,
                os.path.join(log_path, args.video_path),
                save_trajectory=False,
                max_steps_per_video=args.max_episode_steps,
            )
        env = FlattenRGBDObservationWrapper(env)
        env = ManiSkillVectorEnv(env)
        return env

    return gym.vector.SyncVectorEnv(
        [
            build_env(
                env,
                video=(j == 0 and args.video),
                seed=j,
                control_mode=control_mode,
            )
            for j in range(num_envs)
        ]
    )


def save_ckpt(tag):
    os.makedirs(f"{log_path}/checkpoints", exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
        },
        f"{log_path}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Config)

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tag = "{:s}_{:d}".format(now, args.seed)
    log_name = os.path.join(args.env, "BC_RGB", tag)
    log_path = os.path.join(args.output_dir, log_name)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    control_mode = os.path.split(args.demo_path)[1].split(".")[2]

    if args.wandb:
        wandb_init(asdict(args))

    ds = ManiSkillDataset(
        args.demo_path,
        device=device,
        load_count=args.load_count,
    )

    gpu_sim = args.sim_backend == "gpu"

    if args.video:
        os.makedirs(os.path.join(log_path, args.video_path))

    envs = make_eval_envs(args.env, args.num_envs, control_mode, gpu_sim)
    obs, _ = envs.reset(seed=args.seed)

    sampler = RandomSampler(ds)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    iter_sampler = IterationBasedBatchSampler(batch_sampler, args.max_timesteps)

    data_loader = DataLoader(ds, batch_sampler=iter_sampler, num_workers=0)
    actor = Actor(ds.states.shape[1], envs.single_action_space.shape[0]).to(
        device=device
    )

    optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    best_sr = -1

    for i, batch in enumerate(data_loader):
        log_dict = {}

        optimizer.zero_grad()
        preds = actor(batch["rgbd"], batch["state"])
        loss = F.mse_loss(preds, batch["action"])
        loss.backward()
        optimizer.step()

        log_dict["losses/total_loss"] = loss.item()

        if i % args.eval_freq == 0:
            successes = []
            el_steps = []
            returns = []
            success_once = np.zeros((args.num_envs,))
            rewards = np.zeros((args.num_envs,))
            obs, _ = envs.reset()

            for _ in range(args.max_episode_steps):
                obs, reward, terminated, truncated, info = envs.step(actor.act(obs))
                if "success" in info:
                    success_once += (
                        info["success"].cpu().numpy() if gpu_sim else info["success"]
                    )
                if gpu_sim:
                    rewards += reward.cpu().numpy()
                else:
                    rewards += reward

                if "final_info" in info:
                    fin_info = info["final_info"]
                    mask = info["_final_info"]
                    if gpu_sim:
                        el_steps.append(fin_info["elapsed_steps"][mask].cpu().numpy())

                        if "success" in fin_info:
                            successes.append(fin_info["success"][mask].cpu().numpy())

                        mask = mask.cpu().numpy()

                    else:
                        fin_info = fin_info[mask]
                        fin_info = {
                            k: np.array([dic[k] for dic in fin_info])
                            for k in fin_info[0]
                        }
                        el_steps.append(fin_info["elapsed_steps"])
                        if "success" in fin_info:
                            successes.append(fin_info["success"])
                    returns.append(rewards[mask])

            log_dict["eval/episode_len"] = np.concatenate(el_steps).mean()
            log_dict["eval/return"] = np.concatenate(returns).mean()
            log_dict["charts/global_step"] = i

            if len(successes) > 0:
                s = np.concatenate(successes)
                log_dict["eval/success_at_end"] = s.mean()
                success_once += s
                if log_dict["eval/success_at_end"] > best_sr:
                    save_ckpt("best_eval_sr")
                    best_sr = log_dict["eval/success_at_end"]

            log_dict["eval/success_once"] = (success_once > 0).mean()
            out = f"Step: {i} " + ", ".join(
                [
                    f"{k.replace('eval/','')}: " + str(round(log_dict[k], 4))
                    for k in log_dict
                ]
            )

            print(out)
            if args.wandb:
                wandb.log(log_dict, i)

        if i % args.save_freq == 0 and i > 0:
            save_ckpt(str(i))
    envs.close()
    wandb.finish()
