import datetime
import os
import random
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
    # training random seed
    seed: int = 42
    # training device
    device: str = "cuda"
    # learning rate
    lr = 3e-4
    # number of parallel eval envs
    num_envs = 50
    # dataset directory
    demo_path: str = ""
    # log to wandb
    wandb: bool = True
    # normalize states
    normalize_states: bool = False
    # number of trajectories to load, max is usually 1000
    load_count: int = 500
    # seed
    seed: int = 2024
    # where experiment outputs should be stored
    output_dir = "./output/"
    # save frequency
    save_freq: int = 10000
    # sim backend
    sim_backend: str = "cpu"
    # max number of evaluation steps per eval episode
    max_episode_steps: int = 100


def set_seed(seed: int, deterministic_torch: bool = False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=log_name.replace(os.path.sep, "__"),
        tags=["bc"],
    )
    wandb.run.save()


def build_env(
    env,
    video=False,
    normalize_states=False,
    state_stats: tuple = (),
    seed=0,
    control_mode="",
):
    def norm_states(state):
        return (state - state_stats[0]) / state_stats[1]

    def inner():

        x = gym.make(
            env,
            obs_mode="state",
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
        if normalize_states:
            x = gym.wrappers.TransformObservation(x, norm_states)
        x = CPUGymWrapper(x)
        x = gym.wrappers.ClipAction(x)

        x.action_space.seed(seed)
        x.observation_space.seed(seed)
        return x

    return inner


def make_eval_envs(env, num_envs, stats, control_mode, gpu_sim=False):
    if gpu_sim:
        env = gym.make(
            env,
            num_envs=num_envs,
            obs_mode="state",
            control_mode=control_mode,
            render_mode="sensors",
            max_episode_steps=args.max_episode_steps,
        )
        env = ManiSkillVectorEnv(env)
        if args.video:
            env = RecordEpisode(
                env,
                os.path.join(log_path, args.video_path),
                save_trajectory=False,
                max_steps_per_video=args.max_episode_steps,
            )
        return env

    return gym.vector.SyncVectorEnv(
        [
            build_env(
                env,
                video=(j == 0 and args.video),
                normalize_states=args.normalize_states,
                state_stats=stats,
                seed=j,
                control_mode=control_mode,
            )
            for j in range(num_envs)
        ]
    )


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


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        device,
        load_count=-1,
        normalize_states=False,
    ) -> None:
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

        self.observations = []
        self.actions = []
        self.dones = []
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

            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            self.observations.append(trajectory["obs"][:-1])
            self.actions.append(trajectory["actions"])
            self.dones.append(trajectory["success"].reshape(-1, 1))
            # print(trajectory.keys())
        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)
        self.dones = np.vstack(self.dones)
        assert self.observations.shape[0] == self.actions.shape[0]
        assert self.dones.shape[0] == self.actions.shape[0]

        if normalize_states:
            mean, std = self.get_state_stats()
            self.observations = (self.observations - mean) / std

        # self.rewards = np.vstack(self.rewards)

    def get_state_stats(self):
        return np.mean(self.observations), np.std(self.observations)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = torch.from_numpy(self.actions[idx]).float().to(device=self.device)
        obs = torch.from_numpy(self.observations[idx]).float().to(device=self.device)
        done = torch.from_numpy(self.dones[idx]).to(device=self.device)
        return obs, action, done


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    @torch.no_grad()
    def act(self, state, gpu_sim=False):
        if gpu_sim:
            return self(state)
        state = torch.from_numpy(state).to(device="cuda:0")
        return self(state).cpu().data.numpy()


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
    set_seed(args.seed)

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tag = "{:s}_{:d}".format(now, args.seed)
    log_name = os.path.join(args.env, "BC", tag)
    log_path = os.path.join(args.output_dir, log_name)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    control_mode = os.path.split(args.demo_path)[1].split(".")[2]

    if args.wandb:
        wandb_init(asdict(args))
    ds = ManiSkillDataset(
        args.demo_path,
        device=device,
        load_count=args.load_count,
        normalize_states=args.normalize_states,
    )
    stats = ds.get_state_stats()

    if args.video:
        os.makedirs(os.path.join(log_path, args.video_path))
    gpu_sim = args.sim_backend == "gpu"
    envs = make_eval_envs(args.env, args.num_envs, stats, control_mode, gpu_sim)
    obs, _ = envs.reset(seed=args.seed)
    sampler = RandomSampler(ds)
    batchsampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    itersampler = IterationBasedBatchSampler(batchsampler, args.max_timesteps)
    dataloader = DataLoader(ds, batch_sampler=itersampler, num_workers=0)
    actor = Actor(
        envs.single_observation_space.shape[0], envs.single_action_space.shape[0]
    )
    actor = actor.to(device=device)
    optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    best_sr = -1

    for i, batch in enumerate(dataloader):
        log_dict = {}
        obs, action, _ = batch
        pred_action = actor(obs)

        optimizer.zero_grad()
        loss = F.mse_loss(pred_action, action)
        loss.backward()
        optimizer.step()

        log_dict["losses/total_loss"] = loss.item()

        if i % args.eval_freq == 0:

            successes = []
            el_steps = []
            returns = []
            success_once = np.zeros((args.num_envs,))
            obs, _ = envs.reset()
            rewards = np.zeros((args.num_envs,))

            for _ in range(args.max_episode_steps):
                obs, reward, terminated, truncated, info = envs.step(
                    actor.act(obs, gpu_sim)
                )

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

            log_dict["charts/global_step"] = i
            log_dict["eval/episode_len"] = np.concatenate(el_steps).mean()
            log_dict["eval/return"] = np.concatenate(returns).mean()

            if len(successes) > 0:
                s = np.concatenate(successes)
                log_dict["eval/success_at_end"] = s.mean().item()
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
