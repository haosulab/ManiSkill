# Import required packages
import argparse
import os.path as osp
from pathlib import Path

import gym
import h5py
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkill2Dataset(Dataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        import h5py

        from mani_skill2.utils.io_utils import load_json

        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.observations = []
        self.actions = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            self.observations.append(trajectory["obs"][:-1])
            self.actions.append(trajectory["actions"])
        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = th.from_numpy(self.actions[idx]).float()
        obs = th.from_numpy(self.observations[idx]).float()
        return obs, action


class Policy(nn.Module):
    def __init__(
        self,
        obs_dims,
        act_dims,
        hidden_units=[128, 128],
        activation=nn.ReLU,
    ):
        super().__init__()
        mlp_layers = []
        prev_units = obs_dims
        for h in hidden_units:
            mlp_layers += [nn.Linear(prev_units, h), activation()]
            prev_units = h
        # attach a tanh regression head since we know all actions are constrained to [-1, 1]
        mlp_layers += [nn.Linear(prev_units, act_dims), nn.Tanh()]
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, observations) -> th.Tensor:
        return self.mlp(observations)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Simple script demonstrating how to train an agent with imitation learning (behavior cloning) using ManiSkill2 environmnets and demonstrations"
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-d", "--demos", type=str, help="path to demonstration dataset .h5py file"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/bc_state",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--steps", type=int, help="numbr of training steps", default=30000
    )
    parser.add_argument(
        "--eval", action="store_true", help="whether to only evaluate policy"
    )
    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env_id = args.env_id
    demo_path = args.demos
    log_dir = args.log_dir
    iterations = args.steps

    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)

    ckpt_dir = osp.join(log_dir, "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    obs_mode = "state"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "dense"
    env = gym.make(
        env_id, obs_mode=obs_mode, control_mode=control_mode, reward_mode=reward_mode
    )
    # RecordEpisode wrapper auto records a new video once an episode is completed
    env = RecordEpisode(env, output_dir=osp.join(log_dir, "videos"))
    env.seed(0)
    if args.eval:
        model_path = args.model_path
        if model_path is None:
            model_path = osp.join(log_dir, "checkpoints/ckpt_latest.pt")
        # Load the saved model
        policy = th.load(model_path)
    else:
        assert (
            demo_path is not None
        ), "Need to provide a demonstration dataset via --demos"
        dataset = ManiSkill2Dataset(demo_path)
        dataloader = DataLoader(
            dataset,
            batch_size=100,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )
        obs, action = dataset[0]
        print("State:", obs.shape)
        print("Action:", action.shape)
        # create our policy
        obs, action = dataset[0]
        policy = Policy(obs.shape[0], action.shape[0], hidden_units=[256, 256])
    # move model to gpu if possible
    device = "cuda" if th.cuda.is_available() else "cpu"
    policy = policy.to(device)
    print(policy)

    loss_fn = nn.MSELoss()

    # a short save function to save our model
    def save_model(policy, path):
        th.save(policy, path)

    def train_step(policy, obs, actions, optim, loss_fn):
        optim.zero_grad()
        # move data to appropriate device first
        obs = obs.to(device)
        actions = actions.to(device)

        pred_actions = policy(obs)

        # compute loss and optimize
        loss = loss_fn(actions, pred_actions)
        loss.backward()
        optim.step()
        return loss.item()

    def evaluate_policy(env, policy, num_episodes=10):
        obs = env.reset(seed=0)
        successes = []
        i = 0
        pbar = tqdm(total=num_episodes, leave=False)
        while i < num_episodes:
            # move to appropriate device and unsqueeze to add a batch dimension
            obs_device = th.from_numpy(obs).float().unsqueeze(0).to(device)
            with th.no_grad():
                action = policy(obs_device).cpu().numpy()[0]
            obs, reward, done, info = env.step(action)
            if done:
                successes.append(info["success"])
                i += 1
                obs = env.reset(seed=i)
                pbar.update(1)
        success_rate = np.mean(successes)
        return success_rate

    if not args.eval:
        writer = SummaryWriter(log_dir)

        optim = th.optim.Adam(policy.parameters(), lr=1e-3)
        best_epoch_loss = np.inf
        pbar = tqdm(dataloader, total=iterations)
        epoch = 0
        steps = 0
        while steps < iterations:
            epoch_loss = 0
            for batch in dataloader:
                steps += 1
                obs, actions = batch
                loss_val = train_step(policy, obs, actions, optim, loss_fn)

                # track the loss and print it
                writer.add_scalar("train/mse_loss", loss_val, steps)
                epoch_loss += loss_val
                pbar.set_postfix(dict(loss=loss_val))
                pbar.update(1)

                # periodically save the policy
                if steps % 2000 == 0:
                    save_model(policy, osp.join(ckpt_dir, f"ckpt_{steps}.pt"))
                if steps >= iterations:
                    break

            epoch_loss = epoch_loss / len(dataloader)

            # save a new model if the average MSE loss in an epoch has improved
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                save_model(policy, osp.join(ckpt_dir, "ckpt_best.pt"))
            if epoch % 50 == 0:
                print("Evaluating")
                success_rate = evaluate_policy(env, policy)
                writer.add_scalar("test/success_rate", success_rate, epoch)
            writer.add_scalar("train/mse_loss_epoch", epoch_loss, epoch)
            epoch += 1
        save_model(policy, osp.join(ckpt_dir, "ckpt_latest.pt"))

    # run a final evaluation
    success_rate = evaluate_policy(env, policy)
    print(f"Final Success Rate {success_rate}")


if __name__ == "__main__":
    main()
