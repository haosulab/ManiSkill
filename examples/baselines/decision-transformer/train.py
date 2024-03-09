import argparse
import random

import gymnasium as gym
import torch
import wandb
from decision_transformer.dataset import ManiSkill2Dataset, load_batch
from decision_transformer.eval import eval_episodes
from decision_transformer.model import DecisionTransformer
from decision_transformer.trainer import SequenceTrainer

import mani_skill.envs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    help="Specifies the env to run the training: PickCube-v0, LiftCube-v0, StackCube-v0",
)
parser.add_argument("--dataset", type=str)
parser.add_argument("--n_layer", type=int, default=4)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--max_iters", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--log_to_wandb", type=bool, default=False)
parser.add_argument("--wandb_key", type=str)

args = parser.parse_args()
env_id = args.env  # @param ["PickCube-v0", "LiftCube-v0", "StackCube-v0"]
env = gym.make(env_id, obs_mode="state", control_mode="pd_ee_delta_pose")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

dataset = ManiSkill2Dataset(args.dataset)
state_mean, state_std = dataset.get_state_stats()

device = "cuda" if torch.cuda.is_available() else "cpu"

variant = {
    "embed_dim": args.embed_dim,
    "n_layer": args.n_layer,
    "n_head": args.n_head,
    "dropout": 0.1,
    "activation_function": "relu",
    "warmup_steps": 10_000,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "max_iters": args.max_iters,
    "K": 30,
    "max_ep_len": 100,
    "num_eval_episodes": 100,
    "state_dim": state_dim,
    "act_dim": action_dim,
    "device": device,
    "state_mean": state_mean,
    "state_std": state_std,
}

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=action_dim,
    max_length=variant["K"],
    max_ep_len=variant["max_ep_len"],
    hidden_size=variant["embed_dim"],
    n_layer=variant["n_layer"],
    n_head=variant["n_head"],
    n_positions=512,  # not used
    resid_pdrop=variant["dropout"],
    attn_pdrop=variant["dropout"],
    n_inner=4 * variant["embed_dim"],
    activation_function=variant["activation_function"],
)


model = model.to(device=device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=variant["learning_rate"],
    weight_decay=variant["weight_decay"],
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
)

trainer = SequenceTrainer(
    model=model,
    optimizer=optimizer,
    dataset=dataset,
    batch_size=args.batch_size,
    get_batch=load_batch,
    scheduler=scheduler,
    loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
    eval_fns=[eval_episodes(tar, env, variant) for tar in [50, 30]],
    state_dim=state_dim,
    device=device,
)


if args.log_to_wandb:
    wandb.login(key=args.wandb_key)

    wandb.init(
        name=f"maniskill2-{env_id}-{random.randint(0, 100000)}",
        group=f"maniskill2-{env_id}",
        project="decision-transformer",
        config=variant,
    )
for iter in range(variant["max_iters"]):
    outputs = trainer.train_iteration(1000, iter_num=iter, print_logs=True)
    if args.log_to_wandb:
        wandb.log(outputs)
