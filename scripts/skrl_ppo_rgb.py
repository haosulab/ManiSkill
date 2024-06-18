import gymnasium as gym
import numpy as np
import os
import random
import wandb
import time
import tyro

from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from skrl_policies import *
from skrl_args import VisualRunArgs



def run():
    """ Visual-based training """
    args = tyro.cli(VisualRunArgs)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.num_envs = 5

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup (evaluation + training)
    env_kwargs = dict(obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="rgb_array", sim_backend="gpu")
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenRGBDObservationWrapper(envs, rgb_only=True)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)

    # Video recording
    if args.capture_video:
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs, 
                output_dir=f"runs/{run_name}/train_videos", 
                save_trajectory=False, 
                save_video_trigger=save_video_trigger, 
                max_steps_per_video=args.num_steps, 
                video_fps=30)

    # For gpu acceleration
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, **env_kwargs)
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")

    # instantiate a memory as rollout buffer (any memory can be used for this)
    MEM_BUFF_SIZE = 1000 #10000
    memory = RandomMemory(memory_size=MEM_BUFF_SIZE, num_envs=envs.num_envs, device=device)

    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = {}

    models["policy"] = VisionBasedTraining.Policy(envs.observation_space, envs.action_space, device, clip_actions=True)
    models["value"] = VisionBasedTraining.Value(envs.observation_space, envs.action_space, device)

    # initialize models' parameters (weights and biases)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = MEM_BUFF_SIZE
    cfg["learning_epochs"] = 10
    cfg["mini_batches"] = 10
    cfg["discount_factor"] = 0.9995
    cfg["lambda"] = 0.95
    cfg["policy_learning_rate"] = 2.5e-4
    cfg["value_learning_rate"] = 2.5e-4
    cfg["grad_norm_clip"] = 10
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = False
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 0.5
    cfg["kl_threshold"] = 0
    cfg["enable_cameras"] = True
    cfg["wandb_log"] = True

    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 10000
    cfg["experiment"]["checkpoint_interval"] = 10000
    cfg["experiment"]["directory"] = "runs/torch/JetBotEnv"

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device
    )

    TIMESTEPS = 500000
    cfg_trainer = {"timesteps": TIMESTEPS, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=envs, agents=agent, wandblog=True)
    
    trainer.train()

    if args.track:
        wandb.finish()





if __name__ == "__main__":
    print("Start")
    
    run()

    print("End")
