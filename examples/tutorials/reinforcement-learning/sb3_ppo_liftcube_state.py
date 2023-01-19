# Import required packages
import argparse
import os.path as osp

import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from tqdm.notebook import tqdm

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and state based Observations"
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=100,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="logs",
        help="path for where logs, checkpoints, and videos are saved",
    )
    args = parser.parse_args()
    return args


def main(args):
    env_id = args.env_id
    num_envs = args.n_envs
    max_episode_steps = args.max_episode_steps
    logs = args.logs
    rollout_steps = 3200

    obs_mode = "state"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "dense"

    def make_env(env_id: str, rank: int, seed: int = 0, record_dir: str = None):
        def _init() -> gym.Env:
            import mani_skill2.envs

            env = gym.make(
                env_id,
                obs_mode=obs_mode,
                reward_mode=reward_mode,
                control_mode=control_mode,
            )
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
            if record_dir is not None:
                env = RecordEpisode(
                    env, record_dir, info_on_video=True, render_mode="cameras"
                )
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    # create one eval environment
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, record_dir=osp.join(logs, "videos")) for i in range(1)]
    )
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.reset()

    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    env = VecMonitor(env)
    env.reset()

    # define callbacks to periodically save our model and evaluate it to help monitor training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=logs,
        log_path=logs,
        eval_freq=10 * rollout_steps // num_envs,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10 * rollout_steps // num_envs,
        save_path=logs,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Define the policy configuration and algorithm configuration
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=rollout_steps // num_envs,
        batch_size=400,
        gamma=0.85,
        n_epochs=15,
        tensorboard_log=logs,
        target_kl=0.05,
    )
    # Train an agent with PPO for 320_000 interactions
    model.learn(
        350_000,
        callback=[checkpoint_callback, eval_callback],
    )
    # Save the final model
    model.save(osp.join(logs, "latest_model"))

    # Load the saved model
    # model = model.load(osp.join(logs, "rl_model_320000_steps"))

    # Evaluate the model
    results = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        n_eval_episodes=10,
    )
    print(results)


if __name__ == "__main__":
    main(parse_args())
