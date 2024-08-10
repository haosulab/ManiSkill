"""
Code to run Reverse Forward Curriculum Learning.
Configs can be a bit complicated, we recommend directly looking at configs/ms2/base_sac_ms2_sample_efficient.yml for what options are available.
Alternatively, go to the file defining each of the nested configurations and see the comments.
"""
import copy
import os
import os.path as osp
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import jax
import numpy as np
import optax
from omegaconf import OmegaConf

from rfcl.agents.sac import SAC, ActorCritic, SACConfig
from rfcl.agents.sac.networks import DiagGaussianActor
from rfcl.data.dataset import ReplayDataset
from rfcl.envs.make_env import EnvConfig, make_env_from_cfg
from rfcl.logger import LoggerConfig
from rfcl.models import NetworkConfig, build_network_from_cfg
from rfcl.utils.parse import parse_cfg
from rfcl.utils.spaces import get_action_dim


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@dataclass
class TrainConfig:
    steps: int
    actor_lr: float
    critic_lr: float
    dataset_path: str
    shuffle_demos: bool
    num_demos: int

    data_action_scale: Optional[float]

@dataclass
class SACNetworkConfig:
    actor: NetworkConfig
    critic: NetworkConfig


@dataclass
class SACExperiment:
    seed: int
    sac: SACConfig
    env: EnvConfig
    eval_env: EnvConfig
    train: TrainConfig
    network: SACNetworkConfig
    logger: Optional[LoggerConfig]
    verbose: int
    algo: str = "sac"
    save_eval_video: bool = True  # whether to save eval videos
    demo_seed: int = None  # fix a seed to fix which demonstrations are sampled from a dataset

    """additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms"""
    demo_type: str = None
    config_type: str = None # "sample_efficient" or "walltime_efficient"


from dacite import from_dict


def main(cfg: SACExperiment):
    np.random.seed(cfg.seed)

    ### Setup the experiment parameters ###

    # Setup training and evaluation environment configs
    env_cfg = cfg.env
    if "env_kwargs" not in env_cfg:
        env_cfg["env_kwargs"] = dict()
    cfg.eval_env = {**env_cfg, **cfg.eval_env}
    cfg = from_dict(data_class=SACExperiment, data=OmegaConf.to_container(cfg))
    env_cfg = cfg.env
    eval_env_cfg = cfg.eval_env

    # change exp name if it exists
    orig_exp_name = cfg.logger.exp_name
    exp_path = osp.join(cfg.logger.workspace, orig_exp_name)
    if osp.exists(exp_path):
        i = 1
        prev_exp_path = exp_path
        while osp.exists(exp_path):
            prev_exp_path = exp_path
            cfg.logger.exp_name = f"{orig_exp_name}_{i}"
            exp_path = osp.join(cfg.logger.workspace, cfg.logger.exp_name)
            i += 1
        warnings.warn(f"{prev_exp_path} already exists. Changing exp_name to {cfg.logger.exp_name}")
    video_path = osp.join(cfg.logger.workspace, cfg.logger.exp_name, "videos")

    cfg.sac.num_envs = cfg.env.num_envs
    cfg.sac.num_eval_envs = cfg.eval_env.num_envs

    ### Create Environments ###
    if cfg.demo_seed is not None:
        np.random.seed(cfg.demo_seed)

    if "reward_mode" in cfg.env.env_kwargs:
        reward_mode = cfg.env.env_kwargs["reward_mode"]
    elif "reward_type" in cfg.env.env_kwargs:
        reward_mode = cfg.env.env_kwargs["reward_type"]
    else:
        raise ValueError("reward_mode is not specified")

    demo_replay_dataset = ReplayDataset(
        cfg.train.dataset_path,
        shuffle=cfg.train.shuffle_demos,
        skip_failed=False,
        num_demos=cfg.train.num_demos,
        reward_mode=reward_mode,
        data_action_scale=cfg.train.data_action_scale,
    )
    if demo_replay_dataset.action_scale is not None:
        env_cfg.action_scale = demo_replay_dataset.action_scale.tolist()
        eval_env_cfg.action_scale = env_cfg.action_scale
    np.random.seed(cfg.seed)

    env, env_meta = make_env_from_cfg(env_cfg, seed=cfg.seed)
    eval_env = None
    if cfg.sac.num_eval_envs > 0:
        eval_env, _ = make_env_from_cfg(
            eval_env_cfg,
            seed=cfg.seed + 1_000_000,
            video_path=video_path if cfg.save_eval_video else None,
        )

    sample_obs, sample_acts = env_meta.sample_obs, env_meta.sample_acts

    # create actor and critics models
    act_dims = get_action_dim(env_meta.act_space)

    def create_ac_model():
        actor = DiagGaussianActor(
            feature_extractor=build_network_from_cfg(cfg.network.actor),
            act_dims=act_dims,
            state_dependent_std=True,
        )
        ac = ActorCritic.create(
            jax.random.PRNGKey(cfg.seed),
            actor=actor,
            critic_feature_extractor=build_network_from_cfg(cfg.network.critic),
            sample_obs=sample_obs,
            sample_acts=sample_acts,
            initial_temperature=cfg.sac.initial_temperature,
            actor_optim=optax.adam(learning_rate=cfg.train.actor_lr),
            critic_optim=optax.adam(learning_rate=cfg.train.critic_lr),
        )
        return ac

    # create our algorithm
    ac = create_ac_model()
    cfg.logger.cfg = asdict(cfg)
    logger_cfg = cfg.logger
    algo = SAC(
        env=env,
        eval_env=eval_env,
        env_type=cfg.env.env_type,
        ac=ac,
        logger_cfg=logger_cfg,
        cfg=cfg.sac,
    )

    # for ManiSkill 3 baselines, try to modify the wandb config to match other baselines env_cfg setups.
    if algo.logger.wandb:
        import wandb as wb
        sim_backend = "cpu"
        if cfg.env.env_type == "gym:cpu":
            sim_backend = "cpu"
        elif cfg.env.env_type == "gym:gpu":
            sim_backend = "gpu"
        def parse_env_cfg(env_cfg):
            return {
                "env_id": cfg.env.env_id,
                "env_kwargs": cfg.env.env_kwargs,
                "num_envs": cfg.env.num_envs,
                "env_horizon": cfg.env.max_episode_steps,
                "sim_backend": sim_backend,
                "reward_mode": cfg.env.env_kwargs.get("reward_mode"),
                "obs_mode": cfg.env.env_kwargs.get("obs_mode"),
                "control_mode": cfg.env.env_kwargs.get("control_mode"),
            }
        fixed_wb_cfgs = {"env_cfg": parse_env_cfg(env_cfg), "eval_env_cfg": parse_env_cfg(eval_env_cfg), "num_demos": cfg.train.num_demos, "demo_type": cfg.demo_type}
        wb.config.update({**fixed_wb_cfgs}, allow_val_change=True)
        algo.logger.wandb_run.tags = ["rlpd", cfg.config_type]
    algo.offline_buffer = demo_replay_dataset  # create offline buffer to oversample from
    rng_key, train_rng_key = jax.random.split(jax.random.PRNGKey(cfg.seed), 2)
    algo.train(
        rng_key=train_rng_key,
        steps=cfg.train.steps,
        verbose=cfg.verbose,
    )
    algo.save(osp.join(algo.logger.model_path, "latest.jx"), with_buffer=False)
    # with_buffer=True means you can use the checkpoint to easily resume training with the same replay buffer data
    # algo.save(osp.join(algo.logger.model_path, "latest.jx"), with_buffer=True)
    env.close(), eval_env.close()

if __name__ == "__main__":
    cfg = parse_cfg(default_cfg_path=sys.argv[1])
    main(cfg)
