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

import gymnasium as gym
import jax
import numpy as np
import optax
from omegaconf import OmegaConf
import wandb as wb

from rfcl.agents.sac import SAC, ActorCritic, SACConfig
from rfcl.agents.sac.networks import DiagGaussianActor
from rfcl.data.dataset import ReplayDataset, get_states_dataset
from rfcl.envs.make_env import EnvConfig, get_initial_state_wrapper, make_env_from_cfg
from rfcl.envs.wrappers.curriculum import ReverseCurriculumWrapper
from rfcl.envs.wrappers.forward_curriculum import SeedBasedForwardCurriculumWrapper
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

    # reverse curriculum wrapper configs
    reverse_step_size: int
    curriculum_method: str
    start_step_sampler: str
    per_demo_buffer_size: int
    demo_horizon_to_max_steps_ratio: float
    train_on_demo_actions: bool

    # forward curriculum configs
    forward_curriculum: str
    staleness_transform: str
    staleness_coef: float
    staleness_temperature: float
    score_transform: str
    score_temperature: float
    num_seeds: int

    # stage 2 training configs
    load_actor: bool
    load_critic: bool
    load_as_offline_buffer: bool
    load_as_online_buffer: bool

    # other configs that are generally used for experimentation
    use_orig_env_for_eval: bool = True
    eval_start_of_demos: bool = False


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
    stage_1_model_path: str = None  # if not None, will load pretrained stage 1 model and skip to stage 2 of training
    save_eval_video: bool = True  # whether to save eval videos
    stage_1_only: bool = False  # stop training after reverse curriculum completes
    stage_2_only: bool = False # skip stage 1 training
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
    video_path = osp.join(cfg.logger.workspace, cfg.logger.exp_name, "stage_1_videos")

    cfg.sac.num_envs = cfg.env.num_envs
    cfg.sac.num_eval_envs = cfg.eval_env.num_envs

    ### Create Environments ###
    if cfg.demo_seed is not None:
        np.random.seed(cfg.demo_seed)

    states_dataset = get_states_dataset(cfg.train.dataset_path, num_demos=cfg.train.num_demos, shuffle=cfg.train.shuffle_demos, skip_failed=True)

    if "reward_mode" in cfg.env.env_kwargs:
        reward_mode = cfg.env.env_kwargs["reward_mode"]
    elif "reward_type" in cfg.env.env_kwargs:
        reward_mode = cfg.env.env_kwargs["reward_type"]
    else:
        raise ValueError("reward_mode is not specified")

    if cfg.train.train_on_demo_actions:
        demo_replay_dataset = ReplayDataset(
            cfg.train.dataset_path,
            shuffle=cfg.train.shuffle_demos,
            skip_failed=False,
            num_demos=cfg.train.num_demos,
            reward_mode=reward_mode,
            eps_ids=states_dataset.keys(), # forces the demo replay dataset used as the offline buffer to use the same demos as the reverse curriculum
            data_action_scale=cfg.train.data_action_scale,
        )
        if demo_replay_dataset.action_scale is not None:
            env_cfg.action_scale = demo_replay_dataset.action_scale.tolist()
            eval_env_cfg.action_scale = env_cfg.action_scale
    InitialStateWrapper = get_initial_state_wrapper(cfg.env.env_id)
    np.random.seed(cfg.seed)
    wrappers = [
        lambda env: InitialStateWrapper(
            env,
            states_dataset=states_dataset,
            demo_horizon_to_max_steps_ratio=cfg.train.demo_horizon_to_max_steps_ratio,
        )
    ]
    env, env_meta = make_env_from_cfg(env_cfg, seed=cfg.seed, wrappers=wrappers)
    eval_env = None
    use_orig_env_for_eval = cfg.train.use_orig_env_for_eval
    if cfg.sac.num_eval_envs > 0:
        eval_wrappers = []
        if not use_orig_env_for_eval:
            eval_wrappers = wrappers
        eval_env, _ = make_env_from_cfg(
            eval_env_cfg,
            seed=cfg.seed + 1_000_000,
            video_path=video_path if cfg.save_eval_video else None,
            wrappers=eval_wrappers,
        )
    # add reverse curriculum wrapper
    link_envs = []
    if not use_orig_env_for_eval:
        eval_env = ReverseCurriculumWrapper(
            eval_env,
            eval_mode=True,
            eval_start_of_demos=cfg.train.eval_start_of_demos,
            states_dataset=states_dataset,
            reverse_step_size=cfg.train.reverse_step_size,
            curriculum_method=cfg.train.curriculum_method,
            per_demo_buffer_size=cfg.train.per_demo_buffer_size,
            start_step_sampler=cfg.train.start_step_sampler,
        )
        link_envs = [eval_env]
    env = ReverseCurriculumWrapper(
        env,
        states_dataset=states_dataset,
        reverse_step_size=cfg.train.reverse_step_size,
        curriculum_method=cfg.train.curriculum_method,
        per_demo_buffer_size=cfg.train.per_demo_buffer_size,
        start_step_sampler=cfg.train.start_step_sampler,
        link_envs=link_envs,
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
        algo.logger.wandb_run.tags = ["rfcl", cfg.config_type]


    ###########################################
    # Stage 1 Training: Reverse Curriculum RL #
    ###########################################

    if cfg.train.train_on_demo_actions:
        algo.offline_buffer = demo_replay_dataset  # create offline buffer to oversample from

    if not cfg.stage_2_only:
        def early_stop_fn(locals):
            # callback function to log reverse curriculum metrics and stop training once reverse curriculum is done
            nonlocal env, algo
            logger = algo.logger
            demo_metadata = env.demo_metadata
            pts = []
            solved_frac = 0
            for k in demo_metadata:
                pts.append(demo_metadata[k].start_step / (demo_metadata[k].total_steps - 1))
                solved_frac += int(demo_metadata[k].solved)
            solved_frac = solved_frac / len(demo_metadata)
            mean_start_step = np.mean(pts)
            logger.tb_writer.add_histogram("train_stats/start_step_frac_dist", pts, algo.state.total_env_steps)
            logger.tb_writer.add_scalar("train_stats/start_step_frac_avg", mean_start_step, algo.state.total_env_steps)
            if logger.wandb:
                import wandb as wb

                logger.wandb_run.log(data={"train_stats/start_step_frac_dist": wb.Histogram(pts)}, step=algo.state.total_env_steps)
                logger.wandb_run.log(data={"train_stats/start_step_frac_avg": mean_start_step}, step=algo.state.total_env_steps)

            if solved_frac > 0.9:
                print("Reverse solved > 0.9 of demos. Stopping stage 1")
                return True
            return False

        if cfg.stage_1_model_path is None:
            rng_key, train_rng_key = jax.random.split(jax.random.PRNGKey(cfg.seed), 2)
            algo.train(
                rng_key=train_rng_key,
                steps=cfg.train.steps,
                callback_fn=early_stop_fn,
                verbose=cfg.verbose,
            )
            algo.save(osp.join(algo.logger.model_path, "stage_1.jx"), with_buffer=True)
            algo.logger.tb_writer.add_scalar("train_stats/stage_1_steps", algo.state.total_env_steps, algo.state.total_env_steps)
            if algo.logger.wandb:
                algo.logger.wandb_run.log(data={"train_stats/stage_1_steps": algo.state.total_env_steps}, step=algo.state.total_env_steps)
        else:
            print(f"Loading stage 1 model: {cfg.stage_1_model_path}")
            algo.load_from_path(cfg.stage_1_model_path)

    if cfg.stage_1_only:
        exit()

    ###############################
    # Stage 2 Training: Normal RL with Forward Curriculums #
    ###############################
    print("Stage 2 Training starting")
    # Optionally load actor/critic networks from stage 1 of training
    ac = create_ac_model()
    if cfg.train.load_actor:
        ac = ac.load(algo.state.ac.state_dict(), load_critic=cfg.train.load_critic)
        algo.state = algo.state.replace(ac=ac)

    if not cfg.stage_2_only:
        # if not stage 2 only, there is a stage 1 replay buffer we can use
        # Load previous model's replay buffer as a separate offline buffer to sample from or directly into the online buffer
        if cfg.train.load_as_offline_buffer:
            print(
                f"Loading replay buffer as offline buffer which contains {algo.replay_buffer.size() * algo.replay_buffer.num_envs} interactions. Reset online buffer"
            )
            algo.offline_buffer = copy.deepcopy(algo.replay_buffer)
            algo.replay_buffer.reset()
        if cfg.train.load_as_online_buffer:
            print(
                f"Loading replay buffer into online buffer which contains {algo.replay_buffer.size() * algo.replay_buffer.num_envs} interactions. No offline buffer"
            )
            algo.offline_buffer = None

    # Switch environments from the reverse curriculum environments to a normal environment
    env.close(), eval_env.close()

    video_path = osp.join(cfg.logger.workspace, cfg.logger.exp_name, "stage_2_videos")
    wrappers = []
    if cfg.train.data_action_scale is not None:
        rescale_action_wrapper = lambda x: gym.wrappers.RescaleAction(x, -demo_replay_dataset.action_scale, demo_replay_dataset.action_scale)
        clip_wrapper = lambda x: gym.wrappers.ClipAction(x)
        wrappers += [rescale_action_wrapper, clip_wrapper]

    env, env_meta = make_env_from_cfg(env_cfg, seed=cfg.seed, wrappers=wrappers)
    eval_env = None
    if cfg.sac.num_eval_envs > 0:
        eval_wrappers = []
        if cfg.train.data_action_scale is not None:
            eval_wrappers += [rescale_action_wrapper, clip_wrapper]
        eval_env, _ = make_env_from_cfg(
            eval_env_cfg,
            seed=cfg.seed + 1_000_000,
            video_path=video_path if cfg.save_eval_video else None,
            wrappers=eval_wrappers,
        )

    print(f"Forward curriculum: {cfg.train.forward_curriculum}")
    if cfg.train.forward_curriculum is not None and cfg.train.forward_curriculum != "None":
        env = SeedBasedForwardCurriculumWrapper(
            env,
            score_transform=cfg.train.score_transform,
            score_temperature=cfg.train.score_temperature,
            staleness_transform=cfg.train.staleness_transform,
            staleness_temperature=cfg.train.staleness_temperature,
            staleness_coef=cfg.train.staleness_coef,
            score_fn=cfg.train.forward_curriculum,
            rho=0,
            nu=0.95,
            num_seeds=cfg.train.num_seeds,
        )
        env.reset(seed=cfg.seed)
    algo.setup_envs(env, eval_env)
    algo.state = algo.state.replace(initialized=False)

    (
        rng_key,
        train_rng_key,
    ) = jax.random.split(jax.random.PRNGKey(cfg.seed), 2)

    # we seed with policy in stage 2 for algo.cfg.num_seed_steps
    algo.cfg.seed_with_policy = True
    algo.cfg.num_seed_steps = algo.state.total_env_steps + algo.cfg.num_seed_steps
    print(f"Seeding until {algo.cfg.num_seed_steps}")
    algo.train(
        rng_key=train_rng_key,
        steps=cfg.train.steps - algo.state.total_env_steps,
        verbose=cfg.verbose,
    )
    algo.save(osp.join(algo.logger.model_path, "latest.jx"), with_buffer=False)


if __name__ == "__main__":
    cfg = parse_cfg(default_cfg_path=sys.argv[1])
    main(cfg)
