import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored
from omegaconf import OmegaConf

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_envs
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger, print_run
import multiprocessing

import gymnasium as gym

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	assert not cfg.multitask, colored('Warning: multi-task models is not currently supported for maniskill.', 'red', attrs=['bold'])
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	
	# Need to initiate logger before make env to wrap record episode wrapper into async vec cpu env
	manager = multiprocessing.Manager()
	video_path = cfg.work_dir / 'eval_video'
	if cfg.save_video_local:
		try:
			os.makedirs(video_path)
		except:
			pass
	logger = Logger(cfg, manager)
	# Init env
	env = make_envs(cfg, cfg.num_envs)
	eval_env = make_envs(cfg, cfg.num_eval_envs, video_path=video_path, is_eval=True, logger=logger)
	print_run(cfg)
	# Init agent
	agent = TDMPC2(cfg)
	# Update wandb config, for control_mode, env_horizon, discount are set after logger init
	if logger._wandb != None:
		logger._wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
	trainer_cls = OnlineTrainer # OfflineTrainer not available
	trainer = trainer_cls(
		cfg=cfg,
		env=env,
		eval_env=eval_env,
		agent=agent,
		buffer=Buffer(cfg),
		logger=logger,
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()