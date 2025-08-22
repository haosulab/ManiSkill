import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_envs
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`env_id`: task name (eg. PickCube-v0)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video_local`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video_local=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes_per_env > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	assert not cfg.multitask, colored('Warning: multi-task models is not currently supported for maniskill.', 'red', attrs=['bold'])
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.env_id}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

	# Make environment
	env = make_envs(cfg, cfg.num_eval_envs, is_eval=True)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)
	
	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.env_id}:', 'yellow', attrs=['bold']))
	if cfg.save_video_local:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.env_id]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		has_success, has_fail = False, False # if task has success or/and fail (added for maniskill)
		ep_rewards, ep_successes, ep_fails = [], [], []
		i = 0
		for _ in range(cfg.eval_episodes_per_env):
			obs, _ = env.reset()
			done = torch.full((cfg.num_eval_envs, ), False, device=('cuda' if cfg.env_type=='gpu' else 'cpu')) # ms3: done is truncated since the ms3 ignore_terminations.
			ep_reward, t = torch.zeros((cfg.num_eval_envs, ), device=('cuda' if cfg.env_type=='gpu' else 'cpu')), 0
			if cfg.save_video_local:
				frames = [env.render().cpu()]
			while not done[0]: # done is truncated and should be the same
				action = agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, terminated, truncated, info = env.step(action)
				done = terminated | truncated
				ep_reward += reward
				t += 1
				if cfg.save_video_local:
					frames.append(env.render().cpu())
			ep_rewards.append(ep_reward.mean().item())
			if 'success' in info: 
				has_success = True
				ep_successes.append(info['final_info']['success'].float().mean().item())
			if 'fail' in info:
				has_fail = True
				ep_fails.append(info['final_info']['fail'].float().mean().item())
			if cfg.save_video_local:
				videos = np.array(frames).transpose([1,0,2,3,4])
				for video in videos:
					imageio.mimsave(
						os.path.join(video_dir, f'{task}-{i}.mp4'), video, fps=15)
					i += 1
		ep_rewards = np.nanmean(ep_rewards)
		ep_successes = np.nanmean(ep_successes)
		ep_fails = np.nanmean(ep_fails)
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		if has_success and has_fail:
			print(colored(f'  {task:<22}' \
				f'\tR: {ep_rewards:.01f}  ' \
				f'\tS: {ep_successes:.02f}' \
				f'\tF: {ep_fails:.02f}', 'yellow'))
		elif has_success:
			print(colored(f'  {task:<22}' \
				f'\tR: {ep_rewards:.01f}  ' \
				f'\tS: {ep_successes:.02f}', 'yellow'))
		elif has_fail:
			print(colored(f'  {task:<22}' \
				f'\tR: {ep_rewards:.01f}  ' \
				f'\tF: {ep_fails:.02f}', 'yellow'))
		
	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
