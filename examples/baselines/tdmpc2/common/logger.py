import os
import datetime
import re
import numpy as np
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf
from mani_skill.utils.visualization.misc import tile_images

from common import TASK_SET


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("episode_fail", "F", "float"),
	("total_time", "T", "time"),
	# Added for maniskill rl baselines matrics
	# ("episode_reward_avg", "RA", "float"), 
	# ("episode_len", "L", "int"), 
	# ("rollout_time", "RT", "float"), 
	# ("rollout_fps", "RF", "float"), 
	# ("update_time", "U", "float"), 

]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
	# Added for maniskill rl baselines matrics
	"time" : "magenta", 
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.env_id),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.env_id, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self.maniskill_video_nrows = int(np.sqrt(cfg.eval_episodes))
		self._save_dir = make_dir(cfg.work_dir / 'eval_video')
		self._wandb = wandb
		self.fps = fps
		self.frames = [] # records only current num_eval_envs # of epsidoes (ep_len, num_eval_envs, h, w, 3)
		self.videos = [] # records all episodes (eval_episodes, ep_len, h, w, 3)
		self.enabled = False

	def init_cur_eps(self, env, enabled=True):
		"""
		Init a new set of episodes in a frame buffer, later to be added together to the video buffer.
		"""
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled,
		self.record_cur_eps(env)

	def record_cur_eps(self, env):
		"""
		Record current episodes' frames to the frame buffer.
		"""
		if self.enabled:
			self.frames.append(env.render().cpu().numpy())

	def save_cur_eps(self):
		"""
		Save current num_envs episodes to the video buffer.
		"""
		if self.enabled and len(self.frames) > 0:
			self.videos.extend(np.array(self.frames).transpose(1,0,2,3,4))
			
	def flush_saved_eps(self, step, num_episodes, key='videos/eval_video'):
		"""
		Flush all episodes in the video buffer to wandb. It will call reset() at the end.
		"""
		if self.enabled and len(self.frames) > 0:
			videos = np.array(self.videos[:num_episodes]).transpose(1,0,2,3,4) # Truncate recorded episodes to self.cf.eval_episodes (ep_len, eval_episodes, h, w, 3)
			videos = np.stack([tile_images(rgbs, nrows=self.maniskill_video_nrows) for rgbs in videos])
			self._wandb.log(
				{key: self._wandb.Video(videos.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)
		self.reset()
		
	def reset(self):
		"""
		Reset video recorder, including the video buffer.
		"""
		self.frames = []
		self.videos = []


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		self.name = cfg.get("wandb_name", "none")
		self.group = cfg.get("wandb_group", "none")
		if not cfg.wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = False
			self._wandb = None
			self._video = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb

		# Modified for Maniskill RL Baseline Logging Convention
		wandb_tags = cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"] + ["tdmpc2"]
		if cfg.setting_tag != 'none':
			wandb_tags += [cfg.setting_tag]
		wandb.init(
			project=self.project,
			entity=self.entity,
			name=self.name,
			group=self.group,
			tags=wandb_tags,
			dir=self._log_dir,
			config=OmegaConf.to_container(cfg, resolve=True),
		)

		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		self._video = (
			VideoRecorder(cfg, self._wandb)
			if self._wandb and cfg.save_video
			else None
		)

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self.group + '-' + str(self._seed) + '-' + str(identifier),
					type='model',
				)
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)

	def finish(self, agent=None):
		try:
			self.save_agent(agent)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.02f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	# def pprint_multitask(self, d, cfg):
	# 	"""Pretty-print evaluation metrics for multi-task training."""
	# 	print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	# 	dmcontrol_reward = []
	# 	metaworld_reward = []
	# 	metaworld_success = []
	# 	for k, v in d.items():
	# 		if '+' not in k:
	# 			continue
	# 		task = k.split('+')[1]
	# 		if task in TASK_SET['mt30'] and k.startswith('episode_reward'): # DMControl
	# 			dmcontrol_reward.append(v)
	# 			print(colored(f'  {task:<22}\tR: {v:.01f}', 'yellow'))
	# 		elif task in TASK_SET['mt80'] and task not in TASK_SET['mt30']: # Meta-World
	# 			if k.startswith('episode_reward'):
	# 				metaworld_reward.append(v)
	# 			elif k.startswith('episode_success'):
	# 				metaworld_success.append(v)
	# 				print(colored(f'  {task:<22}\tS: {v:.02f}', 'yellow'))
	# 	dmcontrol_reward = np.nanmean(dmcontrol_reward)
	# 	d['episode_reward+avg_dmcontrol'] = dmcontrol_reward
	# 	print(colored(f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}', 'yellow', attrs=['bold']))
	# 	if cfg.task == 'mt80':
	# 		metaworld_reward = np.nanmean(metaworld_reward)
	# 		metaworld_success = np.nanmean(metaworld_success)
	# 		d['episode_reward+avg_metaworld'] = metaworld_reward
	# 		d['episode_success+avg_metaworld'] = metaworld_success
	# 		print(colored(f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}', 'yellow', attrs=['bold']))
	# 		print(colored(f'  {"metaworld":<22}\tS: {metaworld_success:.02f}', 'yellow', attrs=['bold']))

	def log(self, d, category="train"):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			if category in {"train", "eval", "time"}:
				xkey = "step"
			elif category == "pretrain":
				xkey = "iteration"
			_d = dict()
			for k, v in d.items():

				# Change wandb logging titles for maniskill rl baselines common metrics
				if k == 'episode_reward_avg':
					k = 'reward' # train/reward, eval/reward
				elif k == 'episode_reward':
					k = 'return' # train/return, eval/return
				elif k == 'episode_success':
					k = 'success' # train/success, eval/success
				elif k == 'episode_fail' :
					k = 'fail' # train/fail, eval/fail
				elif k == 'episode_len':
					pass # train/episode_len, eval/episode_len
				elif k == 'step':
					pass # train/step, eval/step
				elif k == 'rollout_time':
					pass # time/rollout_time
				elif k == 'rollout_fps':
					pass # time/rollout_fps
				elif k == 'update_time':
					pass

				_d[category + "/" + k] = v
			self._wandb.log(_d, step=d[xkey])
		if category == "eval" and self._save_csv:
			keys = ["step", "episode_reward"]
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				self._log_dir / "eval.csv", header=keys, index=None
			)
		if category != 'time':
			self._print(d, category)
