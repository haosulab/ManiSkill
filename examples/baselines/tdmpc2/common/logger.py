import os
import datetime
import re
import numpy as np
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf
from mani_skill.utils.visualization.misc import tile_images
import wandb
from common import TASK_SET


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("return", "R", "float"),
	("success_once", "S", "float"),
	("fail_once", "F", "float"),
	("total_time", "T", "time"),
	# Added for maniskill rl baselines matrics
	# ("reward", "RET", "float"), 
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


def print_run(cfg): # this function has to be called after make_env
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
		("sim backend", cfg.env_type),
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
class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg, manager = None):
		self.cfg = cfg
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		self.save_video_local = cfg.save_video_local
		# Set up wandb
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		self.name = cfg.get("wandb_name", "none")
		self.group = cfg.get("wandb_group", "none")
		if not cfg.wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			self._wandb = None
		else:
			print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
			os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
			# Modified for Maniskill RL Baseline Logging Convention
			wandb_tags = cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"] + ["tdmpc2"]
			if cfg.setting_tag != 'none':
				wandb_tags += [cfg.setting_tag]
			self._wandb = wandb.init(
				project=self.project,
				entity=self.entity,
				name=self.name,
				group=self.group,
				tags=wandb_tags,
				dir=self._log_dir,
				config=OmegaConf.to_container(cfg, resolve=True),
			)
		
		self.wandb_videos = manager.list()
		self.lock = manager.Lock()

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp)
			if self._wandb:
				artifact = wandb.Artifact(
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

	def add_wandb_video(self, frames: np.ndarray): # (num_envs, num_frames, h, w, 3)
		with self.lock:
			if self.cfg.wandb and len(frames) > 0:
				self.wandb_videos.extend(frames)
            

	def log_wandb_video(self, step, fps=15, key='videos/eval_video'):
		with self.lock:
			if self.cfg.wandb and len(self.wandb_videos) > 0 :
				nrows = int(np.sqrt(len(self.wandb_videos)))
				wandb_video = np.stack(self.wandb_videos)
				wandb_video = wandb_video.transpose(1, 0, 2, 3, 4)
				wandb_video = [tile_images(rgbs, nrows=nrows) for rgbs in wandb_video]
				wandb_video = np.stack(wandb_video)
				self.wandb_videos[:] = []
				return self._wandb.log(
					{key: wandb.Video(wandb_video.transpose(0, 3, 1, 2), fps=fps, format='mp4')}, step=step
				)

	def log(self, d, category="train"):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			if category in {"train", "eval", "time"}:
				xkey = "step"
			elif category == "pretrain":
				xkey = "iteration"
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			self._wandb.log(_d, step=d[xkey])
		if category == "eval" and self._save_csv:
			keys = ["step", "return"]
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				self._log_dir / "eval.csv", header=keys, index=None
			)
		if category != 'time':
			self._print(d, category)
