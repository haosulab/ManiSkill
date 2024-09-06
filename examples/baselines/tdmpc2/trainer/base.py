from tdmpc2 import TDMPC2
from common.buffer import Buffer
from common.logger import Logger

class Trainer:
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, env, eval_env, agent, buffer, logger):
		self.cfg = cfg
		self.env = env
		self.eval_env = eval_env
		self.agent: TDMPC2 = agent
		self.buffer: Buffer = buffer
		self.logger: Logger = logger
		print('Architecture:', self.agent.model)
		print("Learnable parameters: {:,}".format(self.agent.model.total_params))

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		raise NotImplementedError

	def train(self):
		"""Train a TD-MPC2 agent."""
		raise NotImplementedError
