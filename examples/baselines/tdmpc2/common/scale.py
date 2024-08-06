import torch


class RunningScale:
	"""Running trimmed scale estimator."""

	def __init__(self, cfg):
		self.cfg = cfg
		self._value = torch.ones(1, dtype=torch.float32, device=torch.device('cuda'))
		self._percentiles = torch.tensor([5, 95], dtype=torch.float32, device=torch.device('cuda'))

	def state_dict(self):
		return dict(value=self._value, percentiles=self._percentiles)

	def load_state_dict(self, state_dict):
		self._value.data.copy_(state_dict['value'])
		self._percentiles.data.copy_(state_dict['percentiles'])

	@property
	def value(self):
		return self._value.cpu().item()

	def _percentile(self, x):
		x_dtype, x_shape = x.dtype, x.shape
		x = x.view(x.shape[0], -1)
		in_sorted, _ = torch.sort(x, dim=0)
		positions = self._percentiles * (x.shape[0]-1) / 100
		floored = torch.floor(positions)
		ceiled = floored + 1
		ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
		weight_ceiled = positions-floored
		weight_floored = 1.0 - weight_ceiled
		d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
		d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
		return (d0+d1).view(-1, *x_shape[1:]).type(x_dtype)

	def update(self, x):
		percentiles = self._percentile(x.detach())
		value = torch.clamp(percentiles[1] - percentiles[0], min=1.)
		self._value.data.lerp_(value, self.cfg.tau)

	def __call__(self, x, update=False):
		if update:
			self.update(x)
		return x * (1/self.value)

	def __repr__(self):
		return f'RunningScale(S: {self.value})'
