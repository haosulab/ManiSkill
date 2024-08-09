import torch
import torch.nn.functional as F


def soft_ce(pred, target, cfg):
	"""Computes the cross entropy loss between predictions and soft targets."""
	pred = F.log_softmax(pred, dim=-1)
	target = two_hot(target, cfg)
	return -(target * pred).sum(-1, keepdim=True)


@torch.jit.script
def log_std(x, low, dif):
	return low + 0.5 * dif * (torch.tanh(x) + 1)


@torch.jit.script
def _gaussian_residual(eps, log_std):
	return -0.5 * eps.pow(2) - log_std


@torch.jit.script
def _gaussian_logprob(residual):
	return residual - 0.5 * torch.log(2 * torch.pi)


def gaussian_logprob(eps, log_std, size=None):
	"""Compute Gaussian log probability."""
	residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
	if size is None:
		size = eps.size(-1)
	return _gaussian_logprob(residual) * size


@torch.jit.script
def _squash(pi):
	return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)


def squash(mu, pi, log_pi):
	"""Apply squashing function."""
	mu = torch.tanh(mu)
	pi = torch.tanh(pi)
	log_pi -= _squash(pi).sum(-1, keepdim=True)
	return mu, pi, log_pi


@torch.jit.script
def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
	"""Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symlog(x)
	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
	bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()
	bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx.float()).unsqueeze(-1)
	soft_two_hot = torch.zeros(x.size(0), cfg.num_bins, device=x.device)
	soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
	soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
	return soft_two_hot


DREG_BINS = None


def two_hot_inv(x, cfg):
	"""Converts a batch of soft two-hot encoded vectors to scalars."""
	global DREG_BINS
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symexp(x)
	if DREG_BINS is None:
		DREG_BINS = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device)
	x = F.softmax(x, dim=-1)
	x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
	return symexp(x)
