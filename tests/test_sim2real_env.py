"""Regression test for GitHub issue #1498: Sim2RealEnv.reset() recursed forever
when the simulation env was wrapped by gymnasium wrappers.

Since #1308, ``base_sim_env`` was a property re-resolving ``sim_env.unwrapped`` on
every access. During ``reset()``/``step()`` the last wrapper's ``.env`` is swapped
with a stand-in whose ``unwrapped`` returns the Sim2RealEnv itself, so the property
resolved to the Sim2RealEnv and ``get_obs()`` called itself endlessly.
The fix resolves ``sim_env.unwrapped`` once in ``__init__`` and caches it.
"""

import pytest

import gymnasium as gym

import mani_skill.envs  # noqa: F401 - registers environments
from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv


class StubRealAgent(BaseRealAgent):
    """Minimal real agent: good enough to prove reset() gets past the recursion."""

    def capture_sensor_data(self, sensor_names=None):
        pass

    def get_qpos(self):
        import torch

        return torch.zeros(self._sim_agent.robot.qpos.shape[0])

    def get_qvel(self):
        import torch

        return torch.zeros(self._sim_agent.robot.qpos.shape[0])


class TestSim2RealEnv(Sim2RealEnv):
    # PickCube's _get_obs_extra reads info["is_grasped"], which a real environment
    # only has via a perception pipeline; not relevant to the recursion bug.
    def _get_obs_extra(self, info):
        return {}


def test_sim2real_env_reset_with_wrappers_does_not_recurse():
    class IdentityWrapper(gym.Wrapper):
        """Any gym wrapper around the sim env triggers the wrapper-swap path."""

        pass

    sim = gym.make("PickCube-v1", obs_mode="rgb", sim_backend="physx_cpu", num_envs=1)
    sim = IdentityWrapper(sim)
    real = TestSim2RealEnv(
        sim_env=sim,
        agent=StubRealAgent(),
        real_reset_function=lambda env, seed=None, options=None: None,
        skip_data_checks=True,
    )
    try:
        obs, info = real.reset()
    except RecursionError as e:
        pytest.fail(f"Sim2RealEnv.reset() recursed with a wrapped sim env: {e}")
    except NotImplementedError:
        # Past the recursion point, computing rgb observations requires real camera
        # hardware (BaseRealAgent.get_sensor_data), which a stub agent cannot provide.
        # Reaching that hook is the expected behavior fixed by #1498.
        pass
