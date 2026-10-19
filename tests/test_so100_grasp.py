import pytest
import sapien
import torch

try:  # pragma: no cover
    # On headless macOS, creating a Vulkan device inside a process spawned
    # without a window-server session can deadlock in GLFW's event loop unless
    # the app opts out of regular activation first. No-op on other platforms.
    import AppKit

    _app = AppKit.NSApplication.sharedApplication()
    _app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyProhibited)
except ImportError:
    pass

import gymnasium as gym

import mani_skill.envs  # noqa: F401


def oracle_grasp(seed: int = 0, num_squeeze_steps: int = 15):
    """Force the cube between the SO100 gripper jaws and check is_grasped.

    The jaws are closed first, the cube is teleported to the gripper TCP (i.e.
    forced between the jaws), and the close action is then held. A physically
    sane gripper drive stalls at a gentle grasp pressure so the cube stays
    grasped; a drive that keeps accumulating squeeze force ejects the cube and
    ``is_grasped`` never fires.
    """
    env = gym.make(
        "PickCubeSO100-v1",
        num_envs=1,
        obs_mode="state",
        sim_backend="cpu",
        render_backend="none",
    )
    try:
        env.reset(seed=seed)
        base = env.unwrapped
        act = torch.zeros(env.action_space.shape)
        act[..., -1] = -1.0  # close the gripper
        for _ in range(20):
            env.step(act)
        # force the cube between the closed jaws at the tool center point
        base.cube.set_pose(sapien.Pose(p=base.agent.tcp_pose.p[0].cpu().numpy()))
        grasped = torch.zeros(num_squeeze_steps, dtype=torch.bool)
        for i in range(num_squeeze_steps):
            obs, _, _, _, info = env.step(act)
            grasped[i] = info["is_grasped"][0]
        return grasped
    finally:
        env.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_so100_sustained_close_holds_grasp(seed):
    """A held close command must keep grasping the cube instead of ejecting it.

    Regression test for https://github.com/mani-skill/ManiSkill/issues/1427:
    the SO100 gripper drive used to accumulate squeeze force until the cube was
    ejected, so ``is_grasped`` (and with it all grasp-gated reward in
    ``PickCubeSO100-v1``) was unreachable.
    """
    try:
        grasped = oracle_grasp(seed=seed)
    except RuntimeError as e:
        if "vulkan" in str(e).lower() or "device" in str(e).lower():
            pytest.skip(f"no render/sim device available: {e}")
        raise
    # the grasp must be engaged for the majority of the sustained squeeze and
    # still engaged at the end (a drive that only touches the cube momentarily
    # before ejecting it does not make grasp-gated rewards learnable)
    assert grasped[:-2].float().mean() > 0.5, f"is_grasped flags: {grasped.tolist()}"
    assert bool(grasped[-1]), f"is_grasped at final step is False: {grasped.tolist()}"
