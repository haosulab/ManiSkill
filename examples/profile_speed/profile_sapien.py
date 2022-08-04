"""CartPole-Swingup."""

import numpy as np
from transforms3d.quaternions import axangle2quat
import gym
from gym.utils import seeding
from gym import spaces

import sapien.core as sapien
from sapien.core import Pose
from sapien.utils.viewer import Viewer
import PIL.Image as Image
from transforms3d.euler import euler2quat
import cv2


class TestSapienEnv(gym.Env):
    """Superclass for Sapien environments."""

    def __init__(self, control_freq, timestep):
        self.control_freq = control_freq  # alias: frame_skip in mujoco_py
        self.timestep = timestep

        self._engine = sapien.Engine()
        self._renderer = sapien.VulkanRenderer(offscreen_only=False)
        self._engine.set_renderer(self._renderer)
        self._scene = self._engine.create_scene()
        self._scene.set_timestep(timestep)
        self._scene.add_ground(-1.0)
        self._build_world()
        self.viewer = None
        self.seed()

    def _build_world(self):
        raise NotImplementedError()

    def _setup_viewer(self):
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #
    # Override gym functions
    # ---------------------------------------------------------------------------- #
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()  # release viewer

    def render(self, mode="human"):
        if self.viewer is None:
            self._setup_viewer()
        if mode == "human":
            self._scene.update_render()
            self.viewer.render()
        else:
            raise NotImplementedError("Unsupported render mode {}.".format(mode))

    # ---------------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------------- #
    def get_actor(self, name) -> sapien.ArticulationBase:
        all_actors = self._scene.get_all_actors()
        print(all_actors)
        actor = [x for x in all_actors if x.name == name]
        if len(actor) > 1:
            raise RuntimeError(f"Not a unique name for actor: {name}")
        elif len(actor) == 0:
            raise RuntimeError(f"Actor not found: {name}")
        return actor[0]

    def get_articulation(self, name) -> sapien.ArticulationBase:
        all_articulations = self._scene.get_all_articulations()
        articulation = [x for x in all_articulations if x.name == name]
        if len(articulation) > 1:
            raise RuntimeError(f"Not a unique name for articulation: {name}")
        elif len(articulation) == 0:
            raise RuntimeError(f"Articulation not found: {name}")
        return articulation[0]

    @property
    def dt(self):
        return self.timestep * self.control_freq


def create_cartpole(scene: sapien.Scene) -> sapien.Articulation:
    builder = scene.create_articulation_builder()

    base = builder.create_link_builder()
    base.set_name("base")

    cart = builder.create_link_builder(base)
    cart.set_name("cart")
    cart.set_joint_name("cart_joint")
    cart_half_size = np.array([0.04, 0.25, 0.125])
    cart.add_box_collision(half_size=cart_half_size, density=100)
    cart.add_box_visual(half_size=cart_half_size, color=[0.8, 0.6, 0.4])

    cart.set_joint_properties(
        "prismatic",
        limits=[[-5, 5]],
        pose_in_parent=sapien.Pose(
            p=[0, 0, 0], q=axangle2quat([0, 0, 1], np.deg2rad(90))
        ),
        pose_in_child=sapien.Pose(
            p=[0, 0, 0], q=axangle2quat([0, 0, 1], np.deg2rad(90))
        ),
    )

    rod = builder.create_link_builder(cart)
    rod.set_name("rod")
    rod.set_joint_name("rod_joint")
    rod_half_size = np.array([0.016, 0.016, 0.5])
    rod.add_box_collision(half_size=rod_half_size, density=100)
    rod.add_box_visual(half_size=rod_half_size, color=[0.8, 0.6, 0.4])

    # The x-axis of the joint frame is the rotation axis of a revolute joint.
    rod.set_joint_properties(
        "revolute",
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(
            p=[-(cart_half_size[0] + 1e-3), 0, (cart_half_size[2])],
            q=[1, 0, 0, 0],
        ),
        pose_in_child=sapien.Pose(
            p=[rod_half_size[0] + 1e-3, 0.0, -rod_half_size[2]],
            q=[1, 0, 0, 0],
        ),
    )

    cartpole = builder.build(fix_root_link=True)
    cartpole.set_name("cartpole")

    return cartpole


class CartPoleSwingUpEnv(TestSapienEnv):
    def __init__(self):
        super().__init__(control_freq=1, timestep=0.01)

        self.cartpole = self.get_articulation("cartpole")
        self.cart = self.cartpole.get_links()[1]
        self.pole = self.cartpole.get_links()[2]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[4], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=[1], dtype=np.float32
        )

        self.theta_threshold = np.deg2rad(12)
        self._setup_camera()

    # ---------------------------------------------------------------------------- #
    # Simulation world
    # ---------------------------------------------------------------------------- #
    def _build_world(self):
        # frictionless
        phy_mtl = self._scene.create_physical_material(0.0, 0.0, 0.0)
        self._scene.default_physical_material = phy_mtl
        create_cartpole(self._scene)
        self._setup_lighting()

    # ---------------------------------------------------------------------------- #
    # RL
    # ---------------------------------------------------------------------------- #
    def step(self, action):
        for _ in range(self.control_freq):
            self.cartpole.set_qf([action[0], 0])
            self._scene.step()

        obs = self._get_obs()

        x, theta = obs[0], obs[1]

        reward = "__TODO17__"

        if theta < -np.pi:
            theta += np.pi * 2
        if theta > np.pi:
            theta -= np.pi * 2
        success = -self.theta_threshold <= theta <= self.theta_threshold

        done = False

        return obs, reward, done, {"success": success}

    def reset(self):
        self.cartpole.set_qpos([0, np.pi])
        self.cartpole.set_qvel([0, 0])
        self._scene.step()
        return self._get_obs()

    def _get_obs(self):
        qpos = self.cartpole.get_qpos()
        qvel = self.cartpole.get_qvel()
        return np.hstack([qpos, qvel])

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #

    def _setup_lighting(self):
        self._scene.set_ambient_light([0.4, 0.4, 0.4])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self._scene.add_point_light([2, 2, 2], [1, 1, 1])
        self._scene.add_point_light([2, -2, 2], [1, 1, 1])
        self._scene.add_point_light([-2, 0, 2], [1, 1, 1])

    def _setup_viewer(self):
        self.viewer = Viewer(self._renderer)
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(-4, 0, 4)
        self.viewer.set_camera_rpy(0, -0.7, 0)
        self.viewer.window.set_camera_parameters(near=0.01, far=100, fovy=1)
        self.viewer.focus_entity(self.cart)

    def _setup_camera(self):
        self.camera = self._scene.add_mounted_camera(
            name="main_camera",
            actor=self.cart,
            pose=sapien.Pose([-3, 0, 0]),
            width=84,
            height=84,
            fovy=1,
            near=0.01,
            far=100,
        )

    def render(self, mode="human"):
        if mode == "human":
            super().render()
            self.camera.take_picture()
            rgba = self.camera.get_float_texture("Color")
        else:
            self._scene.update_render()
            self.camera.take_picture()
            rgba = self.camera.get_float_texture("Color")


def main():
    import time

    env = CartPoleSwingUpEnv()
    env.reset()
    sim_time = 0.0
    render_time = 0.0
    num_steps = 10000
    for step in range(num_steps):
        # env.render(mode="human")
        action = env.action_space.sample()
        t = time.time()
        obs, reward, done, info = env.step(action)
        sim_time += time.time() - t
        t = time.time()
        env.render("rgbd")
        render_time += time.time() - t
        if done:
            obs = env.reset()

    print("Num steps:", num_steps)
    print("Simulation time:", sim_time, "FPS:", num_steps / sim_time)
    print("Render time:", render_time, "FPS:", num_steps / render_time)

    env.close()


if __name__ == "__main__":
    main()
