from typing import Dict, List, Optional, Tuple, Union
import os
import sys
import mani_skill2
import copy
import typing
import ctypes

from mani_skill2 import PACKAGE_ASSET_DIR

warp_path = os.path.join(os.path.dirname(mani_skill2.__file__), "..", "warp_maniskill")
warp_path = os.path.normpath(warp_path)
if warp_path not in sys.path:
    sys.path.append(warp_path)

# build warp if warp.so does not exist
from warp_maniskill.build_lib import build_path, build

dll = os.path.join(build_path, "bin/warp.so")

from collections import OrderedDict
import numpy as np
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2quat
import sapien.core as sapien
from warp_maniskill.mpm.mpm_simulator import (
    Simulator as MPMSimulator,
    Mesh as MPMMesh,
    DenseVolume as MPMVolume,
)
from warp_maniskill.mpm.mpm_model import MPMModelBuilder

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.envs.mpm.utils import actor2meshes, trimesh2sdf
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.utils.logging_utils import logger
import hashlib
import trimesh

import warp as wp


def task(meshes):
    if meshes is None or len(meshes) == 0:
        return None
    bbox = trimesh.util.concatenate(meshes).bounds
    length = np.max(bbox[1] - bbox[0])
    dx = min(0.01, length / 40)  # dx should at most be 0.01
    margin = max(
        dx * 3, 0.01
    )  # margin should be greater than 3 dx and at least be 0.01
    return trimesh2sdf(meshes, margin, dx)


class MPMBaseEnv(BaseEnv):
    # fmt: off
    SUPPORTED_OBS_MODES = ("none", "image")
    # fmt: on

    def __init__(
        self,
        *args,
        sim_freq=500,
        mpm_freq=2000,
        max_particles=65536,
        **kwargs,
    ):

        if not os.path.isfile(dll):

            class ARGS:
                verbose = False
                mode = "release"
                cuda_path = None
                verify_fp = False

            build(ARGS)

        wp.init()

        if "shader_dir" in kwargs:
            logger.warning("`shader_dir` is ignored for soft-body environments.")
            kwargs.pop("shader_dir")
        shader_dir = os.path.join(os.path.dirname(__file__), "shader", "point")

        self.sim_crashed = False

        self._mpm_step_per_sapien_step = mpm_freq // sim_freq
        self._mpm_dt = 1 / mpm_freq
        self.max_particles = max_particles

        self.sdf_cache = os.path.join(
            os.path.dirname(__file__), self.__class__.__name__ + ".sdf"
        )

        super().__init__(*args, shader_dir=shader_dir, **kwargs)

    # ---------------------------------------------------------------------------- #
    # Setup
    # ---------------------------------------------------------------------------- #

    def reconfigure(self):
        self._clear()

        self._setup_scene()
        self._load_agent()
        self._load_actors()
        self._load_articulations()
        self._setup_cameras()
        self._setup_lighting()

        self._setup_mpm()
        self._setup_render_particles()

        if self._viewer is not None:
            self._setup_viewer()

        # Cache actors and articulations
        self._actors = self.get_actors()
        self._articulations = self.get_articulations()

        self._load_background()

    def _load_actors(self):
        self._scene.add_ground(altitude=0.0, render=False)
        if self.bg_name is None:
            b = self._scene.create_actor_builder()
            b.add_visual_from_file(str(PACKAGE_ASSET_DIR / "maniskill2-scene-2.glb"))
            b.build_kinematic()

    def _get_coupling_actors(
        self,
    ):
        return self.agent.robot.get_links()

    def _setup_mpm(self):
        self.model_builder = MPMModelBuilder()
        self.model_builder.set_mpm_domain(
            domain_size=[0.5, 0.5, 0.5], grid_length=0.005
        )
        self.model_builder.reserve_mpm_particles(count=self.max_particles)

        self._setup_mpm_bodies()

        self.mpm_simulator = MPMSimulator(device="cuda")
        self.mpm_model = self.model_builder.finalize(device="cuda")
        self.mpm_model.gravity = np.array((0.0, 0.0, -9.81), dtype=np.float32)
        self.mpm_model.struct.ground_normal = wp.vec3(0.0, 0.0, 1.0)
        self.mpm_model.struct.particle_radius = 0.005
        self.mpm_states = [
            self.mpm_model.state() for _ in range(self._mpm_step_per_sapien_step + 1)
        ]

    def _setup_mpm_bodies(self):
        # convert actors to meshes
        self._coupled_actors = []
        actor_meshes = []
        actor_primitives = []
        for actor in self._get_coupling_actors():
            visual = False
            if isinstance(actor, tuple) or isinstance(actor, list):
                type = actor[1]
                actor = actor[0]
                if type == "visual":
                    visual = True

            meshes, primitives = actor2meshes(
                actor, visual=visual, return_primitives=True
            )
            if meshes or primitives:
                self._coupled_actors.append(actor)
                actor_meshes.append(meshes)
                actor_primitives.append(primitives)

        if not self._coupled_actors:
            return

        # compute signature for current meshes
        hash = hashlib.sha256()
        hash.update(bytes("v2", encoding="utf-8"))
        for meshes in actor_meshes:
            for m in meshes:
                hash.update(m.vertices.tobytes())
                hash.update(m.faces.tobytes())
        signature = hash.digest()
        sdfs = None

        # load cached sdfs or recompute sdfs
        import pickle

        if os.path.isfile(self.sdf_cache):
            with open(self.sdf_cache, "rb") as f:
                cache = pickle.load(f)
                if "signature" in cache and cache["signature"] == signature:
                    logger.info("load sdfs from file")
                    sdfs = cache["sdfs"]
                    if len(sdfs) != len(actor_meshes):
                        sdfs = None

        if sdfs is None:
            import tqdm
            from multiprocessing import Pool

            print("generating cached SDF volumes")
            with Pool(8) as p:
                sdfs = list(
                    tqdm.tqdm(
                        p.imap(task, actor_meshes),
                        total=len(actor_meshes),
                    )
                )

            with open(self.sdf_cache, "wb") as f:
                meshes = [
                    [(np.array(m.vertices), np.array(m.faces)) for m in ms]
                    for ms in actor_meshes
                ]
                pickle.dump({"signature": signature, "sdfs": sdfs, "meshes": meshes}, f)

        # convert sdfs to dense volumes
        for actor, sdf, meshes, primitives in zip(
            self._coupled_actors, sdfs, actor_meshes, actor_primitives
        ):
            id = self.model_builder.add_body(
                origin=wp.transform((0.0, 0.0, 0.0), wp.quat_identity())
            )
            if sdf is not None:
                data = np.concatenate(
                    [sdf["normal"], sdf["sdf"].reshape(sdf["sdf"].shape + (1,))], -1
                )
                volume = MPMVolume(
                    data,
                    sdf["position"],
                    sdf["scale"],
                    mass=1,
                    I=np.eye(3),
                    com=np.zeros(3),
                )
                self.model_builder.add_shape_dense_volume(id, volume=volume)
            for type, params, pose in primitives:
                if type == "box":
                    self.model_builder.add_shape_box(
                        id,
                        pos=tuple(pose.p),
                        rot=(pose.q[3], *pose.q[:3]),
                        hx=params[0],
                        hy=params[1],
                        hz=params[2],
                    )
                if type == "capsule":
                    self.model_builder.add_shape_capsule(
                        id,
                        pos=tuple(pose.p),
                        rot=(pose.q[3], *pose.q[:3]),
                        radius=params[0],
                        half_width=params[1],
                    )

            cmass = actor.cmass_local_pose
            R = quat2mat(cmass.q)
            self.model_builder.set_body_mass(
                id, actor.mass, R @ np.diag(actor.inertia) @ R.T, cmass.p
            )

    def _setup_render_particles(self):
        N = self.mpm_model.max_n_particles
        scales = np.ones(N, dtype=np.float32) * 0.005
        colors = np.zeros((N, 4))
        colors[:, :3] = self.mpm_model.mpm_particle_colors

        self.pcd = self._scene.add_particle_entity(np.zeros((N, 3), dtype=np.float32))
        self.pcd.visual_body.set_attribute("color", colors.astype(np.float32))
        self.pcd.visual_body.set_attribute("scale", scales)

        self.vertices_ptr = sapien.pysapien.dlpack.dl_ptr(
            self.pcd.visual_body.dl_vertices
        )
        self.vertices_shape = sapien.pysapien.dlpack.dl_shape(
            self.pcd.visual_body.dl_vertices
        )

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(1.0, 0.0, 1.2)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _clear(self):
        super()._clear()

        self.vertices_ptr = None
        self.vertices_shape = None
        self.mpm_model = None
        self.mpm_states = None
        self.pcd = None

    # ---------------------------------------------------------------------------- #
    # Initialization
    # ---------------------------------------------------------------------------- #

    def reset(self, *args, **kwargs):
        self.sim_crashed = False
        return super().reset(*args, **kwargs)

    def initialize_episode(self):
        super().initialize_episode()
        self._initialize_mpm()
        self._initialize_render_particles()

    def _initialize_mpm(self):
        self.model_builder.clear_particles()

        E = 1e5
        nu = 0.3
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

        # 0 for von-mises, 1 for drucker-prager
        type = 1

        # von-mises
        ys = 1e4

        # drucker-prager
        friction_angle = 1.0
        cohesion = 0.1

        x = 0.05
        y = 0.05
        z = 0.05
        cell_x = 0.005
        self.model_builder.add_mpm_grid(
            pos=(-0.0, 0.0, 0.05),
            vel=(0.0, 0.0, 0.0),
            dim_x=int(x // cell_x),
            dim_y=int(y // cell_x),
            dim_z=int(z // cell_x),
            cell_x=cell_x,
            cell_y=cell_x,
            cell_z=cell_x,
            density=3.0e3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.0),
            type=type,
            jitter=True,
            placement_x="center",
            placement_y="center",
            placement_z="start",
            color=(0.65237011, 0.14198029, 0.02201299),
            random_state=self._episode_rng,
        )
        self.model_builder.init_model_state(self.mpm_model, self.mpm_states)
        self.mpm_model.struct.static_ke = 100.0
        self.mpm_model.struct.static_kd = 0.0
        self.mpm_model.struct.static_mu = 1.0
        self.mpm_model.struct.static_ka = 0.0

        self.mpm_model.struct.body_ke = 100.0
        self.mpm_model.struct.body_kd = 0.0
        self.mpm_model.struct.body_mu = 1.0
        self.mpm_model.struct.body_ka = 0.0

        self.mpm_model.adaptive_grid = False

        self.mpm_model.struct.body_sticky = 1
        self.mpm_model.struct.ground_sticky = 1
        self.mpm_model.particle_contact = True
        self.mpm_model.grid_contact = True

    def _initialize_render_particles(self):
        n = self.mpm_model.struct.n_particles

        self.pcd.visual_body.set_rendered_point_count(n)

        buffer = np.empty((n, 4), dtype=np.float32)
        buffer[:, 0] = self.mpm_model.struct.particle_radius * 1.0
        buffer[:, 1:] = self.mpm_model.mpm_particle_colors

        # transfer
        scale_colors = wp.array(buffer, dtype=float, device="cuda")
        width = spitch = 16
        dpitch = self.vertices_shape[1] * 4
        height = n
        wp.context.runtime.core.memcpy2d_d2d(
            ctypes.c_void_p(self.vertices_ptr + 12),
            ctypes.c_size_t(dpitch),
            ctypes.c_void_p(scale_colors.ptr),
            ctypes.c_size_t(spitch),
            ctypes.c_size_t(width),
            ctypes.c_size_t(height),
        )
        wp.synchronize()

    # ---------------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------------- #
    def get_obs(self):
        mpm_state = self.get_mpm_state()

        if self._obs_mode == "particles":
            obs = OrderedDict(
                particles=mpm_state,
                agent=self._get_obs_agent(),
                extra=self._get_obs_extra(),
            )
        else:
            obs = super().get_obs()

        self._last_obs = obs
        return obs

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        return obs

    def copy_array_to_numpy(self, array, length):
        dtype = np.dtype(array.__cuda_array_interface__["typestr"])
        shape = array.__cuda_array_interface__["shape"]
        assert shape[0] >= length
        result = np.empty((length,) + shape[1:], dtype=dtype)
        assert result.strides == array.__cuda_array_interface__["strides"]

        wp.context.runtime.core.memcpy_d2h(
            ctypes.c_void_p(result.__array_interface__["data"][0]),
            ctypes.c_void_p(array.ptr),
            ctypes.c_size_t(result.strides[0] * length),
        )

        return result

    # -------------------------------------------------------------------------- #
    # Simulation state
    # -------------------------------------------------------------------------- #
    def get_mpm_state(self):
        n = self.mpm_model.struct.n_particles

        return OrderedDict(
            x=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_q, n),
            v=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_qd, n),
            F=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_F, n),
            C=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_C, n),
            vc=self.copy_array_to_numpy(
                self.mpm_states[0].struct.particle_volume_correction, n
            ),
        )

    def set_mpm_state(self, state):
        self.mpm_states[0].struct.particle_q.assign(state["x"])
        self.mpm_states[0].struct.particle_qd.assign(state["v"])
        self.mpm_states[0].struct.particle_F.assign(state["F"])
        self.mpm_states[0].struct.particle_C.assign(state["C"])
        self.mpm_states[0].struct.particle_volume_correction.assign(state["vc"])

    def get_sim_state(self):
        sapien_state = super().get_sim_state()
        mpm_state = self.get_mpm_state()
        return OrderedDict(sapien=sapien_state, mpm=mpm_state)

    def set_sim_state(self, state):
        super().set_sim_state(state["sapien"])
        self.set_mpm_state(state["mpm"])

    def get_state(self):
        """Get environment state. Override to include task information (e.g., goal)"""
        # with sapien.ProfilerBlock("get_state"):
        ret_state = []
        sim_state = self.get_sim_state()
        for key, value in sim_state.items():
            if key == "mpm":
                for key, value in value.items():
                    ret_state.append(value.reshape((1, -1)).squeeze())
            else:
                ret_state.append(value.reshape((1, -1)).squeeze())

        ret_state = np.concatenate(ret_state)
        return ret_state

    @property
    def n_particles(self):
        return self.mpm_model.struct.n_particles

    def set_state(self, state: np.ndarray):
        """Set environment state. Override to include task information (e.g., goal)"""
        sim_state = OrderedDict()
        mpm_state = OrderedDict()
        n = self.mpm_model.struct.n_particles

        sim_state["sapien"] = state[: -n * 25]
        mpm_state["x"] = state[-n * 25 : -n * 22].reshape((n, 3))
        mpm_state["v"] = state[-n * 22 : -n * 19].reshape((n, 3))
        mpm_state["F"] = state[-n * 19 : -n * 10].reshape((n, 3, 3))
        mpm_state["C"] = state[-n * 10 : -n].reshape((n, 3, 3))
        mpm_state["vc"] = state[-n:].reshape((n,))
        sim_state["mpm"] = mpm_state

        return self.set_sim_state(sim_state)

    # -------------------------------------------------------------------------- #
    # Visualization
    # -------------------------------------------------------------------------- #
    def update_render(self):
        width = spitch = 12
        dpitch = self.vertices_shape[1] * 4
        height = self.mpm_model.struct.n_particles
        wp.context.runtime.core.memcpy2d_d2d(
            ctypes.c_void_p(self.vertices_ptr),
            ctypes.c_size_t(dpitch),
            ctypes.c_void_p(self.mpm_states[0].struct.particle_q.ptr),
            ctypes.c_size_t(spitch),
            ctypes.c_size_t(width),
            ctypes.c_size_t(height),
        )
        wp.synchronize()
        super().update_render()

    def _setup_lighting(self):
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=True, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #

    def sync_actors(self):
        if not self._coupled_actors:
            return

        body_q = np.empty((len(self._coupled_actors), 7), dtype=np.float32)
        body_qd = np.empty((len(self._coupled_actors), 6), dtype=np.float32)
        for i, a in enumerate(self._coupled_actors):
            pose = a.pose
            p = pose.p
            q = pose.q
            body_q[i, :3] = p
            # different quaternion convention
            body_q[i, 3:6] = q[1:]
            body_q[i, 6] = q[0]

            body_qd[i, :3] = a.angular_velocity
            body_qd[i, 3:] = a.velocity

        for s in self.mpm_states:
            s.body_q.assign(body_q)
            s.body_qd.assign(body_qd)

    def sync_actors_prepare(self):
        if not self._coupled_actors:
            return

        self.body_q = np.empty((len(self._coupled_actors), 7), dtype=np.float32)
        self.body_qd = np.empty((len(self._coupled_actors), 6), dtype=np.float32)
        for i, a in enumerate(self._coupled_actors):
            pose = a.pose
            p = pose.p
            q = pose.q
            self.body_q[i, :3] = p
            # different quaternion convention
            self.body_q[i, 3:6] = q[1:]
            self.body_q[i, 6] = q[0]

            self.body_qd[i, :3] = a.angular_velocity
            self.body_qd[i, 3:] = a.velocity

    def sync_actors_state(self, state):
        state.body_q.assign(self.body_q)
        state.body_qd.assign(self.body_qd)

    def step(self, action: Union[None, np.ndarray, Dict]):
        if not self.sim_crashed:
            obs, rew, done, info = super().step(action)
            info["crashed"] = False
            return obs, rew, done, info

        logger.warn("simulation has crashed!")
        info = self.get_info(obs=self._last_obs)
        info["crashed"] = True
        return self._last_obs, -10, True, info

    def step_action(self, action: np.ndarray):
        if action is None:
            pass
        elif isinstance(action, np.ndarray):
            self.agent.set_action(action)
        elif isinstance(action, dict):
            if action["control_mode"] != self.agent.control_mode:
                self.agent.set_control_mode(action["control_mode"])
            self.agent.set_action(action["action"])
        else:
            raise TypeError(type(action))

        for _ in range(self._sim_steps_per_control):
            self.sync_actors()
            for mpm_step in range(self._mpm_step_per_sapien_step):
                self.mpm_simulator.simulate(
                    self.mpm_model,
                    self.mpm_states[mpm_step],
                    self.mpm_states[mpm_step + 1],
                    self._mpm_dt,
                )

            self.agent.before_simulation_step()

            # apply wrench
            tfs = [s.ext_body_f.numpy() for s in self.mpm_states[:-1]]
            tfs = np.mean(tfs, 0)

            if np.isnan(tfs).any():
                self.sim_crashed = True
                return

            if self.mpm_states[-1].struct.error.numpy()[0] == 1:
                self.sim_crashed = True
                return

            for actor, tf in zip(self._coupled_actors, tfs):
                if actor.type not in ["kinematic", "static"]:
                    actor.add_force_torque(tf[3:], tf[:3])

            self._scene.step()
            self.mpm_states = [self.mpm_states[-1]] + self.mpm_states[
                :-1
            ]  # rotate states

    # -------------------------------------------------------------------------- #
    # Utils
    # -------------------------------------------------------------------------- #
    def _get_bbox(self, percentile=None):
        x = self.mpm_states[0].struct.particle_q.numpy()
        if percentile is None:
            return np.array([x.min(0), x.max(0)])
        assert percentile < 50
        return np.array(
            [np.percentile(x, percentile, 0), np.percentile(x, 100 - percentile, 0)]
        )

    def _add_draw_box(self, bbox):
        from sapien.core import renderer as R

        context: R.Context = self._renderer._internal_context
        # fmt:off
        lines = context.create_line_set(
            [
                -1, -1, -1, -1, -1, 1,
                -1, -1, -1, -1, 1, -1,
                -1, 1, 1, -1, -1, 1,
                -1, 1, 1, -1, 1, -1,
                1, -1, -1, 1, -1, 1,
                1, -1, -1, 1, 1, -1,
                1, 1, 1, 1, -1, 1,
                1, 1, 1, 1, 1, -1,
                1, 1, 1, -1, 1, 1,
                1, -1, 1, -1, -1, 1,
                1, 1, -1, -1, 1, -1,
                1, -1, -1, -1, -1, -1,
            ],
            [1, 1, 1, 1] * 24,
        )
        # fmt:on
        lineset: R.LineSetObject = (
            self._scene.get_renderer_scene()._internal_scene.add_line_set(lines)
        )

        lineset.set_scale((bbox[1] - bbox[0]) / 2)
        lineset.set_position((bbox[1] + bbox[0]) / 2)
        return lineset

    def _remove_draw_box(self, lineset):
        self._scene.get_renderer_scene()._internal_scene.remove_node(lineset)

    def render(self, mode="human", draw_box=False):
        if draw_box:
            bbox = self._get_bbox(5)
            box = self._add_draw_box(bbox)
        img = super().render(mode)
        if draw_box:
            self._remove_draw_box(box)
        return img
