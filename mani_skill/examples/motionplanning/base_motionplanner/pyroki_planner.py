"""Pyroki-based motion planning backend.

This is a cross-platform alternative to `mplib` for the example motion
planning solvers. Unlike mplib (which does not build on macOS / Windows),
this backend is pure Python on top of JAX and therefore runs everywhere
JAX runs, including macOS (CPU).

The backend solves inverse kinematics with PyRoki
(https://github.com/chungmin99/pyroki) and expands the solution into a
time-parameterized joint-space path that is interface-compatible with the
results returned by `mplib.Planner` (`{"status", "position", "velocity"}`),
so solvers built on `BaseMotionPlanningSolver` can switch backends without
code changes.

Limitations compared to mplib:
- point-cloud obstacles are accepted but currently ignored (no collision-aware
  planning yet), and plans are straight-line joint-space interpolations rather
  than RRT paths.
"""

from typing import List, Optional

import numpy as np
import transforms3d

import jax
import jax.numpy as jnp
import jaxlie
import jaxls

try:
    import pyroki as pk

    _HAS_PYROKI = True
except ImportError:  # pragma: no cover
    _HAS_PYROKI = False


def _pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """[p(3), q(4, wxyz)] -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = transforms3d.quaternions.quat2mat(np.asarray(pose7[3:], dtype=np.float64))
    T[:3, 3] = np.asarray(pose7[:3], dtype=np.float64)
    return T


def _solve_ik_jax(
    robot: "pk.Robot",
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    rest_pose: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=200.0,
            ori_weight=50.0,
        ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose[None],
            jnp.array([0.02]),
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]


class PyrokiPlanner:
    """Minimal drop-in replacement for the subset of `mplib.Planner` used by
    the ManiSkill example motion planning solvers."""

    def __init__(self, urdf_path: str, move_group: str = "eef", joint_names: Optional[List[str]] = None):
        import yourdfpy

        self.move_group = move_group
        urdf = yourdfpy.URDF.load(urdf_path)
        self.robot = pk.Robot.from_urdf(urdf)
        # only the actuated joints are decision variables in pyroki
        self.joint_names = list(self.robot.joints.actuated_names)
        if move_group in self.robot.links.names:
            self.target_link_index = self.robot.links.names.index(move_group)
        else:
            # fall back to the last link
            self.target_link_index = len(self.robot.links.names) - 1

        # like mplib, only plan over the actuated joints on the kinematic chain
        # from the robot base to the move-group link (gripper finger joints are
        # handled separately by the solvers via the gripper action).
        chain_joint_names = self._chain_joint_names(urdf, move_group)
        chain_pk_indices = [
            self.joint_names.index(n) for n in chain_joint_names if n in self.joint_names
        ]

        # joint-space limits (same attributes mplib exposes and that
        # BaseMotionPlanningSolver scales in setup_planner)
        self.joint_vel_limits = np.asarray(self.robot.joints.velocity_limits)[chain_pk_indices]
        self.joint_acc_limits = np.asarray(self.robot.joints.velocity_limits)[chain_pk_indices] / 0.1

        # mapping from the caller's joint ordering (SAPIEN active joints) to
        # pyroki's actuated joint ordering, used to reorder IK solutions.
        self._user_to_pk = None
        if joint_names is not None:
            self._user_joint_names = list(joint_names)
            chain_pk_set = set(chain_pk_indices)
            self._user_to_pk = [
                self.joint_names.index(n) if n in self.joint_names and self.joint_names.index(n) in chain_pk_set else -1
                for n in joint_names
            ]

    def _chain_joint_names(self, urdf, move_group: str) -> List[str]:
        """Names of the actuated joints on the kinematic chain from the root
        link to `move_group`, ordered from root to leaf."""
        if move_group not in urdf.link_map:
            return list(self.joint_names)
        parent_of_link = {}
        for joint in urdf.joint_map.values():
            parent_of_link[joint.child] = joint
        names: List[str] = []
        link = move_group
        while link in parent_of_link:
            joint = parent_of_link[link]
            if joint.type != "fixed":
                names.append(joint.name)
            link = joint.parent
        return list(reversed(names))

    # -- interface parity with mplib.Planner ---------------------------------
    def set_base_pose(self, base_pose: np.ndarray):
        # Stored for API parity; planning is done wrt the robot base frame,
        # which matches the current mplib solvers (base at origin).
        self.base_pose = np.asarray(base_pose)

    def update_point_cloud(self, point_cloud: np.ndarray):
        # Accepted for API parity; obstacle-aware planning is not implemented yet.
        self.point_cloud = np.asarray(point_cloud)

    def _to_user_qpos(self, cfg: np.ndarray, current_qpos: np.ndarray) -> np.ndarray:
        """Reorder a pyroki joint configuration into the caller's joint ordering.
        Joints pyroki does not model (fixed or mimic joints, e.g. the second
        finger of a parallel gripper) are dropped: mimic joints are driven by
        SAPIEN itself, matching the mplib behavior where plans only contain the
        move-group joints."""
        if self._user_to_pk is None:
            return cfg
        mapped = [
            (user_idx, pk_idx)
            for user_idx, pk_idx in enumerate(self._user_to_pk)
            if pk_idx >= 0
        ]
        out = np.zeros(len(mapped))
        for i, (user_idx, pk_idx) in enumerate(mapped):
            out[i] = cfg[pk_idx]
        return out

    def _to_pk_qpos(self, user_qpos: np.ndarray) -> np.ndarray:
        """Reorder the caller's joint configuration into pyroki's actuated-joint
        ordering; joints pyroki does not model fall back to the middle of their
        limits (pyroki's default configuration)."""
        n_pk = self.robot.joints.num_actuated_joints
        rest_pk = np.asarray(self.robot.joint_var_cls.default_factory(), dtype=np.float64).reshape(-1)
        assert rest_pk.shape == (n_pk,)
        if self._user_to_pk is None:
            return rest_pk
        for user_idx, pk_idx in enumerate(self._user_to_pk):
            if pk_idx >= 0 and user_idx < len(user_qpos):
                rest_pk[pk_idx] = user_qpos[user_idx]
        return rest_pk

    def _plan(self, target_pose_7, qpos, time_step) -> dict:
        target_position = np.asarray(target_pose_7[:3], dtype=np.float64)
        target_wxyz = np.asarray(target_pose_7[3:], dtype=np.float64)  # wxyz, same as sapien

        # express the target in the robot base frame, matching mplib's
        # wrt_world=True planning against the configured base pose.
        T_base = _pose7_to_matrix(self.base_pose)
        T_world = _pose7_to_matrix(np.concatenate([target_position, target_wxyz]))
        T_rel = np.linalg.inv(T_base) @ T_world
        target_position = T_rel[:3, 3]
        target_wxyz = transforms3d.quaternions.mat2quat(T_rel[:3, :3])

        rest_pose = np.asarray(qpos, dtype=np.float64)
        rest_pk = self._to_pk_qpos(rest_pose)
        cfg = _solve_ik_jax(
            self.robot,
            jnp.array(self.target_link_index, dtype=jnp.int32),
            jnp.array(target_wxyz),
            jnp.array(target_position),
            jnp.array(rest_pk),
        )
        cfg = np.asarray(jax.device_get(cfg))

        q_goal = self._to_user_qpos(cfg, qpos)
        q_start = self._to_user_qpos(self._to_pk_qpos(qpos), qpos)

        # time-parameterize as a straight-line joint-space path that respects
        # the (scaled) per-joint velocity limits, sampled at the env control
        # timestep. The 1.5x factor accounts for the smoothstep easing below,
        # whose peak speed is 1.5x the average speed.
        dq = q_goal - q_start
        vel_limits = np.abs(self.joint_vel_limits)
        ratios = np.abs(dq) / np.where(vel_limits > 1e-6, vel_limits * time_step, 1e-6)
        n_steps = max(1, int(np.ceil(1.5 * float(np.max(ratios)) / 0.9)))
        t = np.linspace(0.0, 1.0, n_steps + 1)[1:]
        # smoothstep easing so the path starts and ends at rest (the pd
        # controller then settles on the final waypoint like with mplib paths).
        alphas = t * t * (3.0 - 2.0 * t)
        positions = np.stack([q_start + a * (q_goal - q_start) for a in alphas])
        # hold the final waypoint for a few extra steps so the (slightly lagging)
        # pd joint-position controller settles on the target configuration.
        n_hold = min(25, max(5, n_steps // 5))
        positions = np.vstack([positions, np.repeat(positions[-1][None], n_hold, axis=0)])
        velocities = np.zeros_like(positions)
        if len(positions) > 1:
            velocities[:-1] = (positions[1:] - positions[:-1]) / time_step
            velocities[-1] = velocities[-2]
        return {
            "status": "Success",
            "position": positions,
            "velocity": velocities,
        }

    def plan_qpos_to_pose(self, target_pose_7, qpos, time_step, use_point_cloud=False, **kwargs) -> dict:
        return self._plan(target_pose_7, np.asarray(qpos).reshape(-1), time_step)

    def plan_screw(self, target_pose_7, qpos, time_step, use_point_cloud=False, **kwargs) -> dict:
        # screw motions are not implemented; fall back to an IK-based plan.
        return self._plan(target_pose_7, np.asarray(qpos).reshape(-1), time_step)
