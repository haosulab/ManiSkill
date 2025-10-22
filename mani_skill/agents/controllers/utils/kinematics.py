"""
Code for kinematics utilities on CPU/GPU
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from typing import Optional, Union

from mani_skill.utils.geometry import rotation_conversions

try:
    import pytorch_kinematics as pk
except ImportError:
    raise ImportError(
        "pytorch_kinematics_ms not installed. Install with pip install pytorch_kinematics_ms"
    )
import torch
from lxml import etree as ET
from sapien.wrapper.pinocchio_model import PinocchioModel

from mani_skill.utils import common
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.articulation_joint import ArticulationJoint
from mani_skill.utils.structs.pose import Pose

# currently fast_kinematics has some bugs on some systems so we use the slower pytorch kinematics package instead.
# try:
#     import fast_kinematics
# except:
#     # not all systems support the fast_kinematics package at the moment
#     fast_kinematics = None


class Kinematics:
    def __init__(
        self,
        urdf_path: str,
        end_link_name: str,
        articulation: Articulation,
        active_joint_indices: torch.Tensor,
    ):
        """
        Initialize the kinematics solver. It will be run on whichever device the articulation is on.

        Args:
            urdf_path (str): path to the URDF file
            end_link_name (str): name of the end-effector link
            articulation (Articulation): the articulation object
            active_joint_indices (torch.Tensor): indices of the active joints that can be controlled
        """

        # NOTE (arth): urdf path with feasible kinematic chain. may not be same urdf used to
        #   build the sapien articulation (e.g. sapien articulation may have added joints for
        #   mobile base which should not be used in IK)
        self.urdf_path = urdf_path
        self.end_link = articulation.links_map[end_link_name]

        self.articulation = articulation
        self.device = articulation.device

        self.active_joint_indices = active_joint_indices

        # note that everything past the end-link is ignored. Any joint whose ancestor is self.end_link is ignored
        cur_link = self.end_link
        active_ancestor_joints: list[ArticulationJoint] = []
        while cur_link is not None:
            if cur_link.joint.active_index is not None:
                active_ancestor_joints.append(cur_link.joint)
            cur_link = cur_link.joint.parent_link
        active_ancestor_joints = active_ancestor_joints[::-1]

        # NOTE (arth): some robots, like Fetch, have dummy joints that can mess with IK solver.
        #   we assume that the urdf_path provides a valid kinematic chain, and prune joints
        #   which are in the ManiSkill articulation but not in the kinematic chain
        with open(self.urdf_path, "r") as f:
            urdf_string = f.read()
        xml = ET.fromstring(urdf_string.encode("utf-8"))
        self._kinematic_chain_joint_names = set(
            node.get("name") for node in xml if node.tag == "joint"
        )
        self._kinematic_chain_link_names = set(
            node.get("name") for node in xml if node.tag == "link"
        )
        self.active_ancestor_joints = [
            x
            for x in active_ancestor_joints
            if x.name in self._kinematic_chain_joint_names
        ]

        if self.device.type == "cuda":
            self.use_gpu_ik = True
            self._setup_gpu()
        else:
            self.use_gpu_ik = False
            self._setup_cpu()

    def _setup_cpu(self):
        """setup the kinematics solvers on the CPU"""
        self.use_gpu_ik = False

        with open(self.urdf_path, "r") as f:
            xml = f.read()

        joint_order = [
            j.name
            for j in self.articulation.active_joints
            if j.name in self._kinematic_chain_joint_names
        ]
        link_order = [
            l.name
            for l in self.articulation.links
            if l.name in self._kinematic_chain_link_names
        ]

        self.pmodel = PinocchioModel(xml, [0, 0, -9.81])
        self.pmodel.set_joint_order(joint_order)
        self.pmodel.set_link_order(link_order)

        controlled_joint_names = [
            self.articulation.active_joints[i].name for i in self.active_joint_indices
        ]
        self.pmodel_controlled_joint_indices = torch.tensor(
            [joint_order.index(cj) for cj in controlled_joint_names],
            dtype=torch.int,
            device=self.device,
        )

        articulation_active_joint_names_to_idx = dict(
            (j.name, i) for i, j in enumerate(self.articulation.active_joints)
        )
        self.pmodel_active_joint_indices = torch.tensor(
            [articulation_active_joint_names_to_idx[jn] for jn in joint_order],
            dtype=torch.int,
            device=self.device,
        )

        # NOTE (arth): pmodel will use urdf_path, set values based on this xml
        self.end_link_idx = link_order.index(self.end_link.name)
        self.qmask = torch.zeros(len(joint_order), dtype=bool, device=self.device)
        self.qmask[self.pmodel_controlled_joint_indices] = 1

    def _setup_gpu(self):
        """setup the kinematics solvers on the GPU"""
        self.use_gpu_ik = True
        with open(self.urdf_path, "rb") as f:
            urdf_str = f.read()

        # NOTE (stao): it seems that the pk library currently always outputs some complaints if there are unknown attributes in a URDF. Hide it with this contextmanager here
        @contextmanager
        def suppress_stdout_stderr():
            """A context manager that redirects stdout and stderr to devnull"""
            with open(devnull, "w") as fnull:
                with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
                    yield (err, out)

        with suppress_stdout_stderr():
            self.pk_chain = pk.build_serial_chain_from_urdf(
                urdf_str,
                end_link_name=self.end_link.name,
            ).to(device=self.device)
        lim = torch.tensor(self.pk_chain.get_joint_limits(), device=self.device)
        self.pik = pk.PseudoInverseIK(
            self.pk_chain,
            joint_limits=lim.T,
            early_stopping_any_converged=True,
            max_iterations=200,
            num_retries=1,
        )

        # initially self.active_joint_indices references active joints that are controlled.
        # we also make the assumption that the active index is the same across all parallel managed joints
        self.active_ancestor_joint_idxs = [
            (x.active_index[0]).cpu().item() for x in self.active_ancestor_joints
        ]
        self.controlled_joints_idx_in_qmask = [
            self.active_ancestor_joint_idxs.index(idx)
            for idx in self.active_joint_indices
        ]

        self.qmask = torch.zeros(
            len(self.active_ancestor_joints), dtype=bool, device=self.device
        )
        self.qmask[self.controlled_joints_idx_in_qmask] = 1

    def compute_ik(
        self,
        pose: Union[Pose, torch.Tensor],
        q0: torch.Tensor,
        is_delta_pose: bool = False,
        current_pose: Optional[Pose] = None,
        solver_config: dict = dict(
            type="levenberg_marquardt", solver_iterations=1, alpha=1.0
        ),
    ):
        """Given a target pose, initial joint positions, this computes the target joint positions that will achieve the target pose.
        For optimization you can also provide the delta pose instead and specify is_delta_pose=True.

        Args:
            pose (Pose): target pose in the world frame. Note this is not relative to the robot base frame!
            q0 (torch.Tensor): initial joint positions of each active joint in the articulation. Note that this function will mask out the joints that are not kinematically relevant.
            is_delta_pose (bool): if True, the `pose` parameter should be a delta pose. It can also be a tensor of shape (N, 6) with the first 3 channels for translation and the last 3 channels for rotation in the euler angle format.
            current_pose (Optional[Pose]): current pose of the controlled link in the world frame. This is used to optimize the function by avoiding computing the current pose from q0 to compute the delta pose. If is_delta_pose is False, this is not used.
            solver_config (dict): configuration for the IK solver. Default is `dict(type="levenberg_marquardt", alpha=1.0)`. type can be one of "levenberg_marquardt" or "pseudo_inverse". alpha is a scaling term applied to the delta joint positions generated by the solver.
        """
        assert (
            isinstance(pose, Pose) or pose.shape[1] == 6
        ), "pose must be a Pose or a tensor with shape (N, 6)"
        if self.use_gpu_ik:
            B = pose.shape[0]
            q0 = q0[:, self.active_ancestor_joint_idxs]
            if not is_delta_pose:
                if current_pose is None:
                    current_pose = self.pk_chain.forward_kinematics(q0).get_matrix()
                    current_pose = Pose.create_from_pq(
                        current_pose[:, :3, 3],
                        rotation_conversions.matrix_to_quaternion(
                            current_pose[:, :3, :3]
                        ),
                    )
                if isinstance(pose, torch.Tensor):
                    target_pos, target_rot = pose[:, 0:3], pose[:, 3:6]
                    target_quat = rotation_conversions.matrix_to_quaternion(
                        rotation_conversions.euler_angles_to_matrix(target_rot, "XYZ")
                    )
                    pose = Pose.create_from_pq(target_pos, target_quat)
                if isinstance(pose, Pose):
                    # the following assumes root_translation:root_aligned_body_rotation control frame
                    translation = pose.p - current_pose.p
                    quaternion = rotation_conversions.quaternion_multiply(
                        pose.q, rotation_conversions.quaternion_invert(current_pose.q)
                    )
                    pose = Pose.create_from_pq(translation, quaternion)
            if isinstance(pose, Pose):
                delta_pose = torch.zeros(
                    (B, 6), device=self.device, dtype=torch.float32
                )
                delta_pose[:, 0:3] = pose.p
                delta_pose[:, 3:6] = rotation_conversions.matrix_to_euler_angles(
                    rotation_conversions.quaternion_to_matrix(pose.q), "XYZ"
                )
            else:
                delta_pose = pose
            jacobian = self.pk_chain.jacobian(q0)[:, :, self.qmask]
            if solver_config["type"] == "levenberg_marquardt":
                lambd = 0.0001  # Regularization parameter to ensure J^T * J is non-singular.
                J_T = jacobian.transpose(1, 2)
                lfs_A = torch.bmm(J_T, jacobian) + lambd * torch.eye(
                    len(self.qmask), device=self.device
                )
                rhs_B = torch.bmm(J_T, delta_pose.unsqueeze(-1))
                delta_joint_pos = torch.linalg.solve(lfs_A, rhs_B)

            elif solver_config["type"] == "pseudo_inverse":
                # NOTE (stao): this method of IK is from https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf by Samuel R. Buss
                delta_joint_pos = torch.linalg.pinv(jacobian) @ delta_pose.unsqueeze(-1)

            return q0[:, self.qmask] + solver_config["alpha"] * delta_joint_pos.squeeze(
                -1
            )
        else:
            q0 = q0[:, self.pmodel_active_joint_indices]
            result, success, error = self.pmodel.compute_inverse_kinematics(
                self.end_link_idx,
                pose.sp,
                initial_qpos=q0.cpu().numpy()[0],
                active_qmask=self.qmask,
                max_iterations=100,
            )
            if success:
                return common.to_tensor(
                    [result[self.pmodel_controlled_joint_indices]], device=self.device
                )
            else:
                return None
