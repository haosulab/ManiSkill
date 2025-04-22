"""
Code for kinematics utilities on CPU/GPU
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from typing import List

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
        active_ancestor_joints: List[ArticulationJoint] = []
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
        target_pose: Pose,
        q0: torch.Tensor,
        pos_only: bool = False,
        action=None,
        use_delta_ik_solver: bool = False,
    ):
        """Given a target pose, via inverse kinematics compute the target joint positions that will achieve the target pose

        Args:
            target_pose (Pose): target pose of the end effector in the world frame. note this is not relative to the robot base frame!
            q0 (torch.Tensor): initial joint positions of every active joint in the articulation
            pos_only (bool): if True, only the position of the end link is considered in the IK computation
            action (torch.Tensor): delta action to be applied to the articulation. Used for fast delta IK solutions on the GPU.
            use_delta_ik_solver (bool): If true, returns the target joint positions that correspond with a delta IK solution. This is specifically
                used for GPU simulation to determine which GPU IK algorithm to use.
        """
        if self.use_gpu_ik:
            q0 = q0[:, self.active_ancestor_joint_idxs]
            if not use_delta_ik_solver:
                tf = pk.Transform3d(
                    pos=target_pose.p,
                    rot=target_pose.q,
                    device=self.device,
                )
                self.pik.initial_config = q0  # shape (num_retries, active_ancestor_dof)
                result = self.pik.solve(
                    tf
                )  # produce solutions in shape (B, num_retries/initial_configs, active_ancestor_dof)
                # TODO return mask for invalid solutions. CPU returns None at the moment
                return result.solutions[:, 0, :]
            else:
                jacobian = self.pk_chain.jacobian(q0)
                # code commented out below is the fast kinematics method
                # jacobian = (
                #     self.fast_kinematics_model.jacobian_mixed_frame_pytorch(
                #         self.articulation.get_qpos()[:, self.active_ancestor_joint_idxs]
                #     )
                #     .view(-1, len(self.active_ancestor_joints), 6)
                #     .permute(0, 2, 1)
                # )
                # jacobian = jacobian[:, :, self.qmask]
                if pos_only:
                    jacobian = jacobian[:, 0:3]

                # NOTE (arth): use only the parts of the jacobian that correspond to the active joints
                jacobian = jacobian[:, :, self.qmask]

                # NOTE (stao): this method of IK is from https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf by Samuel R. Buss
                delta_joint_pos = torch.linalg.pinv(jacobian) @ action.unsqueeze(-1)

                return q0[:, self.qmask] + delta_joint_pos.squeeze(-1)
        else:
            q0 = q0[:, self.pmodel_active_joint_indices]
            result, success, error = self.pmodel.compute_inverse_kinematics(
                self.end_link_idx,
                target_pose.sp,
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
