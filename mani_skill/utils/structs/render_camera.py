from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Union

from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import torch

from mani_skill.render import SAPIEN_RENDER_SYSTEM
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose

if SAPIEN_RENDER_SYSTEM == "3.1":
    sapien.render.RenderCameraGroup = "oldtype"  # type: ignore

# NOTE (stao): commented out functions are functions that are not confirmed to be working in the wrapped class but the original class has


@dataclass
class RenderCamera:
    """
    Wrapper around sapien.render.RenderCameraComponent
    """

    _render_cameras: List[sapien.render.RenderCameraComponent]
    name: str
    # NOTE (stao): I cannot seem to use ManiSkillScene as a type here, it complains it is undefined despite using TYPE_CHECKING variable. Without typchecking there is a ciruclar import error
    scene: Any
    camera_group: sapien.render.RenderCameraGroup = None
    mount: Union[Actor, Link] = None

    # we cache model and extrinsic matrices since the code here supports computing these when the camera is mounted and these are always changing
    _cached_model_matrix: torch.Tensor = None
    _cached_extrinsic_matrix: torch.Tensor = None
    # NOTE (stao): default @cache from functools seems to cause CPU memory leaks in dataclasses, so we cache ourselves here
    _cached_intrinsic_matrix: torch.Tensor = None
    _cached_local_pose: Pose = None

    @classmethod
    def create(
        cls,
        render_cameras: List[sapien.render.RenderCameraComponent],
        scene: Any,
        mount: Union[Actor, Link] = None,
    ):
        w, h = (
            render_cameras[0].width,
            render_cameras[0].height,
        )  # Currently camera groups must have all the same shape
        shared_name = "_".join(render_cameras[0].name.split("_")[1:])
        for render_camera in render_cameras:
            assert (render_camera.width, render_camera.height) == (
                w,
                h,
            ), "all passed in render cameras must have the same width and height"
        return cls(
            _render_cameras=render_cameras, scene=scene, name=shared_name, mount=mount
        )

    def get_name(self) -> str:
        return self.name

    def __hash__(self):
        return self._render_cameras[0].__hash__()

    # -------------------------------------------------------------------------- #
    # Functions from RenderCameraComponent
    # -------------------------------------------------------------------------- #
    def get_extrinsic_matrix(self):
        if self.scene.gpu_sim_enabled:
            if self._cached_extrinsic_matrix is not None:
                return self._cached_extrinsic_matrix
            ros2opencv = torch.tensor(
                [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                device=self.scene.device,
                dtype=torch.float32,
            ).T
            res = (
                ros2opencv @ self.get_global_pose().inv().to_transformation_matrix()
            )[:, :3, :4]
            if self.mount is None:
                self._cached_extrinsic_matrix = res
            return res
        else:
            return common.to_tensor(self._render_cameras[0].get_extrinsic_matrix())[
                None, :
            ]

    def get_far(self) -> float:
        return self._render_cameras[0].get_far()

    def get_global_pose(self) -> Pose:
        return self.global_pose

    def get_height(self) -> int:
        return self._render_cameras[0].get_height()

    def get_intrinsic_matrix(self):
        if self._cached_intrinsic_matrix is None:
            self._cached_intrinsic_matrix = common.to_tensor(
                np.array([cam.get_intrinsic_matrix() for cam in self._render_cameras]),
                device=self.scene.device,
            )
        return self._cached_intrinsic_matrix

    def get_local_pose(self) -> Pose:
        if self._cached_local_pose is None:
            if self.scene.gpu_sim_enabled:
                ps = np.array(
                    [
                        np.concatenate([cam.get_local_pose().p, cam.get_local_pose().q])
                        for cam in self._render_cameras
                    ]
                )
                self._cached_local_pose = Pose.create(
                    common.to_tensor(ps), device=self.scene.device
                )
            else:
                self._cached_local_pose = Pose.create_from_pq(
                    self._render_cameras[0].get_local_pose().p,
                    self._render_cameras[0].get_local_pose().q,
                )
        return self._cached_local_pose

    def get_model_matrix(self):

        if self.scene.gpu_sim_enabled:
            if self._cached_model_matrix is not None:
                return self._cached_model_matrix
            # NOTE (stao): This code is based on SAPIEN. It cannot expose GPU buffers of this data of all cameras directly so
            # we have to compute it here based on how SAPIEN does it:
            POSE_GL_TO_ROS = Pose.create_from_pq(p=[0, 0, 0], q=[-0.5, -0.5, 0.5, 0.5])
            pose = self.get_global_pose() * POSE_GL_TO_ROS
            b = len(pose.raw_pose)
            qmat = torch.zeros((b, 4, 4), device=self.scene.device)
            qmat[:, :3, :3] = quaternion_to_matrix(pose.q)
            qmat[:, -1, -1] = 1
            pmat = torch.eye(4, device=self.scene.device)[None, ...].repeat(
                len(qmat), 1, 1
            )
            pmat[:, :3, 3] = pose.p
            res = pmat @ qmat
            if self.mount is None:
                self._cached_model_matrix = res
            return res
        else:
            return common.to_tensor(self._render_cameras[0].get_model_matrix())[None, :]

    def get_near(self) -> float:
        return self._render_cameras[0].get_near()

    def get_picture(self, names: Union[str, List[str]]) -> List[torch.Tensor]:
        if isinstance(names, str):
            names = [names]
        if self.scene.gpu_sim_enabled and not self.scene.parallel_in_single_scene:
            if SAPIEN_RENDER_SYSTEM == "3.0":
                return [
                    self.camera_group.get_picture_cuda(name).torch() for name in names
                ]
            elif SAPIEN_RENDER_SYSTEM == "3.1":
                return [x.torch() for x in self.camera_group.get_cuda_pictures(names)]
        else:
            if self.scene.backend.render_backend == "sapien_cuda":
                return [
                    self._render_cameras[0].get_picture_cuda(name).torch()[None, ...]
                    for name in names
                ]
            else:
                return [
                    common.to_tensor(self._render_cameras[0].get_picture(name))[
                        None, ...
                    ]
                    for name in names
                ]

    # def get_picture_cuda(self, name: str):
    #     return self._render_cameras[0].get_picture_cuda(name)

    # def get_picture_names(self) -> list[str]:
    #     return self._render_cameras[0].get_picture_names()

    def get_projection_matrix(self):
        return self._render_cameras[0].get_projection_matrix()

    def get_skew(self) -> float:
        return self._render_cameras[0].get_skew()

    def get_width(self) -> int:
        return self._render_cameras[0].get_width()

    def set_far(self, far: float) -> None:
        for obj in self._render_cameras:
            obj.set_far(far)

    def set_focal_lengths(self, fx: float, fy: float) -> None:
        for obj in self._render_cameras:
            obj.set_focal_lengths(fx, fy)

    def set_fovx(self, fov: float, compute_y: bool = True) -> None:
        for obj in self._render_cameras:
            obj.set_fovx(fov, compute_y)

    def set_fovy(self, fov: float, compute_x: bool = True) -> None:
        for obj in self._render_cameras:
            obj.set_fovy(fov, compute_x)

    def set_gpu_pose_batch_index(self, arg0: int) -> None:
        for obj in self._render_cameras:
            obj.set_gpu_pose_batch_index(arg0)

    def set_local_pose(self, arg0: sapien.Pose) -> None:
        for obj in self._render_cameras:
            obj.set_local_pose(arg0)
        self._cached_local_pose = None

    def set_near(self, near: float) -> None:
        for obj in self._render_cameras:
            obj.set_near(near)

    def set_perspective_parameters(
        self,
        near: float,
        far: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        skew: float,
    ) -> None:
        for obj in self._render_cameras:
            obj.set_perspective_parameters(near, far, fx, fy, cx, cy, skew)

    def set_principal_point(self, cx: float, cy: float) -> None:
        for obj in self._render_cameras:
            obj.set_principal_point(cx, cy)

    # @typing.overload
    # def set_property(self, name: str, value: float) -> None:
    #     self._render_cameras[0].set_property(name, value)

    def set_property(self, name: str, value: Any) -> None:
        """change properties of the camera. This is not well documented at the moment and is a heavily overloaded function.

        At the moment you can do this:

        - set_property("toneMapper", value) where value is 0 (gamma), 1 (sRGB), 2 (filmic) change the color management used. Default is 0 (gamma)
        - set_property("exposure", value) where value is the exposure. Default is 1.0
        """
        self._render_cameras[0].set_property(name, value)

    def set_skew(self, skew: float) -> None:
        for obj in self._render_cameras:
            obj.set_skew(skew)

    # def set_texture(self, name: str, texture: RenderTexture) -> None:
    #     self._render_cameras[0].set_texture(name, texture)

    # def set_texture_array(self, name: str, textures: list[RenderTexture]) -> None:
    #     self._render_cameras[0].set_texture_array(name, textures)

    def take_picture(self) -> None:
        if self.scene.gpu_sim_enabled:
            self.camera_group.take_picture()
        else:
            self._render_cameras[0].take_picture()

    @property
    def _cuda_buffer(self):
        return self._render_cameras[0]._cuda_buffer

    @property
    def cx(self) -> float:
        return self._render_cameras[0].cx

    @property
    def cy(self) -> float:
        return self._render_cameras[0].cy

    @property
    def far(self) -> float:
        return self._render_cameras[0].far

    @far.setter
    def far(self, arg1: float) -> None:
        for obj in self._render_cameras:
            obj.far = arg1

    @property
    def fovx(self) -> float:
        return self._render_cameras[0].fovx

    @property
    def fovy(self) -> float:
        return self._render_cameras[0].fovy

    @property
    def fx(self) -> float:
        return self._render_cameras[0].fx

    @property
    def fy(self) -> float:
        return self._render_cameras[0].fy

    @property
    def global_pose(self) -> sapien.Pose:
        if self.scene.gpu_sim_enabled:
            if self.mount is not None:
                return self.mount.pose * self.get_local_pose()
            return self.get_local_pose()
        else:
            return Pose.create_from_pq(
                self._render_cameras[0].get_global_pose().p,
                self._render_cameras[0].get_global_pose().q,
            )

    # TODO (stao): These properties should be torch tensors in the future
    @property
    def height(self) -> int:
        return self._render_cameras[0].height

    @property
    def local_pose(self) -> sapien.Pose:
        return self._render_cameras[0].local_pose

    @local_pose.setter
    def local_pose(self, arg1: sapien.Pose) -> None:
        for obj in self._render_cameras:
            obj.local_pose = arg1

    @property
    def near(self) -> float:
        return self._render_cameras[0].near

    @near.setter
    def near(self, arg1: float) -> None:
        for obj in self._render_cameras:
            obj.near = arg1

    @property
    def skew(self) -> float:
        return self._render_cameras[0].skew

    @skew.setter
    def skew(self, arg1: float) -> None:
        for obj in self._render_cameras:
            obj.skew = arg1

    @property
    def width(self) -> int:
        return self._render_cameras[0].width
