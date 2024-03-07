from dataclasses import dataclass
from functools import cache
from typing import List

import sapien
import sapien.physx as physx
import sapien.render
import torch

from mani_skill.utils import sapien_utils

# NOTE (stao): commented out functions are functions that are not confirmed to be working in the wrapped class but the original class has


@dataclass
class RenderCamera:
    """
    Wrapper around sapien.render.RenderCameraComponent
    """

    _render_cameras: List[sapien.render.RenderCameraComponent]
    name: str
    camera_group: sapien.render.RenderCameraGroup = None

    @classmethod
    def create(cls, render_cameras: List[sapien.render.RenderCameraComponent]):
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
        return cls(_render_cameras=render_cameras, name=shared_name)

    def get_name(self) -> str:
        return self.name

    def __hash__(self):
        return self._render_cameras[0].__hash__()

    # -------------------------------------------------------------------------- #
    # Functions from RenderCameraComponent
    # -------------------------------------------------------------------------- #
    # TODO (stao): support extrinsic matrix changing
    @cache
    def get_extrinsic_matrix(self):
        return sapien_utils.to_tensor(self._render_cameras[0].get_extrinsic_matrix())[
            None, :
        ]

    def get_far(self) -> float:
        return self._render_cameras[0].get_far()

    def get_global_pose(self) -> sapien.Pose:
        return self._render_cameras[0].get_global_pose()

    def get_height(self) -> int:
        return self._render_cameras[0].get_height()

    @cache
    def get_intrinsic_matrix(self):
        return sapien_utils.to_tensor(self._render_cameras[0].get_intrinsic_matrix())[
            None, :
        ]

    def get_local_pose(self) -> sapien.Pose:
        return self._render_cameras[0].get_local_pose()

    @cache
    def get_model_matrix(self):
        return sapien_utils.to_tensor(self._render_cameras[0].get_model_matrix())[
            None, :
        ]

    def get_near(self) -> float:
        return self._render_cameras[0].get_near()

    def get_picture(self, name: str):
        if physx.is_gpu_enabled():
            return self.camera_group.get_picture_cuda(name).torch()
        else:
            return sapien_utils.to_tensor(self._render_cameras[0].get_picture(name))[
                None, ...
            ]

    def get_picture_cuda(self, name: str):
        return self._render_cameras[0].get_picture_cuda(name)

    def get_picture_names(self) -> list[str]:
        return self._render_cameras[0].get_picture_names()

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

    # @typing.overload
    # def set_property(self, name: str, value: int) -> None:
    #     self._render_cameras[0].set_property(name, value)

    def set_skew(self, skew: float) -> None:
        for obj in self._render_cameras:
            obj.set_skew(skew)

    # def set_texture(self, name: str, texture: RenderTexture) -> None:
    #     self._render_cameras[0].set_texture(name, texture)

    # def set_texture_array(self, name: str, textures: list[RenderTexture]) -> None:
    #     self._render_cameras[0].set_texture_array(name, textures)

    def take_picture(self) -> None:
        if physx.is_gpu_enabled():
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
        return self._render_cameras[0].global_pose

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
