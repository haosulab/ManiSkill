"""
Utilities that work with the simulation / SAPIEN
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, TypeVar

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import sapien.wrapper.urdf_loader

from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.envs.scene import ManiSkillScene

import torch

from mani_skill.utils.structs.types import Array, Device

T = TypeVar("T")


def get_obj_by_name(objs: List[T], name: str, is_unique=True):
    """Get a object given the name.

    Args:
        objs (List[T]): objs to query. Expect these objects to have a get_name function. These may be sapien.Entity, physx.PhysxArticulationLink etc.
        name (str): name for query.
        is_unique (bool, optional):
            whether the name should be unique. Defaults to True.

    Raises:
        RuntimeError: The name is not unique when @is_unique is True.

    Returns:
        T or List[T]:
            matched T or Ts. None if no matches.
    """
    matched_objects = [x for x in objs if x.get_name() == name]
    if len(matched_objects) > 1:
        if not is_unique:
            return matched_objects
        else:
            raise RuntimeError(f"Multiple objects with the same name {name}.")
    elif len(matched_objects) == 1:
        return matched_objects[0]
    else:
        return None


def get_objs_by_names(objs: List[T], names: List[str]) -> List[T]:
    """Get a list of objects given a list of names from a larger list of objects (objs). The returned list is in the order of the names given

    Args:
        objs (List[T]): objs to query. Expect these objects to have a get_name function. These may be sapien.Entity, physx.PhysxArticulationLink etc.
        name (str): names to query.

    Returns:
        T or List[T]:
            matched T or Ts. None if no matches.
    """
    assert isinstance(objs, (list, tuple)), type(objs)
    ret = [None for _ in names]

    for obj in objs:
        name = obj.get_name()
        if name in names:
            ret[names.index(name)] = obj
    return ret


def get_obj_by_type(objs: List[T], target_type: T, is_unique=True):
    matched_objects = [x for x in objs if type(x) == target_type]
    if len(matched_objects) > 1:
        if not is_unique:
            return matched_objects
        else:
            raise RuntimeError(f"Multiple objects with the same type {target_type}.")
    elif len(matched_objects) == 1:
        return matched_objects[0]
    else:
        return None


def check_urdf_config(urdf_config: dict):
    """Check whether the urdf config is valid for SAPIEN.

    Args:
        urdf_config (dict): dict passed to `sapien.URDFLoader.load`.
    """
    allowed_keys = ["material", "density", "link"]
    for k in urdf_config.keys():
        if k not in allowed_keys:
            raise KeyError(
                f"Not allowed key ({k}) for `sapien.URDFLoader.load`. Allowed keys are f{allowed_keys}"
            )

    allowed_keys = ["material", "density", "patch_radius", "min_patch_radius"]
    for k, v in urdf_config.get("link", {}).items():
        for kk in v.keys():
            # In fact, it should support specifying collision-shape-level materials.
            if kk not in allowed_keys:
                raise KeyError(
                    f"Not allowed key ({kk}) for `sapien.URDFLoader.load`. Allowed keys are f{allowed_keys}"
                )


def parse_urdf_config(config_dict: dict) -> Dict:
    """Parse config from dict for SAPIEN URDF loader.

    Args:
        config_dict (dict): a dict containing link physical properties.
        scene (ManiSkillScene): the simulation scene

    Returns:
        Dict: urdf config passed to `sapien.URDFLoader.load`.
    """
    # urdf_config = deepcopy(config_dict)
    urdf_config = dict()

    # Create the global physical material for all links
    if "material" in config_dict:
        urdf_config["material"] = physx.PhysxMaterial(**config_dict["material"])

    # Create link-specific physical materials
    materials = {}
    if "_materials" in config_dict:
        for k, v in config_dict["_materials"].items():
            materials[k] = physx.PhysxMaterial(**v)

    # Specify properties for links
    if "link" in config_dict:
        urdf_config["link"] = dict()
        for k, link_config in config_dict["link"].items():
            urdf_config["link"][k] = link_config.copy()
            # substitute with actual material
            urdf_config["link"][k]["material"] = materials[link_config["material"]]
    return urdf_config


def apply_urdf_config(loader: sapien.wrapper.urdf_loader.URDFLoader, urdf_config: dict):
    if "link" in urdf_config:
        for name, link_config in urdf_config["link"].items():
            if "material" in link_config:
                mat: physx.PhysxMaterial = link_config["material"]
                loader.set_link_material(
                    name, mat.static_friction, mat.dynamic_friction, mat.restitution
                )
            if "patch_radius" in link_config:
                loader.set_link_patch_radius(name, link_config["patch_radius"])
            if "min_patch_radius" in link_config:
                loader.set_link_min_patch_radius(name, link_config["min_patch_radius"])
            if "density" in link_config:
                loader.set_link_density(name, link_config["density"])
    if "material" in urdf_config:
        mat: physx.PhysxMaterial = urdf_config["material"]
        loader.set_material(mat.static_friction, mat.dynamic_friction, mat.restitution)
    if "patch_radius" in urdf_config:
        loader.set_patch_radius(urdf_config["patch_radius"])
    if "min_patch_radius" in urdf_config:
        loader.set_min_patch_radius(urdf_config["min_patch_radius"])
    if "density" in urdf_config:
        loader.set_density(urdf_config["density"])


# -------------------------------------------------------------------------- #
# Entity state
# -------------------------------------------------------------------------- #
def get_actor_state(actor: sapien.Entity):
    pose = actor.get_pose()
    component = actor.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if component is None or component.kinematic:
        vel = np.zeros(3)
        ang_vel = np.zeros(3)
    else:
        vel = component.get_linear_velocity()  # [3]
        ang_vel = component.get_angular_velocity()  # [3]
    return np.hstack([pose.p, pose.q, vel, ang_vel])


def get_articulation_state(articulation: physx.PhysxArticulation):
    root_link = articulation.get_links()[0]
    pose = root_link.get_pose()
    vel = root_link.get_linear_velocity()  # [3]
    ang_vel = root_link.get_angular_velocity()  # [3]
    qpos = articulation.get_qpos()
    qvel = articulation.get_qvel()
    return np.hstack([pose.p, pose.q, vel, ang_vel, qpos, qvel])


def get_articulation_padded_state(articulation: physx.PhysxArticulation, max_dof: int):
    state = get_articulation_state(articulation)
    qpos, qvel = np.split(state[13:], 2)
    nq = len(qpos)
    assert max_dof >= nq, (max_dof, nq)
    padded_state = np.zeros(13 + 2 * max_dof, dtype=np.float32)
    padded_state[:13] = state[:13]
    padded_state[13 : 13 + nq] = qpos
    padded_state[13 + max_dof : 13 + max_dof + nq] = qvel
    return padded_state


# TODO (stao): Synchronize the contacts APIs as well as getting forces/impulses
# -------------------------------------------------------------------------- #
# Contact
#
# Note that for simplicity, we always compare contact by using entitiy objects
# and check if the entity is the same
# -------------------------------------------------------------------------- #
def get_pairwise_contacts(
    contacts: List[physx.PhysxContact], actor0: sapien.Entity, actor1: sapien.Entity
) -> List[Tuple[physx.PhysxContact, bool]]:
    """
    Given a list of contacts, return the list of contacts involving the two actors
    """
    pairwise_contacts = []
    for contact in contacts:
        if contact.bodies[0].entity == actor0 and contact.bodies[1].entity == actor1:
            pairwise_contacts.append((contact, True))
        elif contact.bodies[0].entity == actor1 and contact.bodies[1].entity == actor0:
            pairwise_contacts.append((contact, False))
    return pairwise_contacts


def get_multiple_pairwise_contacts(
    contacts: List[physx.PhysxContact],
    actor0: sapien.Entity,
    actor1_list: List[sapien.Entity],
) -> Dict[sapien.Entity, List[Tuple[physx.PhysxContact, bool]]]:
    """
    Given a list of contacts, return the dict of contacts involving the one actor and actors
    This function is used to avoid double for-loop when using `get_pairwise_contacts` with multiple actors
    """
    pairwise_contacts = {actor: [] for actor in actor1_list}
    for contact in contacts:
        if (
            contact.bodies[0].entity == actor0
            and contact.bodies[1].entity in actor1_list
        ):
            pairwise_contacts[contact.bodies[1].entity].append((contact, True))
        elif (
            contact.bodies[0].entity in actor1_list
            and contact.bodies[1].entity == actor0
        ):
            pairwise_contacts[contact.bodies[0].entity].append((contact, False))
    return pairwise_contacts


def compute_total_impulse(contact_infos: List[Tuple[physx.PhysxContact, bool]]):
    total_impulse = np.zeros(3)
    for contact, flag in contact_infos:
        contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
        # Impulse is applied on the first component
        total_impulse += contact_impulse * (1 if flag else -1)
    return total_impulse


def get_pairwise_contact_impulse(
    contacts: List[physx.PhysxContact], actor0: sapien.Entity, actor1: sapien.Entity
):
    pairwise_contacts = get_pairwise_contacts(contacts, actor0, actor1)
    total_impulse = compute_total_impulse(pairwise_contacts)
    return total_impulse


def get_cpu_actor_contacts(
    contacts: List[physx.PhysxContact], actor: sapien.Entity
) -> List[Tuple[physx.PhysxContact, bool]]:
    entity_contacts = []
    for contact in contacts:
        if contact.bodies[0].entity == actor:
            entity_contacts.append((contact, True))
        elif contact.bodies[1].entity == actor:
            entity_contacts.append((contact, False))
    return entity_contacts


def get_cpu_actors_contacts(
    contacts: List[physx.PhysxContact], actors: List[sapien.Entity]
) -> Dict[sapien.Entity, List[Tuple[physx.PhysxContact, bool]]]:
    """
    This function is used to avoid double for-loop when using `get_actor_contacts` with multiple actors
    """
    entity_contacts = {actor: [] for actor in actors}
    for contact in contacts:
        if contact.bodies[0].entity in actors:
            entity_contacts[contact.bodies[0].entity].append((contact, True))
        elif contact.bodies[1].entity in actors:
            entity_contacts[contact.bodies[1].entity].append((contact, False))
    return entity_contacts


# -------------------------------------------------------------------------- #
# Camera
# -------------------------------------------------------------------------- #
def sapien_pose_to_opencv_extrinsic(sapien_pose_matrix: np.ndarray) -> np.ndarray:
    sapien2opencv = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ex = sapien2opencv @ np.linalg.inv(sapien_pose_matrix)  # world -> camera

    return ex


def look_at(eye, target, up=(0, 0, 1)) -> Pose:
    """Get the camera pose in SAPIEN by the Look-At method.

    Note:
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        The SAPIEN camera follows the convention: (forward, right, up) = (x, -y, z)
        while the OpenGL camera follows (forward, right, up) = (-z, x, y)
        Note that the camera coordinate system (OpenGL) is left-hand.

    Args:
        eye: camera location
        target: looking-at location
        up: a general direction of "up" from the camera.

    Returns:
        Pose: camera pose
    """
    # only accept batched input as tensors
    # accept all other input as 1 dimensional
    if not isinstance(eye, torch.Tensor):
        eye = torch.tensor(eye, dtype=torch.float32)
        assert eye.ndim == 1, eye.ndim
        assert len(eye) == 3, len(eye)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)
        assert target.ndim == 1, target.ndim
        assert len(target) == 3, len(target)
    if not isinstance(up, torch.Tensor):
        up = torch.tensor(up, dtype=torch.float32)
        assert up.ndim == 1, up.ndim
        assert len(up) == 3, len(up)

    def normalize_tensor(x, eps=1e-6):
        x = x.view(-1, 3)
        norm = torch.linalg.norm(x, dim=-1)
        zero_vectors = norm < eps
        x[zero_vectors] = torch.zeros(3).float()
        x[~zero_vectors] /= norm[~zero_vectors].view(-1, 1)
        return x

    forward = normalize_tensor(target - eye)
    up = normalize_tensor(up)
    left = torch.cross(up, forward, dim=-1)
    left = normalize_tensor(left)
    up = torch.cross(forward, left, dim=-1)
    rotation = torch.stack([forward, left, up], dim=-1)
    return Pose.create_from_pq(p=eye, q=matrix_to_quaternion(rotation))


def hex2rgba(h, correction=True):
    # https://stackoverflow.com/a/29643643
    h = h.lstrip("#")
    r, g, b = tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    rgba = np.array([r, g, b, 1])
    if correction:  # reverse gamma correction in sapien
        rgba = rgba**2.2
    return rgba


def set_render_material(material: sapien.render.RenderMaterial, **kwargs):
    for k, v in kwargs.items():
        if k == "color":
            material.set_base_color(v)
        else:
            setattr(material, k, v)
    return material


def set_articulation_render_material(articulation: physx.PhysxArticulation, **kwargs):
    # TODO: (stao): Avoid using this function as it does not play nice with render server
    # we should edit the urdf files to have the correct materials in the first place and remove this function in the future
    for link in articulation.get_links():
        component = link.entity.find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        if component is None:
            continue
        for s in component.render_shapes:
            if type(s) == sapien.render.RenderShapeTriangleMesh:
                for part in s.parts:
                    mat = part.material
                    set_render_material(mat, **kwargs)


# -------------------------------------------------------------------------- #
# Misc
# -------------------------------------------------------------------------- #
def check_joint_stuck(
    articulation: physx.PhysxArticulation,
    active_joint_idx: int,
    pos_diff_threshold: float = 1e-3,
    vel_threshold: float = 1e-4,
):
    actual_pos = articulation.get_qpos()[active_joint_idx]
    target_pos = articulation.get_drive_target()[active_joint_idx]
    actual_vel = articulation.get_qvel()[active_joint_idx]

    return (
        abs(actual_pos - target_pos) > pos_diff_threshold
        and abs(actual_vel) < vel_threshold
    )


def check_actor_static(actor: Actor, lin_thresh=1e-3, ang_thresh=1e-2):
    return torch.logical_and(
        torch.linalg.norm(actor.linear_velocity, axis=1) <= lin_thresh,
        torch.linalg.norm(actor.angular_velocity, axis=1) <= ang_thresh,
    )


def is_state_dict_consistent(state_dict: dict):
    """Checks if the given state dictionary (generated via env.get_state_dict()) is consistent where each actor/articulation has the same batch dimension"""
    batch_size = None
    for name in ["actors", "articulations"]:
        if name in state_dict:
            for k, v in state_dict[name].items():
                if batch_size is None:
                    batch_size = v.shape[0]
                else:
                    if v.shape[0] != batch_size:
                        return False
    return True
