from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import sapien.wrapper.urdf_loader
from transforms3d.quaternions import mat2quat

if TYPE_CHECKING:
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.envs.scene import ManiSkillScene

import torch

from mani_skill.utils.structs.types import Array, Device, get_backend_name


# TODO (stao): this code can be simplified
def to_tensor(array: Union[torch.Tensor, np.array, Sequence], device: Device = None):
    """
    Maps any given sequence to a torch tensor on the CPU/GPU. If physx gpu is not enabled then we use CPU, otherwise GPU, unless specified
    by the device argument

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None and to_tensor will put the device on the GPU if physx is enabled
            and CPU otherwise

    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v) for k, v in array.items()}
    if get_backend_name() == "torch":
        if isinstance(array, np.ndarray):
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            ret = torch.from_numpy(array)
            if ret.dtype == torch.float64:
                ret = ret.float()
        elif isinstance(array, torch.Tensor):
            ret = array
        else:
            ret = torch.Tensor(array)
        if device is None:
            return ret.cuda()
        else:
            return ret.to(device)
    elif get_backend_name() == "numpy":
        if isinstance(array, np.ndarray):
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            ret = torch.from_numpy(array)
            if ret.dtype == torch.float64:
                ret = ret.float()
        elif isinstance(array, list) and isinstance(array[0], np.ndarray):
            ret = torch.from_numpy(np.array(array))
            if ret.dtype == torch.float64:
                ret = ret.float()
        elif np.iterable(array):
            ret = torch.Tensor(array)
        else:
            ret = torch.Tensor(array)
        if device is None:
            return ret
        else:
            return ret.to(device)


def _to_numpy(array: Union[Array, Sequence]) -> np.ndarray:
    if isinstance(array, (dict)):
        return {k: _to_numpy(v) for k, v in array.items()}
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    if (
        isinstance(array, np.ndarray)
        or isinstance(array, bool)
        or isinstance(array, str)
        or isinstance(array, float)
        or isinstance(array, int)
    ):
        return array
    else:
        return np.array(array)


def to_numpy(array: Union[Array, Sequence], dtype=None) -> np.ndarray:
    array = _to_numpy(array)
    if dtype is not None:
        return array.astype(dtype)
    return array


def _unbatch(array: Union[Array, Sequence]):
    if isinstance(array, (dict)):
        return {k: _unbatch(v) for k, v in array.items()}
    if isinstance(array, str):
        return array
    if isinstance(array, torch.Tensor):
        return array.squeeze(0)
    if isinstance(array, np.ndarray):
        if array.shape == (1,):
            return array.item()
        if np.iterable(array) and array.shape[0] == 1:
            return array.squeeze(0)
    if isinstance(array, list):
        if len(array) == 1:
            return array[0]
    return array


def unbatch(*args: Tuple[Union[Array, Sequence]]):
    x = [_unbatch(x) for x in args]
    if len(args) == 1:
        return x[0]
    return tuple(x)


def _batch(array: Union[Array, Sequence]):
    if isinstance(array, (dict)):
        return {k: _batch(v) for k, v in array.items()}
    if isinstance(array, str):
        return array
    if isinstance(array, torch.Tensor):
        return array[None, :]
    if isinstance(array, np.ndarray):
        if array.shape == ():
            return array.reshape(1, 1)
        return array[None, :]
    if isinstance(array, list):
        if len(array) == 1:
            return [array]
    if (
        isinstance(array, float)
        or isinstance(array, int)
        or isinstance(array, bool)
        or isinstance(array, np.bool_)
    ):
        return np.array([[array]])
    return array


def batch(*args: Tuple[Union[Array, Sequence]]):
    """Adds one dimension in front of everything. If given a dictionary, every leaf in the dictionary
    has a new dimension. If given a tuple, returns the same tuple with each element batched"""
    x = [_batch(x) for x in args]
    if len(args) == 1:
        return x[0]
    return tuple(x)


def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    if norm < eps:
        return np.zeros_like(x)
    else:
        return x / norm


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


def parse_urdf_config(config_dict: dict, scene: ManiSkillScene) -> Dict:
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
        urdf_config["material"] = scene.create_physical_material(
            **config_dict["material"]
        )

    # Create link-specific physical materials
    materials = {}
    if "_materials" in config_dict:
        for k, v in config_dict["_materials"].items():
            materials[k] = scene.create_physical_material(**v)

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
        for name, link_cfg in urdf_config["link"].items():
            if "material" in link_cfg:
                mat: physx.PhysxMaterial = link_cfg["material"]
                loader.set_link_material(
                    name, mat.static_friction, mat.dynamic_friction, mat.restitution
                )
            if "patch_radius" in link_cfg:
                loader.set_link_patch_radius(name, link_cfg["patch_radius"])
            if "min_patch_radius" in link_cfg:
                loader.set_link_min_patch_radius(name, link_cfg["min_patch_radius"])
            if "density" in link_cfg:
                loader.set_link_density(name, link_cfg["density"])
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


def get_actor_contacts(
    contacts: List[physx.PhysxContact], actor: sapien.Entity
) -> List[Tuple[physx.PhysxContact, bool]]:
    entity_contacts = []
    for contact in contacts:
        if contact.bodies[0].entity == actor:
            entity_contacts.append((contact, True))
        elif contact.bodies[1].entity == actor:
            entity_contacts.append((contact, False))
    return entity_contacts


def get_actors_contacts(
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


def get_articulation_contacts(
    contacts: List[physx.PhysxContact],
    articulation: physx.PhysxArticulation,
    excluded_entities: Optional[List[sapien.Entity]] = None,
    included_links: Optional[List[physx.PhysxArticulationLinkComponent]] = None,
) -> List[Tuple[physx.PhysxContact, bool]]:
    articulation_contacts = []
    links = articulation.get_links()
    if excluded_entities is None:
        excluded_entities = []
    if included_links is None:
        included_links = links
    for contact in contacts:
        if contact.bodies[0] in included_links:
            if contact.bodies[1] in links:
                continue
            if contact.bodies[1].entity in excluded_entities:
                continue
            articulation_contacts.append((contact, True))
        elif contact.bodies[1] in included_links:
            if contact.bodies[0] in links:
                continue
            if contact.bodies[0].entity in excluded_entities:
                continue
            articulation_contacts.append((contact, False))
    return articulation_contacts


def compute_max_impulse_norm(contact_infos: List[Tuple[physx.PhysxContact, bool]]):
    max_impulse_norms = [0]
    for contact, flag in contact_infos:
        max_impulse_norm = max(
            [np.linalg.norm(point.impulse) for point in contact.points]
        )
        max_impulse_norms.append(max_impulse_norm)
    return max(max_impulse_norms)


def get_articulation_max_impulse_norm(
    contacts: List[physx.PhysxContact],
    articulation: physx.PhysxArticulation,
    excluded_entities: Optional[List[sapien.Entity]] = None,
):
    articulation_contacts = get_articulation_contacts(
        contacts, articulation, excluded_entities
    )
    max_impulse_norm = compute_max_impulse_norm(articulation_contacts)
    return max_impulse_norm


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


def look_at(eye, target, up=(0, 0, 1)) -> sapien.Pose:
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
        sapien.Pose: camera pose
    """
    forward = normalize_vector(np.array(target) - np.array(eye))
    up = normalize_vector(up)
    left = np.cross(up, forward)
    up = np.cross(forward, left)
    rotation = np.stack([forward, left, up], axis=1)
    return sapien.Pose(p=eye, q=mat2quat(rotation))


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
