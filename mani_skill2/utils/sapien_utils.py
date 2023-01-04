from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.quaternions import mat2quat


def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    if norm < eps:
        return np.zeros_like(x)
    else:
        return x / norm


def vectorize_pose(pose: sapien.Pose):
    return np.hstack([pose.p, pose.q])


def set_actor_visibility(actor: sapien.Actor, visibility):
    for v in actor.get_visual_bodies():
        v.set_visibility(visibility)


@contextmanager
def set_default_physical_material(
    material: sapien.PhysicalMaterial, scene: sapien.Scene
):
    """Set default physical material within the context.

    Args:
        material (sapien.PhysicalMaterial): physical material to use as default.
        scene (sapien.Scene): scene instance.

    Yields:
        sapien.PhysicalMaterial: original default physical material.

    Example:
        with set_default_physical_material(material, scene):
            ...
    """
    old_material = scene.default_physical_material
    scene.default_physical_material = material
    try:
        yield old_material
    finally:
        scene.default_physical_material = old_material


def get_entity_by_name(entities, name: str, is_unique=True):
    """Get a Sapien.Entity given the name.

    Args:
        entities (List[sapien.Entity]): entities (link, joint, ...) to query.
        name (str): name for query.
        is_unique (bool, optional):
            whether the name should be unique. Defaults to True.

    Raises:
        RuntimeError: The name is not unique when @is_unique is True.

    Returns:
        sapien.Entity or List[sapien.Entity]:
            matched entity or entities. None if no matches.
    """
    matched_entities = [x for x in entities if x.get_name() == name]
    if len(matched_entities) > 1:
        if not is_unique:
            return matched_entities
        else:
            raise RuntimeError(f"Multiple entities with the same name {name}.")
    elif len(matched_entities) == 1:
        return matched_entities[0]
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


def parse_urdf_config(config_dict: dict, scene: sapien.Scene) -> Dict:
    """Parse config from dict for SAPIEN URDF loader.

    Args:
        config_dict (dict): a dict containing link physical properties.
        scene (sapien.Scene): simualtion scene

    Returns:
        Dict: urdf config passed to `sapien.URDFLoader.load`.
    """
    urdf_config = deepcopy(config_dict)

    # Create the global physical material for all links
    mtl_cfg = urdf_config.pop("material", None)
    if mtl_cfg is not None:
        urdf_config["material"] = scene.create_physical_material(**mtl_cfg)

    # Create link-specific physical materials
    materials = {}
    for k, v in urdf_config.pop("_materials", {}).items():
        materials[k] = scene.create_physical_material(**v)

    # Specify properties for links
    for link_config in urdf_config.get("link", {}).values():
        # Substitute with actual material
        link_config["material"] = materials[link_config["material"]]

    return urdf_config


# -------------------------------------------------------------------------- #
# Entity state
# -------------------------------------------------------------------------- #
def get_actor_state(actor: sapien.Actor):
    pose = actor.get_pose()
    if actor.type == "static":
        vel = np.zeros(3)
        ang_vel = np.zeros(3)
    else:
        vel = actor.get_velocity()  # [3]
        ang_vel = actor.get_angular_velocity()  # [3]
    return np.hstack([pose.p, pose.q, vel, ang_vel])


def set_actor_state(actor: sapien.Actor, state: np.ndarray):
    assert len(state) == 13, len(state)
    actor.set_pose(Pose(state[0:3], state[3:7]))
    if actor.type != "static" and actor.type != "kinematic":
        actor.set_velocity(state[7:10])
        actor.set_angular_velocity(state[10:13])


def get_articulation_state(articulation: sapien.Articulation):
    root_link = articulation.get_links()[0]
    pose = root_link.get_pose()
    vel = root_link.get_velocity()  # [3]
    ang_vel = root_link.get_angular_velocity()  # [3]
    qpos = articulation.get_qpos()
    qvel = articulation.get_qvel()
    return np.hstack([pose.p, pose.q, vel, ang_vel, qpos, qvel])


def set_articulation_state(articulation: sapien.Articulation, state: np.ndarray):
    articulation.set_root_pose(Pose(state[0:3], state[3:7]))
    articulation.set_root_velocity(state[7:10])
    articulation.set_root_angular_velocity(state[10:13])
    qpos, qvel = np.split(state[13:], 2)
    articulation.set_qpos(qpos)
    articulation.set_qvel(qvel)


def get_articulation_padded_state(articulation: sapien.Articulation, max_dof: int):
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
# -------------------------------------------------------------------------- #
def get_pairwise_contacts(
    contacts: List[sapien.Contact], actor0: sapien.ActorBase, actor1: sapien.ActorBase
) -> List[Tuple[sapien.Contact, bool]]:
    pairwise_contacts = []
    for contact in contacts:
        if contact.actor0 == actor0 and contact.actor1 == actor1:
            pairwise_contacts.append((contact, True))
        elif contact.actor0 == actor1 and contact.actor1 == actor0:
            pairwise_contacts.append((contact, False))
    return pairwise_contacts


def compute_total_impulse(contact_infos: List[Tuple[sapien.Contact, bool]]):
    total_impulse = np.zeros(3)
    for contact, flag in contact_infos:
        contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
        # Impulse is applied on the first actor
        total_impulse += contact_impulse * (1 if flag else -1)
    return total_impulse


def get_pairwise_contact_impulse(
    contacts: List[sapien.Contact], actor0: sapien.ActorBase, actor1: sapien.ActorBase
):
    pairwise_contacts = get_pairwise_contacts(contacts, actor0, actor1)
    total_impulse = compute_total_impulse(pairwise_contacts)
    return total_impulse


def get_actor_contacts(
    contacts: List[sapien.Contact], actor: sapien.ActorBase
) -> List[Tuple[sapien.Contact, bool]]:
    actor_contacts = []
    for contact in contacts:
        if contact.actor0 == actor:
            actor_contacts.append((contact, True))
        elif contact.actor1 == actor:
            actor_contacts.append((contact, False))
    return actor_contacts


def get_articulation_contacts(
    contacts: List[sapien.Contact],
    articulation: sapien.Articulation,
    excluded_actors: Optional[List[sapien.Actor]] = None,
    included_links: Optional[List[sapien.Link]] = None,
) -> List[Tuple[sapien.Contact, bool]]:
    articulation_contacts = []
    links = articulation.get_links()
    if excluded_actors is None:
        excluded_actors = []
    if included_links is None:
        included_links = links
    for contact in contacts:
        if contact.actor0 in included_links:
            if contact.actor1 in links:
                continue
            if contact.actor1 in excluded_actors:
                continue
            articulation_contacts.append((contact, True))
            # print(contact.actor0, contact.actor1)
        elif contact.actor1 in included_links:
            if contact.actor0 in links:
                continue
            if contact.actor0 in excluded_actors:
                continue
            articulation_contacts.append((contact, False))
            # print(contact.actor0, contact.actor1)
    return articulation_contacts


def compute_max_impulse_norm(contact_infos: List[Tuple[sapien.Contact, bool]]):
    max_impulse_norms = [0]
    for contact, flag in contact_infos:
        max_impulse_norm = max(
            [np.linalg.norm(point.impulse) for point in contact.points]
        )
        max_impulse_norms.append(max_impulse_norm)
    return max(max_impulse_norms)


def get_articulation_max_impulse_norm(
    contacts: List[sapien.Contact],
    articulation: sapien.Articulation,
    excluded_actors: Optional[List[sapien.Actor]] = None,
):
    articulation_contacts = get_articulation_contacts(
        contacts, articulation, excluded_actors
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


def set_render_material(material: sapien.RenderMaterial, **kwargs):
    for k, v in kwargs.items():
        if k == "color":
            material.set_base_color(v)
        else:
            setattr(material, k, v)
    return material


def set_articulation_render_material(articulation: sapien.Articulation, **kwargs):
    for link in articulation.get_links():
        for b in link.get_visual_bodies():
            for s in b.get_render_shapes():
                mat = s.material
                set_render_material(mat, **kwargs)
                # s.set_material(mat)


# -------------------------------------------------------------------------- #
# Misc
# -------------------------------------------------------------------------- #
def check_joint_stuck(
    articulation: sapien.Articulation,
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


def check_actor_static(actor: sapien.Actor, lin_thresh=1e-3, ang_thresh=1e-2):
    return (
        np.linalg.norm(actor.velocity) <= lin_thresh
        and np.linalg.norm(actor.angular_velocity) <= ang_thresh
    )
