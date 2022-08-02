from typing import List

import numpy as np
import sapien.core as sapien
import trimesh


def get_actor_meshes(actor: sapien.ActorBase):
    """Get actor meshes in the actor frame."""
    meshes = []
    for col_shape in actor.get_collision_shapes():
        geom = col_shape.geometry
        if isinstance(geom, sapien.BoxGeometry):
            mesh = trimesh.creation.box(extents=2 * geom.half_lengths)
        elif isinstance(geom, sapien.CapsuleGeometry):
            mesh = trimesh.creation.capsule(
                height=2 * geom.half_length, radius=geom.radius
            )
        elif isinstance(geom, sapien.SphereGeometry):
            mesh = trimesh.creation.icosphere(radius=geom.radius)
        elif isinstance(geom, sapien.PlaneGeometry):
            continue
        elif isinstance(
            geom, (sapien.ConvexMeshGeometry, sapien.NonconvexMeshGeometry)
        ):
            vertices = geom.vertices  # [n, 3]
            faces = geom.indices  # [m * 3]
            faces = [faces[i : i + 3] for i in range(0, len(faces), 3)]
            vertices = vertices * geom.scale
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            raise TypeError(type(geom))
        mesh.apply_transform(col_shape.get_local_pose().to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None


def get_actor_mesh(actor: sapien.ActorBase, to_world_frame=True):
    mesh = merge_meshes(get_actor_meshes(actor))
    if mesh is None:
        return None
    if to_world_frame:
        T = actor.pose.to_transformation_matrix()
        mesh.apply_transform(T)
    return mesh


def get_articulation_meshes(
    articulation: sapien.ArticulationBase, exclude_link_names=()
):
    """Get link meshes in the world frame."""
    meshes = []
    for link in articulation.get_links():
        if link.name in exclude_link_names:
            continue
        mesh = get_actor_mesh(link, True)
        if mesh is None:
            continue
        meshes.append(mesh)
    return meshes
