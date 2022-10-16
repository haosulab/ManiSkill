from collections import OrderedDict
from typing import Union

import h5py
import numpy as np
import sapien.core as sapien
import trimesh


def actor2meshes(actor: sapien.Actor, visual=False, return_primitives=False):
    primitives = []
    meshes = []
    if visual:
        bodies = actor.get_visual_bodies()
        for body in bodies:
            if body.type == "mesh":
                for shape in body.get_render_shapes():
                    mesh = trimesh.Trimesh(
                        shape.mesh.vertices * body.scale,
                        shape.mesh.indices.reshape((-1, 3)),
                    )
                    mesh.apply_transform(body.local_pose.to_transformation_matrix())
                    meshes.append(mesh)
            elif body.type == "box":
                half_lengths = body.half_lengths
                if return_primitives:
                    primitives.append(("box", half_lengths, body.local_pose))
                else:
                    mesh = trimesh.creation.box(half_lengths * 2)
                    meshes.append(mesh)
            elif body.type == "capsule":
                radius = body.radius
                half_length = body.half_length
                if return_primitives:
                    primitives.append(
                        ("capsule", [radius, half_length], body.local_pose)
                    )
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
    else:
        collisions = actor.get_collision_shapes()
        meshes = []
        for s in collisions:
            s: sapien.CollisionShape
            g: sapien.CollisionGeometry = s.geometry

            if isinstance(g, sapien.ConvexMeshGeometry):
                mesh = trimesh.Trimesh(g.vertices * g.scale, g.indices.reshape(-1, 3))
                assert np.allclose(g.rotation, np.array([1, 0, 0, 0]))
                mesh.apply_transform(s.get_local_pose().to_transformation_matrix())
                meshes.append(mesh)
            elif isinstance(g, sapien.BoxGeometry):
                half_lengths = g.half_lengths
                if return_primitives:
                    primitives.append(("box", half_lengths, s.get_local_pose()))
                else:
                    mesh = trimesh.creation.box(half_lengths * 2)
                    meshes.append(mesh)
            elif isinstance(g, sapien.CapsuleGeometry):
                radius = g.radius
                half_length = g.half_length
                if return_primitives:
                    primitives.append(
                        ("capsule", [radius, half_length], s.get_local_pose())
                    )
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    if return_primitives:
        return meshes, primitives
    return meshes


def trimesh2sdf(meshes, margin, dx, bbox=None):
    if meshes is None:
        return None
    mesh = trimesh.util.concatenate(meshes)

    if bbox is None:
        bbox = mesh.bounds.copy()

    sdfs = []
    normals = []
    for mesh in meshes:
        center = (bbox[0] + bbox[1]) / 2
        res = np.ceil((bbox[1] - bbox[0] + margin * 2) / dx).astype(int)
        lower = center - res * dx / 2.0

        points = np.zeros((res[0], res[1], res[2], 3))
        x = np.arange(0.5, res[0]) * dx + lower[0]
        y = np.arange(0.5, res[1]) * dx + lower[1]
        z = np.arange(0.5, res[2]) * dx + lower[2]

        points[..., 0] += x[:, None, None]
        points[..., 1] += y[None, :, None]
        points[..., 2] += z[None, None, :]

        points = points.reshape((-1, 3))

        query = trimesh.proximity.ProximityQuery(mesh)
        sdf = query.signed_distance(points) * -1.0

        surface_points, _, tri_id = query.on_surface(points)
        face_normal = mesh.face_normals[tri_id]
        normal = (points - surface_points) * np.sign(sdf)[..., None]
        length = np.linalg.norm(normal, axis=-1)
        mask = length < 1e6
        normal[mask] = face_normal[mask]
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
        sdf = sdf.reshape(res)
        normal = normal.reshape((res[0], res[1], res[2], 3))

        sdfs.append(sdf)
        normals.append(normal)

    if len(sdfs) == 1:
        sdf = sdfs[0]
        normal = normals[0]
    else:
        sdfs = np.stack(sdfs)
        normals = np.stack(normals)
        index = np.expand_dims(sdfs.argmin(0), 0)
        sdf = np.take_along_axis(sdfs, index, 0)[0]
        normal = np.take_along_axis(normals, np.expand_dims(index, -1), 0)[0]

    return {
        "sdf": sdf,
        "normal": normal,
        "position": lower,
        "scale": np.ones(3) * dx,
        "dim": res,
    }


def load_h5_as_dict(h5file: Union[h5py.File, h5py.Group]):
    out = OrderedDict()
    for key in h5file.keys():
        if isinstance(h5file[key], h5py.Group):
            out[key] = load_h5_as_dict(h5file[key])
        else:
            out[key] = h5file[key][:]
    return out
