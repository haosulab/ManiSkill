from copy import deepcopy
from typing import Union, List

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


def np2pcd(points, colors=None, normals=None):
    """Convert numpy array to open3d PointCloud."""
    if isinstance(points, o3d.geometry.PointCloud):
        return points
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.copy())
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


O3D_GEOMETRIES = (o3d.geometry.Geometry3D, o3d.t.geometry.Geometry,
                  rendering.TriangleMeshModel)
ANY_O3D_GEOMETRY = Union[O3D_GEOMETRIES]
def transform_geometry(geometry: ANY_O3D_GEOMETRY, T: np.ndarray) -> ANY_O3D_GEOMETRY:
    """Apply transformation to o3d geometry, always returns a copy

    :param T: transformation matrix, [4, 4] np.floating np.ndarray
    """
    if isinstance(geometry, rendering.TriangleMeshModel):
        out_geometry = rendering.TriangleMeshModel()
        out_geometry.meshes = [
            rendering.TriangleMeshModel.MeshInfo(
                deepcopy(mesh_info.mesh).transform(T),
                mesh_info.mesh_name,
                mesh_info.material_idx
            )
            for mesh_info in geometry.meshes
        ]
        out_geometry.materials = geometry.materials
    elif isinstance(geometry, (o3d.geometry.Geometry3D, o3d.t.geometry.Geometry)):
        out_geometry = deepcopy(geometry).transform(T)
    else:
        raise TypeError(f"Unknown o3d geometry type: {type(geometry)}")
    return out_geometry


O3D_GEOMETRY_LIST = Union[tuple(List[t] for t in O3D_GEOMETRIES)]
def merge_geometries(geometries: O3D_GEOMETRY_LIST) -> ANY_O3D_GEOMETRY:
    """Merge a list of o3d geometries, must be of same type"""
    geometry_types = set([type(geometry) for geometry in geometries])
    assert len(geometry_types) == 1, f"Not the same geometry type: {geometry_types = }"

    merged_geometry = next(iter(geometry_types))()
    for i, geometry in enumerate(geometries):
        if isinstance(geometry, rendering.TriangleMeshModel):
            num_materials = len(merged_geometry.materials)
            merged_geometry.meshes += [
                rendering.TriangleMeshModel.MeshInfo(
                    deepcopy(mesh_info.mesh),
                    f"mesh_{i}_{mesh_info.mesh_name}".strip('_'),
                    mesh_info.material_idx + num_materials
                )
                for mesh_info in geometry.meshes
            ]
            merged_geometry.materials += geometry.materials
        elif isinstance(geometry, (o3d.geometry.Geometry3D, o3d.t.geometry.Geometry)):
            merged_geometry += geometry
        else:
            raise TypeError(f"Unknown o3d geometry type: {type(geometry)}")
    return merged_geometry
