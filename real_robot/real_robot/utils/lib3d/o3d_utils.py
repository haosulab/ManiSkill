import numpy as np
import open3d as o3d


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
