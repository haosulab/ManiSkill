import numpy as np
import scipy.spatial
import trimesh


def generate_convex_stl(input_stl_path, output_stl_path):

    # Load the mesh
    original_mesh = trimesh.load(input_stl_path)
    vertices = original_mesh.vertices
    hull = scipy.spatial.ConvexHull(vertices)
    convex_vertices = vertices[hull.vertices]
    convex_faces = hull.simplices
    # Create mapping from original vertex indices to new indices in convex_vertices
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}

    # Remap face indices to use new vertex ordering
    convex_faces = np.array(
        [[vertex_map[idx] for idx in face] for face in hull.simplices]
    )
    convex_mesh = trimesh.Trimesh(vertices=convex_vertices, faces=convex_faces)

    # Export as STL
    convex_mesh.export(output_stl_path)
    print(f"Convex hull saved to {output_stl_path}")


if __name__ == "__main__":
    import argparse
    import glob
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing mesh files to process")
    args = parser.parse_args()

    # Get all .stl and .ply files in the folder
    mesh_files = glob.glob(os.path.join(args.folder, "*.stl"))
    mesh_files.extend(glob.glob(os.path.join(args.folder, "*.ply")))

    for input_path in mesh_files:
        # Skip if file is already a convex hull
        if input_path.endswith(".convex.stl"):
            continue

        # Generate output path by inserting .convex before .stl
        base, ext = os.path.splitext(input_path)
        output_path = input_path + ".convex.stl"

        print(f"Processing {input_path}...")
        generate_convex_stl(input_path, output_path)
