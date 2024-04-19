"""
Converts assets from .stl/.obj to .glb

Useful bash commands:

Convert all files in a directory of given endings

find /path/to/your/directory -iname "*.STL" -exec python manualtest/convert_assets_to_glb.py -f {} \;


"""
import argparse
import os.path as osp

import bpy
import numpy as np


# Some basic blender py utilities
def del_obj(obj_name):
    bpy.data.objects[obj_name].select_set(True)
    bpy.ops.object.delete()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="File to convert to .glb"
    )
    args = parser.parse_args()
    return args


def main(args):
    file_path = args.file
    if not osp.exists(file_path):
        raise FileNotFoundError(f"Could not find file {file_path}")
    for obj in ["Cube", "Camera", "Light"]:
        if obj in bpy.data.objects:
            del_obj(obj)
    base_file_name, ext = osp.splitext(osp.basename(file_path))
    if ext.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=file_path)
    elif ext.lower() == ".stl":
        bpy.ops.wm.stl_import(filepath=file_path)
    else:
        raise RuntimeError(
            f"converting object with extension {ext} is not supported at the moment."
        )

    for obj in bpy.data.objects.keys():
        bpy.data.objects[obj].rotation_euler[0] = np.pi / 2

    output_path = osp.join(osp.dirname(file_path), f"{base_file_name}.glb")
    bpy.ops.export_scene.gltf(filepath=output_path)
    bpy.ops.object.delete()


if __name__ == "__main__":
    main(parse_args())
