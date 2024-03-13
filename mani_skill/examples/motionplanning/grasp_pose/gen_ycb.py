import numpy as np
from utils import compute_grasp_poses

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.envs.pick_and_place.pick_single import load_json

YCB_DIR = ASSET_DIR / "mani_skill2_ycb"


def main():
    model_db = load_json(YCB_DIR / "info_pick_v0.json")
    model_ids = [
        "019_pitcher_base",
        "024_bowl",
        "025_mug",
        "035_power_drill",
        "052_extra_large_clamp",
        "065-f_cups",
        "065-g_cups",
        "065-h_cups",
        "065-i_cups",
        "065-j_cups",
        "072-a_toy_airplane",
        "072-b_toy_airplane",
    ]
    output_dir = YCB_DIR / "grasp_poses_info_pick_v0_panda_v2"
    output_dir.mkdir(exist_ok=True)

    gripper_urdf = str(PACKAGE_ASSET_DIR / "robots/panda/panda_v2_gripper.urdf")
    gripper_srdf = str(PACKAGE_ASSET_DIR / "robots/panda/panda_v2.srdf")

    for model_id in model_ids:
        scales = model_db[model_id]["scales"]
        assert len(scales) == 1
        mesh_path = str(YCB_DIR / f"models/{model_id}/collision.obj")

        n_pts = 2048
        n_angles = 12
        if model_id == "019_pitcher_base":
            n_angles = 24
        if model_id == "072-a_toy_airplane":
            n_angles = 24

        while True:
            grasp_poses = compute_grasp_poses(
                mesh_path,
                int(n_pts),
                gripper_urdf,
                gripper_srdf,
                "panda_hand_tcp",
                n_angles=n_angles,
                mesh_scale=scales[0],
                octree_resolution=0.005,
                add_ground=True,
                # vis=True,
            )
            if len(grasp_poses) < 16:
                n_pts *= 2
            elif len(grasp_poses) > 100:
                n_pts /= 2
            else:
                break
            print(n_pts)
            if n_pts > 8096 or n_pts < 512:
                break

        print(model_id, len(grasp_poses))
        np.save(output_dir / f"{model_id}.npy", grasp_poses)


if __name__ == "__main__":
    main()
