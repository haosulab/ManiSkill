import argparse
import numpy as np
import gymnasium as gym
import mani_skill.envs  # Register ManiSkill environments

"""Check whether PickCubeDiscreteInit and TableScanDiscreteNoRobotEnv
return *consistent* cube positions when the same discrete index is used.

Usage
-----
python -m map_rl.check_discrete_env_alignment --grid-dim 10 --indices 0 42 99
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Check discrete-env alignment")
    parser.add_argument("--grid-dim", type=int, default=30,
                        help="grid_dim (N) for the N×N discrete grid")
    parser.add_argument("--indices", type=int, nargs="*", default=[0, 65, 532, 334],
                        help="List of global indices to test")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="Tolerance for position mismatch (m)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Create both environments (single-env each)
    # ------------------------------------------------------------------
    env_pick = gym.make(
        "PickCubeDiscreteInit-v1",
        num_envs=1,
        grid_dim=args.grid_dim,
        robot_uids="xarm6_robotiq",
        obs_mode="state",
        render_mode="none",
    )

    env_scan = gym.make(
        "TableScanDiscreteNoRobot-v0",
        num_envs=1,
        grid_dim=args.grid_dim,
        robot_uids="none",
        obs_mode="state",
        render_mode="none",
    )

    ok_all = True
    for idx in args.indices:
        # ----------------- Reset with identical global_idx --------------
        _ = env_pick.reset(options={"global_idx": [idx]})
        _ = env_scan.reset(options={"global_idx": [idx]})

        cube_pos_pick = env_pick.unwrapped.cube.pose.p[0]  # shape (3,)
        cube_pos_scan = env_scan.unwrapped.cube.pose.p[0]
        diff = np.linalg.norm(cube_pos_pick - cube_pos_scan)
        same = diff < args.tol
        ok_all &= same
        status = "OK" if same else "MISMATCH"
        print(f"Idx {idx:3d}: Δ={diff:.6f}  -> {status}")

    env_pick.close()
    env_scan.close()

    if ok_all:
        print("\nAll tested indices consistent ✅")
    else:
        print("\nSome indices mismatched ❌")


if __name__ == "__main__":
    main()
