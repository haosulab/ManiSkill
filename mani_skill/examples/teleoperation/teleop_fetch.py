import gymnasium as gym
import numpy as np
import sapien.core as sapien
import sapien.utils.viewer

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import Pose

import tyro
from dataclasses import dataclass
import time
import string # For checking digit keys

@dataclass
class Args:
    env_id: str = "ArchitecTHOR_SceneManipulation-v1"
    """Environment ID."""
    obs_mode: str = "none"
    """Observation mode."""
    # control_mode: str = "pd_ee_delta_pose_plus_pd_gripper_pos" # Let's try this composite mode
    control_mode: str = "pd_joint_delta_pos" # Start with simpler mode first, check action space
    """Control mode for the agent."""
    render_mode: str = "human"
    """Render mode."""
    shader: str = "default"
    """Shader for rendering."""
    seed: int = 42
    """Random seed."""

def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)

    # Note: ArchitecTHOR_SceneManipulation-v1 likely defaults to Fetch
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode="none",
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        shader_dir=args.shader, # Corrected arg name? Check gym.make signature if error
        # Assuming Fetch is default, no need for robot_uids unless overriding
    )

    print("Environment created.")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space) # Should print Box(..., (13,), ...)
    print("Control mode:", env.unwrapped.control_mode) # Should print pd_joint_delta_pos

    action_dim = env.action_space.shape[0]
    print(f"Detected Action Dimension: {action_dim}") # Confirm it's 13

    # --- Get Controlled Joint Names and Indices ---
    controlled_joint_names = []
    gripper_action_index = -1
    base_x_idx, base_rz_idx = -1, -1
    torso_lift_idx = -1
    arm_joint_indices = [-1] * 7 # Indices for the 7 arm joints

    try:
        controller = env.unwrapped.agent.controller
        joint_names_source = None
        # Try different ways to get the ordered list of controlled joints
        if hasattr(controller, 'controlled_joint_names'):
             joint_names_source = controller.controlled_joint_names
        elif hasattr(controller, 'joint_names'):
             joint_names_source = controller.joint_names
        elif hasattr(controller, 'configs'):
             if isinstance(controller.configs, dict):
                 temp_names = []
                 for ctrl_cfg in controller.configs.values():
                     if hasattr(ctrl_cfg, 'joint_names'):
                         temp_names.extend(ctrl_cfg.joint_names)
                 joint_names_source = temp_names
             elif hasattr(controller.configs, 'joint_names'):
                 joint_names_source = controller.configs.joint_names

        if joint_names_source is None:
             print("Warning: Could not find joint names from controller. Falling back to robot active joints.")
             joint_names_source = [j.name for j in env.unwrapped.agent.robot.active_joints]
        
        controlled_joint_names = joint_names_source

        print("\n--- Controlled Joints ---")
        if len(controlled_joint_names) == action_dim:
             arm_joint_count = 0
             for i, name in enumerate(controlled_joint_names):
                 print(f"  Action Index {i}: {name}")
                 # Identify gripper
                 if "gripper" in name and gripper_action_index == -1:
                     gripper_action_index = i
                 # Identify base/torso
                 elif name == "root_x_axis_joint": base_x_idx = i
                 elif name == "root_z_rotation_joint": base_rz_idx = i
                 elif name == "torso_lift_joint": torso_lift_idx = i
                 # Identify arm joints (assuming they appear sequentially)
                 elif arm_joint_count < 7:
                     # Check against known Fetch arm joint names if needed for robustness
                     # known_arm_joints = ["shoulder_pan", "shoulder_lift", "upperarm_roll", "elbow_flex", "forearm_roll", "wrist_flex", "wrist_roll"]
                     # if any(known in name for known in known_arm_joints):
                     arm_joint_indices[arm_joint_count] = i
                     arm_joint_count += 1

             if gripper_action_index != -1: print(f"==> Gripper Index: {gripper_action_index}")
             else: print("==> Warning: Gripper index not found.")
             print(f"==> Base/Torso Indices (X={base_x_idx}, Rz={base_rz_idx}, Lift={torso_lift_idx})")
             print(f"==> Arm Joint Indices: {arm_joint_indices}")
             if -1 in arm_joint_indices: print("==> Warning: Not all arm joints mapped.")

        else:
            print(f"Warning: Mismatch between joint names ({len(controlled_joint_names)}) and action dim ({action_dim}). Using fallbacks.")
            if action_dim == 13: # Previous fallback assumption
                 arm_joint_indices = list(range(7)) # 0-6
                 gripper_action_index = 7
                 # Body joints (head pan, head tilt, torso lift?) - assume order
                 torso_lift_idx = 10 # index 10? Need confirmation
                 # Base joints (X, Y, Rz?) - assume order
                 base_x_idx = 11 # index 11?
                 base_rz_idx = 12 # index 12?
                 print(f"Using fallback indices: Arm={arm_joint_indices}, Gripper={gripper_action_index}, Lift={torso_lift_idx}, BaseX={base_x_idx}, BaseRz={base_rz_idx}")

    except Exception as e:
        print(f"Error determining controlled joint names/indices: {e}")
        # Fallback guess if exception occurred
        if action_dim == 13:
             arm_joint_indices = list(range(7))
             gripper_action_index = 7
             torso_lift_idx = 10
             base_x_idx = 11
             base_rz_idx = 12
             print(f"Using fallback indices due to error: Arm={arm_joint_indices}, Gripper={gripper_action_index}, Lift={torso_lift_idx}, BaseX={base_x_idx}, BaseRz={base_rz_idx}")

    obs, _ = env.reset(seed=args.seed)
    viewer = env.render()

    if viewer is None:
        print("Error: Human rendering mode requires a graphical display.")
        env.close()
        return

    viewer.paused = False

    # Control parameters
    base_move_speed = 0.5 # Delta position per step for base X
    base_rot_speed = 0.1  # Delta angle per step for base Rz
    torso_lift_speed = 0.1 # Delta position per step for torso lift
    arm_joint_delta_speed = 0.1 # Delta angle per step for arm joints
    gripper_step = 0.01 # How much to change gripper target per key press
    gripper_min = -0.01
    gripper_max = 0.05
    current_gripper_target = gripper_max # Start open

    print("\n--- Teleop Controls ---")
    print(" J/L: Move Base Forward/Backward")
    print(" I/K: Move Torso Up/Down")
    print(" Q/E: Rotate Base Left/Right")
    print(" O/P : Decrease/Increase Gripper Opening")
    print(" 1-7: Arm Joints +Delta")
    print(" z-m: Arm Joints -Delta (z->1, x->2, c->3, v->4, b->5, n->6, m->7)") # Updated arm keys
    print(" R: Reset Environment")
    print(" Esc: Quit")
    print("-----------------------\n")

    action = np.zeros(action_dim, dtype=np.float32)
    step_counter = 0

    # Map negative arm keys to joint indices
    neg_arm_keys = {'z': 0, 'x': 1, 'c': 2, 'v': 3, 'b': 4, 'n': 5, 'm': 6}

    while True:
        if viewer.closed:
            break

        # Check keyboard inputs
        should_quit = viewer.window.key_press('escape')
        should_reset = viewer.window.key_press('r')

        # Base/Torso movement keys
        move_forward = viewer.window.key_down('j')
        move_backward = viewer.window.key_down('l')
        torso_up = viewer.window.key_down('i')
        torso_down = viewer.window.key_down('k')
        rotate_left = viewer.window.key_down('q')
        rotate_right = viewer.window.key_down('e')

        # Gripper keys
        close_gripper_step = viewer.window.key_down('o')
        open_gripper_step = viewer.window.key_down('p')

        # Arm keys (Check 1-7 for positive, z-m for negative)
        arm_deltas = np.zeros(7)
        active_arm_key = False
        # Positive delta
        for i in range(7):
            key_code = str(i + 1)
            if viewer.window.key_down(key_code):
                arm_deltas[i] = arm_joint_delta_speed
                active_arm_key = True
        # Negative delta
        for key_code, joint_index in neg_arm_keys.items():
             if joint_index < 7 and viewer.window.key_down(key_code):
                 # Prevent conflicting inputs for the same joint
                 if arm_deltas[joint_index] == 0:
                    arm_deltas[joint_index] = -arm_joint_delta_speed
                    active_arm_key = True
                 else: # If both positive and negative keys pressed, cancel out
                    arm_deltas[joint_index] = 0


        if should_quit:
            break
        if should_reset:
            print("Resetting environment...")
            action.fill(0.0)
            obs, _ = env.reset()
            current_gripper_target = gripper_max
            step_counter = 0
            print("Reset complete.")
            continue

        # Reset all actions except gripper target
        action.fill(0.0)
        action_taken_this_step = False

        # --- Base and Torso Control ---
        if base_x_idx != -1 and move_forward:
             action[base_x_idx] = base_move_speed
             action_taken_this_step = True
        if base_x_idx != -1 and move_backward:
             action[base_x_idx] = -base_move_speed
             action_taken_this_step = True
        if torso_lift_idx != -1 and torso_up:
             action[torso_lift_idx] = torso_lift_speed
             action_taken_this_step = True
        if torso_lift_idx != -1 and torso_down:
             action[torso_lift_idx] = -torso_lift_speed
             action_taken_this_step = True
        if base_rz_idx != -1 and rotate_left:
             action[base_rz_idx] = base_rot_speed
             action_taken_this_step = True
        if base_rz_idx != -1 and rotate_right:
             action[base_rz_idx] = -base_rot_speed
             action_taken_this_step = True

        # --- Arm Control ---
        if active_arm_key:
             for i in range(7):
                 if arm_joint_indices[i] != -1: # Check if index is valid
                     action[arm_joint_indices[i]] = arm_deltas[i]
             action_taken_this_step = True


        # --- Gripper Control (Incremental Target, using O and P) ---
        if gripper_action_index != -1:
            new_gripper_target = current_gripper_target
            gripper_target_changed = False
            if close_gripper_step:
                new_gripper_target -= gripper_step
                action_taken_this_step = True
                gripper_target_changed = True
            if open_gripper_step:
                new_gripper_target += gripper_step
                action_taken_this_step = True
                gripper_target_changed = True

            new_gripper_target_clamped = np.clip(new_gripper_target, gripper_min, gripper_max)
            if new_gripper_target_clamped != current_gripper_target:
                 current_gripper_target = new_gripper_target_clamped
                 print(f"Gripper Target Updated: {current_gripper_target:.3f}")
            elif gripper_target_changed:
                 if not getattr(main, f'_gripper_limit_warned_{"min" if close_gripper_step else "max"}', False):
                    print(f"Gripper reached {'minimum' if close_gripper_step else 'maximum'} limit: {current_gripper_target:.3f}")
                    setattr(main, f'_gripper_limit_warned_{"min" if close_gripper_step else "max"}', True)
            else:
                 setattr(main, '_gripper_limit_warned_min', False)
                 setattr(main, '_gripper_limit_warned_max', False)

            action[gripper_action_index] = current_gripper_target


        # --- Step the environment ---
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            step_counter += 1

            if truncated and not getattr(main, '_truncated_warned', False):
                 print(f"Info: Episode truncated at step {step_counter}. Max steps likely reached. Use 'R' to reset or 'Q' to quit.")
                 main._truncated_warned = True
            if not truncated:
                 main._truncated_warned = False

        except Exception as e:
            print(f"Error during env.step: {e}")
            print("Action attempted:", action)
            break

        env.render()

    print("Closing environment.")
    env.close()

if __name__ == "__main__":
    # Set default control mode here
    Args.control_mode = "pd_joint_delta_pos"
    args = tyro.cli(Args)
    main(args)
