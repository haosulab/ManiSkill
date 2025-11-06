#!/usr/bin/env python3
# run_peg_in_hole_scripted.py
import time
import numpy as np
import gymnasium as gym

import mani_skill                  # ensure registry loads
import peg_in_hole                 # registers PegInHole-v1

from gymnasium.wrappers import ClipAction, RescaleAction

# -------- Tunables --------
FPS         = 60.0

HOVER_Z     = 0.22
PREGRASP_Z  = 0.115          # above peg
GRASP_Z     = 0.100          # grasp band on upper body
LIFT_Z      = 0.18
INSERT_Z    = 0.02

STEP_COARSE = 0.012
STEP_FINE   = 0.0030
ROT_FINE    = 0.03

XY_TOL      = 0.008
Z_TOL       = 0.004

OPEN_CMD    = +0.05          # keep within RescaleAction bounds
PREOPEN     = +0.015
CLOSE_CMD   = -0.05
HOLD_STEPS  = 120            # let PD converge while closing

# Preload (fight skating) + micro push
PRELOAD_Z   = -0.018         # 18 mm down; raises normal force
PUSH_IN     = 0.006          # tiny lateral push during close

# Spiral & tilt insertion
SPIRAL_R      = 0.0035
SPIRAL_STEPS  = 24
DITHER_XY     = 0.0010
DITHER_YAW    = 0.01
TILT_ROLL_MAX = 0.12
TILT_DECAY_STEPS = 60


def make_env():
    env = gym.make(
        "PegInHole-v1",
        obs_mode="state_dict",
        render_mode="human",
        control_mode="pd_ee_delta_pose",   # 7D: dx dy dz dR dP dY + gripper last dim
        num_envs=1,
    )
    env = ClipAction(env)
    env = RescaleAction(env, -0.05, 0.05)
    return env


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def get_xyz(pose7):
    return np.asarray(pose7[..., :3], dtype=np.float32)


def step_pose(env, cur_xyz, target_xyz, grip, step_xyz, step_rot=0.0, rot_dxyz=(0, 0, 0)):
    d = target_xyz - cur_xyz
    dx, dy, dz = (clamp(d[i], -step_xyz, step_xyz) for i in range(3))
    dr, dp, dyaw = (clamp(rot_dxyz[i], -step_rot, step_rot) for i in range(3))
    # Action layout per ManiSkill controller docs (EE delta pose + gripper). :contentReference[oaicite:3]{index=3}
    act = np.array([dx, dy, dz, dr, dp, dyaw, grip], np.float32)
    obs, _, terminated, truncated, info = env.step(act)
    return obs, terminated, truncated, info


def main():
    env = make_env()
    obs, info = env.reset(seed=0)
    dt = 1.0 / FPS

    def tcp():     return get_xyz(obs["extra"]["tcp_pose"][0])
    def peg():     return get_xyz(obs["extra"]["peg_pose"][0])
    def socket():  return get_xyz(obs["extra"]["socket_pose"][0])

    state, hold, k, tilt_step, retries = "hover_over_peg", 0, 0, 0, 0

    try:
        while True:
            cur = tcp()

            if state == "hover_over_peg":
                p = peg()
                target = np.array([p[0], p[1], HOVER_Z], np.float32)
                obs, *_ = step_pose(env, cur, target, OPEN_CMD, STEP_COARSE)
                if np.linalg.norm(cur[:2]-target[:2]) < XY_TOL and abs(cur[2]-target[2]) < Z_TOL:
                    state = "pregrasp_over_center"

            elif state == "pregrasp_over_center":
                p = peg()
                target = np.array([p[0], p[1], PREGRASP_Z], np.float32)
                obs, *_ = step_pose(env, cur, target, OPEN_CMD, STEP_FINE)
                if abs(cur[2]-PREGRASP_Z) < Z_TOL:
                    state = "preload_down"

            elif state == "preload_down":
                # Increase normal force to raise friction during closing
                target = np.array([cur[0], cur[1], cur[2] + PRELOAD_Z], np.float32)
                obs, *_ = step_pose(env, cur, target, PREOPEN, STEP_FINE, ROT_FINE, (0, 0, +0.02))
                if abs(cur[2]-target[2]) < 1.5*Z_TOL:
                    state = "micro_push_close"
                    hold = HOLD_STEPS

            elif state == "micro_push_close":
                # Tiny lateral push WHILE closing to avoid symmetric shove that skates the peg
                p = peg()
                xbias = np.sign((p[0] - cur[0]) + 1e-6) * PUSH_IN
                target = np.array([p[0] + xbias, p[1], cur[2]], np.float32)
                obs, *_ = step_pose(env, cur, target, CLOSE_CMD, STEP_FINE, ROT_FINE, (0, 0, +0.03))
                hold -= 1
                if hold <= 0:
                    state = "verify_lift"

            elif state == "verify_lift":
                before = peg()[2]
                target = np.array([cur[0], cur[1], cur[2] + 0.012], np.float32)  # 12 mm up
                obs, *_ = step_pose(env, cur, target, CLOSE_CMD, STEP_FINE)
                after = peg()[2]
                reached = abs(tcp()[2] - target[2]) < 2*Z_TOL
                if reached:
                    if (after - before) > 0.007:
                        state = "lift_with_peg"
                        retries = 0
                    else:
                        # Not grasped: retry with deeper preload up to twice
                        retries += 1
                        if retries <= 2:
                            obs, *_ = step_pose(env, tcp(), np.array([tcp()[0], tcp()[1], PREGRASP_Z], np.float32), OPEN_CMD, STEP_FINE)
                            globals()['PRELOAD_Z'] = float(min(-0.022, PRELOAD_Z - 0.004))
                            state = "preload_down"
                        else:
                            obs, info = env.reset(seed=None)
                            state, retries = "hover_over_peg", 0

            elif state == "lift_with_peg":
                target = np.array([cur[0], cur[1], LIFT_Z], np.float32)
                obs, *_ = step_pose(env, cur, target, CLOSE_CMD, STEP_FINE)
                if abs(cur[2]-LIFT_Z) < Z_TOL:
                    state, k, tilt_step = "move_over_socket", 0, 0

            elif state == "move_over_socket":
                s = socket()
                target = np.array([s[0], s[1], LIFT_Z], np.float32)
                obs, *_ = step_pose(env, cur, target, CLOSE_CMD, STEP_COARSE)
                if np.linalg.norm(cur[:2]-target[:2]) < XY_TOL:
                    state, k, tilt_step = "spiral_align", 0, 0

            elif state == "spiral_align":
                s = socket()
                ang = 2.0*np.pi*(k % SPIRAL_STEPS)/SPIRAL_STEPS
                target = np.array([s[0] + SPIRAL_R*np.cos(ang),
                                   s[1] + SPIRAL_R*np.sin(ang),
                                   LIFT_Z], np.float32)
                obs, *_ = step_pose(env, cur, target, CLOSE_CMD, STEP_FINE)
                k += 1
                if k >= SPIRAL_STEPS:
                    state, k, tilt_step = "descend_insert_tilt", 0, 0

            elif state == "descend_insert_tilt":
                # Tilt-then-straighten with tiny dither + yaw; classic peg-in-hole search. :contentReference[oaicite:4]{index=4}
                s = socket()
                roll_tilt = TILT_ROLL_MAX * max(0.0, 1.0 - (tilt_step / float(TILT_DECAY_STEPS)))
                phase = k * 0.5
                dxy  = np.array([DITHER_XY*np.cos(phase), DITHER_XY*np.sin(phase)], np.float32)
                dyaw = DITHER_YAW*np.sin(phase)
                target = np.array([s[0] + dxy[0], s[1] + dxy[1], INSERT_Z], np.float32)
                obs, *_ = step_pose(env, cur, target, CLOSE_CMD, STEP_FINE, ROT_FINE, (roll_tilt, 0.0, dyaw))
                tilt_step += 1; k += 1
                if abs(cur[2]-INSERT_Z) < Z_TOL or tilt_step > (TILT_DECAY_STEPS + 60):
                    state, hold = "open_release", 40

            elif state == "open_release":
                obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, OPEN_CMD], np.float32))
                hold -= 1
                if hold <= 0:
                    state = "retreat"

            elif state == "retreat":
                target = np.array([cur[0], cur[1], LIFT_Z], np.float32)
                obs, *_ = step_pose(env, cur, target, OPEN_CMD, STEP_COARSE)

            env.render()
            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
