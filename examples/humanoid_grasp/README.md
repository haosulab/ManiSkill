# Guidance: learning a grasp policy for the Unitree G1 humanoid (issue #1465)

This directory is a companion to the discussion in
[mani-skill/ManiSkill#1465](https://github.com/mani-skill/ManiSkill/issues/1465).
It contains a minimal, runnable reference environment that demonstrates the
recommendations below:

- [`g1_tabletop_grasp_guidance.py`](./g1_tabletop_grasp_guidance.py) — a
  self-contained env (`G1TabletopGraspGuidance-v1`) with a gated reaching
  reward, contact-force based grasp check, gated lift/hold terms,
  root-relative observations, and a pregrasp-style reset. Run the built-in CPU
  smoke test with:

  ```bash
  python examples/humanoid_grasp/g1_tabletop_grasp_guidance.py
  ```

## Review of the setup in the issue

### 1. "Policy hovers near the object" — gate the reaching reward

Yes — your instinct is right. In both approaches the reaching reward keeps
paying at a maximum once the hand is at the object, so the gradient for
discovering grasping is tiny by comparison. Two fixes (both used in
`PickSingleYCBEnv`, see
`mani_skill/envs/tasks/tabletop/pick_single_ycb.py::compute_dense_reward`):

1. **Gate the reaching term by `(1 - is_grasped)`** so its value drops to zero
   the moment a grasp is detected; the policy then must move to the grasp term
   to keep collecting reward.
2. **Make every downstream term multiplicative on `is_grasped`** (lift pays
   only while grasped, hold pays only while grasped *and* lifted). In your
   Approach 2 this is already mostly correct; in Approach 1 the success bonus
   was the *only* grasp-related term, which is far too sparse for PPO to
   discover on a 43-DoF humanoid.

Note that saturating/reducing the reaching reward when close (your suggestion)
also works, but gating by `is_grasped` is cleaner: it does not punish useful
fine positioning before contact.

### 2. `is_grasped` threshold

`UnitreeG1UpperBody.right_hand_is_grasping` defaults to `min_force=0.5` N and
`max_angle=85°` (`mani_skill/agents/robots/unitree_g1/g1_upper_body.py`). For
a jenga-block-sized object we recommend:

- raise `min_force` to ~1.5 N so light brushing does not count as a grasp;
- require the *grasp* pair check (one contact on each side, as the helper does)
  rather than any-single-finger contact;
- optionally AND it with a gripper-closing signal
  (`right_hand_dist_to_open_grasp()` small) so the policy must actively close
  the fingers, and use a hard threshold version of it in `evaluate()` while
  keeping a soft (continuous) version in the dense reward.

### 3. Missing observations

Your Approach 2 obs set (root-relative EE pose, object pose, palm-relative
object pose, gripper width, `is_grasped`) is good. Suggestions:

- add **relative pose diffs** (`object_pose_palm` you already have is the most
  useful one — keep it);
- add object **linear/angular velocity** (helps the hold term and stability);
- for a small object, a **wrist/head camera** eventually helps a lot, but
  train state-based first;
- `is_grasped` in obs is fine (it is also in `info`), but keep it as a float,
  not bool, when concatenating.

### 4. `normalize_action` and controller choice

Two concrete, empirically verified points:

- **`normalize_action=True` matters.** Actions are per-joint and the G1
  configs set per-joint delta limits `[-0.2, 0.2]` for arm joints but
  `[-0.5, 0.5]` for finger joints. Without normalization, a PPO
  `tanh`-squashed action of the same magnitude moves fingers 2.5× faster than
  arm joints, which makes fine finger closure hard to learn. For
  `pd_joint_delta_pos` on the G1 the configs do not override the default, and
  `PDJointPosControllerConfig.normalize_action` defaults to `True`
  (`mani_skill/agents/controllers/pd_joint_pos.py`), so delta actions are
  normalized — good. But the G1 `pd_joint_pos` config sets
  `normalize_action=False` explicitly
  (`mani_skill/agents/robots/unitree_g1/g1_upper_body.py`); if you switch to
  `pd_joint_pos` (recommended below), enable normalization in your own
  subclass or check that your training script rescales actions accordingly.
- **`pd_joint_delta_pos` drifts under gravity.** With the default
  `use_target=False` delta controller
  (`mani_skill/agents/controllers/pd_joint_pos.py::set_action`), the drive
  target is reset to the *current* qpos every step, so any gravity-induced sag
  is never corrected: in our smoke test the same pregrasp pose held at
  0.19 m tcp-to-object distance with `pd_joint_pos` but sagged to 0.68 m with
  `pd_joint_delta_pos` under zero actions. **For a precision grasp task we
  recommend `pd_joint_pos`** (or a delta controller configured with
  `use_target=True`), plus `balance_passive_force=True` (already set).

### 5. PPO vs `ppo_fast` vs SAC

- `ppo` (examples/baselines/ppo) is the reference implementation;
  `ppo_fast` is a faster variant of the same algorithm — results should be
  equivalent, use whichever runs faster on your hardware.
- SAC generally spends far more samples on a 40+ DoF humanoid before showing
  grasps; PPO with dense shaping is the better first choice here. Your
  hyperparameters (300M steps, 32 steps/update, 8 epochs) are reasonable, but
  with the gated reward above you should see grasping attempts far sooner; try
  a smaller reach first (~50–100M) before committing to 300M.

### 6. Object/table difficulty

Yes — the G1 place task uses a taller table (`kitchen_scene_scale=0.82`,
apple-like object), which puts the target inside a comfortable part of the arm
workspace. A 25 mm jenga block on a low table is at the edge of the simplified
G1 upper-body's reach: in our reference env, a grid search over arm poses could
bring the `right_tcp_link` only to ~0.19 m from the object center (fingers
reach the rest of the way). If your object is similarly marginal, raise the
surface, move the object closer to the robot, or use a larger object first —
and evaluate the actual workspace with an IK/geometry check before training.

### 7. Reset

The IK-generated pregrasp reset (your setup 2) is the right idea; keep it but
add small pose/qpos noise so the policy does not overfit one configuration.
The reference env in this directory embeds a small grid-searched pregrasp
keyframe as an example.

## Summary of concrete changes to try

1. Gate the reaching reward by `(1 - is_grasped)`; make lift/hold terms
   multiplicative on `is_grasped`.
2. Use `right_hand_is_grasping(obj, min_force=1.5)` for the grasp check.
3. Switch to `pd_joint_pos` (or a `use_target=True` delta controller); if you
   do, make sure actions stay normalized.
4. Keep root-relative + palm-relative obs; add object velocities.
5. Start from a (noisy) pregrasp reset; raise the table or enlarge the object
   if it is near the workspace boundary.

A runnable reference implementing 1, 2, 4 and 5 is
[`g1_tabletop_grasp_guidance.py`](./g1_tabletop_grasp_guidance.py).
