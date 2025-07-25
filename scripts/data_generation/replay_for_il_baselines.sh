# Script used for replaying downloaded demonstrations to include the relevant observation 
# data and action space (controller) data used for the imitation learning benchmarking in ManiSkill.
# Note that we specify here the controller mode to use, different from the original stored in the datasets. 
# The strategy here is to use the simplest controller possible such that the task is still solvable.

# We do not upload the replayed demonstrations here as they can be extremely large due to image data 
# being saved. Uploaded demonstrations typically only keep environment state data which is much smaller.


### State-based demonstration replay ###

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackPyramid-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o state \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o state \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
  --use-env-states -c pd_ee_delta_pose -o state \
  --save-traj --num-envs 1024 -b physx_cuda

### RGB-based demonstration replay ###

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgb \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgb \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgb \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o rgb \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/DrawTriangle-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgb \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/StackPyramid-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o rgb \
  --save-traj --num-envs 10 -b physx_cpu

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pos.physx_cuda.h5 \
  --use-env-states -c pd_ee_delta_pos -o rgb \
  --save-traj --num-envs 256 -b physx_cuda