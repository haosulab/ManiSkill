# Motion planning based approach to heuristically generate demonstrations

## Panda Arm Solutions
for env_id in PushCube-v1 PickCube-v1 StackCube-v1 PegInsertionSide-v1 PlugCharger-v1 PullCubeTool-v1 LiftPegUpright-v1 PullCube-v1 StackPyramid-v1
do
    python -m mani_skill.examples.motionplanning.panda.run --env-id $env_id \
      --traj-name="trajectory" --only-count-success --save-video -n 1 \
      --shader="rt" # generate sample videos
    mv demos/$env_id/motionplanning/0.mp4 demos/$env_id/motionplanning/sample.mp4
    python -m mani_skill.examples.motionplanning.panda.run --env-id $env_id --traj-name="trajectory" -n 1000 --only-count-success
done

## Panda Stick Solutions ##
for env_id in DrawTriangle-v1
do
    python -m mani_skill.examples.motionplanning.panda.run --env-id $env_id \
      --traj-name="trajectory" --only-count-success --save-video -n 1 \
      --shader="rt" # generate sample videos
    mv demos/$env_id/motionplanning/0.mp4 demos/$env_id/motionplanning/sample.mp4
    python -m mani_skill.examples.motionplanning.panda.run --env-id $env_id --traj-name="trajectory" -n 1000 --num-procs 10 --only-count-success
done