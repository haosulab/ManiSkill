# Performance Benchmarking


## ManiSkill

To benchmark ManiSkill + SAPIEN, after following the setup instructions on this repository's README.md, run

```
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickCube-v1" -n=4096 -o=state --control-freq=50
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickCube-v1" -n=1536 -o=rgbd --control-freq=50
# note we use --control-freq=50 as this is the control frequency isaac sim based repos tend to use
```

These are the expected state-based only results on a single 4090 GPU:
```
env.step: 277840.711 steps/s, 67.832 parallel steps/s, 100 steps in 1.474s
env.step+env.reset: 239463.964 steps/s, 58.463 parallel steps/s, 1000 steps in 17.105s
```

These are the expected visual observations/rendering results on a single 4090 GPU:
```
env.step: 18549.002 steps/s, 12.076 parallel steps/s, 100 steps in 8.281s
env.step+env.reset: 18146.848 steps/s, 11.814 parallel steps/s, 1000 steps in 84.643s
```

On 4090's generally the bottle neck is the memory available to spawn more cameras in parallel scenes. Results on high memory GPUs will be published later.
