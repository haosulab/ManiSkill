# nice renders for website


```
env_id=AnymalC-Reach-v1
env_id=PickCube-v1
env_id=OpenCabinetDrawer-v1
env_id=UnitreeG1PlaceAppleInBowl-v1
python ppo_for_renders.py --env_id="${env_id}" \
  --evaluate --checkpoint=pretrained/${env_id}/final.pt \
  --num_eval_envs=16 --num-eval-steps=10
```