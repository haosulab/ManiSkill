# nice renders for website


```
python ppo_for_renders.py --env_id="AnymalC-Reach-v1" \
  --evaluate --checkpoint=runs/AnymalC-Reach-v1__ppo__1__1717018060/ckpt_101.pt \
  --num_eval_envs=16 --num-eval-steps=1000
```