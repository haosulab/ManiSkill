# Action Repeat

A wrapper that repeats the same action `repeat` times when stepping in the environment.  
Special care was taken to make sure that this wrapper works properly with parallel environments, this means that it handles the fact that some sub-environments might finish before other sub-environments.   
The wrapper works with both single environments and parallel environments.    

---

The behavior of the wrapper is shown by the following figure :   

<img src="https://raw.githubusercontent.com/haosulab/ManiSkill/refs/heads/main/docs/source/user_guide/wrappers/images/action_repeat.svg"/>

The figure illustrates an example when using `num_envs=3` (so 3 parallel envs) and an action repeat with `repeat=3`.  
Until all environments are done or until the action (`action_1`) has been repeated 3 times, we perform a step in the environment by re-using the same action (`action_1`).  
When an environment becomes done (`done=True`), subsequent step data is discarded/ignored and does not contribute to the final output (for the affected environments.).   
The final output is obtained as follow, for the observation, we take the last **valid** observation received (illustrated in green and orange), any observation received after a sub-environment was already done is ignored and considered an invalid observation for the affected sub-environments (illustrated in red). The same logic applies for the done information and the info dict. For the reward, we return the sum of the rewards but only for the valid steps (illustrated in green and orange). Therefore the reward obtained when performing extra steps in a sub-environment that was already done is discarded and not used. In the example, we can see that the sub-environment #1 has `done=False, done=False, done=True`, this means that the reward will be the sum of the 3 rewards obtained (this includes the reward from `done=True`). For the sub-environment #2 we see that it becomes `done=True` after the second step call, therefore the final reward will be `r_t+1:t+2 = r_t+1 + r_t+2` (we discard the reward from the third step in red).  
This implementation ensures that only valid data is taken into account and that we handle parallel environments properly.  

## Basic Usage  
Some examples for the `repeat` parameter and what it represents in practice :   
- `repeat=1` : We use the action once to perform a step (no action repeat). This is the same as not using the wrapper.  
- `repeat=2` : We use the same action twice, therefore 2 steps is performed with the same action.  
- `repeat=3` : We use the same action three times, therefore 3 steps is performed.  
```python
import mani_skill.envs
from mani_skill.utils.wrappers import ActionRepeatWrapper
import gymnasium as gym

env = gym.make("PickCube-v1", obs_mode="state_dict")
env = ActionRepeatWrapper(env, repeat=2)
```
