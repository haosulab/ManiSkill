import gymnasium as gym
class VecSeqActionWrapper(gym.Wrapper):
    def step(self, action_seq):
        rew_sum = 0
        for action in action_seq:
            obs, rew, terminated, truncated, info = self.env.step(action)
            rew_sum += rew
            assert truncated.any() == truncated.all(), "Envs are assumed to not be partially reset and are all truncated at the same time for fair evaluation"
            if truncated.any():
                break
        return obs, rew_sum, terminated, truncated, info
class VecContinuousTaskWrapper(gym.Wrapper):
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return obs, rew, terminated.zeros_(), truncated, info
