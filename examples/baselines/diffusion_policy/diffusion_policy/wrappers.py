import gymnasium as gym
class SeqActionWrapper(gym.Wrapper):
    def step(self, action_seq):
        rew_sum = 0
        for action in action_seq:
            obs, rew, terminated, truncated, info = self.env.step(action)
            rew_sum += rew
            if terminated or truncated:
                break
        return obs, rew_sum, terminated, truncated, info
class ContinuousTaskWrapper(gym.Wrapper):
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return obs, rew, False, truncated, info
