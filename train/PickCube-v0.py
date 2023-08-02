import gym
import mani_skill2.envs  # import to register all environments in gym

env = gym.make("PickCube-v0", obs_mode="state", control_mode="pd_ee_delta_pose")
print("Observation space", env.observation_space)
print("Action space", env.action_space)

env.seed(0)  # specify a seed for randomness
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()  # a display is required to render
env.close()