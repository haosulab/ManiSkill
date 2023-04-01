import gymnasium as gym

from mani_skill2.envs import mpm

ENV_IDS = ["Excavate-v0", "Fill-v0", "Pour-v0", "Hang-v0", "Write-v0", "Pinch-v0"]


if __name__ == "__main__":
    for env_id in ENV_IDS:
        env = gym.make(env_id)
        env.reset()
        env.step(None)
        env.close()
