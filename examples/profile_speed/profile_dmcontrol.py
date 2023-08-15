import time

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import tqdm
from dm_control import suite

num_steps = 10000

width = 84
height = 84
images = []

env = suite.load(domain_name="cartpole", task_name="swingup_sparse")

action_spec = env.action_spec()
time_step = env.reset()

sim_time, render_time = 0, 0
for i in range(num_steps):
    start = time.time()
    action = np.random.uniform(
        action_spec.minimum, action_spec.maximum, size=action_spec.shape
    )
    sim_time += time.time() - start

    start = time.time()
    images.append(env.physics.render(height, width, camera_id=0))
    render_time += time.time() - start

print("Num steps:", num_steps)
print("Simulation time:", sim_time, "FPS:", num_steps / sim_time)
print("Render time:", render_time, "FPS:", num_steps / render_time)
