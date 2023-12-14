"""
The code at the moment supports scenes created by [AI2-THOR](https://ai2thor.allenai.org/) stored in this 
Hugging Face Dataset: https://huggingface.co/datasets/hssd/ai2thor-hab/tree/main

To download the dataset, run `python -m mani_skill2.utils.scene_builder.ai2thor.download <HUGGING_FACE_TOKEN>` 
and make sure to pass in your hugging face API token. Note that you must create a hugging face account 
and accept the terms of use for the dataset. Alternatively, if you can download the data through other means, simply
save it to `data/scene_datasets/ai2thor` and the code should run.

To learn how scenes are imported and built in ManiSkill, check out mani_skill2/utils/scene_builder module, there are some prebuilt scenes, 
including code that imports scenes in the AI2THOR set of scenes and format, as well as code to build simple table-top scenes commonly used in
ManiSkill.

PickObjectScene-v0 selects a scene randomly from the given SceneBuilder and
instantiates a robot randomly and selects a random object for the robot to find and pick up.
render_mode="human" opens up a viewer, convex_decomposition="none" makes scene loading fast (but not well simulated)
set convex_decomposition="coacd" to use CoACD to get better collision meshes

Code is setup so that if you press the "r" key, a new scene is loaded and shown. You can run this file by running 
`python -m mani_skill2.examples.demo_scenes` and explore around.
"""
import gymnasium as gym
import numpy as np
import sapien.render

import mani_skill2.envs
from mani_skill2.utils.scene_builder.ai2thor import (
    ArchitecTHORSceneBuilder,
    ProcTHORSceneBuilder,
    RoboTHORSceneBuilder,
    iTHORSceneBuilder,
)

if __name__ == "__main__":
    # specify we want to sample from the ArchitecTHOR set of scenes. Other SceneBuilders are imported above and can be used
    env = gym.make(
        "PickObjectScene-v0",
        render_mode="human",
        scene_builder_cls=ArchitecTHORSceneBuilder,
        convex_decomposition="none",
        fixed_scene=True,
    )

    # optionally set these to make it more realistic
    sapien.render.set_camera_shader_dir("rt2")
    sapien.render.set_viewer_shader_dir("rt2")
    sapien.render.set_ray_tracing_samples_per_pixel(4)
    sapien.render.set_ray_tracing_path_depth(2)
    sapien.render.set_ray_tracing_denoiser("optix")

    env.reset(seed=np.random.randint(2**31), options=dict(reconfigure=True))
    viewer = env.render()
    while True:
        env.render()
        if viewer.window.key_down("r"):
            env.reset(options=dict(reconfigure=True))
            viewer = env.render()
