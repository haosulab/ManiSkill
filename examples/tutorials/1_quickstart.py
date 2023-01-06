# Import required packages
import argparse

import gym
from tqdm.notebook import tqdm
import numpy as np
import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
np.set_printoptions(suppress=True, precision=3)


def watch_random(env_id, obs_mode, reward_mode, control_mode, render=True):
   
    env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode) 
    env = RecordEpisode(
        env,
        "./videos", # the directory to save replay videos and trajectories to
        render_mode="cameras", # cameras - three camera images + depth images, rgb_array - single camera image
        info_on_video=True # when True, will add informative text onto the replay video such as step counter, reward, and other metrics 
    )

    # step through the environment with random actions
    obs = env.reset()
    for i in tqdm(range(100)):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render() # will render with a window if possible
    env.flush_video() # Save the video
    env.close()


import h5py
from mani_skill2.utils.io_utils import load_json
def load_demonstrations(dataset_path):
    # Load the trajectory data from the .h5 file
    traj_path = f"{dataset_path}/trajectory.h5"
    h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)

    return h5_file, json_data

def watch_demonstration(episodes, h5_file, env_kwargs, traj_id):
    eps = episodes[traj_id]
    traj = h5_file[f"traj_{traj_id}"]

    # create our environment and reset with the same seed used for the trajectory
    env = gym.make(env_id, **env_kwargs)
    env = RecordEpisode(env, "./videos", render_mode="cameras", info_on_video=True)
    env.reset(seed=eps["episode_seed"])

    # replay the actions and save a video
    for i in tqdm(range(len(traj["actions"]))):
        action = traj["actions"][i]
        obs, reward, done, info = env.step(action)
        env.render() # will render with a window if possible
    env.flush_video() # Save the video
    env.close()


if __name__ == "__main__":
    
    # Can be any env_id from the list of Rigid-Body envs: https://github.com/haosulab/ManiSkill2/wiki/Rigid-Body-Environments
    # and Soft-Body envs: https://github.com/haosulab/ManiSkill2/wiki/Soft-Body-Environments
    env_id = "PickCube-v0"

    # choose an observation type and space, see https://github.com/haosulab/ManiSkill2/wiki/Observation-Space for details
    obs_mode = "rgbd" # can be one of pointcloud, rgbd, state_dict and state

    # choose a controller type / action space, see https://github.com/haosulab/ManiSkill2/wiki/Controllers for a full list
    control_mode = "pd_ee_delta_pose"

    reward_mode = "dense" # can be one of sparse, dense

    
    # Try out the functions from the notebook here yourself by uncommenting the lines below
    
    #####
    ## live watch with a interactive viewer a random policy interact in the environment, and save a video to the videos folder
    # watch_random(env_id=env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode)
    #####

    #####
    # To download datasets: find the particular link for the environment you want in this drive: https://drive.google.com/drive/folders/1pd9Njg2sOR1VSSmp-c1mT7zCgJEnF8r7
    # You can then download e.g. the PickCube demonstrations by running the following
    # !gdown https://drive.google.com/drive/folders/1WgYpQjqnZBbyXqlqtQfoNlCKAVPdeRIx?usp=share_link --folder
    #####

    #####
    ## loads the h5 demonstration data file as well as the associated json data
    # h5_file, json_data = load_demonstrations(dataset_path=env_id)

    ## get information about the demonstrations in order to use them, replay them etc.
    # episodes = json_data["episodes"] # each episode/demonstration meta data
    # env_info = json_data["env_info"]
    # env_id = env_info["env_id"]
    # env_kwargs = env_info["env_kwargs"]

    ## watch any demonstration
    # watch_demonstration(episodes, h5_file, env_kwargs, traj_id=10)
    #####

    ##### 
    # Following command can replay a trajectory and convert the demonstrations to a target observation and control mode
    """ 
    python ../../tools/replay_trajectory.py --traj-path ./PickCube-v0/trajectory.h5 \
        --save-traj --target-control-mode pd_ee_delta_pose --obs-mode rgbd --num-procs 10
    """
    #####


    # To try and manually control a robot agent with your keyboard, you can run https://github.com/haosulab/ManiSkill2/blob/main/examples/demo_manual_control.py