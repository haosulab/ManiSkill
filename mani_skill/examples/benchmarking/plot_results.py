import argparse
import numpy as np
import pandas as pd
import os
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", required=True, help="ID of the environment")
    parser.add_argument("-f", "--files", nargs='+', required=True, help="Paths to the benchmark result files to be plotted")
    return parser.parse_args()

COLOR_PALLETE = [
    "#e02b35",
    "#59a89c",
    "#082a54",
    "#a559aa"
    "#f0c571"
]

def main(args):
    import matplotlib.pyplot as plt

    data: dict[str, pd.DataFrame] = dict()

    for file in args.files:
        df = pd.read_csv(file)
        df = df[df["env_id"] == args.env_id]
        exp_name = os.path.basename(file).split('.')[0]
        data[exp_name] = df
    # modify matplotlib settings for higher quality images
    plt.rcParams["figure.figsize"] = [10, 6]  # set figure size
    plt.rcParams["figure.dpi"] = 200  # set figure dpi
    plt.rcParams["savefig.dpi"] = 200  # set savefig dpi


    ### RENDERING RESULTS ###
    # generate plot of RGB FPS against number of parallel environments with 1x 128x128 camera
    cam_sizes = [80, 128, 160, 224, 256, 512]
    for cam_size in cam_sizes:
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_title(f"{args.env_id}: RGB FPS vs Number of Parallel Envs. 1x{cam_size}x{cam_size} RGB Camera")
        ax.set_xlabel("Number of Parallel Envs")
        ax.set_ylabel("FPS")
        for i, (exp_name, df) in enumerate(data.items()):
            df = df[(df["obs_mode"] == "rgb") & (df["camera_width"] == cam_size) & (df["camera_height"] == cam_size) & (df["num_cameras"] == 1) & (df["env.step/gpu_mem_use"] < 16 * 1024 * 1024 * 1024)]
            if len(df) == 0: continue
            df = df.sort_values("num_envs")
            for j, (x, y) in enumerate(zip(df["num_envs"], df["env.step/fps"])):
                ax.annotate(f'{df["env.step/gpu_mem_use"].iloc[j] / (1024 * 1024 * 1024):0.1f} GB', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)
            ax.plot(df["num_envs"], df["env.step/fps"], '-o', label=exp_name, color=COLOR_PALLETE[i % len(COLOR_PALLETE)])
        plt.legend()
        plt.tight_layout()
        fig.savefig(f"benchmark_results/fps:num_envs_1x{cam_size}x{cam_size}_rgb.png")

    # generate plot of RGB FPS against camera width under 16GB of GPU memory
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel("Camera Width/Height")
    ax.set_ylabel("FPS")
    ax.set_title(f"{args.env_id}: Highest RGB FPS vs Camera Size under 16GB GPU memory")
    for i, (exp_name, df) in enumerate(data.items()):
        df = df[df["env.step/gpu_mem_use"] < 16 * 1024 * 1024 * 1024]
        df = df[(df["obs_mode"] == "rgb")]
        df = df[df["num_cameras"] == 1]
        if len(df) == 0: continue
        ids = df.groupby("camera_width").idxmax()["env.step/fps"].to_list()
        df = df.loc[ids]
        df = df.sort_values("camera_width")
        for j, (x, y) in enumerate(zip(df["camera_width"], df["env.step/fps"])):
            ax.annotate(f'{df["num_envs"].iloc[j]} envs', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
        ax.plot(df["camera_width"], df["env.step/fps"], '-o', label=exp_name, color=COLOR_PALLETE[i % len(COLOR_PALLETE)])
    plt.legend()
    plt.tight_layout()
    fig.savefig("benchmark_results/fps:camera_size_rgb.png")


    # generate plot of RGB FPS against number of 128x128 cameras under 16GB of GPU memory
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel("Number of Cameras")
    ax.set_ylabel("FPS")
    ax.set_title(f"{args.env_id}: Highest RGB FPS vs Number of 128x128 Cameras under 16GB GPU memory")
    for i, (exp_name, df) in enumerate(data.items()):
        df = df[df["env.step/gpu_mem_use"] < 16 * 1024 * 1024 * 1024]
        df = df[(df["obs_mode"] == "rgb")]
        df = df[df["camera_width"] == 128]
        ids = df.groupby("num_cameras").idxmax()["env.step/fps"].to_list()
        df = df.loc[ids]
        df = df.sort_values("camera_width")
        if len(df) == 0: continue
        for j, (x, y) in enumerate(zip(df["num_cameras"], df["env.step/fps"])):
            ax.annotate(f'{df["num_envs"].iloc[j]} envs', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
        ax.plot(df["num_cameras"], df["env.step/fps"], '-o', label=exp_name, color=COLOR_PALLETE[i % len(COLOR_PALLETE)])
    plt.legend()
    plt.tight_layout()
    fig.savefig("benchmark_results/fps:num_cameras_rgb.png")

    ### State results ###
    # generate plot of state FPS against number of parallel environments
    fig, ax = plt.subplots()
    ax.grid(True, axis='y')
    ax.set_title(f"{args.env_id}: State FPS vs Number of Parallel Envs")
    ax.set_xlabel("Number of Parallel Envs")
    ax.set_ylabel("FPS")
    width = 0.8 / len(data)

    for i, (exp_name, df) in enumerate(data.items()):
        df = df[(df["obs_mode"] == "state") & (df["num_envs"] >= 32)]
        if len(df) == 0: continue
        x = np.arange(len(df)) + i * width
        ax.bar(x, df["env.step/fps"], label=exp_name, color=COLOR_PALLETE[i % len(COLOR_PALLETE)], width=width)
        num_envs_list = df["num_envs"]
        for j, (x_val, y_val, mem_use) in enumerate(zip(x, df["env.step/fps"], df["env.step/gpu_mem_use"])):
            ax.annotate(f'{mem_use / (1024 * 1024 * 1024):0.1f} GB', (x_val, y_val), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)
    ax.set_xticks(x - width / 2, num_envs_list)
    plt.legend()
    plt.tight_layout()
    fig.savefig("benchmark_results/fps:num_envs_state.png")

# To use this script, run it from the command line with the paths to the benchmark result files as arguments.
# For example:
# python plot_results.py -f file1.csv file2.csv file3.csv
    return
if __name__ == "__main__":
    main(parse_args())
