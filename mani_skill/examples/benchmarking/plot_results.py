"""
Run
python plot_results.py -e CartpoleBalanceBenchmark-v1 -f benchmark_results/maniskill.csv benchmark_results/isaac_lab.csv
"""
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import os
import os.path as osp
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", required=True, help="ID of the environment")
    parser.add_argument("-f", "--files", nargs='+', required=True, help="Paths to the benchmark result files to be plotted")
    return parser.parse_args()

COLOR_PALLETE = [
    "#e02b35",
    "#59a89c",
    "#4190F1",
    "#a559aa"
    "#f0c571"
]
COLOR_MAP = {
    "ManiSkill3": "#e02b35",
    "Isaac Lab": "#59a89c"
}

def filter_df(df, df_filter):
    for k, v in df_filter.items():
        parts = k.split("$:")
        if parts[0] == "<":
            k = parts[1]
            df = df[df[k] < v]
        else:
            df = df[df[k] == v]
    return df

def draw_bar_plot_envs_vs_fps(ax, data, df_filter, xname="num_envs", yname="env.step/fps", annotate_label=None):
    ax.set_xlabel("Number of Parallel Environments")
    ax.set_ylabel("FPS")
    width = 0.8 / len(data)

    num_envs_list = []
    plotted_bars = 0
    for i, (exp_name, df) in enumerate(data.items()):
        df = filter_df(df, df_filter)
        if len(df) == 0: continue
        df = df.sort_values(xname)
        xs = np.arange(len(df)) + i * width
        ax.bar(xs, df[yname], label=exp_name, color=COLOR_MAP[exp_name], width=width, zorder=3)
        plotted_bars += 1
        if len(df[xname]) > len(num_envs_list):
            global_xs = np.arange(len(df)) + i * width
            num_envs_list = df[xname]
        if annotate_label is not None:
            for j, (x_val, y_val, annotate_data) in enumerate(zip(xs, df[yname], df[annotate_label])):
                if "gpu_mem_use" in annotate_label:
                    ax.annotate(f'{annotate_data / (1024 * 1024 * 1024):0.1f} GB', (x_val, y_val), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)
                else:
                    ax.annotate(annotate_data, (x_val, y_val), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)
    ax.set_xticks(np.arange(len(num_envs_list)) + (plotted_bars - 1) * width / 2, num_envs_list)
    plt.legend()
    ax.grid(True, axis='y', zorder=0)
    plt.tight_layout()
def draw_line_plot_envs_vs_fps(ax, data, df_filter, xname="num_envs", yname="env.step/fps", annotate_label=None):
    ax.set_xlabel("Number of Parallel Environments")
    ax.set_ylabel("FPS")
    for i, (exp_name, df) in enumerate(data.items()):
        df = filter_df(df, df_filter)
        df = df.sort_values(xname)
        if len(df) == 0: continue
        if annotate_label is not None:
            for j, (x, y) in enumerate(zip(df[xname], df[yname])):
                if "gpu_mem_use" in annotate_label:
                    ax.annotate(f'{df[annotate_label].iloc[j] / (1024 * 1024 * 1024):0.1f} GB', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)
                else:
                    ax.annotate(df[annotate_label].iloc[j], (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)
        ax.plot(df[xname], df[yname], '-o', label=exp_name, color=COLOR_MAP[exp_name], zorder=3)
    plt.legend()
    ax.grid(True, zorder=0)
    plt.tight_layout()
def main(args):

    data: dict[str, pd.DataFrame] = dict()

    for file in args.files:
        df = pd.read_csv(file)
        exp_name = os.path.basename(file).split('.')[0]
        if exp_name == "maniskill":
            exp_name = "ManiSkill3"
        if exp_name == "isaac_lab":
            exp_name = "Isaac Lab"
        data[exp_name] = df
    # modify matplotlib settings for higher quality images
    plt.rcParams["figure.figsize"] = [10, 4]  # set figure size
    plt.rcParams["figure.dpi"] = 200  # set figure dpi
    plt.rcParams["savefig.dpi"] = 200  # set savefig dpi

    root_save_path = f"benchmark_results/{'_'.join([os.path.basename(file).split('.')[0] for file in args.files])}/{args.env_id}"
    # Create root_save_path if it doesn't exist
    os.makedirs(root_save_path, exist_ok=True)
    print(f"Saving figures to {root_save_path}")

    ### RENDERING RESULTS ###
    # generate plot of RGB FPS against number of parallel environments with 1x 128x128 camera
    for obs_mode in ["rgb", "rgb+depth", "depth"]:
        cam_sizes = [80, 128, 160, 224, 256, 512]
        for cam_size in cam_sizes:
            fig, ax = plt.subplots()
            ax.set_title(f"{args.env_id}: {obs_mode} FPS vs Number of Parallel Envs. 1x{cam_size}x{cam_size} Camera")
            draw_bar_plot_envs_vs_fps(
                ax, data,
                {"env_id": args.env_id, "obs_mode": obs_mode, "camera_width": cam_size, "camera_height": cam_size, "num_cameras": 1}, annotate_label="env.step/gpu_mem_use")
            save_path = f"fps_num_envs_1x{cam_size}x{cam_size}_{obs_mode}.png"
            fig.savefig(osp.join(root_save_path, save_path))
            plt.close(fig)
            print(f"Saved figure to {save_path}")

    # generate plot of RGB FPS against square cameras and camera width under 16GB of GPU memory
    for obs_mode in ["rgb", "rgb+depth"]:
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_xlabel("Camera Width/Height")
        ax.set_ylabel("FPS")
        ax.set_title(f"{args.env_id}: Highest RGB FPS vs Camera Size under 16GB GPU memory")
        for i, (exp_name, df) in enumerate(data.items()):
            df = df[df["env_id"] == args.env_id]
            df = df[df["env.step/gpu_mem_use"] < 16 * 1024 * 1024 * 1024]
            df = df[(df["obs_mode"] == obs_mode)]
            df = df[df["num_cameras"] == 1]
            df = df[df["camera_height"] == df["camera_width"]]
            if len(df) == 0: continue
            ids = df.groupby("camera_width").idxmax()["env.step/fps"].to_list()
            df = df.loc[ids]
            df = df.sort_values("camera_width")
            for j, (x, y) in enumerate(zip(df["camera_width"], df["env.step/fps"])):
                ax.annotate(f'{df["num_envs"].iloc[j]} envs', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
            ax.plot(df["camera_width"], df["env.step/fps"], '-o', label=exp_name, color=COLOR_PALLETE[i % len(COLOR_PALLETE)])
        plt.legend()
        plt.tight_layout()
        save_path = osp.join(root_save_path, f"fps_camera_size_{obs_mode}.png")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved figure to {save_path}")


    # generate plot of RGB FPS against number of 128x128 cameras under 16GB of GPU memory
    for camera_size in [80, 128, 160, 224, 256, 512]:
        for obs_mode in ["rgb", "rgb+depth"]:
            fig, ax = plt.subplots()
            ax.grid(True)
            ax.set_xlabel("Number of Cameras")
            ax.set_ylabel("FPS")
            ax.set_title(f"{args.env_id}: Highest RGB FPS vs Number of {camera_size}x{camera_size} Cameras under 16GB GPU memory")
            for i, (exp_name, df) in enumerate(data.items()):
                df = df[df["env_id"] == args.env_id]
                df = df[df["env.step/gpu_mem_use"] < 16 * 1024 * 1024 * 1024]
                df = df[(df["obs_mode"] == obs_mode)]
                df = df[df["camera_width"] == camera_size]
                df = df[df["camera_height"] == camera_size]
                ids = df.groupby("num_cameras").idxmax()["env.step/fps"].to_list()
                df = df.loc[ids]
                df = df.sort_values("camera_width")
                if len(df) == 0: continue
                for j, (x, y) in enumerate(zip(df["num_cameras"], df["env.step/fps"])):
                    ax.annotate(f'{df["num_envs"].iloc[j]} envs', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
                ax.plot(df["num_cameras"], df["env.step/fps"], '-o', label=exp_name, color=COLOR_PALLETE[i % len(COLOR_PALLETE)])
            plt.legend()
            plt.tight_layout()
            save_path = osp.join(root_save_path, f"fps_num_cameras_{camera_size}x{camera_size}_{obs_mode}.png")
            fig.savefig(save_path)
            print(f"Saved figure to {save_path}")
            plt.close(fig)

    # generate plot for RT/google dataset settings, which is 1x 640x480 cameras
    for obs_mode in ["RGB", "Depth"]:
        fig, ax = plt.subplots()
        ax.set_title(f"{args.env_id}: FPS with 1x 640x480 {obs_mode} Cameras")
        draw_bar_plot_envs_vs_fps(ax, data, {"env_id": args.env_id, "obs_mode": obs_mode.lower(), "num_cameras": 1, "camera_width": 640, "camera_height": 480}, annotate_label="env.step/gpu_mem_use")
        plt.legend()
        plt.tight_layout()
        save_path = osp.join(root_save_path, f"fps_rt_dataset_setup_{obs_mode.lower()}_bar.png")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved figure to {save_path}")

    # generate plot for droit dataset settings, which is 3x 320x180 cameras
    for obs_mode in ["RGB", "Depth"]:
        fig, ax = plt.subplots()
        ax.set_title(f"{args.env_id}: FPS with 3x 320x180 {obs_mode} Cameras")
        draw_bar_plot_envs_vs_fps(ax, data, {"env_id": args.env_id, "obs_mode": obs_mode.lower(), "num_cameras": 3, "camera_width": 320, "camera_height": 180}, annotate_label="env.step/gpu_mem_use")
        save_path = osp.join(root_save_path, f"fps_droid_dataset_setup_{obs_mode.lower()}.png")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved figure to {save_path}")

    ### State results ###
    # generate plot of state FPS against number of parallel environments
    fig, ax = plt.subplots()
    ax.set_title(f"{args.env_id} random actions: State FPS vs Number of Parallel Environments")
    draw_bar_plot_envs_vs_fps(ax, data, {"env_id": args.env_id, "obs_mode": "state"}, annotate_label="env.step/gpu_mem_use")
    save_path = osp.join(root_save_path, f"fps_num_envs_state.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved figure to {save_path}")

    # Print column names of first entry in data
    first_key = list(data.keys())[0]
    first_df = data[first_key]
    fixed_trajectory_cols = []
    for col in first_df.columns:
        if "_env.step/fps" in col:
            # special fixed trajectory runs
            fixed_trajectory_cols.append(col)
    for col in fixed_trajectory_cols:
        fixed_name = '_'.join(col.split('_')[:-1])
        fig, ax = plt.subplots()
        ax.set_title(f"{args.env_id} {fixed_name} actions: State FPS vs Number of Parallel Environments")
        draw_bar_plot_envs_vs_fps(ax, data, {"env_id": args.env_id, "obs_mode": "state"}, yname=col, annotate_label="env.step/gpu_mem_use")
        save_path = osp.join(root_save_path, f"fps_num_envs_state_{fixed_name}.png")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved figure to {save_path}")


    ### Special figures for maniskill ###
    if "ManiSkill3" in data.keys():
        # Generate line plots of rendering FPS for different env_ids against number of parallel environments
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.grid(True)
        ax.set_xlabel("Number of Parallel Environments")
        ax.set_ylabel("FPS")
        ax.set_title("Simulation+Rendering FPS vs Number of Parallel Environments for Different Tasks")
        df = data["ManiSkill3"]
        df = df[(df["obs_mode"] == "rgb") & (df["num_envs"] >= 16) & (df["num_envs"] <= 1024) & (df["num_cameras"] == 1) & (df["camera_width"] == 128)]
        env_ids = df["env_id"].unique()
        for i, env_id in enumerate(env_ids):
            env_df = df[df["env_id"] == env_id].sort_values("num_envs")
            ax.plot(env_df["num_envs"], env_df["env.step/fps"], '-o', label=env_id, color=COLOR_PALLETE[i % len(COLOR_PALLETE)])

            for x, y, mem_use in zip(env_df["num_envs"], env_df["env.step/fps"], env_df["env.step/gpu_mem_use"]):
                ax.annotate(f'{mem_use / (1024 * 1024 * 1024):0.1f} GB', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)

        ax.legend()
        plt.tight_layout()
        fig.savefig("benchmark_results/fps_vs_num_envs_different_tasks.png")

# To use this script, run it from the command line with the paths to the benchmark result files as arguments.
# For example:
# python plot_results.py -f file1.csv file2.csv file3.csv
    return
if __name__ == "__main__":
    main(parse_args())
