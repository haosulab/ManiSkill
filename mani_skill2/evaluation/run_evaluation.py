import os
import sys

from tqdm import tqdm

from mani_skill2.evaluation.evaluator import BaseEvaluator
from mani_skill2.utils.io_utils import dump_json, load_json, write_txt
from mani_skill2.utils.wrappers import RecordEpisode


class Evaluator(BaseEvaluator):
    """Local evaluation."""

    def __init__(self, output_dir: str, record_dir=None):
        if os.path.exists(output_dir):
            print(f"{output_dir} already exists.")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        self.record_dir = record_dir

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        if self.record_dir is not None:
            self.env = RecordEpisode(self.env, self.record_dir, clean_on_close=False)

    def submit(self):
        # Export per-episode results
        json_path = os.path.join(self.output_dir, "episode_results.json")
        dump_json(json_path, self.result)
        print("The per-episode evaluation result is saved to {}.".format(json_path))

        # Export average result
        json_path = os.path.join(self.output_dir, "average_metrics.json")
        merged_metrics = self.merge_result()
        self.merged_metrics = merged_metrics
        dump_json(json_path, merged_metrics)
        print("The averaged evaluation result is saved to {}.".format(json_path))

    def error(self, *args):
        write_txt(os.path.join(self.output_dir, "error.log"), args)


class TqdmCallback:
    def __init__(self, n: int):
        self.n = n
        self.pbar = tqdm(total=n)

    def __call__(self, i, metrics):
        self.pbar.update()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--env-id", type=str, required=True, help="Environment ID"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the config file. If None, use the dummy config.",
    )
    # For debug only
    parser.add_argument("-n", "--num-episodes", type=int, help="Number of episodes.")
    parser.add_argument(
        "--use-random-policy",
        action="store_true",
        help="Whether to use a random policy.",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        help="Directory to record videos and trajectories. If it is '@', use the output directory.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.record_dir == "@":
        args.record_dir = args.output_dir
    evaluator = Evaluator(args.output_dir, record_dir=args.record_dir)

    # ---------------------------------------------------------------------------- #
    # Load evaluation configuration
    # ---------------------------------------------------------------------------- #
    try:
        if args.config_file is not None:
            config = load_json(args.config_file)
            config_env_id = config["env_info"]["env_id"]
            assert config_env_id == args.env_id, (config_env_id, args.env_id)
        else:  # For debug
            config = evaluator.generate_dummy_config(args.env_id, args.num_episodes)
    except:
        exc_info = sys.exc_info()
        print("Fail to load evaluation configuration.", exc_info[:-1])
        evaluator.error("Fail to load evaluation configuration.", str(exc_info[0]))
        exit(1)

    # ---------------------------------------------------------------------------- #
    # Import user policy
    # ---------------------------------------------------------------------------- #
    if args.use_random_policy:
        from mani_skill2.evaluation.solution import RandomPolicy

        UserPolicy = RandomPolicy
    else:
        try:
            from user_solution import UserPolicy
        except:
            exc_info = sys.exc_info()
            print("Fail to import UserPolicy", exc_info[:-1])
            evaluator.error("Fail to import UserPolicy", str(exc_info[0]))
            exit(2)

    # ---------------------------------------------------------------------------- #
    # Main
    # ---------------------------------------------------------------------------- #
    env_kwargs = config["env_info"].get("env_kwargs")
    evaluator.setup(args.env_id, UserPolicy, env_kwargs)

    try:
        episodes = config["episodes"]
        if args.num_episodes is not None:
            episodes = episodes[: args.num_episodes]
        cb = TqdmCallback(len(episodes))
        evaluator.evaluate_episodes(episodes, callback=cb)
    except:
        exc_info = sys.exc_info()
        print("Error during evaluation", exc_info[:-1])
        evaluator.error("Error during evaluation", str(exc_info[0]))
        exit(3)

    evaluator.submit()
    evaluator.close()


if __name__ == "__main__":
    main()
