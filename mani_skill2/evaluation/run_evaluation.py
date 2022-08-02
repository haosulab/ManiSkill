import sys

from mani_skill2.evaluation.evaluator import Evaluator
from mani_skill2.utils.io_utils import load_json


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("--cfg", dest="episode_cfgs_json", type=str)
    parser.add_argument("-n", "--num-episodes", type=int, default=1)
    parser.add_argument("--use-random-policy", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    evaluator = Evaluator(args.env_id, args.output_dir)

    # Get the user policy
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
            exit(1)

    evaluator.setup(UserPolicy)

    try:
        # Only used for debug
        if args.episode_cfgs_json is None:
            episode_cfgs = evaluator.generate_episode_configs(args.num_episodes)
        else:
            episode_cfgs = load_json(args.episode_cfgs_json)
    except:
        print("Fail to load episode configs.")
        evaluator.error("Fail to load episode configs.")
        exit(2)

    try:
        evaluator.evaluate_episodes(episode_cfgs)
    except:
        exc_info = sys.exc_info()
        print("Error during evaluation", exc_info[:-1])
        evaluator.error("Error during evaluation", str(exc_info[0]))
        exit(3)

    evaluator.submit()


if __name__ == "__main__":
    main()
