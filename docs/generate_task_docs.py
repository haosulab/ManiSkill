# Code to generate task documentation automatically

import mani_skill.envs
from mani_skill.utils.registration import REGISTERED_ENVS

TASK_CATEGORIES_TO_INCLUDE = [
    "tabletop",
    "humanoid",
    "mobile_manipulation", "quadruped", "control", "drawing"
]

TASK_CATEGORIES_NAME_MAP = {
    "tabletop": "table_top_gripper"
}

GLOBAL_TASK_HEADER = """[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg
[dense-reward-badge]: https://img.shields.io/badge/dense%20reward-yes-green.svg
[sparse-reward-badge]: https://img.shields.io/badge/sparse%20reward-yes-green.svg
[no-dense-reward-badge]: https://img.shields.io/badge/dense%20reward-no-red.svg
[no-sparse-reward-badge]: https://img.shields.io/badge/sparse%20reward-no-red.svg
"""

TASK_CATEGORIES_HEADERS = {
    "tabletop": """# Table-Top 2 Finger Gripper Tasks

These are tasks situated on table and involve a two-finger gripper arm robot manipulating objects on the surface.""",

    "humanoid": """# Humanoid Tasks
Both real-world humanoids and the Mujoco humanoid are supported in ManiSkill, and we are still in the process of adding more tasks. Humanoid category of tasks generally considers control of robots with legs and two arms.""",

    "mobile_manipulation": """# Mobile Manipulation Tasks

These are tasks where a mobile manipulator is used to manipulate objects. This cateogry primarily uses robots with mobile bases like Fetch or Stretch robots""",

    "quadruped": """# Quadruped Tasks

These are tasks where a quadruped robot is used for locomotion and/or manipulation. This cateogry primarily uses robots with four legs like the ANYmal or Unitree go robots""",

    "control": """# Control Tasks

These are classic control tasks where the objective is to control a robot to reach a particular state, similar to the [DM Control suite](https://github.com/deepmind/dm_control) but with GPU parallelized simulation and rendering""",

    "drawing": """# Drawing Tasks

These are tasks where the robot is controlled to draw a specific shape or pattern""",
}

def main():
    import os
    import importlib
    import inspect
    from pathlib import Path

    # Get the path to mani_skill/envs/tasks
    tasks_dir = Path(mani_skill.envs.__file__).parent / "tasks"

    # Dictionary to store task info
    task_info = {}

    # Walk through all subfolders in tasks directory
    for root, dirs, files in os.walk(tasks_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                # Get relative import path
                rel_path = os.path.relpath(os.path.join(root, file), tasks_dir.parent)
                module_path = rel_path.replace(os.sep, '.')[:-3]  # Remove .py

                # Import the module
                try:
                    module = importlib.import_module(f"mani_skill.envs.{module_path}")

                    # Find all classes defined in this module
                    classes = inspect.getmembers(module, inspect.isclass)

                    # Store classes that are defined in this module (not imported)
                    local_classes = [cls for name, cls in classes
                                   if cls.__module__ == f"mani_skill.envs.{module_path}"]

                    if local_classes:
                        task_info[module_path] = local_classes

                except Exception as e:
                    print(f"Error importing {module_path}: {e}")
    # Filter to only include registered environment classes
    filtered_task_info = {}
    for module_path, classes in task_info.items():
        registered_classes = []
        for cls in classes:
            # Check if this class is registered as an environment
            for env_id, env_spec in REGISTERED_ENVS.items():
                if env_spec.cls == cls:
                    registered_classes.append(dict(env_id=env_id, cls=cls))
                    break
        if registered_classes:
            filtered_task_info[module_path] = registered_classes

    task_info = filtered_task_info
    # Categorize tasks by their type
    categorized_tasks = {k: [] for k in TASK_CATEGORIES_TO_INCLUDE}

    for module_path in task_info.keys():
        parts = module_path.split('.')
        if parts[0] == 'tasks':
            category = parts[1]
            if category in categorized_tasks:
                categorized_tasks[category].append(module_path)

    # Generate documentation for each category and module
    print("\nTask Documentation:")
    for category, modules in categorized_tasks.items():
        print(f"\n{category}:")
        # Create directory if it doesn't exist
        category_name = TASK_CATEGORIES_NAME_MAP.get(category, category)
        os.makedirs(f"source/tasks_generated/{category_name}", exist_ok=True)

        # Delete existing index.md file for this category
        if os.path.exists(f"source/tasks_generated/{category_name}/index.md"):
            os.remove(f"source/tasks_generated/{category_name}/index.md")
        if category in TASK_CATEGORIES_HEADERS:
            with open(f"source/tasks_generated/{category_name}/index.md", "w") as f:
                f.write(GLOBAL_TASK_HEADER)
                f.write(TASK_CATEGORIES_HEADERS[category])
        for module in sorted(modules):
            environment_data = task_info[module]
            classes = [env_data["cls"] for env_data in environment_data]
            env_ids = [env_data["env_id"] for env_data in environment_data]
            # print(f"\n  {module}:")
            for cls, env_id in zip(classes, env_ids):
                # Check if dense reward function is overridden
                from mani_skill.envs.sapien_env import BaseEnv
                has_custom_dense = (
                    hasattr(cls, "compute_dense_reward") and
                    cls.compute_dense_reward != BaseEnv.compute_dense_reward
                )
                if has_custom_dense and "dense" not in cls.SUPPORTED_REWARD_MODES:
                    print(f"Warning: {cls.__name__}, {env_id} has custom dense reward but dense not in SUPPORTED_REWARD_MODES")
                does_not_have_custom_dense = not has_custom_dense and "dense" in cls.SUPPORTED_REWARD_MODES
                if does_not_have_custom_dense:
                    print(f"Warning: {cls.__name__}, {env_id} does not have custom dense reward but dense is in SUPPORTED_REWARD_MODES")

                # Extract docstring
                if cls.__doc__:
                    with open(f"source/tasks_generated/{category_name}/index.md", "a") as f:
                        f.write(f"\n## {env_id}\n\n")
                        # Write reward modes if available
                        if hasattr(cls, 'SUPPORTED_REWARD_MODES'):
                            if "dense" in cls.SUPPORTED_REWARD_MODES:
                                f.write("![dense-reward][dense-reward-badge]\n")
                            else:
                                f.write("![no-dense-reward][no-dense-reward-badge]\n")
                            if "sparse" in cls.SUPPORTED_REWARD_MODES:
                                f.write("![sparse-reward][sparse-reward-badge]\n")
                            else:
                                f.write("![no-sparse-reward][no-sparse-reward-badge]\n")
                        """:::{dropdown} Task Card\n:icon: note\n:color: primary"""
                        # Clean up docstring and write to file
                        f.write(":::{dropdown} Task Card\n:icon: note\n:color: primary\n\n")
                        doc_lines = [line.strip() for line in cls.__doc__.split('\n')]
                        while doc_lines and not doc_lines[0]:
                            doc_lines.pop(0)
                        while doc_lines and not doc_lines[-1]:
                            doc_lines.pop()
                        if doc_lines:
                            f.write("\n".join(doc_lines))
                            f.write("\n")
                        f.write(":::\n")
                        # Add video link if available
                        if hasattr(cls, '_sample_video_link') and cls._sample_video_link is not None:
                            f.write("\n<div style=\"display: flex; justify-content: center;\">\n")
                            f.write("<video preload=\"auto\" controls=\"True\" width=\"100%\" style=\"max-width: min(100%, 512px);\">\n")
                            f.write(f"<source src=\"{cls._sample_video_link}\" type=\"video/mp4\">\n")
                            f.write("</video>\n")
                            f.write("</div>\n")
                        else:
                            print(f"Warning: {cls.__name__}, {env_id} has no sample video link")
                else:
                    print(f"Warning: {cls.__name__}, {env_id} has no docstring")
                    pass
                    # print("      No documentation available")

if __name__ == "__main__":
    main()
