from setuptools import setup, find_packages

setup(
    name="behavior_cloning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorboard",
        "wandb",
        "mani_skill"
    ],
    description="A minimal setup for behavior cloning for ManiSkill",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
