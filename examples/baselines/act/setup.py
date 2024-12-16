from setuptools import setup, find_packages

setup(
    name="act",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torchvision",
        "diffusers",
        "tensorboard",
        "wandb",
        "mani_skill"
    ],
    description="A minimal setup for ACT for ManiSkill",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
