from setuptools import setup, find_namespace_packages

setup(
    name="mani_skill2",
    version="0.3.0",
    author="SU Lab at UC San Diego",
    packages=find_namespace_packages(include=["mani_skill2*", "warp_maniskill*"]),
)
