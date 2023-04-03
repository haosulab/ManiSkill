from setuptools import find_packages, setup

long_description = """ManiSkill2 is a unified benchmark for learning generalizable robotic manipulation skills powered by [SAPIEN](https://sapien.ucsd.edu/). **It features 20 out-of-box task families with 2000+ diverse object models and 4M+ demonstration frames**. Moreover, it empowers fast visual input learning algorithms so that **a CNN-based policy can collect samples at about 2000 FPS with 1 GPU and 16 processes on a workstation**. The benchmark can be used to study a wide range of algorithms: 2D & 3D vision-based reinforcement learning, imitation learning, sense-plan-act, etc.

Please refer our [documentation](https://haosulab.github.io/ManiSkill2) to learn more information."""


def read_requirements():
    with open("requirements.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]
    install_requires = list(filter(None, lines))
    return install_requires


setup(
    name="mani_skill2",
    version="0.4.2",
    description="ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ManiSkill2 contributors",
    url="https://github.com/haosulab/ManiSkill2",
    packages=find_packages(
        include=["mani_skill2*", "warp_maniskill*"],
        exclude=["warp_maniskill.warp.tests"],
    ),
    python_requires=">=3.7",
    setup_requires=["setuptools>=62.3.0"],
    install_requires=read_requirements(),
    # Glob patterns do not automatically match dotfiles
    package_data={
        "mani_skill2": [
            "assets/**",
            "envs/mpm/shader/**",
            "envs/mpm/RopeInit.pkl",
        ],
        "warp_maniskill.warp": ["native/*", "native/nanovdb/*"],
    },
    exclude_package_data={"": ["*.convex.stl"]},
    extras_require={
        "tests": ["pytest", "black", "isort"],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
            # Copy button for code snippets
            "sphinx_copybutton",
            # Markdown parser
            "myst-parser",
            "sphinx-subfigure",
        ],
    },
)
