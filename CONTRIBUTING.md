# Contributing

Thank you for your interest in contributing to ManiSkill2! To get started, follow the setup and installation instructions below.

To start contributing, raise an issue at https://github.com/haosulab/ManiSkill2/issues describing what your proposed changes/contributions or comment on an existing issue. Once one of the maintainers gives a thumbs up, you can make a pull request, and our team will review it.

## Setup and Installation

We recommend using Python 3.9 to build and develop on ManiSkill2 (MS2), although we currently aim to support versions 3.8 and above. To get started you must set up a conda/mamba environment which can be done as follows

```
conda create -n "ms2_dev" "python==3.9"
git clone https://github.com/haosulab/ManiSkill2.git
cd ManiSkill2
pip install -e . # install MS2 locally
pip install pytest coverage stable-baselines3 # add development dependencies for testing purposes
```

## Testing

Testing is currently semi-automated and a WIP. We currently rely on coverage.py and pytest to test ManiSkill2.

After you make changes, be sure to add any necessary tests to cover any new code in the `tests/` folder and run all the tests with the following command

```
coverage run --source=mani_skill2/ -a -m pytest tests # run tests
coverage html --include=mani_skill2/**/*.py # see the test coverage results
```


## Building

Adapted from https://packaging.python.org/en/latest/tutorials/packaging-projects/. For some reason running build directly does not work, you have to pass in -s and -w.

```
python3 -m build -s -w
python3 -m twine upload --repository testpypi dist/*
```