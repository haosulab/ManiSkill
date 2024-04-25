# Contributing

Thank you for your interest in contributing to ManiSkill! To get started, follow the setup and installation instructions below.

To start contributing, raise an issue at https://github.com/haosulab/ManiSkill/issues describing what your proposed changes/contributions or comment on an existing issue. Once one of the maintainers gives a thumbs up, you can make a pull request, and our team will review it.

## Setup and Installation

We recommend using Python 3.9 to build and develop on ManiSkill, although we currently aim to support versions 3.8 and above. To get started you must set up a conda/mamba environment which can be done as follows

```
conda create -n "ms_dev" "python==3.9"
git clone https://github.com/haosulab/ManiSkill.git
cd ManiSkill
pip install -e .[build] # install ManiSkill locally with testing dependencies
```

Then to setup pre-commit, run

```
pip install pre-commit
pre-commit install
```

## Testing

Testing is currently semi-automated and a WIP. We currently rely on coverage.py and pytest to test ManiSkill.

After you make changes, be sure to add any necessary tests to cover any new code in the `tests/` folder and run all the tests with the following command

```
pytest tests/ -m "not slow and gpu_sim"
pytest tests/ -m "not slow and not gpu_sim"
```

Note that we add a "not slow" argument which is to prevent testing on slow tests like download utility testing. There is also the "gpu_sim" pytest mark, which marks some tests as having to use the GPU simulation. These tests are separated as CPU simulation cannot run once GPU simulation has started and vice versa. With that in mind, any test that uses GPU simulation must add the `@pytest.mark.gpu_sim` decorator.

<!-- ```
coverage run --source=mani_skill/ -a -m pytest tests -m "not slow" # run tests
coverage html --include=mani_skill/**/*.py # see the test coverage results
``` -->
<!-- 
To skip generating a coverage report and also for easy debugging you can just run
```
pytest tests/ --pdb --pdbcls=IPython.terminal.debugger:Pdb -m "not slow"
``` -->

## Building

Adapted from https://packaging.python.org/en/latest/tutorials/packaging-projects/. For some reason running build directly does not work, you have to pass in -s and -w.

```
python -m build -s -w
python -m twine upload --repository testpypi dist/*
```

To install the test package
```
python -m pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mani_skill
```

To upload to the actual pypi repository
```
python -m twine upload dist/*
```

## Adding New Tasks

ManiSkill is built to support building your own custom tasks easily. The documentation on how to use the ManiSkill API to do so is here: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks.html

We encourage users to either create their own repositories with their own tasks for others to use, or to submit to this ManiSkill repo to be part of the official, *vetted*, task list. For tasks in this repo, we do a number of checks to ensure they are of high quality and are well documented. For detailed information on how to add new tasks, see https://maniskill.readthedocs.io/en/latest/contributing/task.html