# Contributing

Thank you for your interest in contributing to ManiSkill2!

## Setup and Installation

We recommend using Python 3.9 to build and develop on ManiSkill2 (MS2), although we currently aim to support versions 3.8 and above. To get started you must set up a conda/mamba environment which can be done as follows

```
conda create -n "ms2_dev" "python==3.9"
git clone https://github.com/haosulab/ManiSkill2.git
cd ManiSkill2
pip install -e .
# TODO combine below into a .[dev] version?
pip install pytest-xdist[psutil]
pip install coverage
```

## Testing

After you make changes, be sure to add any necessary tests to cover any new code in the `tests/` folder and run all the tests with the following command

```
pytest -n auto tests
coverage run --source=mani_skill2/ -a -m pytest tests
coverage html --include=mani_skill2/**/*.py
```