#!/bin/bash

# Test script to run all pytests with different python versions via docker
# Assumes you are running from the root directory of ManiSkill2 repository and you have downloaded all assets via
# python -m mani_skill2.utils.download_asset all

TESTED_PYTHON_VERSIONS=('3.8' '3.9' '3.10' '3.11')

for PYTHON_VERSION in "${TESTED_PYTHON_VERSIONS[@]}"
do
    echo "=== Building Docker with Python Version ${PYTHON_VERSION} ==="
    docker build -t "haosulab/mani-skill2_py$PYTHON_VERSION" --build-arg PYTHON_VERSION="$PYTHON_VERSION" docker
done

for PYTHON_VERSION in "${TESTED_PYTHON_VERSIONS[@]}"
do
    echo "=== Testing Python Version ${PYTHON_VERSION} ==="
    container_name="maniskill2_test_${PYTHON_VERSION}"

    # stop and delete container if it is up already, ignore if it doesn't exist
    docker container stop ${container_name} > /dev/null 2>&1 || true
    docker container rm ${container_name} > /dev/null 2>&1 || true

    docker run -d --rm -t --gpus all --name ${container_name} \
        -v "$(pwd)":/root/ \
        "haosulab/mani-skill2_py$PYTHON_VERSION"
    # uninstall the pypi mani_skill2, install the local version and pytest, then run pytest
    echo "Installing Dependencies"
    docker exec ${container_name} /bin/bash -c "cd ~ && pip uninstall -y mani_skill2 && pip install -e . && pip install pytest-xdist[psutil]" > /dev/null 2>&1
    docker exec ${container_name} /bin/bash -c "cd ~ && pytest -n auto tests"
    docker container stop ${container_name}
    docker container rm ${container_name}
done