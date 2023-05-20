#!/bin/bash

# Test script to run all pytests with docker images
TESTED_PYTHON_VERSIONS=('3.8' '3.9' '3.10' '3.11')

for PYTHON_VERSION in "${TESTED_PYTHON_VERSIONS[@]}"
do
    echo "=== Building Docker with Python Version ${PYTHON_VERSION} ==="
    cd docker && docker build -t "haosulab/mani-skill2_py$PYTHON_VERSION" --build-arg PYTHON_VERSION="$PYTHON_VERSION" .
done

for PYTHON_VERSION in "${TESTED_PYTHON_VERSIONS[@]}"
do
    echo "=== Testing Python Version ${PYTHON_VERSION} ==="
    docker run -d --rm -t --gpus all --name "maniskill2_test" \
        -v "$(pwd)":/root/ \
        "haosulab/mani-skill2_py$PYTHON_VERSION"
    # uninstall the pypi mani_skill2, install the local version and pytest and run pytest
    echo "Installing Dependencies"
    docker exec "maniskill2_test" /bin/bash -c "cd ~ && pip uninstall -y mani_skill2 && pip install -e . && pip install pytest" > /dev/null 2>&1
    docker exec "maniskill2_test" /bin/bash -c "pytest tests"
done