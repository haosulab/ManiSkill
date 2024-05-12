NAME=$1
DOCKER_FOLDER=$2

if [ -z "${NAME}" ]; then
    echo "No docker name has been specified, exiting"
    exit
fi

if [ -z "${DOCKER_FOLDER}" ]; then
    echo "No Dockerfile has been specified, exiting"
    exit
fi

echo "Docker name is ${NAME}"
echo "Dockerfile is at ${DOCKER_FOLDER}"

# Get your host's UID and GID
export UID=$(id -u)
export GID=$(id -g)

docker build --build-arg USER=$USER --build-arg UID=$UID --build-arg GID=$GID -t nvcr.io/filip.grigorov/$NAME -f $DOCKER_FOLDER/Dockerfile .