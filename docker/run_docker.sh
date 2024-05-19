NAME=$1

docker run --gpus all -p 8888:8888 -p 16006:6006 -v $(pwd):/home/filip-grigorov/code -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it nvcr.io/filip.grigorov/$NAME

