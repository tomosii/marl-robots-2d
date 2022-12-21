set -x 
IMAGE_NAME_TAG=marl-robots-2d:1.0
#docker run  --rm --net=host -it --gpus '"device=0"' $IMAGE_NAME_TAG bash
sudo docker run -u `id -u`:`id -g` --rm --net=host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --gpus all  -v /tmp:/host/tmp -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v `pwd`:/workspace $IMAGE_NAME_TAG bash

# #!/bin/bash
# HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
# GPU=$1
# name=${USER}_pymarl_GPU_${GPU}_${HASH}

# echo "Launching container named '${name}' on GPU '${GPU}'"
# # Launches a docker container using our image, and runs the provided command

# if hash nvidia-docker 2>/dev/null; then
#   cmd=nvidia-docker
# else
#   cmd=docker
# fi

# NV_GPU="$GPU" ${cmd} run \
#     --name $name \
#     --user $(id -u):$(id -g) \
#     -v `pwd`:/pymarl \
#     -t pymarl:1.0 \
#     ${@:2}