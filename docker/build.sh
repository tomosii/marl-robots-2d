#!/bin/bash

# sudo bash build.sh

IMAGE_NAME_TAG=marl-robots-2d:1.0
DOCKER_FILE=Dockerfile

echo 'Building Dockerfile with tag: ' ${IMAGE_NAME_TAG}
docker build -t ${IMAGE_NAME_TAG} -f ${DOCKER_FILE} .
