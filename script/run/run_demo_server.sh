#!/bin/bash

# build a demo server image
sudo docker build -t demo_server:v1 -f ./docker/demo/Dockerfile .

# run a demo server container
sudo docker run --rm -it --net=host demo_server:v1