#!/bin/bash

# build an app server image
sudo docker build -t app_server:v1 -f ./docker/app/Dockerfile .

# run an app server container
sudo docker run --rm -it --net=host app_server:v1
