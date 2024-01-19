#!/bin/bash

echo "비 Docker 환경에서 Torchserve가 작동하는지를 확인하기 위한 테스트 shell script 입니다."

PWD=$(pwd)
model_name="NERmodel"

torchserve --stop
torchserve --start --model-store model_store \
        --models ner=${model_name}.mar \
        --ts-config $PWD/docker/ml/config.properties