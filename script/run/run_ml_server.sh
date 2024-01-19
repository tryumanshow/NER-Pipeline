#!/bin/bash

PWD=$(pwd)
server='/ml_server'

# build a ml server image
sudo docker build -t ml_server:v1 -f ./docker/ml/Dockerfile .

echo "Dockerfile을 먼저 빌드하였기 때문에, 해당 파일을 실행시 자동으로 torchserve가 모델 서빙을 시작합니다."
echo "Bash shell로 진입하고 싶을 시 맨 마지막 줄에 주석처리 된 /bin/bash를 활성화 해주세요."

# run a ml server container
sudo docker run -d --rm -it --net=host \
    --name mar \
    -v $PWD/model_store:$server/model_store ml_server:v1 
    # /bin/bash

# docker exec -it mar /bin/bash