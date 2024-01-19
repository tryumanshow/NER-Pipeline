FROM python:3.8-slim

RUN echo " Start building docker container of API server ... "

ENV DIR_NAME="/app"

WORKDIR ${DIR_NAME}

# A lot of dependencies b/w the model-related stuffs and FastAPI
COPY requirements.txt ${DIR_NAME}/requirements.txt
COPY deploy/app_server.py ${DIR_NAME}/deploy/app_server.py
COPY train ${DIR_NAME}/train
COPY common ${DIR_NAME}/common
COPY inference ${DIR_NAME}/inference
COPY deploy ${DIR_NAME}/deploy
COPY config.yaml ${DIR_NAME}

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "deploy/app_server.py"]

# sudo docker build -t app_server:v1 -f ./docker/app/Dockerfile .