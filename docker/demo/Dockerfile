FROM python:3.8-slim

RUN echo " Start building docker container of demo server ... "

ENV DIR_NAME="/demo"

WORKDIR ${DIR_NAME}

COPY demo ${DIR_NAME}/demo
COPY docker/demo/requirements.txt ${DIR_NAME}/requirements.txt
COPY config.yaml ${DIR_NAME}

RUN pip3 install --no-cache-dir -r ${DIR_NAME}/requirements.txt

CMD ["streamlit", "run", "demo/streamlit.py"]

# sudo docker build -t demo_server:v1 -f ./docker/demo/Dockerfile .