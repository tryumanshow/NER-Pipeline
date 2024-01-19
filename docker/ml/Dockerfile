FROM pytorch/torchserve:0.8.1-gpu

USER root

RUN echo " Start building docker container of ml server ... "
RUN mkdir model_store
RUN apt-get update && apt-get install -y vim  # To check logs from torchserve

ENV DIR_NAME="/ml_server"

# Working Directory
WORKDIR ${DIR_NAME}

# Change Permissions
RUN chmod -R 777 ${DIR_NAME}
RUN chown -R root:model-server ${DIR_NAME}

USER model-server

# Copy necessary files to WD
COPY docker/ml/requirements.txt ${DIR_NAME}/requirements.txt
COPY docker/ml/config.properties ${DIR_NAME}/config.properties
COPY train ${DIR_NAME}/train
COPY deploy ${DIR_NAME}/deploy
COPY common ${DIR_NAME}/common

RUN pip3 install --no-cache-dir -r requirements.txt

# Ports for ML Serving 
# 8080: inference / 8081: management / 8082: metrics 
EXPOSE 8080

# Launch TorchServe
CMD ["torchserve", "--start",  "--ncs", "--model-store", \
    "${DIR_NAME}/model_store",  "--models", "ner=NERmodel.mar", \
    "--ts-config", "config.properties"]