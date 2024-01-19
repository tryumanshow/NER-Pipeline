#!/bin/bash

# Run 'bash script/model-archiver.sh' on the root directory.
# You will mount a '.mar' file on a docker container.

PWD=$(pwd)
model_store="model_store"
tokenizer_dict="common/tokenizer"

if [ ! -d "$model_store" ]; then
    mkdir -p "$model_store"
fi

python deploy/serialize.py

# Archive PyTorch model & make `.mar` file
# Encountered Error: Backend worker monitoring thread interrupted or backend worker process died.

torch-model-archiver -f --model-name NERmodel \
    --version 1.0 \
    --model-file $PWD/train/models/modules/ner_submodules.py \
    --serialized-file $PWD/pretrained_ner_script/NERmodel.pt \
    --export-path $model_store \
    --handler $PWD/deploy/handler \
    -r $PWD/requirements.txt \
    --extra-files "$PWD/$tokenizer_dict/kocharelectra_tokenizer.py,$PWD/$tokenizer_dict/vocab.txt"
    
echo "나의 Local에 .mar 파일이 생성되었습니다."