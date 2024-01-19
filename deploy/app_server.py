import os
import sys
import json
import uvicorn
import httpx
from typing import Dict
from fastapi import FastAPI, Body, HTTPException
from omegaconf import OmegaConf

model_path = os.path.join(os.getcwd(), 'train')
sys.path.append(model_path)

import models

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)

from inference.batch import BatchInferencer
from deploy.utils.fastapi_utils import (
    model_input, 
    model_forward, 
    extract_keyword
) 
from deploy.utils.models import (
    UserInput, 
    DataOutput
)
from common.tokenizer.kocharelectra_tokenizer import KoCharElectraTokenizer


config = OmegaConf.load('./config.yaml').stage4_cfg
ml_server_exist = config.ml_server_exist

app = FastAPI()


if not ml_server_exist:
    """
    Without the necessity of re-declaring the model-related stuffs, 
    I re-used the information that I previously used.
    """
    BI = BatchInferencer(config)
    model, tokenizer = BI.model, BI.tokenizer
    del BI

else:
    tokenizer = KoCharElectraTokenizer.from_pretrained(f"./common/tokenizer")
    ml_server = config.ml_server
    host, port = ml_server.host, str(ml_server.port)
    address = host + ':' + port


@app.post("/predict/job_keyword", response_model=DataOutput)
def predict(text: UserInput = Body(...)):
    text = text.sentence
    transformed = model_input(text, tokenizer)
    preds = model_forward(transformed, model)
    output = extract_keyword(preds, text, tokenizer)
    return output


@app.post("/ml_server", response_model=DataOutput)
async def send_request(text: UserInput = Body(...)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url=f'http://{address}/predictions/ner', 
                                         data = text.sentence, 
                                         headers={'Content-Type': 'text/plain'})
            status_code = response.status_code
            if status_code == 200:
                preds = response.json()
                result = extract_keyword(preds['response'], text.sentence, tokenizer)
                return result
            else:
                raise HTTPException(status_code = status_code, 
                                    detail="Error on ML Server.")

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, 
                            detail=f'Request error: {str(e)}')


# NOTE: Command on the root directory: python deploy/app_server.py
if __name__ == '__main__':

    config = OmegaConf.load('./config.yaml').stage4_cfg
    host, port = list(config.app_server.values())

    uvicorn.run(app, host=host, port=port)