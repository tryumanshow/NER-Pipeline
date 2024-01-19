import os
import sys
import json
import torch
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

working_dir = os.getcwd()
sys.path.append(working_dir)

logger.info(f'Current Working Directory: {working_dir}')

from kocharelectra_tokenizer import KoCharElectraTokenizer
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context


class NERModelHandler(BaseHandler):
    """
    Custom Handler
    """

    def __init__(self):
        super(NERModelHandler, self).__init__()
        self.initialized = False
        self.model = None

    def initialize(self, context: Context):
        """
        Context definition: https://github.com/pytorch/serve/blob/master/ts/context.py
        """
        manifest = context.manifest
        properties = context.system_properties
        self.model_dir = properties.get("model_dir")

        logging.info(f'Model dir: {self.model_dir}')
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        model_file = manifest['model']
        serialized_file = model_file['serializedFile']

        model_pt_path = os.path.join(self.model_dir, serialized_file)
        vocab_path = os.path.join(self.model_dir, 'vocab.txt')

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
    
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        self.model.eval()
        logging.info(f'Model checkpoint of {serialized_file} is successfully loaded !')

        self.tokenizer = KoCharElectraTokenizer.from_pretrained(vocab_path, torchscript=True)
        logging.info(f'Tokenizer is successfully loaded !')

        self.initialized = True

    def preprocess(self, requests: List) -> Dict:
        data = requests[0].get('body')
        data = self.tokenizer(data)
        tensor_data = dict()
        for k, v in data.items():
            v_update = torch.LongTensor(v).unsqueeze(0)
            tensor_data[k] = v_update
        return tensor_data

    def inference(self, model_input: Dict) -> torch.tensor:
        with torch.no_grad():
            output = self.model(model_input)
        return output

    def postprocess(self, inference_output: torch.tensor) -> Dict:
        output = torch.argmax(inference_output, -1).squeeze(0)[1:-1]
        output = output.cpu().numpy().tolist()
        output = {
            'response': output
        }
        return output
    
    def handle(self, data: List, context: Context) -> List:
        net_input = self.preprocess(data)
        net_output = self.inference(net_input)
        net_output = self.postprocess(net_output)
        net_output = json.dumps(net_output)
        return [ net_output ]


