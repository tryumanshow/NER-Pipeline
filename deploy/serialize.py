import os
import sys
import logging
import torch
import torch.nn as nn
import hydra

logger = logging.getLogger(__name__)

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)
sys.path.append(dirname+'/train')

from common.tokenizer.kocharelectra_tokenizer import KoCharElectraTokenizer
from inference.batch import BatchInferencer

@hydra.main(version_base=None, config_path='../', config_name='config')
def serialize(config):

    model_save_path = config.serialize.model_save_path
    model_name = config.serialize.model_name
    serialized_path = model_save_path + f'/{model_name}.pt'
    
    os.system(f'mkdir -p {model_save_path}')

    config.stage2_cfg.pretrain.model.script = True

    model = BatchInferencer(config.stage2_cfg).model    # Load pretrained one
    tokenizer = KoCharElectraTokenizer.from_pretrained(f"./common/tokenizer", torchscript=True)

    text = '나는 삼성전자에서 2년을 근무했다.'
    tokenized_text = tokenizer(text)
    input_ids = torch.LongTensor([tokenized_text['input_ids']])
    token_type_ids = torch.LongTensor([tokenized_text['token_type_ids']])
    attention_mask = torch.LongTensor([tokenized_text['attention_mask']])

    # script does not accept an input with * or **
    # torch.jit.frontend.NotSupportedError: keyword-arg expansion is not supported:
    model.eval()
    scripted_model = torch.jit.trace(model, ({
        'input_ids': input_ids,
        'token_type_ids': token_type_ids, 
        'attention_mask': attention_mask
        })
    )
    torch.jit.save(scripted_model, serialized_path)

    logger.info('모델 Serialize가 완료되었습니다.')
    logger.info(f'Serialized 모델 경로: {serialized_path}')

if __name__ == '__main__':
    serialize()