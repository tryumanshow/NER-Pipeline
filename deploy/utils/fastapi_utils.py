import os
import sys
import torch
from typing import Generic, TypeVar, Dict, List

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.dirname(os.path.abspath(os.path.dirname(dirname)))

sys.path.append(dirname)

from common.constants import LABEL, LABEL_INV

O_idx = LABEL['O']

T = TypeVar('T')

def model_input(text: str, tokenizer: Generic[T]) -> Dict:
    if len(text) > 510:
        text = text[:510][::-1]
        period_idx = text.find('.')
        text = text[period_idx+1:][::-1].strip()
    return tokenizer(text)

def make_tensor(text: Dict) -> Dict:
    for key, value in text.items():
        text[key] = torch.LongTensor(value).unsqueeze(0)
    return text

@torch.inference_mode()
def model_forward(text, model: Generic[T]) -> List:
    text = make_tensor(text)
    logits = model(text)
    pred = torch.argmax(logits, -1).squeeze(0)[1:-1].cpu().numpy().tolist()
    return pred

def extract_keyword(pred: List, text: str, tokenizer: Generic[T]) -> Dict:
    result_dict = dict()

    tokenized = tokenizer.tokenize(text)
    
    assert len(pred) == len(tokenized), 'Something wrong on your post processing.'

    start, end = 0, 0

    flag = False
    while start < len(pred):
        if pred[start] != O_idx:
            label_type = LABEL_INV[pred[start]][:3]
            for end in range(start+1, len(pred)):
                if end == len(pred)-1:
                    flag = True
                    break
                if pred[end+1] == O_idx:
                    selected = ''.join(tokenized[start:end+1])
                    result_dict[selected] = label_type
                    start = end
                    break
        start += 1

        if flag:
            break

    return { 'response': result_dict }