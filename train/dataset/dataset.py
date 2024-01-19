import os
import sys
import torch
import logging
import torch.nn.functional as F
from typing import TypeVar, Generic, List, Dict
from sqlalchemy import create_engine, text

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.dirname(dirname)
sys.path.append(dirname)

from common.utils.file_utils import *
from common.constants import LABEL, LABEL_INV

logger = logging.getLogger(__name__)

T = TypeVar('T')

artificial_label = max(list(LABEL_INV.keys())) + 1

"""
0: [PAD]
1: [UNK]
2: [CLS]
3: [SEP]
4: [MASK]
5: ' '
"""

class NERDataset(torch.utils.data.Dataset):
    
    def __init__(
        self, 
        input_path: str, 
        split: str, 
        tokenizer: Generic[T]
    ):   
        
        file_path = os.path.join(input_path, f'{split}_data.parquet')
        self.data = read_file(file_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict:
        text, label = self.data.iloc[index, :]
        tokenized = self.tokenizer(text)
        label = label.split(' ')

        assert len(tokenized['input_ids'])-2 == len(label), 'Sth wrong on data'

        input_ids = torch.LongTensor(tokenized['input_ids'])
        token_type_ids = torch.LongTensor(tokenized['token_type_ids'])
        attention_mask = torch.LongTensor(tokenized['attention_mask'])
        label = F.pad(torch.LongTensor([LABEL[x] for x in label]), 
                      pad=(1, 1), value=artificial_label)

        return {
            'input_ids': input_ids, 
            'token_type_ids': token_type_ids, 
            'attention_mask': attention_mask, 
            'label': label
        }
    
    def _pad(self, tensors: torch.tensor) -> torch.tensor:
        padded = []
        for tensor in tensors:
            temp = F.pad(tensor, (0, self.max_seq-len(tensor)))
            padded.append(temp)
        padded = torch.vstack(padded)
        return padded
    
    def collator(self, samples: List) -> Dict:

        final_input = dict()

        input_ids = [sample['input_ids'] for sample in samples]
        token_type_ids = [sample['token_type_ids'] for sample in samples]
        attention_mask = [sample['attention_mask'] for sample in samples]
        label = [sample['label'] for sample in samples]

        self.max_seq = max([x.size(0) for x in input_ids])

        input_ids = self._pad(input_ids)
        token_type_ids = self._pad(token_type_ids)
        attention_mask = self._pad(attention_mask)
        label = self._pad(label)

        final_input['net_input'] = {
            'input_ids': input_ids, 
            'token_type_ids': token_type_ids, 
            'attention_mask': attention_mask
        }
        final_input['net_output'] = {
            'label': label
        }

        return final_input
    
    
#%%

class BatchNERDataset(NERDataset):

    def __init__(
        self, 
        input_path: str, 
        split: str, 
        tokenizer: Generic[T]
    ):
        
        file_path = os.path.join(input_path, f'{split}_data.parquet')
        self.data = read_file(file_path)
        self.tokenizer = tokenizer


#%%

class InstanceNERDataset:

    def __init__(
        self,
        backend_cfg: Dict, 
        tokenizer: Generic[T]
    ):
        self.tokenizer = tokenizer

        engine_name, table_name, row_index = list(backend_cfg.values())
        aws_pg_engine = create_engine(engine_name)

        datum = pd.read_sql(f'select * from {table_name} limit 1 offset {row_index}', 
                            aws_pg_engine, 
                            index_col = 'index')

        self.datum = self.post_process(datum)

    def tensorize(self, input_dict: Dict) -> Dict:

        for key, value in input_dict.items():
            if isinstance(value[0], int):
                input_dict[key] = torch.LongTensor(value)[None, :]
            else:   # str
                input_dict[key] = F.pad(torch.LongTensor([LABEL[x] for x in value]), 
                                        pad=(1, 1), value=artificial_label)
        return input_dict

    def post_process(self, datum: pd.DataFrame) -> Dict:
          
        final_input = {}

        text = datum.Text.values.item()
        label = datum.Label.values.item()
        label = label.split(' ')
        
        tokenized = self.tensorize(self.tokenizer(text))
        label = self.tensorize({'label': label})
        
        assert tokenized['input_ids'].shape[1] == label['label'].shape[0], 'Sth wrong on data.'

        final_input['net_input'] = tokenized
        final_input['net_output'] = label

        return final_input