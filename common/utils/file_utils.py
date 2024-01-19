import os
import sys
import json
import pickle
import pandas as pd
from typing import Generic, TypeVar, Dict, Any
from itertools import chain

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)


T = TypeVar('T')


def read_file(path: str) -> Any:
    ext = path.split('.')[-1]

    if ext == 'pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        f.close()

    elif ext == 'json':
        with open(path, 'r') as f:
            data = json.load(f) 
        f.close()

    elif ext in 'parquet':
        data = pd.read_parquet(path)      

    else:
        with open(path, 'r') as f:
            data = f.readlines()
        f.close() 

    return data


def write_file(data: pd.DataFrame, 
               path: str, 
               format: str) -> None:

    assert format in ['pickle', 'parquet', 'txt', 'json'], 'Pickle & Parquet for now.'

    if format == 'pickle':
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(data, f)
        f.close()

    elif format == 'txt':
        assert len(data) == 2, 'Devised only for this case.'
        data_cat = list(zip(*data))
        data_cat = list(chain(*data_cat))
        data_cat = [x + '\n' for x in data_cat]
        with open(path + '/outlier.txt', 'w') as f:
            f.writelines(data_cat)

    elif format == 'json':
        os.system(f'mkdir -p {os.path.split(path)[0]}')
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        f.close()

    else:
        data.to_parquet(path + '.parquet')


def log_results(path: str, 
                epoch: int, 
                logger: Generic[T], 
                sample: Dict, 
                tokenizer: Generic[T]):
       
    os.system(f'mkdir -p {path}')
    path = path + f'/epoch{epoch}.csv'
  
    pred = logger.loss_logger.pred_instance
    label = logger.loss_logger.label_instance
    sequence = sample['net_input']['input_ids'][-1]
    sequence = tokenizer.batch_decode(sequence)

    assert len(pred) == len(label) == len(sequence), 'Strange-!!'

    non_zero_idx = [i for i, x in enumerate(label) if x != 0]
    pred = list(map(lambda x: pred[x], non_zero_idx))
    label = list(map(lambda x: label[x], non_zero_idx))
    sequence = list(map(lambda x: sequence[x], non_zero_idx))

    df = pd.DataFrame({
        'Sequence': sequence, 
        'Pred': pred, 
        'Label': label
    })

    df.to_csv(path)