import os
import sys
import hydra
import logging
import multiprocessing
from typing import Dict
# from Korpora import Korpora
from datasets import load_dataset
from joblib import Parallel, delayed
from typing import List
from common.tokenizer.kocharelectra_tokenizer import KoCharElectraTokenizer
from common.utils.preprocess_utils.main_utils import *
from common.utils.file_utils import read_file, write_file


cpu_cores = multiprocessing.cpu_count()
logger = logging.getLogger(__name__)

file_names = [
    'naver', 
    'nikl', 
    'kmou',
    'klue'
]


class SyllableTokenizer:

    def __init__(self):
        self.tokenizer = KoCharElectraTokenizer.from_pretrained(f"./common/tokenizer")

    def tokenize(self, x: str) -> List:
        return self.tokenizer.tokenize(x)


# @hydra.main(version_base=None, config_path='../', config_name='config')
def perform_pre(config: Dict) -> None:

    """
    -----------------------------------------
    | Source | Train | Valid | Test | Total | 
    | ------ | ----- | ----- | ---- | ----- |
    | Naver  | 90000 |   -   |   -  | 90000 |
    | NIKL   | 12162 |   -   |   -  | 12612 |
    | KMOU   | 2928  |  366  |  366 | 3660  |  
    | KLUE   | 21008 |  5000 |   -  | 26008 |
    -----------------------------------------
    """
    
    config = config.stage1_cfg
    in_path, out_path, fig_path, need_debug = config.values()

    tokenizer = SyllableTokenizer()

    num_cores = cpu_cores if 'pydevd' not in sys.modules else 1

    ############
    # NaverNER #
    ###########

    """
    NAVERx창원대 NER데이터 : 품질이 상당히 저조 -> 배제 결정
    """

    # Korpora.fetch("naver_changwon_ner")

    # corpus1 = Korpora.load("naver_changwon_ner")

    # Parallel(n_jobs=num_cores, require='sharedmem')(
    #     delayed(iterate_naver)(item, need_debug)
    #     for item in list(corpus1.train)
    # )

    # logger.info('Preprocessing for NaverNER dataset is successfully done.')

    ########
    # NIKL #
    ########

    nikl_path, post_path = f'{in_path}/raw/NIKL_NE_2022_v1', 'XNE2202211218.json'
    corpus2 = []
    # for x in ['M', 'N', 'S']:
    for x in ['N']:
        file_path = f'{nikl_path}/{x}{post_path}'
        temp_corp = read_file(file_path)
        corpus2.append(temp_corp)
        if len(corpus2) == 1:   # M, S는 영양가 없는 대화데이터가 많아서 배제
            corpus2 = corpus2[0]['document']
    
    Parallel(n_jobs=num_cores, require='sharedmem')(
        delayed(iterate_nikl)(item['sentence'], tokenizer, need_debug)
        for item in corpus2
    )

    logger.info('Preprocessing for NIKL dataset is successfully done.')

    #####################
    # KMOU (한국해양데이터) #
    #####################

    corpus3 = []
    for identity in ['train', 'dev', 'test']:
        instance = read_file(f'{in_path}/raw/KMOU/ner.{identity}')
        corpus3.extend(instance)

    corpus3 = split_by_enter(corpus3)

    Parallel(n_jobs=num_cores, require='sharedmem')(
        delayed(iterate_kmou)(item, tokenizer, need_debug)
        for item in corpus3
    )

    logger.info('Preprocessing for KMOU dataset is successfully done.')

    ########
    # KLUE #
    ########
    
    corpus4 = load_dataset("klue", "ner")
    
    for identity in ['train', 'validation']:
        Parallel(n_jobs=num_cores, require='sharedmem')(
            delayed(iterate_klue)(item, tokenizer, need_debug)
            for item in corpus4[identity]
    )

    logger.info('Preprocessing for KLUE dataset is successfully done.')

    #################
    # Concat & Save #
    #################

    os.system(f'mkdir -p {out_path}')
    for i, fn in enumerate(file_names):
        if i == 0:  # Abandon NAVERx창원대
            continue
        path = os.path.join(out_path, f'{fn}')
        contents = (out_dicts[f'c{i+1}_s'], out_dicts[f'c{i+1}_t'], out_dicts[f'c{i+1}_w'])
        write_file(contents, path, 'pickle')

    logger.info('Entire preprocessing is successfully done.')

    #############
    # Get Stats #
    #############

    logger.info('Drawing a figure...')

    get_public_dset_stats(out_path, fig_path)

    logger.info('Drawing a figure is successfully done.')

    data_split_and_gather(in_path)

    calculate_unk(in_path, tokenizer)


if __name__ == '__main__':
    perform_stage1()