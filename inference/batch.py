import os
import sys
import logging
import pprint
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, TypeVar
from torch.nn.parallel.data_parallel import DataParallel

logger = logging.getLogger(__name__)

model_path = os.path.join(os.getcwd(), 'train')
sys.path.append(model_path)

# import inference.models
import models
from dataset import BatchNERDataset

from common.metrics import NERMetric
from common.tokenizer.kocharelectra_tokenizer import KoCharElectraTokenizer
from common.utils.trainer_utils import (
    rename_logger,
    Scheduler
)
from common.utils.file_utils import write_file


T = TypeVar('T')


class BatchInferencer:

    def __init__(self, cfg: Dict):
        # Chunky
        self.cfg = cfg
        self.common_cfg = self.cfg.common
        self.model_size = self.cfg.pretrain.model.size
        _, _, self.train_cfg, self.eval_cfg = list(self.cfg[self.cfg.task].values())
        self.eval_set = self.eval_cfg.valid_set.split(', ')

        # Path config
        self.data_path = self.train_cfg.in_path
        self.model_save_dir = self.common_cfg.model_path; os.makedirs(self.model_save_dir, exist_ok=True)
        self.result_dir = self.common_cfg.result_path.replace('train', 'inference')

        # Train config
        self.batch_size = self.train_cfg.batch_size
        self.lr = self.train_cfg.lr
        self.tokenizer = KoCharElectraTokenizer.from_pretrained(f"./common/tokenizer")
        self.use_fp16 = self.common_cfg.use_fp16
        self.use_clipping = self.common_cfg.use_clipping

        self.exp_name = f'{self.cfg.task}-KoCharELECTRA_{self.model_size}-lr_{str(self.lr)}-bsz_{self.batch_size}-use_fp16_{self.use_fp16}'
        self.result_dir = os.path.join(self.result_dir, self.exp_name)

        logger.info(pprint.pformat(self.cfg))
   
        if self.train_cfg.scheduler.which is not None:
            self.exp_name += f'-{self.train_cfg.scheduler.which}'
        if self.use_clipping:
            self.exp_name += f'-grad_clipping_{self.train_cfg.grad_norm}'

        self.model = models.build_model(self.cfg)    
                  
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, device_ids=[0])
            self.device = f'cuda:{self.model.device_ids[0]}'
            self.model.to(self.device)
        self.device = torch.device(self.device) if torch.cuda.is_available() else 'cpu'
        
        self.load_pretrained()
        self.load_dataset()

    def load_pretrained(self):
        model_path = os.path.join(self.model_save_dir, self.exp_name + '_checkpoint_best.pt')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    def load_dataset(self):
        dataset = BatchNERDataset(
            input_path=self.data_path,
            split='valid',
            tokenizer=self.tokenizer
        )
    
        self.eval_loader = DataLoader(
            dataset, 
            collate_fn = dataset.collator, 
            batch_size = self.eval_cfg.batch_size, 
            num_workers = 4 if 'pydevd' not in sys.modules else 1, 
            shuffle = False,
            pin_memory = True,
            drop_last = False
        )
        
    def inference(self):  
        
        logger.info("Begin validation on batchwise inference")
        metric_logger = NERMetric(0, 'valid', len(self.eval_loader)) 

        self.model.eval()
        with torch.no_grad():
            for i , sample in enumerate(tqdm(self.eval_loader, desc=f'Iteration time for Batchwise Inference')):
                net_output = self.model(sample['net_input'])
                loss, features = metric_logger.return_loss(net_output, sample['net_output']['label'])
                metric_logger.update(loss, *features)

        eval_log = metric_logger.log_info

        write_file(eval_log, self.result_dir + '/batch_inference.json', 'json')

        with rename_logger(logger, 'inference'):
            logger.info("epoch: {}, loss: {:.3f}, acc: {:.3f}, pure acc: {:.3f}".format(*eval_log.values()))

        logger.info("[Batch Version] Validation process is over.")