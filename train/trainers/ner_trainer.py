from contextlib import nullcontext
import os
import sys
import logging
import pprint
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from typing import Dict, List
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel

logger = logging.getLogger(__name__)

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)

import models
from dataset import NERDataset

dirname = os.path.dirname(os.path.abspath(os.path.dirname(dirname)))
sys.path.append(dirname)

from common.metrics import NERMetric
from common.tokenizer.kocharelectra_tokenizer import KoCharElectraTokenizer
from common.utils.trainer_utils import (
    rename_logger,
    should_stop_early, 
    count_parameters,
    Scheduler
)
from common.utils.file_utils import log_results


class NERTrainer:

    def __init__(self, cfg: Dict):
        # Chunky
        self.cfg = cfg
        self.common_cfg = self.cfg.common
        self.model_size = self.cfg.pretrain.model.size
        self.do_train, _, self.train_cfg, self.eval_cfg = list(self.cfg[self.cfg.task].values())
        self.eval_set = self.eval_cfg.valid_set.split(', ')

        # Path config
        self.data_path = self.train_cfg.in_path
        self.model_save_dir = self.common_cfg.model_path; os.makedirs(self.model_save_dir, exist_ok=True)
        self.result_dir = self.common_cfg.result_path 

        # Train config
        self.batch_size = self.train_cfg.batch_size
        self.n_epochs = self.train_cfg.epochs
        self.lr = self.train_cfg.lr
        self.patience = self.train_cfg.patience
        self.grad_norm = self.train_cfg.grad_norm
        self.tokenizer = KoCharElectraTokenizer.from_pretrained(f"./common/tokenizer")
        self.use_fp16 = self.common_cfg.use_fp16
        self.use_clipping = self.common_cfg.use_clipping

        self.exp_name = f'{self.cfg.task}-KoCharELECTRA_{self.model_size}-lr_{str(self.lr)}-bsz_{self.batch_size}-use_fp16_{self.use_fp16}'
        self.result_dir = os.path.join(self.result_dir, self.exp_name)

        logger.info(pprint.pformat(self.cfg))

        self.model = models.build_model(self.cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.train_cfg.scheduler.which is not None:
            self.scheduler = Scheduler(self.optimizer, self.train_cfg.scheduler).lr_scheduler
            self.exp_name += f'-{self.train_cfg.scheduler.which}'
        
        if self.use_clipping:
            self.exp_name += f'-grad_clipping_{self.train_cfg.grad_norm}'

        if self.train_cfg.load_pretrained: 
            self.model = self.load_pretrained(self.model)
            
        table = count_parameters(self.model)
        logger.info(self.model)
        logger.info("task: {}".format(self.cfg.task))
        logger.info("model: {}".format(self.model.__class__.__name__))
        logger.info(
            "num. of model params: {:,} (num. to be trained: {:,})".format(
                sum(p.numel() for p in self.model.parameters()),
                sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            )
        )
        logger.info(
            f"num. of model params in detail: {table[0]}"
        )
      
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, device_ids=[0])
            self.device = f'cuda:{self.model.device_ids[0]}'
            self.model.to(self.device)
        self.enabled = True if self.use_fp16 else False
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)

        self.data_loaders = {}
        for subset in ['train', 'valid']:
            self.load_dataset(subset)

    def load_pretrained(self, model):
        raise NotImplementedError()

    def load_dataset(self, split: str):
        dataset = NERDataset(
            input_path=self.data_path,
            split=split,
            tokenizer=self.tokenizer
        )

        self.batch_size = {
            'train': self.train_cfg.batch_size, 
            'valid': self.eval_cfg.batch_size
        }

        self.shuffle = {
            'train': True,
            'valid': False
        } 

        self.drop_last = {
            'train': True, 
            'valid': False
        }
    
        self.data_loaders[split] = DataLoader(
            dataset, 
            collate_fn=dataset.collator, 
            batch_size=self.batch_size[split], 
            num_workers=4 if 'pydevd' not in sys.modules else 1, 
            shuffle=self.shuffle[split],
            pin_memory=True,
            drop_last=self.drop_last[split]
        )
        
    def grad_update(self, loss: torch.tensor) -> None:
        if self.use_fp16:
            self.grad_scaler.scale(loss).backward()
            if self.use_clipping:
                self.grad_scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            if self.use_clipping:
                clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()

    def train(self):

        if torch.cuda.is_available():
            torch.cuda.set_device(self.model.device_ids[0])

        if 'pydevd' not in sys.modules:
            wandb.init(project=self.common_cfg.exp_name, entity=self.common_cfg.entity)
            wandb.config.update(dict(self.cfg))
            wandb.run.name = self.exp_name
            self.wandb = wandb
    
        for epoch in range(1, self.n_epochs + 1):
            logger.info(f"Begin training epoch {epoch}")
            
            if self.do_train:

                self.optimizer.zero_grad(set_to_none=True)

                metric_logger = NERMetric(epoch, 'train', len(self.data_loaders['train'])) 

                self.model.train()
                logger.info('Trainer loop start')
                for i , sample in enumerate(tqdm(self.data_loaders['train'], desc='Iteration time for train set')):
                   
                    with torch.cuda.amp.autocast(enabled=self.enabled):
                        net_output = self.model(sample['net_input'])
                        loss, features = metric_logger.return_loss(net_output, sample['net_output']['label'])
                        if i == len(self.data_loaders['train'])-1:
                            log_results(self.result_dir + '/train', epoch, metric_logger, sample, self.tokenizer)
  
                    if 'pydevd' not in sys.modules:
                        self.wandb.log({'Train Loss': loss})

                    self.grad_update(loss)

                    with torch.no_grad():
                        metric_logger.update(loss, *features)

                    del loss, features
                    
                train_log = metric_logger.log_info
                
                if 'pydevd' not in sys.modules:
                    self.wandb.log(train_log)
                
                with rename_logger(logger, "train"):
                    logger.info("epoch: {}, loss: {:.3f}, acc: {:.3f}, pure acc: {:.3f}".format(*train_log.values()))

                if self.train_cfg.scheduler.which is not None:
                    self.scheduler.step()

            should_stop = self.validate_and_save(epoch, self.eval_set)
           
            if should_stop:
                sys.exit(f'No enhancement on accuracy for {self.patience} consecutive runs. System is exited.')


    def validate(
        self,
        epoch: int,
        subset: str
    ):  
        
        logger.info("Begin validation on '{}' subset".format(subset))
        metric_logger = NERMetric(epoch, subset, len(self.data_loaders[subset])) 

        self.model.eval()
        with torch.no_grad():
            for i , sample in enumerate(tqdm(self.data_loaders[subset], 
                                                desc=f'Iteration time for {subset} set')):
                net_output = self.model(sample['net_input'])
                loss, features = metric_logger.return_loss(net_output, sample['net_output']['label'])
                if i == len(self.data_loaders[subset])-1 :
                    log_results(self.result_dir + f'/{subset}', epoch, metric_logger, sample, self.tokenizer)

                metric_logger.update(loss, *features)

        eval_log = metric_logger.log_info
        
        if 'pydevd' not in sys.modules:
            self.wandb.log(eval_log)

        with rename_logger(logger, subset):
            logger.info("epoch: {}, loss: {:.3f}, acc: {:.3f}, pure acc: {:.3f}".format(*eval_log.values()))

        return metric_logger.pure_accuracy

    def validate_and_save(
        self,
        epoch: int,
        valid_subsets: List
    ):
        accuracies = []
        for subset in valid_subsets:
            subset_acc = self.validate(epoch, subset)
            accuracies.append(subset_acc)

        prev_best = getattr(should_stop_early, "best", None)

        should_stop = False
        should_stop |= should_stop_early(self.patience, accuracies[0])


        if (
            (self.patience <= 0 or 
                prev_best is None or 
                prev_best < accuracies[0])
        ):
            
            logger.info(
                "Saving checkpoint to {}".format(
                    os.path.join(self.model_save_dir, f'{self.exp_name}_checkpoint_best.pt')
                )
            )
            torch.save(
                {
                    'model_state_dict': self.model.module.state_dict() if (
                        isinstance(self.model, DataParallel)
                    ) else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epochs': epoch,
                    'args': self.cfg,
                },
                os.path.join(self.model_save_dir, f'{self.exp_name}_checkpoint_best.pt')
            )
            logger.info(
                "Finished saving checkpoint to {}".format(
                    os.path.join(self.model_save_dir, f'{self.exp_name}_checkpoint_best.pt')
                )
            )

        return should_stop