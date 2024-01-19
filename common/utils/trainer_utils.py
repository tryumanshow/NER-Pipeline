import torch
import logging
import torch.nn as nn
import pandas as pd
from contextlib import contextmanager
from prettytable import PrettyTable
from typing import Generic, TypeVar, Dict
from torch.optim.lr_scheduler import StepLR


T = TypeVar('T')

logger = logging.getLogger(__name__)


class CE_CALCULATION(nn.Module):

    def __init__(self, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.fn = nn.CrossEntropyLoss(ignore_index=ignore_index, 
                                    reduction='none')
        
    def forward(self, 
                yhat: torch.tensor, 
                y: torch.tensor) -> torch.tensor:
        loss = self.fn(yhat, y)
        loss = torch.sum(loss) / (len(y) - torch.sum(y == self.ignore_index))
        return loss
    

class Scheduler:
    
    def __init__(self, 
                 optimizer: Generic[T], 
                 scheduler_cfg: Dict, ):
        
        self.scheduler = StepLR(optimizer,
                                **scheduler_cfg[scheduler_cfg.which])

    @property
    def lr_scheduler(self):
        return self.scheduler


def align_single_batch_result(net_output: torch.tensor, 
                              sample: Dict, 
                              tokenizer: Generic[T]) -> pd.DataFrame:

    pred = torch.argmax(net_output, -1).squeeze(0).cpu().numpy().tolist()
    label = sample['net_output']['label'].cpu().numpy().tolist()
    sequence = sample['net_input']['input_ids'].squeeze(0).cpu().numpy().tolist()
    sequence = tokenizer.batch_decode(sequence)

    align_dict = pd.DataFrame({
        'sequence': sequence, 
        'pred': pred, 
        'label': label
    })

    return align_dict


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return table, total_params


def should_stop_early(patience, valid_criteria: float, standard='higher') -> bool:
    
    assert valid_criteria is not None, 'Error'
    
    if patience <= 0:
        return False

    # add flexibility for applying various metrics in the future (e.g. loss, ...)
    def is_better(a, b):
        return a > b if standard == 'higher' else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if (prev_best is None) or (is_better(valid_criteria, prev_best)):
        should_stop_early.best = valid_criteria
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    patience
                )
            )
            return True
        else:
            return False


@contextmanager
def rename_logger(logger, new_name):
    old_name = logger.name
    if new_name is not None:
        logger.name = new_name
    yield logger
    logger.name = old_name