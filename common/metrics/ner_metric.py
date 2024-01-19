import torch
from typing import Dict
from constants import LABEL, LABEL_INV
from criterions import NERLoss
from dataset.dataset import artificial_label
from sklearn.metrics import confusion_matrix

LABEL_INV[0] = '[PAD]'
LABEL_INV[artificial_label] = '[CLS]+[SEP]'
maximum_idx = artificial_label-1

class NERMetric:

    def __init__(self, 
                epoch: int, 
                subset: str, 
                total_bsz: int):
        
        self.loss_logger = NERLoss(total_bsz)
        
        self.epoch = epoch
        self.subset = subset.capitalize()
        self.total_bsz = total_bsz

        self.cum_step = 0
        self.total_loss = 0.
        self.pred = []
        self.label = []

        self.dict_keys = [
            'Epoch', 
            f'Avg {self.subset} Loss', 
            f'Avg {self.subset} PredAcc', 
            f'Avg {self.subset} PredAcc w/o [O] tag', 
            f'{self.subset} Precision per Classes', 
            f'{self.subset} Recall per Classes',
            f'{self.subset} F1 Score per Classes'
        ]

        self.precision = dict()
        self.recall = dict()

    @property
    def accuracy(self) -> float:
        acc = sum(list(map(lambda x, y: x==y, self.pred, self.label))) / len(self.pred)
        return acc * 100
    
    @property
    def pure_accuracy(self) -> float:
        notOidx = [i for i, x in enumerate(self.label) if x not in [LABEL['O'], artificial_label]]
        notOlabel = list(map(lambda x: self.label[x], notOidx))
        notOpred = list(map(lambda x: self.pred[x], notOidx))
        acc = sum(list(map(lambda x, y: x == y, notOpred, notOlabel))) / len(notOlabel)
        return acc * 100
    
    @property
    def confusion_matrix(self):
        confusion_dict = dict()
        mat = confusion_matrix(self.label, self.pred)
        for i in range(mat.shape[0]):
            tp = mat[i,i]
            fp = mat[:,i].sum() - tp
            tn = mat[i,:].sum() - tp
            fn = mat.sum() - (tp + fp + tn)
            confusion_dict[LABEL_INV[i]] = (tp, fp, tn, fn)
        return confusion_dict
    
    def update(self, 
            loss: torch.tensor, 
            pred: torch.tensor, 
            label: torch.tensor) -> None:
        self.total_loss += loss.item()
        self.pred += pred.cpu().numpy().tolist()
        self.label += label.cpu().numpy().tolist()

    def return_loss(self, 
                    net_output: torch.tensor, 
                    label: torch.tensor) -> torch.tensor:
        self.cum_step += 1
        return self.loss_logger.get_loss(net_output, label, self.cum_step)

    def get_nonzero(self):
        non_zero_idx = [i for i, x in enumerate(self.label) if x != 0]
        self.pred = list(map(lambda x: self.pred[x], non_zero_idx))
        self.label = list(map(lambda x: self.label[x], non_zero_idx))

    def get_pr(self):
        confusion_dict = self.confusion_matrix
        for i in range(maximum_idx+1):
            tp, fp, _, fn = confusion_dict[LABEL_INV[i]]
            self.precision[LABEL_INV[i]] = tp / (tp + fp)
            self.recall[LABEL_INV[i]] = tp / (tp + fn)
            
    @property
    def log_info(self) -> Dict:
        self.get_pr()
        self.get_nonzero()
        log_dict = {}
        for idx, stats in enumerate([self.epoch, 
                                     self.total_loss / self.cum_step, 
                                     self.accuracy, 
                                     self.pure_accuracy, 
                                     self.precision, 
                                     self.recall]):
            log_dict[self.dict_keys[idx]] = stats
        return log_dict