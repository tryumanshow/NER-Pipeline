import torch
from typing import Tuple
from utils.trainer_utils import CE_CALCULATION

class NERLoss:

    def __init__(self, total_bsz: int):
        self.loss_fn = CE_CALCULATION()
        self.total_bsz = total_bsz

    def get_loss(self, 
                 logits: torch.tensor, 
                 label: torch.tensor, 
                 cum_step: int) -> Tuple:

        device = logits.device

        label = label.to(device)
        pred = torch.argmax(logits, -1)

        # For Logging
        if cum_step == self.total_bsz:
            self.label_instance = label[-1].cpu().numpy().tolist()
            self.pred_instance = pred[-1].cpu().numpy().tolist()
            
        logits = logits.reshape(-1, logits.size(-1))
        label = label.flatten()    
        pred = pred.flatten()

        loss = self.loss_fn(logits, label) 

        return loss, (pred, label)