import logging
import torch
import torch.nn as nn
from typing import Dict
from models import register_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)

@register_model("ner_with_electra")
class NERModel(nn.Module):
    
    def __init__(self, cfg: Dict):
        super().__init__()
        self.common_cfg = cfg.common
        self.task_cfg = cfg[cfg.task]
        self.model = self._obtain_ner_model.build_model(self.task_cfg)

    @classmethod
    def build_model(cls, cfg: Dict):
        return cls(cfg)
    
    @property
    def _obtain_ner_model(self):
        if self.task_cfg.model.model is not None:
            return MODEL_REGISTRY[self.task_cfg.model.model]
        else:
            return None

    def forward(self, net_input: Dict) -> torch.tensor: 
        net_output = self.model(net_input)
        return net_output
    

