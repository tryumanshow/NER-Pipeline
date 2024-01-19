import os
import sys
import torch
import torch.nn as nn
from typing import Dict
from models import register_model
from collections import OrderedDict
from transformers import AutoModel
from dataset.dataset import artificial_label


@register_model("ner_with_head")
class NERModel(nn.Module):
    
    model_size = 'small'
    script = False

    def __init__(self, cfg: Dict = None):
        super().__init__()

        if cfg is not None:
            self.model_size = cfg.model.size
            self.script = cfg.model.script
    
        self.model = AutoModel.from_pretrained(f"monologg/kocharelectra-{self.model_size}-discriminator", 
                                               torchscript=self.script)
        
        self.head = nn.Sequential(
            OrderedDict([
                ('dense', nn.Linear(self.model.config.hidden_size, 
                                    self.model.config.hidden_size)), 
                ('dense_prediction', nn.Linear(self.model.config.hidden_size,
                                            artificial_label + 1))
            ])
        )       

    @classmethod
    def build_model(cls, cfg: Dict):
        return cls(cfg)
    
    def forward(self, input: Dict) -> torch.tensor:
        net_output = self.model(**input)
        net_output = net_output.last_hidden_state if not self.script else net_output[0]
        net_output = self.head(net_output)
        return net_output