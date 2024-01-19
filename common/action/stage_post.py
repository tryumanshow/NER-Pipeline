import os
import sys
import torch
import random
import logging
import numpy as np
import torch.multiprocessing as mp

from typing import Dict

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)

from common.utils.file_utils import *

logger = logging.getLogger(__name__)

#%%

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


# @hydra.main(version_base=None, config_path='../', config_name='config')
def perform_post(config: Dict) -> None:
    
    stage = config.stage

    ############
    # Training #
    ############
    if stage == 2:

        config = config.stage2_cfg
        common_cfg = config.common

        mp.set_sharing_strategy('file_system')
        
        random.seed(common_cfg.seed)
        np.random.seed(common_cfg.seed)
        torch.manual_seed(common_cfg.seed)
        torch.cuda.manual_seed(common_cfg.seed)
        torch.cuda.manual_seed_all(common_cfg.seed)  
        torch.backends.cudnn.deterministic = True

        if not common_cfg.use_ddp:
            from train.trainers import NERTrainer as Trainer
        else:
            raise NotImplementedError()

        trainer = Trainer(config)
        trainer.train()

    ######################################
    # Batchwise / Instancewise Inference #
    ######################################s
    elif stage == 3:

        config = config.stage3_cfg
        unit = config.unit

        if unit == 'batch':
            from inference.batch import BatchInferencer as Inferencer
        else:
            from inference.instance import InstanceInferencer as Inferencer

        trainer = Inferencer(config)
        trainer.inference()

    ##########
    # Deploy #
    ##########
    else:
    
        config = config.stage4_cfg

        # streamlit run demo/streamlit.py
        # python deploy/app_server.py