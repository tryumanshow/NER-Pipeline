import hydra
import logging
import logging.config

from common.action import (
    perform_pre, # Data Processing
    perform_post # Training, Batch / Instance Inference, Deploy
)

logger = logging.getLogger(__name__)


mapping = {
    1: 'pre'
}
for idx in range(2, 5):
    mapping[idx] = 'post'


@hydra.main(version_base=None, config_path='./', config_name='config')
def execute(config):
    
    stage = config.stage

    globals()[f'perform_{mapping[stage]}'](config)


if __name__ == '__main__':
    execute()