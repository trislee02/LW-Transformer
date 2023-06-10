import os
import argparse
import torch
import random
import numpy as np
from config import get_cfg_defaults
from utils.logger import setup_logger
from data_loaders import make_dataloader
from models import make_model
from loss import make_loss
from optimizers import make_optimizer
from schedulers import make_scheduler
from processor import do_train

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(config):
    set_seed(config.SOLVER.SEED)

    output_dir = config.OUTPUT_DIR

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger = setup_logger("LW-Transformer")
    
    logger.info('Saving model in the path: {}'.format(output_dir))

    os.environ['CUDA_VISIBLE_DEVICES'] = config.MODEL.DEVICE_ID

    train_loader, val_loader, num_classes = make_dataloader(config)

    model = make_model(config, num_classes, feature_only=False)

    loss_function = make_loss(config)

    optimizer = make_optimizer(config, model)
   
    scheduler = make_scheduler(config)

    do_train(config, 
            model=model, 
            train_dataloader=train_loader, 
            val_dataloader=val_loader,
            loss_fn=loss_function, 
            optimizer=optimizer,
            scheduler=scheduler)
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LW-Transformer')
    parser.add_argument('-c', '--config', default=None, type=str,
                help="config file path (default: None)")
    parser.add_argument('opts', help='Modify config options using the command-line', default=None,
                nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_cfg_defaults()
    if args.config != "":
        config.merge_from_file(args.config)
    config.merge_from_list(args.opts)
    config.freeze()
    
    main(config)
    