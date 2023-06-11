import os
import argparse
import torch
import random
import numpy as np
from config import get_cfg_defaults
from processor import do_test
from data_loaders import make_dataloader
from models import make_model
import sys

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

def main(config, model_path, feature_type):
  set_seed(config.SOLVER.SEED)

  os.environ['CUDA_VISIBLE_DEVICES'] = config.MODEL.DEVICE_ID

  train_loader, val_loader, query_loader, gallery_loader, num_classes = make_dataloader(config)

  model = make_model(config, num_classes, feature_only=True, feature_type=feature_type)

  do_test(config, model, model_path, query_loader, gallery_loader)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='LW-Transformer')
  parser.add_argument('-c', '--config', default=None, type=str,
              help="config file path (default: None)")
  parser.add_argument('-m', '--model', default="", type=str,
              help="model file path (default: None)")
  parser.add_argument('-f', '--feature', default="gl", type=str,
              help="Feature type (default: Global and Local)")
  parser.add_argument('opts', help='Modify config options using the command-line', default=None,
              nargs=argparse.REMAINDER)
  args = parser.parse_args()

  config = get_cfg_defaults()
  if args.config != "":
    config.merge_from_file(args.config)
  config.merge_from_list(args.opts)
  config.freeze()

  if args.model == "" or args.model == None:
    print("You must specify a model path!");
    sys.exit(0);
  
  main(config, model_path=args.model, feature_type=args.feature);