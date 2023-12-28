import argparse
import collections
import shutil

import torch
import numpy as np
from data_loader import data_loaders
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from torch.utils.data import DataLoader
import pandas as pd
import os, sys
from tqdm import tqdm

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
from utils import inf_loop

def train(config):
    logger = config.get_logger('train')
    # setup data_loader instances
    dataset = data_loaders.PointWiseDataset(config['data_loader']['args'])
    data_loader = data_loaders.MyDataLoader(dataset, config['data_loader']['args'])
    # valid_data_loader = data_loader.split_validation()
    valid_data_loader = data_loader
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    shutil.copy('model/model.py', config.save_dir)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print("train on {} samples".format(len(data_loader.dataset)))
    print("valid on {} samples".format(len(valid_data_loader.dataset)))
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      len_epoch=len(data_loader)*20
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: None, means all available GPUs)')
    args.add_argument('-id', '--run_id', default=None, type=str,
                      help='Default set to timestamp')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--vs', '--validation_split'], type=float, target='data_loader;args;validation_split')
    ]

    config = ConfigParser.from_args(args, options)
    train(config)

