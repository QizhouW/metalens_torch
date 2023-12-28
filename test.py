import argparse
import torch
from tqdm import tqdm
from data_loader.data_loaders import *
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import json, os
from utils import prepare_device


def test(model_path):
    config = json.load(open(os.path.join(model_path, 'config.json')))
    config['data_loader']['args']['shuffle'] = False
    config['data_loader']['args']['batchsize'] = 64
    config['data_loader']['args']['image_sample_pts'] = 32
    config['data_loader']['args']['avg_weight_bond'] = [1, 1]
    config['data_loader']['args']['img_transforms']["disable_transforms"] = True
    dataset = PointWiseDataset(config['data_loader']['args'])
    data_loader = MyDataLoader(dataset, config['data_loader']['args'])
    model = getattr(module_arch, config["arch"]["type"])(**config["arch"]["args"])
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    checkpoint = torch.load(os.path.join(model_path, 'model_best.pth'))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    model.eval()
    Xs=[]
    Ys=[]
    HSIs= []
    RGBs=[]
    sample_names = []
    for idx in range(len(dataset)):
        s,x,y= dataset[idx]
        sample_names.append(s)
        Xs.append(x)
        Ys.append(y)
        HSIs.append(dataset._hsi[idx])
        RGBs.append(dataset._rgb[idx])
    Xs = torch.stack(Xs).float().to(device)
    Ys = torch.stack(Ys).float().to(device)
    pred = model(Xs)
    loss = loss_fn(pred, Ys)
    print("loss: ", loss)
    metric = metric_fns[0](pred, Ys)
    print("metric: ", metric)
    return

if __name__ == '__main__':
    test('/home/wjoe/projects/digilens/code/res/models/Avodaco/regression/')