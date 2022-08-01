import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import TestDataset
from experiment import SRexperiment
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--test_data_dir", type=str, default="dataset/Urban100")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--loss_fn", type=str, 
                        choices=["MSE", "L1", "SmoothL1"], default="L1")

    return parser.parse_args()





def sample(net, device, dataset, cfg):
    scale = cfg.scale
    for step, (hr, lr, name) in enumerate(dataset):
        print(name)
        if "DIV2K" in dataset.name:
            t1 = time.time()
            h, w = lr.size()[1:]
            #print(lr.size())
            #print(hr.size())
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave
             
            lr_patch = torch.zeros((4, 3, h_chop, w_chop), dtype=torch.float)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(device)
            with torch.no_grad():
                sr = net(lr_patch, cfg.scale).detach()
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale

            result = torch.zeros((3, h, w), dtype=torch.float).to(device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result
            t2 = time.time()
        else:
            t1 = time.time()
            lr = lr.unsqueeze(0).to(device)
            with torch.no_grad():
                sr = net(lr, cfg.scale).detach().squeeze(0)
            lr = lr.squeeze(0)
            t2 = time.time()
        print(sr.size())
        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
        sr_dir = os.path.join(cfg.sample_dir,"SR")
        print(sr_dir)
        #hr_dir = os.path.join(cfg.sample_dir,"HR")
        
        os.makedirs(sr_dir, exist_ok=True)
        #os.makedirs(hr_dir, exist_ok=True)

        sr_path = os.path.join(sr_dir, name)
        print(sr_path)
        #hr_im_path = os.path.join(hr_dir, "{}".format(name))

        sr.cpu().detach().numpy().tofile(sr_path)
        #save_image(hr, hr_im_path)
        #print("Saved {} ({}x{} -> {}x{}, {:.3f}s)"
        #    .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1))


def main(cfg):

    net = importlib.import_module("model.{}".format(cfg.model)).Net
    model=SRexperiment(net,cfg)
    print("0")
    checkpoint = torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
    print("1")
    model.load_state_dict(checkpoint['state_dict'])
    print("2")
    dataset = TestDataset(cfg.test_data_dir, cfg.scale)
    print("3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample(model, device, dataset, cfg)
    print("4")
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
