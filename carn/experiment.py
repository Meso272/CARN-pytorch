import os
import random
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
from tensorboardX import SummaryWriter
import math
import torch
import torch.nn as nn
from torch import optim
from utils import data_loader
import pytorch_lightning as pl
#from torchvision import transforms
#import torchvision.utils as vutils
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
class SRexperiment(pl.LightningModule):

    def __init__(self,
                 model,
                 cfg) -> None:
        super(SRexperiment, self).__init__()
        self.cfg = cfg
        if cfg.scale > 0:
            self.model = model(scale=cfg.scale, 
                                 group=cfg.group)
        else:
            self.model = model(multi_scale=True, 
                                 group=cfg.group)
        if cfg.loss_fn in ["MSE"]: 
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]: 
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        #self.cur_scale=self.cfg.scale
        
        self.curr_device = None
        self.hold_graph = False
  


       

    def forward(self, input, scale,**kwargs):
        return self.model(input, scale,**kwargs)

    def training_step(self, batch, batch_idx):
        if self.cfg.scale > 0:
            scale = self.cfg.scale
            hr, lr = batch[-1][0], batch[-1][1]
        else:
                   
            scale = random.randint(2, 4)
            hr, lr = inputs[scale-2][0], inputs[scale-2][1]
            #self.cur_scale=scale
        self.curr_device = hr.device



        sr = self.forward(lr,scale)
        train_loss = self.loss_fn(sr,hr)
       
       
        return train_loss

    '''
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss
        
        return 0

    def validation_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
        
        return 0
    '''


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            self.cfg.lr)
        optims.append(optimizer)
       

        try:
            if self.cfg.gamma is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.cfg.gamma)
                scheds.append(scheduler)

                
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        train_dataset=TrainDataset(self.cfg.train_data_path, 
                                       scale=self.cfg.scale, 
                                       size=self.cfg.patch_size,fix_length=self.cfg.dslen)
        return DataLoader(train_dataset,
                                       batch_size=self.cfg.batch_size,
                                       num_workers=32,
                                       shuffle=True, drop_last=True)
    '''
    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()
        self.params['epsilon']=float(self.params['epsilon'])

        if self.params['dataset'] == 'celeba':
            celeba=CelebA(root = self.params['data_path'],split = "test",transform=transform,download=True)
            self.sample_dataloader =  DataLoader(celeba,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'cesm':
            #print(self.params['img_size'])
            dataset=CLDHGH(path=self.params['data_path'],start=50,end=52,size=self.params['img_size'],normalize=True)
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] =='cesm_new':
            dataset=CESM(path=self.params['data_path'],start=50,end=52,size=self.params['img_size'],field=self.params['field'],global_max=self.params['max'],global_min=self.params['min'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)

            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] =='nyx':

            dataset=NYX(path=self.params['data_path'],start=3,end=4,size=self.params['img_size'],field=self.params['field'],log=self.params['log'],global_max=self.params['max'],global_min=self.params['min'],epsilon=self.params['epsilon'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)

            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] =='exafel':
            dataset=EXAFEL(path=self.params['data_path'],start=300,end=310,size=self.params['img_size'],global_max=self.params['max'],global_min=self.params['min'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)

            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] =='hurricane':
            dataset=Hurricane(path=self.params['data_path'],start=41,end=42,size=self.params['img_size'],field=self.params['field'],global_max=self.params['max'],global_min=self.params['min'])  
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)     
        elif self.params['dataset'] == 'exaalt':
            dataset=EXAALT(path=self.params['data_path'],start=4000,end=4400)
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'aramco':
            dataset=ARAMCO(path=self.params['data_path'],start=1500,end=1503,size=self.params['img_size'],global_max=0.0386,global_min=-0.0512,cache_size=self.params['cache_size'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'qmcpack':
            dataset=QMCPACK(path=self.params['data_path'],start=0,end=2,size=self.params['img_size'],global_max=20.368572,global_min=-21.25822,epsilon=self.params['epsilon'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'miranda':
            dataset=MIRANDA(path=self.params['data_path'],start=0,end=1,size=self.params['img_size'],global_max=3,global_min=0.99,epsilon=self.params['epsilon'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'turbulence':
            dataset=Turbulence_Val(self.params['img_size'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 64,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'aps':
            dataset=APS(path=self.params['data_path'],start=50,end=51,size=self.params['img_size'],global_max=65535.0,global_min=0.0,cache_size=-1,epsilon=self.params['epsilon'])
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 64,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] =='hacc':
            dataset=HACC(path=self.params['data_path'],start=487,end=489,size=self.params['img_size'],field=self.params['field'],global_max=self.params['max'],global_min=self.params['min'],epsilon=self.params['epsilon'])  
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 32,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)     
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            transform =  SetRange
            #raise ValueError('Undefined dataset type')
        return transform
    '''