
import argparse
import numpy as np
import os
#from models import *
from experiment import SRexperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
#from pytorch_lightning.loggers import TestTubeLogger

from pytorch_lightning.callbacks import ModelCheckpoint
    
if __name__=='__main__':

    #trainer = Trainer()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_name", type=str)
    
    
    parser.add_argument("--train_data_path", type=str, 
                        default="dataset/DIV2K_train.h5")
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoint")
    parser.add_argument("--ckpt_file", type=str,
                        default=None)
    
    
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--scale", type=int, default=2)

    parser.add_argument("--verbose", action="store_true", default="store_true")

    parser.add_argument("--group", type=int, default=1)
    arser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_interval", type=int, default=10)
    
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=10.0)

    parser.add_argument("--loss_fn", type=str, 
                        choices=["MSE", "L1", "SmoothL1"], default="L1")

    args = parser.parse_args()
   
    ckpt_save_dir=args.ckpt_dir
    if not os.path.exists(ckpt_save_dir):
        os.mkdir(ckpt_save_dir)
     
    #gpus=config['trainer_params']['gpus']
    #print(','.join([str(idx) for idx in gpus]))
    #os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(idx) for idx in gpus])
   
    cudnn.deterministic = True
    cudnn.benchmark = False

    net = importlib.import_module("model.{}".format(args.model)).Net
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #USE_CUDA = torch.cuda.is_available()
    #print(USE_CUDA)
    #device = torch.device("cuda:0" if USE_CUDA else "cpu")
    
   
    #model = nn.DataParallel(model)
    #model= model.to(device)
    experiment = SRexperiment(net,
                              args)

# DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_save_dir,
        save_top_k=-1,
        verbose=True,
        #save_last=True,
        #monitor='loss',
        #mode='min',
        
        period=args.save_interval
    )
    gpu_list=list(range(args.num_gpu))
    runner = Trainer(min_epochs=1,
                 resume_from_checkpoint=args.ckpt_file,
                 #train_percent_check=1.,
                 #val_percent_check=1.,
                 checkpoint_callback=True,
                 callbacks=checkpoint_callback,
                 accelerator='ddp',
                 gpus=gpu_list,
                 max_epochs=args.epoch

                 )

    print(f"======= Training  =======")
    runner.fit(experiment)
    runner.save_checkpoint(config['logging_params']['ckpt_save_dir']+"/last.ckpt")