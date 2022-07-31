import yaml
import argparse
import numpy as np
import os
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from pytorch_lightning.callbacks import ModelCheckpoint
    
if __name__=='__main__':

    trainer = Trainer()
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
    parser.add_argument('--acc',  '-a',
                    dest="accelerator",
                    metavar='ACC',
                    help =  'accelerator',
                    default='ddp')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    ckpt_save_dir=config['logging_params']['ckpt_save_dir']
    if not os.path.exists(ckpt_save_dir):
        os.mkdir(ckpt_save_dir)
     
    #gpus=config['trainer_params']['gpus']
    #print(','.join([str(idx) for idx in gpus]))
    #os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(idx) for idx in gpus])
    
    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )

# For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = vae_models[config['model_params']['name']](**config['model_params'])
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #USE_CUDA = torch.cuda.is_available()
    #print(USE_CUDA)
    #device = torch.device("cuda:0" if USE_CUDA else "cpu")
    
   
    #model = nn.DataParallel(model)
    #model= model.to(device)
    experiment = VAEXperiment(model,
                              config['exp_params'])

# DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging_params']['ckpt_save_dir'],
        save_top_k=-1,
        verbose=True,
        #save_last=True,
        #monitor='loss',
        #mode='min',
        
        period=20
    )

    runner = Trainer(min_epochs=1,
                 logger=tt_logger,
                 log_every_n_steps=100000,
                 #train_percent_check=1.,
                 #val_percent_check=1.,
                 num_sanity_val_steps=5,
                 checkpoint_callback=True,
                 callbacks=checkpoint_callback,
                 accelerator=args.accelerator,
                 **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
    runner.save_checkpoint(config['logging_params']['ckpt_save_dir']+"/last.ckpt")