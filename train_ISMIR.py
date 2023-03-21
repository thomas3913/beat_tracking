import os, argparse
import numpy as np
import torch
import torch.nn as nn
import warnings
from beat_tracking.helper_functions import *
from beat_tracking.data_loading import *
from beat_tracking.models import MyMadmom
from beat_tracking.modules import MyMadmomModule

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings("ignore")

# TRAINING LOOP

def train(args):
    data = MyDataModule(args)
    
    checkpoints_dir = args.checkpoints_dir
    figures_dir = args.figures_dir
    dataset = args.dataset
    epochs = args.epochs
    
    model = MyMadmomModule()
    
    wandb_logger = WandbLogger(project="ISMIR-training-"+dataset)
    
    trainer = pl.Trainer(
        
        logger=wandb_logger,
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        max_epochs=int(epochs),
        #gpus=gpus,
        accelerator='gpu',
        devices=[0]
    )
    
    trainer.fit(model, data)
    
    num_epochs = 6
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train ISMIR.')

    parser.add_argument('--checkpoints_dir', type=str, help='Where to store the results.')
    parser.add_argument('--figures_dir', type=str, help='Where to store the plots.')
    parser.add_argument('--dataset', type=str, help='Which dataset?')
    parser.add_argument('--mode', type=str, help='ismir/pm2s')
    parser.add_argument('--epochs', type=str, help='How many epochs?')

    args = parser.parse_args()

    train(args)