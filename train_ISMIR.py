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
    #data = MyDataModule(args)
    data = CombinedDataModule(args)
    
    dataset = args.dataset
    epochs = args.epochs
    full_train = args.full_train
    stepsize = args.stepsize
    
    model = MyMadmomModule(args)
    #model = MyMadmomModule.load_from_checkpoint(args=args, checkpoint_path="/home/thomass/beat_tracking/ISMIR-training-all-pretty_midi-step_size_20-only_beats/9pjqc8mu/checkpoints/epoch=20-val_loss=0.09-val_f1=0.00.ckpt")
    
    wandb_logger = WandbLogger(project="ISMIR-training-"+dataset+"-full_train_"+full_train+"-step_size_"+str(stepsize))
    
    trainer = pl.Trainer(
        default_root_dir="pl_checkpoints/",
        logger=wandb_logger,
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        max_epochs=int(epochs),
        #gpus=gpus,
        accelerator='gpu',
        devices=[0]
    )
    
    trainer.fit(model, data)
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train ISMIR.')

    parser.add_argument('--dataset', type=str, help='Which dataset?')
    parser.add_argument('--epochs', type=str, help='How many epochs?')
    parser.add_argument('--full_train',type=bool,help="Train on the whole dataset")
    parser.add_argument('--stepsize',  type=int, help='Adjust learning rate after ... epochs')

    args = parser.parse_args()

    train(args)