import os, argparse
import numpy as np
import torch
import torch.nn as nn
import warnings
from beat_tracking.helper_functions import *
from beat_tracking.data_loading import *
from beat_tracking.modules import AudioModule, MyMadmomModule

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings("ignore")

# TRAINING LOOP

def train(args):
    data = Audio_and_pr_DataModule(args)
    
    dataset = args.dataset
    mode = args.mode
    epochs = args.epochs
    stepsize = args.stepsize
    full_train = args.full_train

    #train_loader = data.train_dataloader()
    #for i,elem in enumerate(train_loader):
    #    print(i,elem[0],elem[3].shape)
    
    if mode == "audio":
        model = AudioModule(args)
    elif mode == "pianorolls":
        model = MyMadmomModule(args)
    #model = MyMadmomModule.load_from_checkpoint(args=args, checkpoint_path="/home/thomass/beat_tracking/ISMIR-training-all-pretty_midi-step_size_20-only_beats/9pjqc8mu/checkpoints/epoch=20-val_loss=0.09-val_f1=0.00.ckpt")
    
    wandb_logger = WandbLogger(project="ISMIR-training-"+dataset+"-full_train_"+str(full_train))
    
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
    parser.add_argument('--mode', type=str, default="audio", help='Audio or pianorolls?')
    parser.add_argument('--epochs', type=str, help='How many epochs?')
    parser.add_argument('--stepsize',  type=int, help='Adjust learning rate after ... epochs')
    parser.add_argument('--full_train', default=False, action='store_true', help='Train on the whole dataset')

    args = parser.parse_args()

    train(args)