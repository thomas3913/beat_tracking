import os, argparse
import numpy as np
import torch
import torch.nn as nn
import warnings
from beat_tracking.helper_functions import *
from beat_tracking.data_loading import *
from beat_tracking.modules import TransformerModule

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings("ignore")

# TRAINING LOOP

def evaluate(args):
    data = CombinedDataModule(args)
    
    #checkpoints_dir = args.checkpoints_dir
    #figures_dir = args.figures_dir
    dataset = args.dataset

    model = TransformerModule(args).load_from_checkpoint(args=args,checkpoint_path="Transformer-training-all-lr_0.0001/byrn45q4/checkpoints/epoch=57-val_loss=4.62-val_f1=0.00.ckpt")
    
    
    trainer = pl.Trainer(
        #callbacks=[]
        default_root_dir="pl_checkpoints/",
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        #gpus=gpus,
        accelerator='gpu',
        devices=[0]
    )
    
    trainer.test(model, data)
    
    #trainer.test(model, data)
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Transformer.')

    #parser.add_argument('--checkpoints_dir', type=str, help='Where to store the results.')
    #parser.add_argument('--figures_dir', type=str, help='Where to store the plots.')
    parser.add_argument('--dataset', type=str, help='Which dataset?')
    parser.add_argument('--learning_rate', type=float, default= 1e-4, help='Learning rate')
    args = parser.parse_args()

    evaluate(args)