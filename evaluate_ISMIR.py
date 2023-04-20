import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import madmom
import partitura as pt
import torch
import warnings
from beat_tracking.helper_functions import *
from beat_tracking.data_loading import *
from beat_tracking.modules import MyMadmomModule

import pytorch_lightning as pl
# TESTING

warnings.filterwarnings("ignore")


def evaluate(args):
    data = MyDataModule(args)
    
    model = MyMadmomModule.load_from_checkpoint(args=args,checkpoint_path="/home/thomass/beat_tracking/ISMIR-training-all-pretty_midi-step_size_20-only_beats/9pjqc8mu/checkpoints/epoch=20-val_loss=0.09-val_f1=0.00.ckpt")


    trainer = pl.Trainer(
        default_root_dir="pl_checkpoints/",
        logger=None,
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        #gpus=gpus,
        accelerator='gpu',
        devices=[0]
    )
    
    trainer.test(model, data)

    
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate ISMIR.')

    parser.add_argument('--results_dir', type=str, help='Where to store the results.')
    parser.add_argument('--dataset', type=str, help='Which dataset?')
    parser.add_argument('--pianorolls', type=str, help='Which pianorolls?')
    parser.add_argument('--only_beats', type=bool, help='Only beats?')
    parser.add_argument('--mode', type=str, help='ismir/pm2s')
    parser.add_argument('--stepsize', type=int, help='stepsize')

    args = parser.parse_args()

    evaluate(args)