from beat_tracking.data_loading import Audio_and_pr_DataModule
from beat_tracking.modules import BeatModule
#import mir_eval
import numpy as np
import argparse
import pytorch_lightning as pl
import warnings

warnings.filterwarnings("ignore")


def evaluate(args):
    data = Audio_and_pr_DataModule(args)

    dataset = args.dataset

    train_loader = data.train_dataloader()

    abc =  next(iter(train_loader))
    print(abc)

    

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

    parser = argparse.ArgumentParser(description='Evaluate Audio/Pianorolls.')

    parser.add_argument('--dataset', type=str,default= "all", help='Which dataset?')

    args = parser.parse_args()

    evaluate(args)