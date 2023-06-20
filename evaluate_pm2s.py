from beat_tracking.data_loading import CombinedDataModule
from beat_tracking.modules import BeatModule
#import mir_eval
import numpy as np
import argparse
import pytorch_lightning as pl
import warnings

warnings.filterwarnings("ignore")


def evaluate(args):
    data = CombinedDataModule(args)
    
    dataset = args.dataset

    model = BeatModule.load_from_checkpoint(args=args,checkpoint_path="PM2S-training-all/a0h12lhr/checkpoints/epoch=21-val_loss=5.44-val_f1=0.00.ckpt")

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

    parser = argparse.ArgumentParser(description='Evaluate PM2S.')

    parser.add_argument('--dataset', type=str,default= "all", help='Which dataset?')

    args = parser.parse_args()

    evaluate(args)