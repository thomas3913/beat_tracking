from beat_tracking.data_loading import Audio_and_pr_DataModule
from beat_tracking.modules import AudioModule, MyMadmomModule_pr
#import mir_eval
import numpy as np
import argparse
import pytorch_lightning as pl
import warnings

warnings.filterwarnings("ignore")


def evaluate(args):
    data = Audio_and_pr_DataModule(args)

    dataset = args.dataset
    mode = args.mode

    if mode == "audio":
        model = AudioModule.load_from_checkpoint(args=args,checkpoint_path="ISMIR-training-all-audio-full_train_False/e1dy1280/checkpoints/epoch=27-val_loss=0.14-val_f1=0.00.ckpt")
    elif mode == "pianorolls":
        model = MyMadmomModule_pr.load_from_checkpoint(args=args,checkpoint_path="")
    

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
    parser.add_argument('--mode', type=str, default="audio", help='Audio or pianorolls?')

    args = parser.parse_args()

    evaluate(args)