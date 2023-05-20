from beat_tracking.data_loading import Pm2sDataModule
from beat_tracking.modules import BeatModule
import pandas as pd
import pretty_midi as pm
#import mir_eval
import madmom
import numpy as np
import argparse
import pytorch_lightning as pl
import warnings

warnings.filterwarnings("ignore")


def evaluate(args):
    data = Pm2sDataModule(args)
    
    dataset = args.dataset

    model = BeatModule.load_from_checkpoint(args=args,checkpoint_path="PM2S-training-all/ffs82z8g/checkpoints/epoch=19-step=16000.ckpt")

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
    
    """
    processor = RNNJointBeatProcessor()
    scores = []
    for i,element in enumerate(val_loader):
        beats_pred = processor.process(element[0][0])
        beats_targ = element[1]
        #beats_pred_trimmed = mir_eval.beat.trim_beats(beats_pred)
        #beats_targ_trimmed = mir_eval.beat.trim_beats(beats_targ)
        #f1 = mir_eval.beat.f_measure(beats_targ_trimmed, beats_pred_trimmed)
        evaluate = madmom.evaluation.beats.BeatEvaluation(beats_targ, beats_pred)
        f1 = evaluate.fmeasure
        scores.append([i,element[0][0],f1])
        if i%10 == 0:
            print(i,element[0][0],f1)
    with open(results_dir+"/evaluate_pm2s_"+dataset+"_"+"val"+".txt","w") as fp:
        sum = 0
        for entry in scores:
            fp.write("%s\t%s\t%s\n" % (entry[0],entry[1],entry[2]))
            sum += entry[2]
        fp.write("%s\n" % "Summary: "+str(sum/len(scores)))
    """

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate PM2S.')

    parser.add_argument('--dataset', type=str,default= "all", help='Which dataset?')

    args = parser.parse_args()

    evaluate(args)