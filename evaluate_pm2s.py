from beat_tracking.models import MyMadmom, RNNJointBeatProcessor
from beat_tracking.data_loading import MyDataModule
import pandas as pd
import pretty_midi as pm
import mir_eval
import numpy as np
import argparse


def evaluate(args):
    data = MyDataModule(args)
    
    results_dir = args.results_dir
    dataset = args.dataset
    
    val_loader = data.val_dataloader()
    
    processor = RNNJointBeatProcessor()
    scores = []
    for i,element in enumerate(val_loader):
        beats_pred = processor.process(element[0][0])
        beats_targ = element[1]
        beats_pred_trimmed = mir_eval.beat.trim_beats(beats_pred)
        beats_targ_trimmed = mir_eval.beat.trim_beats(beats_targ)
        f1 = mir_eval.beat.f_measure(beats_targ_trimmed, beats_pred_trimmed)
        scores.append([i,element[0][0],f1])
        if i%10 == 0:
            print(i,element[0][0],f1)
    with open(results_dir+"/evaluate_pm2s_"+dataset+"_"+"val"+".txt","w") as fp:
        sum = 0
        for entry in scores:
            fp.write("%s\t%s\t%s\n" % (entry[0],entry[1],entry[2]))
            sum += entry[2]
        fp.write("%s\n" % "Summary: "+str(sum/len(scores)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate PM2S.')

    parser.add_argument('--results_dir', type=str, help='Where to store the results.')
    parser.add_argument('--dataset', type=str, help='Which dataset?')
    parser.add_argument('--mode', type=str, help='ismir/pm2s')

    args = parser.parse_args()

    evaluate(args)