import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import sys
import time
import mir_eval
import madmom
import partitura as pt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
from beat_tracking.helper_functions import *
from beat_tracking.data_loading import *
from beat_tracking.models import my_madmom

import pytorch_lightning as pl

AMAPS, AMAPS_path_dict = get_midi_filelist(["AMAPS"])
CPM, CPM_path_dict = get_midi_filelist(["CPM"])
ASAP, ASAP_path_dict = get_midi_filelist(["ASAP"])

df2 = pd.read_csv("metadata.csv")
all_datasets, all_datasets_path_dict = get_midi_filelist(["ASAP","CPM","AMAPS"])

# Dataloaders:
train_loader_asap = get_dataloader(get_pianoroll_dataset(ASAP,ASAP_path_dict),"train",shuffle=True)
train_loader_amaps = get_dataloader(get_pianoroll_dataset(AMAPS,AMAPS_path_dict),"train",shuffle=True)
train_loader_cpm = get_dataloader(get_pianoroll_dataset(CPM,CPM_path_dict),"train",shuffle=True)
val_loader_asap = get_dataloader(get_pianoroll_dataset(ASAP,ASAP_path_dict),"val",shuffle=False)
val_loader_amaps = get_dataloader(get_pianoroll_dataset(AMAPS,AMAPS_path_dict),"val",shuffle=False)
val_loader_cpm = get_dataloader(get_pianoroll_dataset(CPM,CPM_path_dict),"val",shuffle=False)
test_loader_asap = get_dataloader(get_pianoroll_dataset(ASAP,ASAP_path_dict),"test",shuffle=False)
test_loader_amaps = get_dataloader(get_pianoroll_dataset(AMAPS,AMAPS_path_dict),"test",shuffle=False)
test_loader_cpm = get_dataloader(get_pianoroll_dataset(CPM,CPM_path_dict),"test",shuffle=False)

print("Lengths of training loaders:",len(train_loader_asap),len(train_loader_amaps),len(train_loader_cpm))
print("Lengths of validation loaders:",len(val_loader_asap),len(val_loader_amaps),len(val_loader_cpm))
print("Lengths of test loaders:",len(test_loader_asap),len(test_loader_amaps),len(test_loader_cpm))

# TESTING

checkpoint_list = ["models_ISMIR/model_iter_96000.pt"]

def evaluate(args):
    results_dir = args.results_dir
    dataset = args.dataset
    split = args.split

    for checkpoint in checkpoint_list:
        try:
            model = torch.load(checkpoint,map_location = torch.device("cpu"))
        except Exception as e:
            print(checkpoint,"error:\n",e)
            continue
        model.eval()
        warnings.filterwarnings("ignore")
        
        device = next(model.parameters()).device
        print("\nCheckpoint:",checkpoint,"--- Device:",device,"\n")

        scores = []
        strr = split+"_loader_"+dataset
        with torch.no_grad():
            for j, datapoint_test in enumerate(globals()[strr]):

                padded_array_test = cnn_pad(datapoint_test[0].float(),2)
                padded_array_test = torch.tensor(padded_array_test)
                
                print("Test sample number",j+1,"/",len(globals()[strr]),"---",datapoint_test[4][0],"--- Input shape:",datapoint_test[0].shape)
                outputs = model(padded_array_test)

                #Calculate F-Score:
                try:
                    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
                    beat_times = proc(outputs[0][:,0].detach().cpu().numpy())
                    evaluate = madmom.evaluation.beats.BeatEvaluation(beat_times, datapoint_test[1][0])
                    f_score_test = evaluate.fmeasure
                except Exception as e:
                    print("Sample can not be processed correctly. Error in beat process:",e)
                    f_score_test = 0

                scores.append(j,datapoint_test[4][0],f_score_test)
                #if j % 10 == 0:
                print(j,datapoint_test[4][0],f_score_test)
        
        with open(results_dir+"/evaluate_ISMIR_"+dataset+"_"+split+".txt","w") as fp:
            sum = 0
            for entry in scores:
                fp.write("%s\t%s\t%s\n" % (entry[0],entry[1],entry[2]))
                sum += entry[2]
            fp.write("%s\n" % "Summary: "+str(sum/len(scores)))
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate ISMIR.')

    parser.add_argument('--results_dir', type=str, help='Where to store the results.')
    parser.add_argument('--dataset', type=str, help='Which dataset?')
    parser.add_argument('--split', type=str, help='Train/val/test.')

    args = parser.parse_args()

    evaluate(args)