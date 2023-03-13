from beat_tracking.models import my_madmom, RNNJointBeatProcessor
from beat_tracking.data_loading import get_midi_filelist, get_pianoroll_dataset, get_dataloader
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pretty_midi as pm
import mir_eval
import numpy as np
import argparse

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

def evaluate(args):
    results_dir = args.results_dir
    dataset = args.dataset
    split = args.split
    
    processor = RNNJointBeatProcessor()
    scores = []
    strr = split+"_loader_"+dataset
    for i,element in enumerate(globals()[strr]):
        beats_pred = processor.process(element[4][0])
        beats_targ = element[1]
        beats_pred_trimmed = mir_eval.beat.trim_beats(beats_pred)
        beats_targ_trimmed = mir_eval.beat.trim_beats(beats_targ)
        f1 = mir_eval.beat.f_measure(beats_targ_trimmed, beats_pred_trimmed)
        scores.append([i,element[4][0],f1])
        if i%10 == 0:
            print(i,element[4][0],f1)
    with open(results_dir+"/evaluate_pm2s_"+dataset+"_"+split+".txt","w") as fp:
        sum = 0
        for entry in scores:
            fp.write("%s\t%s\t%s\n" % (entry[0],entry[1],entry[2]))
            sum += entry[2]
        fp.write("%s\n" % "Summary: "+str(sum/len(scores)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate PM2S.')

    parser.add_argument('--results_dir', type=str, help='Where to store the results.')
    parser.add_argument('--dataset', type=str, help='Which dataset?')
    parser.add_argument('--split', type=str, help='Train/val/test.')

    args = parser.parse_args()

    evaluate(args)