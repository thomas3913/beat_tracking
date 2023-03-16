import pytorch_lightning as pl
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import os
from torch.utils.data import Dataset
from beat_tracking.helper_functions import beat_list_to_array, get_note_sequence_and_annotations_from_midi

# Experimental configurations during model training.
learning_rate = 1e-3
dropout = 0.15
max_length = 500 # maximum length of input sequence

batch_size = 1
gpus = [0,1,2,3]
# batch_size = 64 # for 2 GPUs
# gpus = [0,1]

num_workers = 4

df2 = pd.read_csv("metadata.csv")

def get_midi_filelist(dataset_list):
    
    for i,el in enumerate(dataset_list):
        dataset_list[i]= el.upper()
    
    for elemnt in dataset_list:
        if elemnt not in ["ASAP","CPM","AMAPS"]:
            print("You can only enter ASAP, CPM or AMAPS")
            raise ValueError
    
    asap_dataset = "data/asap-dataset"
    AMAPS = "data/A-MAPS_1.2"
    CPM = "data/Piano-MIDI/midis"
    ACPAS = "data/ACPAS"

    df = pd.read_csv(Path(asap_dataset,"metadata.csv"))

    all_datasets = list()
    
    ASAP = list()
    for element in df["midi_performance"]:
        #midi_performances.append(str(Path(PROJECT_PATH,element)))
        ASAP.append(element)
        if "ASAP" in dataset_list:
            all_datasets.append(os.path.join(asap_dataset,element))
    
    AMAPS_files = list()
    for (dirpath, dirnames, filenames) in os.walk(AMAPS):
        AMAPS_files += [os.path.join(dirpath, file) for file in filenames if file[-3:] == "mid"]
    AMAPS_files = sorted(AMAPS_files)
    if "AMAPS" in dataset_list:
        all_datasets += [AMAPS_files[i] for i in range(len(AMAPS_files))]

    PIANO_MIDI_files = list()
    for (dirpath, dirnames, filenames) in os.walk(CPM):
        PIANO_MIDI_files += [os.path.join(dirpath, file) for file in filenames if file[-3:] in ["mid","MID"]]
    PIANO_MIDI_files = sorted(PIANO_MIDI_files)
    if "CPM" in dataset_list:
        all_datasets += [PIANO_MIDI_files[i] for i in range(len(PIANO_MIDI_files))]


    element_path_dict = {}
    for _ in range(2):
        error_list = list()
        for i, element in enumerate(all_datasets):
            if element.split("/")[1] == "A-MAPS_1.2":
                element_path = "../../MIDI-Datasets"+element[4:]
            elif element.split("/")[1] == "Piano-MIDI":
                element_path = "../../MIDI-Datasets"+element[4:]
            elif element.split("/")[1] == "asap-dataset":
                element_path = "../.."+element[4:]
            element_path_dict[element] = element_path
            try:
                split = df2.loc[df2["midi_perfm"] == element_path]["split"].iloc[0]
            except Exception as e:
                error_list.append(element)
                all_datasets.remove(element)
                del element_path_dict[element]

    return all_datasets, element_path_dict

def get_split_lists(filelist,path_dict):
    train_list = list()
    validation_list = list()
    test_list = list()
    not_used = list()
    for element in filelist:
        try:
            split = df2.loc[df2["midi_perfm"] == path_dict[element]]["split"].iloc[0]
        except:
            continue
        if split == "train":
            train_list.append(element)
        elif split == "valid":
            validation_list.append(element)
        elif split == "test":
            test_list.append(element)
        else:
            not_used.append(element)
    return train_list, validation_list, test_list

class Audio_Dataset(Dataset):
    def __init__(self,file_list,path_dict,mode):
        self.file_list = file_list
        self.path_dict = path_dict
        self.mode = mode

    def __getitem__(self, index):
        if df2.loc[df2["midi_perfm"] == self.path_dict[self.file_list[index]]]["source"].iloc[0] == "ASAP":  
            annotation_file = self.file_list[index][:-4]+"_annotations.txt"
            beats = beat_list_to_array(annotation_file,"annotations","beats")
            downbeats = beat_list_to_array(annotation_file,"annotations","downbeats")
        else:
            annot_from_midi = get_note_sequence_and_annotations_from_midi(self.file_list[index])
            beats = annot_from_midi[1]['beats']
            downbeats = annot_from_midi[1]['downbeats']

        if self.mode == "ismir":
            pr = np.load(self.file_list[index][:-4]+"_pianoroll.npy")
            return self.file_list[index], beats, downbeats, index, pr
        else:
            return self.file_list[index], beats, downbeats, index
    
    def __len__(self):
        return len(self.file_list)
        

class MyDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        
        # Parameters from input arguments
        self.dataset = args.dataset
        self.mode = args.mode
        
        if self.dataset == "all":
            self.file_list, self.path_dict = get_midi_filelist(["AMAPS","ASAP","CPM"])
        else:
            self.file_list,self.path_dict = get_midi_filelist([self.dataset])

        self.df2 = pd.read_csv("metadata.csv")

    def _get_dataset(self, split):
        train_list,validation_list,test_list = get_split_lists(self.file_list,self.path_dict)
        split_dict = {"train":train_list,"valid":validation_list,"test":test_list}
        dataset = Audio_Dataset(split_dict[split],self.path_dict,self.mode)
        return dataset

    def train_dataloader(self):
        dataset = self._get_dataset(split='train')
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = self._get_dataset(split='valid')
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True
        )
        return dataloader

    def test_dataloader(self):
        dataset = self._get_dataset(split='test')
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            drop_last=False
        )
        return dataloader