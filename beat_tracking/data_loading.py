import pytorch_lightning as pl
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import os
from torch.utils.data import Dataset
from beat_tracking.helper_functions import beat_list_to_array, get_note_sequence_from_midi, get_annotations_from_annot_file, get_note_sequence_and_annotations_from_midi
from collections import defaultdict
import random

# ========== data representation related constants ==========
## quantisation resolution
resolution = 0.01  # quantization resolution: 0.01s = 10ms
tolerance = 0.05  # tolerance for beat alignment: 0.05s = 50ms
ibiVocab = int(4 / resolution) + 1  # vocabulary size for inter-beat-interval: 4s = 4/0.01s + 1, index 0 is ignored during training

# ========== post-processing constants ==========
min_bpm = 40
max_bpm = 240
ticks_per_beat = 240

# =========== time signature definitions ===========
tsDenominators = [0, 2, 4, 8]  # 0 for others
tsDeno2Index = {0: 0, 2: 1, 4: 2, 8: 3}
tsIndex2Deno = {0: 0, 1: 2, 2: 4, 3: 8}
tsDenoVocabSize = len(tsDenominators)

tsNumerators = [0, 2, 3, 4, 6]  # 0 for others
tsNume2Index = {0: 0, 2: 1, 3: 2, 4: 3, 6: 4}
tsIndex2Nume = {0: 0, 1: 2, 2: 3, 3: 4, 4: 6}
tsNumeVocabSize = len(tsNumerators)

# =========== key signature definitions ==========
# key in sharps in mido
keySharps2Name = {0: 'C', 1: 'G', 2: 'D', 3: 'A', 4: 'E', 5: 'B', 6: 'F#',
                  7: 'C#m', 8: 'G#m', 9: 'D#m', 10: 'Bbm', 11: 'Fm', 12: 'Cm',
                  -11: 'Gm', -10: 'Dm', -9: 'Am', -8: 'Em', -7: 'Bm', -6: 'F#m',
                  -5: 'Db', -4: 'Ab', -3: 'Eb', -2: 'Bb', -1: 'F'}
keyName2Sharps = dict([(name, sharp) for sharp, name in keySharps2Name.items()])
# key in numbers in pretty_midi
keyNumber2Name = [
    'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B',
    'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'Bbm', 'Bm',
]
keyName2Number = dict([(name, number) for number, name in enumerate(keyNumber2Name)])
keySharps2Number = dict([(sharp, keyName2Number[keySharps2Name[sharp]]) for sharp in keySharps2Name.keys()])
keyNumber2Sharps = dict([(number, keyName2Sharps[keyNumber2Name[number]]) for number in range(len(keyNumber2Name))])
keyVocabSize = len(keySharps2Name) // 2  # ignore minor keys in key signature prediction!

# =========== onset musical & note value definitions ===========
# proposed model
N_per_beat = 24  # 24 resolution per beat
max_note_value = 4 * N_per_beat  # 4 beats
omVocab = N_per_beat
nvVocab = max_note_value

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

    #df = pd.read_csv(Path(asap_dataset,"metadata.csv"))

    midi_filelist = list()
    
    for element in df2["midi_perfm"]:
        if "AMAPS" in dataset_list:
            if df2.loc[df2["midi_perfm"]==element]["source"].iloc[0] == "MAPS":
                midi_filelist.append(element)
        if "ASAP" in dataset_list:
            if df2.loc[df2["midi_perfm"]==element]["source"].iloc[0] == "ASAP":
                midi_filelist.append(element)
        if "CPM" in dataset_list:
            if df2.loc[df2["midi_perfm"]==element]["source"].iloc[0] == "CPM":
                midi_filelist.append(element)

    return midi_filelist

def get_split_lists(filelist):
    train_list = list()
    validation_list = list()
    test_list = list()
    not_used = list()
    for element in filelist:
        try:
            split = df2.loc[df2["midi_perfm"] == element]["split"].iloc[0]
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
    def __init__(self,file_list,mode,pianorolls=None):
        self.file_list = file_list
        self.mode = mode
        self.pianorolls = pianorolls

    def __getitem__(self, index):
        if df2.loc[df2["midi_perfm"] == self.file_list[index]]["source"].iloc[0] == "ASAP":  
            annotation_file = self.file_list[index][:-4]+"_annotations.txt"
            beats = beat_list_to_array(annotation_file,"annotations","beats")
            downbeats = beat_list_to_array(annotation_file,"annotations","downbeats")
        else:
            annot_from_midi = get_note_sequence_and_annotations_from_midi(self.file_list[index])
            beats = annot_from_midi[1]['beats']
            downbeats = annot_from_midi[1]['downbeats']

        if self.mode == "ismir":
            if self.pianorolls == "partitura":
                pr = np.load(self.file_list[index][:-4]+"_pianoroll.npy")
            else:
                pr = np.load(self.file_list[index][:-4]+"_pianoroll_pm.npy")
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
        if self.mode == "ismir":
            self.pianorolls = args.pianorolls
        
        if self.dataset == "all":
            self.file_list = get_midi_filelist(["AMAPS","ASAP","CPM"])
        else:
            self.file_list = get_midi_filelist([self.dataset])

        self.df2 = pd.read_csv("metadata.csv")

    def _get_dataset(self, split):
        train_list,validation_list,test_list = get_split_lists(self.file_list)
        split_dict = {"train":train_list,"valid":validation_list,"test":test_list}
        if self.mode == "ismir":
            dataset = Audio_Dataset(split_dict[split],self.mode,self.pianorolls)
        else:
            dataset = Audio_Dataset(split_dict[split],self.mode)
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



#PM2S

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, split, from_asap=True):

        # parameters
        self.split = split

        # Get metadata by split
        metadata = pd.read_csv('metadata.csv')
        if split == 'all':
            self.metadata = metadata
        else:
            self.metadata = metadata[metadata['split'] == split]
        if not from_asap:
            self.metadata = self.metadata[self.metadata['source'] != 'ASAP']
        self.metadata.reset_index(inplace=True)

        # Get distinct pieces
        self.piece2row = defaultdict(list)
        for i, row in self.metadata.iterrows():
            self.piece2row[row['piece_id']].append(i)
        self.pieces = list(self.piece2row.keys())

        # Initialise data augmentation
        #self.dataaug = DataAugmentation()

    def __len__(self):
        if self.split == 'train' or self.split == 'all':
            # constantly update 200 steps per epoch, not related to training dataset size
            return batch_size * len(gpus) * 200

        elif self.split == 'valid':
            # by istinct pieces in validation set
            return batch_size * len(self.piece2row)  # valid dataset size

        elif self.split == 'test':
            return len(self.metadata)

    def _sample_row(self, idx):
        # Sample one row from the metadata
        if self.split == 'train' or self.split == 'all':
            piece_id = random.choice(list(self.piece2row.keys()))   # random sampling by piece
            row_id = random.choice(self.piece2row[piece_id])
        elif self.split == 'valid':
            piece_id = self.pieces[idx // batch_size]    # by istinct pieces in validation set
            row_id = self.piece2row[piece_id][idx % batch_size % len(self.piece2row[piece_id])]
        elif self.split == 'test':
            row_id = idx
        row = self.metadata.iloc[row_id]

        return row

    def _load_data(self, row):
        # Get feature
        #note_sequence, annotations = pickle.load(open(str(Path(self.feature_folder, row['feature_file'])), 'rb'))
        path = str(Path(row['feature_file'])).replace("\\","/")

        if row['source'] == 'ASAP':
            # get note sequence
            note_sequence = get_note_sequence_from_midi(row['midi_perfm'])
            # get annotations dict (beats, downbeats, key signatures, time signatures)
            annotations = get_annotations_from_annot_file(row['midi_perfm'][:-4]+"_annotations.txt")
        else:
            # get note sequence and annotations dict
            # (beats, downbeats, key signatures, time signatures, musical onset times, note value in beats, hand parts)
            note_sequence, annotations = get_note_sequence_and_annotations_from_midi(row['midi_perfm'])

        #note_sequence, annotations = pickle.load(open(path, 'rb'))

        # Data augmentation
        #if self.split == 'train' or self.split == 'all':
        #    note_sequence, annotations = self.dataaug(note_sequence, annotations)

        # Randomly sample a segment that is at most max_length long
        if self.split == 'train' or self.split == 'all':
            start_idx = random.randint(0, len(note_sequence)-1)
            end_idx = start_idx + max_length
        elif self.split == 'valid':
            start_idx, end_idx = 0, max_length  # validate on the segment starting with the first note
        elif self.split == 'test':
            start_idx, end_idx = 0, len(note_sequence)  # test on the whole note sequence

        if end_idx > len(note_sequence):
            end_idx = len(note_sequence)

        note_sequence = note_sequence[start_idx:end_idx]
        for key in annotations.keys():
            if key in ['onsets_musical', 'note_value', 'hands', 'hands_mask'] and annotations[key] is not None:
                annotations[key] = annotations[key][start_idx:end_idx]

        return note_sequence, annotations



class BeatDataset(BaseDataset):

    def __init__(self, split):
        super().__init__(split)

    def __getitem__(self, idx):

        row = self._sample_row(idx)
        note_sequence, annotations = self._load_data(row)

        # Get model output data
        # beats downbeats
        beats = annotations['beats']
        downbeats = annotations['downbeats']

        # time to beat/downbeat/inter-beat-interval dictionaries
        end_time = max(beats[-1], note_sequence[-1][1] + note_sequence[-1][2]) + 1.0
        time2beat = np.zeros(int(np.ceil(end_time / resolution)))
        time2downbeat = np.zeros(int(np.ceil(end_time / resolution)))
        time2ibi = np.zeros(int(np.ceil(end_time / resolution)))
        for idx, beat in enumerate(beats):
            l = np.round((beat - tolerance) / resolution).astype(int)
            r = np.round((beat + tolerance) / resolution).astype(int)
            time2beat[l:r+1] = 1.0

            ibi = beats[idx+1] - beats[idx] if idx+1 < len(beats) else beats[-1] - beats[-2]
            l = np.round((beat - tolerance) / resolution).astype(int) if idx > 0 else 0
            r = np.round((beat + ibi) / resolution).astype(int) if idx+1 < len(beats) else len(time2ibi)
            if ibi > 4:
                # reset ibi to 0 if it's too long, index 0 will be ignored during training
                ibi = np.array(0)
            time2ibi[l:r+1] = np.round(ibi / resolution)
        
        for downbeat in downbeats:
            l = np.round((downbeat - tolerance) / resolution).astype(int)
            r = np.round((downbeat + tolerance) / resolution).astype(int)
            time2downbeat[l:r+1] = 1.0
        
        # get beat probabilities at note onsets
        beat_probs = np.zeros(len(note_sequence), dtype=np.float32)
        downbeat_probs = np.zeros(len(note_sequence), dtype=np.float32)
        ibis = np.zeros(len(note_sequence), dtype=np.float32)
        for i in range(len(note_sequence)):
            onset = note_sequence[i][1]
            beat_probs[i] = time2beat[np.round(onset / resolution).astype(int)]
            downbeat_probs[i] = time2downbeat[np.round(onset / resolution).astype(int)]
            ibis[i] = time2ibi[np.round(onset / resolution).astype(int)]
        
        # pad if length is shorter than max_length
        length = len(note_sequence)
        if len(note_sequence) < max_length:
            note_sequence = np.concatenate([note_sequence, np.zeros((max_length - len(note_sequence), 4))])
            beat_probs = np.concatenate([beat_probs, np.zeros(max_length - len(beat_probs))])
            downbeat_probs = np.concatenate([downbeat_probs, np.zeros(max_length - len(downbeat_probs))])
            ibis = np.concatenate([ibis, np.zeros(max_length - len(ibis))])
            
        return (
            note_sequence, 
            beat_probs, 
            downbeat_probs, 
            ibis, 
            length,
        )



class Pm2sDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        
        # Parameters from input arguments
        self.dataset = args.dataset
        self.full_train = False

    def _get_dataset(self, split):
        dataset = BeatDataset(split)
        return dataset

    def train_dataloader(self):
        if self.full_train:
            dataset = self._get_dataset(split='all')
        else:
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