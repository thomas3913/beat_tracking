import pytorch_lightning as pl
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import os
from torch.utils.data import Dataset
from beat_tracking.helper_functions import beat_list_to_array, get_note_sequence_and_annotations_from_midi
from collections import defaultdict

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
    def __init__(self,file_list,path_dict,mode,pianorolls=None):
        self.file_list = file_list
        self.path_dict = path_dict
        self.mode = mode
        self.pianorolls = pianorolls

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
            self.file_list, self.path_dict = get_midi_filelist(["AMAPS","ASAP","CPM"])
        else:
            self.file_list,self.path_dict = get_midi_filelist([self.dataset])

        self.df2 = pd.read_csv("metadata.csv")

    def _get_dataset(self, split):
        train_list,validation_list,test_list = get_split_lists(self.file_list,self.path_dict)
        split_dict = {"train":train_list,"valid":validation_list,"test":test_list}
        if self.mode == "ismir":
            dataset = Audio_Dataset(split_dict[split],self.path_dict,self.mode,self.pianorolls)
        else:
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





#PM2S

class DataAugmentation():
    def __init__(self, 
        tempo_change_prob=1.0,
        tempo_change_range=(0.8, 1.2),
        pitch_shift_prob=1.0,
        pitch_shift_range=(-12, 12),
        extra_note_prob=0.5,
        missing_note_prob=0.5):

        if extra_note_prob + missing_note_prob > 1.:
            extra_note_prob, missing_note_prob = extra_note_prob / (extra_note_prob + missing_note_prob), \
                                                missing_note_prob / (extra_note_prob + missing_note_prob)
            print('INFO: Reset extra_note_prob and missing_note_prob to', extra_note_prob, missing_note_prob)
        
        self.tempo_change_prob = tempo_change_prob
        self.tempo_change_range = tempo_change_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.extra_note_prob = extra_note_prob
        self.missing_note_prob = missing_note_prob

    def __call__(self, note_sequence, annotations):
        # tempo change
        if random.random() < self.tempo_change_prob:
            note_sequence, annotations = self.tempo_change(note_sequence, annotations)

        # pitch shift
        if random.random() < self.pitch_shift_prob:
            note_sequence, annotations = self.pitch_shift(note_sequence, annotations)

        # extra note or missing note
        extra_or_missing = random.random()
        if extra_or_missing < self.extra_note_prob:
            note_sequence, annotations = self.extra_note(note_sequence, annotations)
        elif extra_or_missing > 1. - self.missing_note_prob:
            note_sequence, annotations = self.missing_note(note_sequence, annotations)

        return note_sequence, annotations

    def tempo_change(self, note_sequence, annotations):
        tempo_change_ratio = random.uniform(*self.tempo_change_range)
        note_sequence[:,1:3] *= 1 / tempo_change_ratio
        annotations['beats'] *= 1 / tempo_change_ratio
        annotations['downbeats'] *= 1 / tempo_change_ratio
        annotations['time_signatures'][:,0] *= 1 / tempo_change_ratio
        annotations['key_signatures'][:,0] *= 1 / tempo_change_ratio
        return note_sequence, annotations

    def pitch_shift(self, note_sequence, annotations):
        shift = round(random.uniform(*self.pitch_shift_range))
        note_sequence[:,0] += shift

        for i in range(len(annotations['key_signatures'])):
            annotations['key_signatures'][i,1] += shift
            annotations['key_signatures'][i,1] %= 24
            
        return note_sequence, annotations

    def extra_note(self, note_sequence, annotations):
        # duplicate
        note_sequence_new = np.zeros([len(note_sequence) * 2, 4])
        note_sequence_new[::2,:] = np.copy(note_sequence)  # original notes
        note_sequence_new[1::2,:] = np.copy(note_sequence)  # extra notes

        if annotations['onsets_musical'] is not None:
            onsets_musical_new = np.zeros(len(note_sequence) * 2)
            onsets_musical_new[::2] = np.copy(annotations['onsets_musical'])
            onsets_musical_new[1::2] = np.copy(annotations['onsets_musical'])

        if annotations['note_value'] is not None:
            note_value_new = np.zeros(len(note_sequence) * 2)
            note_value_new[::2] = np.copy(annotations['note_value'])
            note_value_new[1::2] = np.copy(annotations['note_value'])

        if annotations['hands'] is not None:
            hands_new = np.zeros(len(note_sequence) * 2)
            hands_new[::2] = np.copy(annotations['hands'])
            hands_new[1::2] = np.copy(annotations['hands'])
            hands_mask = np.ones(len(note_sequence) * 2)  # mask out hand for extra notes during training
            hands_mask[1::2] = 0

        # pitch shift for extra notes (+-12)
        shift = ((np.round(np.random.random(len(note_sequence_new))) - 0.5) * 24).astype(int)
        shift[::2] = 0
        note_sequence_new[:,0] += shift
        note_sequence_new[:,0][note_sequence_new[:,0] < 0] += 12
        note_sequence_new[:,0][note_sequence_new[:,0] > 127] -= 12

        # keep a random ratio of extra notes
        ratio = random.random() * 0.3
        probs = np.random.random(len(note_sequence_new))
        probs[::2] = 0.
        remaining = probs < ratio
        note_sequence_new = note_sequence_new[remaining]
        if annotations['onsets_musical'] is not None:
            annotations['onsets_musical'] = onsets_musical_new[remaining]
        if annotations['note_value'] is not None:
            annotations['note_value'] = note_value_new[remaining]
        if annotations['hands'] is not None:
            annotations['hands'] = hands_new[remaining]
            annotations['hands_mask'] = hands_mask[remaining]

        return note_sequence_new, annotations

    def missing_note(self, note_sequence, annotations):
        # find successing concurrent notes
        candidates = np.diff(note_sequence[:,1]) < tolerance

        # randomly select a ratio of candidates to be removed
        ratio = random.random()
        candidates_probs = candidates * np.random.random(len(candidates))
        remaining = np.concatenate([np.array([True]), candidates_probs < (1 - ratio)])

        # remove selected candidates
        note_sequence = note_sequence[remaining]
        if annotations['onsets_musical'] is not None:
            annotations['onsets_musical'] = annotations['onsets_musical'][remaining]
        if annotations['note_value'] is not None:
            annotations['note_value'] = annotations['note_value'][remaining]
        if annotations['hands'] is not None:
            annotations['hands'] = annotations['hands'][remaining]

        return note_sequence, annotations



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
        self.dataaug = DataAugmentation()

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
        note_sequence, annotations = pickle.load(open(path, 'rb'))

        # Data augmentation
        if self.split == 'train' or self.split == 'all':
            note_sequence, annotations = self.dataaug(note_sequence, annotations)

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