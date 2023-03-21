import torch
import torch.nn as nn

from beat_tracking.pm2s_files.constants import ibiVocab
from beat_tracking.pm2s_files.utils import get_in_features, encode_note_sequence
from beat_tracking.pm2s_files.constants import min_bpm, tolerance

import pretty_midi
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_features, hidden_size=512, kernel_size=9, dropout=0.15):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_size // 4,
                kernel_size=(kernel_size, in_features),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=hidden_size // 4,
                out_channels=hidden_size // 2,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=hidden_size // 2,
                out_channels=hidden_size,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(hidden_size),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        # x: (batch_size, sequence_length, in_features)
        x = x.unsqueeze(1)   # (batch_size, 1, sequence_length, in_features)
        x = self.conv(x)   # (batch_size, hidden_size, sequence_length, 1)
        x = x.squeeze(3).transpose(1, 2)  # (batch_size, sequence_length, hidden_size)
        return x


class GRUBlock(nn.Module):
    def __init__(self, in_features, hidden_size=512, gru_layers=2, dropout=0.15):
        super().__init__()

        self.grus_beat = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)

        x, _ = self.grus_beat(x)  # (batch_size, sequence_length, hidden_size*2)
        x = self.linear(x)  # (batch_size, sequence_length, hidden_size)

        return x


class LinearOutput(nn.Module):
    def __init__(self, in_features, out_features, activation_type='sigmoid', dropout=0.15):
        super().__init__()

        self.activation_type = activation_type

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features, out_features)

        if activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'softmax':
            self.activation = nn.LogSoftmax(dim=2)
        elif activation_type == 'softplus':
            self.activation = nn.Softplus()

    def forward(self, x):
        # x: (batch_size, sequence_length, in_features)

        x = self.dropout(x)  # (batch_size, sequence_length, in_features)
        x = self.linear(x)  # (batch_size, sequence_length, out_features)
        x = self.activation(x)  # (batch_size, sequence_length, out_features)

        return x

def read_note_sequence(midi_file):
    """
    Load MIDI file into note sequence.

    Parameters
    ----------
    midi_file : str
        Path to MIDI file.

    Returns
    -------
    note_seq: (numpy.array) in the shape of (n, 4), where n is the number of notes, and 4 is the number of features including (pitch, onset, offset, velocity). The note sequence is sorted by onset time.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        notes.extend(instrument.notes)
    notes = sorted(notes, key=lambda x: x.start)
    note_seq = np.array([[note.pitch, note.start, note.end - note.start, note.velocity] for note in notes])
    return note_seq

class MyMadmom(nn.Module):

    def __init__(self):
        super(MyMadmom,self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=20,kernel_size=(3,3),padding="valid")
        self.act1 = torch.nn.ELU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(1,3))
        self.dropout1 = torch.nn.Dropout(0.15)
        
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=20,kernel_size=(1,10),padding="valid")
        self.act2 = torch.nn.ELU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(1,3))
        self.dropout2 = torch.nn.Dropout(0.15)
        
        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=20,kernel_size=(3,3),padding="valid")
        self.act3 = torch.nn.ELU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(1,3))
        self.dropout3 = torch.nn.Dropout(0.15)
        
        self.name = 'tcn'
        self.dropout_rate = 0.15
        self.activation = 'elu'
        self.dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.kernel_size = 5
        self.num_filters = [20] * len(self.dilations)
        self.padding = 'same'
        
        self.res_x = torch.nn.ModuleList()
        self.layer_list = torch.nn.ModuleList()
        self.layer_list_2 = torch.nn.ModuleList()
        self.activation4 = torch.nn.ModuleList()
        self.dropout4 = torch.nn.ModuleList()
        self.conv1d = torch.nn.ModuleList()
        
        for i in self.dilations:
            self.res_x.append(torch.nn.Conv1d(in_channels=20,out_channels=20, kernel_size=1, padding='same'))
            self.layer_list.append(torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=self.kernel_size, dilation=i, padding=self.padding))
            self.layer_list_2.append(torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=self.kernel_size, dilation=i*2, padding=self.padding))
            self.activation4.append(torch.nn.ELU())
            self.dropout4.append(torch.nn.Dropout(0.15))
            self.conv1d.append(torch.nn.Conv1d(in_channels=40, out_channels=20,kernel_size=1, padding='same'))

        self.activation5 = torch.nn.ELU()

        self.beats1 = torch.nn.Dropout(0.15)
        self.beats2 = torch.nn.Linear(in_features=20,out_features=1)
        self.beats3 = torch.nn.Sigmoid()
        
        self.downbeats1 = torch.nn.Dropout(0.15)
        self.downbeats2 = torch.nn.Linear(in_features=20,out_features=1)
        self.downbeats3 = torch.nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = torch.reshape(x,(20,-1))
                      
        # TCN:
        indx = 0
        for i, num_filters in zip(self.dilations, self.num_filters):
            # feed the output of the previous layer into the next layer
            # increase dilation rate for each consecutive layer
            
            name = self.name + '_dilation_%d' % i
            # 1x1 conv. of input (so it can be added as residual)
            res_x = self.res_x[indx](x)
                        
            conv_1 = self.layer_list[indx](x)
            conv_2 = self.layer_list_2[indx](x)
    
            # concatenate the output of the two dilations
            concat = torch.cat((conv_1,conv_2))
            
            # apply activation function
            x = self.activation4[indx](concat)
            
            # apply spatial dropout
            x = torch.transpose(x,0,1)
            x = self.dropout4[indx](x)
            x = torch.transpose(x,0,1)
            
            # 1x1 conv. to obtain a representation with the same size as the residual
            x = self.conv1d[indx](x)
            
            # add the residual to the processed data and also return it as skip connection
            x = torch.add(x,res_x)

            indx += 1
            
        # activate the output of the TCN stack
        x = self.activation5(x)
        
        beats = self.beats1(x)
        beats = torch.transpose(beats,0,1)
        beats = self.beats2(beats)
        beats = self.beats3(beats)   
        
        downbeats = self.downbeats1(x)
        downbeats = torch.transpose(downbeats,0,1)
        downbeats = self.downbeats2(downbeats)
        downbeats = self.downbeats3(downbeats) 
        
        return beats, downbeats
    

class MIDIProcessor(object):
    """
    Abstract base class for processing MIDI data.
    """
    
    def __init__(self, model_state_dict_path=None, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
        """
        self._kwargs = kwargs

        self.load(model_state_dict_path)
        self._model.eval()

    def load(self, model_state_dict_path):
        """
        Load the processor from a model checkpoint file.

        Parameters
        ----------
        path : str
            Path to dump file.
        """
        raise NotImplementedError("Subclasses must implement this method.")
        
    def process(self, midi_file, **kwargs):
        """
        Process the input MIDI file.

        Parameters
        ----------
        midi_file : str
        
        Returns
        -------
        Depends on the processor.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, midi_file, **kwargs):
        """
        Process the input MIDI file.

        Parameters
        ----------
        midi_file : str

        Returns
        -------
        Depends on the processor.
        """
        return self.process(midi_file, **kwargs)
    
    
class RNNJointBeatModel(nn.Module):

    def __init__(self, hidden_size=512):
        super().__init__()

        in_features = get_in_features()

        self.convs = ConvBlock(in_features=in_features)

        self.gru_beat = GRUBlock(in_features=hidden_size)
        self.gru_downbeat = GRUBlock(in_features=hidden_size)
        self.gru_tempo = GRUBlock(in_features=hidden_size)

        self.out_beat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')
        self.out_downbeat = LinearOutput(in_features=hidden_size, out_features=1, activation_type='sigmoid')
        self.out_tempo = LinearOutput(in_features=hidden_size, out_features=ibiVocab, activation_type='softmax')

    def forward(self, x):
        # x: (batch_size, seq_len, len(features)==4)
        x = encode_note_sequence(x)

        x = self.convs(x)  # (batch_size, seq_len, hidden_size)

        x_gru_beat = self.gru_beat(x)  # (batch_size, seq_len, hidden_size)
        x_gru_downbeat = self.gru_downbeat(x_gru_beat)  # (batch_size, seq_len, hidden_size)
        x_gru_tempo = self.gru_tempo(x_gru_downbeat)  # (batch_size, seq_len, hidden_size)

        y_beat = self.out_beat(x_gru_beat)  # (batch_size, seq_len, 1)
        y_downbeat = self.out_downbeat(x_gru_downbeat)  # (batch_size, seq_len, 1)
        y_tempo = self.out_tempo(x_gru_tempo)  # (batch_size, seq_len, ibiVocab)

        # squeeze and transpose
        y_beat = y_beat.squeeze(2)  # (batch_size, seq_len)
        y_downbeat = y_downbeat.squeeze(2)  # (batch_size, seq_len)
        y_tempo = y_tempo.transpose(1, 2)  # (batch_size, ibiVocab, seq_len)

        return y_beat, y_downbeat, y_tempo
    
    
class RNNJointBeatProcessor(MIDIProcessor):

    def __init__(self, model_state_dict_path='_model_state_dicts/beat/RNNJointBeatModel.pth', **kwargs):
        super().__init__(model_state_dict_path, **kwargs)

    def load(self, state_dict_path):
        if state_dict_path:
            self._model = RNNJointBeatModel()
            self._model.load_state_dict(torch.load(state_dict_path))
        else:
            self._model = RNNJointBeatModel()

    def process(self, midi_file, **kwargs):
        # Read MIDI file into note sequence
        note_seq = read_note_sequence(midi_file)
        x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        beat_probs, downbeat_probs, _ = self._model(x)

        # Post-processing
        beat_probs = beat_probs.squeeze(0).detach().numpy()
        downbeat_probs = downbeat_probs.squeeze(0).detach().numpy()
        onsets = note_seq[:, 1]
        beats = self.pps_dp(beat_probs, downbeat_probs, onsets)
        
        return beats

    @staticmethod
    def pps_dp(beat_probs, downbeat_probs, onsets, penalty=1.0):
        """
        Post-processing with dynamic programming.

        Parameters
        ----------
        beat_probs: np.ndarray, shape=(n_notes,)
            beat probabilities at each note
        downbeat_probs: np.ndarray, shape=(n_notes,)
            downbeat probabilities at each note
        onsets: np.ndarray, shape=(n_notes,)
            onset times of each note

        Returns
        -------
        beats: np.ndarray, shape=(n_beats,)
            beat times
        """
        N_notes = len(beat_probs)

        # ========== Dynamic thrsholding ==========
        # window length in seconds
        wlen_beats = (60. / min_bpm) * 4
        wlen_downbeats = (60. / min_bpm) * 8

        # initialize beat and downbeat thresholds
        thresh_beats = np.ones(N_notes) * 0.5
        thresh_downbeats = np.ones(N_notes) * 0.5
        
        l_b, r_b, l_db, r_db = 0, 0, 0, 0  # sliding window indices
        
        for i, onset in enumerate(onsets):
            # udpate pointers
            while onsets[l_b] < onset - wlen_beats / 2:
                l_b += 1
            while r_b < N_notes and onsets[r_b] < onset + wlen_beats / 2:
                r_b += 1
            while onsets[l_db] < onset - wlen_downbeats / 2:
                l_db += 1
            while r_db < N_notes and onsets[r_db] < onset + wlen_downbeats / 2:
                r_db += 1
            # update beat and downbeat thresholds
            thresh_beats[i] = np.max(beat_probs[l_b:r_b]) * 0.5
            thresh_downbeats[i] = np.max(downbeat_probs[l_db:r_db]) * 0.5

        # threshold beat and downbeat probabilities
        beats = onsets[beat_probs > thresh_beats]
        downbeats = onsets[downbeat_probs > thresh_downbeats]

        # ========= Remove in-note-beats that are too close to each other =========
        beats_min = beats[np.concatenate([[True], np.abs(np.diff(beats)) > tolerance * 2])]
        beats_max = beats[::-1][np.concatenate([[True], np.abs(np.diff(beats[::-1])) > tolerance * 2])][::-1]
        beats = np.mean([beats_min, beats_max], axis=0)
        downbeats_min = downbeats[np.concatenate([[True], np.abs(np.diff(downbeats)) > tolerance * 2])]
        downbeats_max = downbeats[::-1][np.concatenate([[True], np.abs(np.diff(downbeats[::-1])) > tolerance * 2])][::-1]
        downbeats = np.mean([downbeats_min, downbeats_max], axis=0)

        # ========= Merge downbeats to beats if they are not in beat prediction =========
        beats_to_merge = []
        for downbeat in downbeats:
            if np.min(np.abs(beats - downbeat)) > tolerance * 2:
                beats_to_merge.append(downbeat)
        beats = np.concatenate([beats, beats_to_merge])
        beats = np.sort(beats)

        # ========= dynamic programming for beat prediction ====================
        # minimize objective function:
        #   O = sum(abs(log((t[k] - t[k-1]) / (t[k-1] - t[k-2]))))      (O1)
        #       + lam1 * insertions                                     (O2)
        #       + lam2 * deletions                                      (O3)
        #   t[k] is the kth beat after dynamic programming.
        # ======================================================================

        # ========= fill up out-of-note beats by inter-beat intervals =========
        # fill up by neighboring beats
        wlen = 5  # window length for getting neighboring inter-beat intervals (+- wlen)
        IBIs = np.diff(beats)
        beats_filled = []

        for i in range(len(beats) - 1):
            beats_filled.append(beats[i])

            # current and neighboring inter-beat intervals
            ibi = IBIs[i]
            ibis_near = IBIs[max(0, i-wlen):min(len(IBIs), i+wlen+1)]
            ibis_near_median = np.median(ibis_near)

            for ratio in [2, 3, 4]:
                if abs(ibi / ibis_near_median - ratio) / ratio < 0.15:
                    for x in range(1, ratio):
                        beats_filled.append(beats[i] + x * ibi / ratio)
        beats = np.sort(np.array(beats_filled))

        # ============= insertions and deletions ======================
        beats_dp = [
            [beats[0], beats[1]],     # no insertion
            [beats[0], beats[1]],     # insert one beat
            [beats[0], beats[1]],     # insert two beats
            [beats[0], beats[1]],     # insert three beats
        ]
        obj_dp = [0, 0, 0, 0]

        for i in range(2, len(beats)):
            beats_dp_new = [0] * len(beats_dp)
            obj_dp_new = [0, 0, 0, 0]

            # insert x beats
            for x in range(4):
                ibi = (beats[i] - beats[i-1]) / (x + 1)
                objs = []
                for x_prev in range(4):
                    o1 = np.abs(np.log(ibi / (beats_dp[x_prev][-1] - beats_dp[x_prev][-2])))
                    o = obj_dp[x_prev] + o1 + penalty * x
                    objs.append(o)

                x_prev_best = np.argmin(objs)
                beats_dp_new[x] = beats_dp[x_prev_best] + [beats[i-1] + ibi * k for k in range(1, x+1)] + [beats[i]]
                obj_dp_new[x] = objs[x_prev_best]

            beats_dp = beats_dp_new
            obj_dp = obj_dp_new

        x_best = np.argmin(obj_dp)
        beats = beats_dp[x_best]
        return np.array(beats)