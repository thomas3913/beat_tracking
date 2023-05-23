import numpy as np
import torch
from beat_tracking.pm2s_files.constants import min_bpm, tolerance
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

class RNNJointBeatProcessor():

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    #def load(self, model):
    #    self._model = model

    def process(self,note_seq, beat_probs, downbeat_probs, **kwargs):
        # Read MIDI file into note sequence

        #note_seq = read_note_sequence(midi_file)
        #x = torch.tensor(note_seq).unsqueeze(0)

        # Forward pass
        #beat_probs, downbeat_probs, _ = self._model(x)

        note_seq = np.array(note_seq[0].detach().cpu())

        # Post-processing
        beat_probs = beat_probs.squeeze(0).detach().cpu().numpy()
        downbeat_probs = downbeat_probs.squeeze(0).detach().cpu().numpy()

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

        #check following again:

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

        # Ask on github about the 2 functions:

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
    
    
# Additional function to get the beat activation function for DBN:
def get_beat_activation_function_from_probs(note_seq,beat_probs,downbeat_probs):
    
    note_seq = np.array(note_seq[0].detach().cpu())
    
    beat_probs = beat_probs.squeeze(0).detach().cpu().numpy()
    downbeat_probs = downbeat_probs.squeeze(0).detach().cpu().numpy()

    onsets = note_seq[:, 1]
        
    assert np.array_equal(onsets,note_seq[:, 1])
    assert len(onsets) > 0

    #determine length of piece by searching max value of (onset+duration):
    length_of_piece = np.max(note_seq[:,1]+note_seq[:,2])

    beat_activation_function = np.zeros(shape=(int(length_of_piece*100),))
    beat_activation_function += 0.0001
    
    for o, p in zip(onsets,beat_probs):
        idx = int(o*100)
        beat_activation_function[idx] = np.maximum(beat_activation_function[idx],p)
        
    beat_activation_function = beat_activation_function / np.max(beat_activation_function)
    
    beat_activation_function_modified = beat_activation_function.copy()
    
    #Modify by adding half of the values left and right of a peak:
    #for i,elem in enumerate(beat_activation_function_modified[2:-2]):
    #    if elem > 0.5:
    #        if beat_activation_function_modified[i-1] < 0.5:
    #            beat_activation_function_modified[i-1] = elem / 2
    #        if beat_activation_function_modified[i+1] < 0.5:
    #            beat_activation_function_modified[i+1] = elem / 2
                
    #gaussian filter_1d:
    #scipy.ndimage.gaussian_filter1d
    #test_array = np.zeros(20)
    #test_array[8] = 0.8
    #print(test_array)
    
    #plt.clf()
    #plt.plot(np.arange(20),test_array)
    #plt.savefig("fig1.png")
    
    #test_array = gaussian_filter1d(test_array,sigma=1)
    #print(test_array)
    
    #plt.clf()
    #plt.plot(np.arange(20),test_array)
    #plt.savefig("fig2.png")
    
    beat_activation_function_modified = gaussian_filter1d(beat_activation_function,sigma=1)
    
        
    #old version:
    #bin_array = np.zeros(shape=onsets.shape)
    #for i in range(len(bin_array)):
    #    bin_array[i] = int(onsets[i]*100)

    #for j,bin in enumerate(bin_array):
    #    indcs = np.where(bin_array==bin)
    #    prob = np.sum(beat_probs[indcs])/len(indcs[0])        
    #    beat_activation_function[int(bin)] = prob

    return beat_activation_function, beat_activation_function_modified