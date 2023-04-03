import numpy as np
import csv
from scipy.ndimage import maximum_filter1d
import json
import matplotlib.pyplot as plt
import pretty_midi as pm
from pathlib import Path
from functools import reduce, cmp_to_key
from beat_tracking.pm2s_files.constants import tolerance
import torch
import pandas as pd

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

def beat_list_to_array(filename, data_type, beat_type):
    beat_list = []
    with open(filename, newline='\n') as f:
        lines = csv.reader(f, delimiter='\t')
        for line in lines:
            # include downbeats to beats!
            if data_type == "detections":
                beat_list.append(np.float64(line[0]))

            elif data_type == "annotations":

                if beat_type == "beats":
                    beat_list.append(np.float64(line[0]))

                elif beat_type == "downbeats":
                    if line[2] == 'db':
                        beat_list.append(np.float64(line[0]))
                else:
                    print("Beat type not specified.")
                    raise TypeError
            else:
                print("Data type not specified.")
                raise TypeError

    beat_array = np.array(beat_list)

    return beat_array


#def cnn_pad(data, pad_frames):
#    """Pad the data by repeating the first and last frame N times."""
#    pad_start = np.repeat(data[:, :1, :], pad_frames, axis=1)
#    pad_stop = np.repeat(data[:, -1:, :], pad_frames, axis=1)
#    return torch.concatenate((pad_start, data, pad_stop), axis=1)


def cnn_pad(data, pad_frames):
    """Pad the data by repeating the first and last frame N times."""
    pad_start = torch.cat([data[:, :1, :].repeat(1, pad_frames, 1)], dim=1)
    pad_stop = torch.cat([data[:, -1:, :].repeat(1, pad_frames, 1)], dim=1)
    return torch.cat([pad_start, data, pad_stop], dim=1)


def show_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return d, h, m, s


def widen_beat_targets(y, size=3, value=0.5):
    np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)


def print_model_statistics(model_checkpoint):
    with open(model_checkpoint, 'r') as f:
        load_dict = json.load(f)

    print("Checkpoint stats loaded:", model_checkpoint)
    d, h, m, s = show_time(load_dict["total_time"])
    print("Epochs trained:", load_dict["epoch"], "--- Iterations:", load_dict["iter"], "--- Total training time:", d,
          "days,", f"{h:02d}:{m:02d}:{s:02d}", "--- Last learning rate:", load_dict["learning_rate"])

    loss_list = load_dict["loss_list"]
    val_loss_list = load_dict["val_loss_list"]
    fscore_list_val_average = load_dict["fscore_list_val_average"]
    fscore_list_val_average_db = load_dict["fscore_list_val_average_db"]
    learning_rates_list = load_dict["learning_rates_list"]

    if len(loss_list) > 1:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 5))
        ax1.plot(np.arange(len(loss_list)), loss_list)
        ax1.set_title("Training loss")
        ax2.plot(np.arange(len(val_loss_list)), val_loss_list, color="green")
        ax2.set_title("Validation summary loss after " + str(len(val_loss_list)) + " validation runs")
        ax3.plot(np.arange(len(fscore_list_val_average)), fscore_list_val_average, color="red")
        ax3.set_title("Validation F-Score (beats) after " + str(len(fscore_list_val_average)) + " validation runs")
        ax3.set_xlabel("Last value: " + str("%.4f" % fscore_list_val_average[-1]) + " --- Max value: " + str(
            "%.4f" % np.max(fscore_list_val_average)) + " (Index " + str(np.argmax(fscore_list_val_average)) + ")")
        ax4.plot(np.arange(len(fscore_list_val_average_db)), fscore_list_val_average_db, color="orange")
        ax4.set_title(
            "Validation F-Score (downbeats) after " + str(len(fscore_list_val_average_db)) + " validation runs")
        ax4.set_xlabel("Last value: " + str("%.4f" % fscore_list_val_average_db[-1]) + " --- Max value: " + str(
            "%.4f" % np.max(fscore_list_val_average_db)) + " (Index " + str(
            np.argmax(fscore_list_val_average_db)) + ")")
        plt.show()

        plt.plot(np.arange(len(learning_rates_list)), learning_rates_list, color="purple")
        plt.title("Learning rate history")
        plt.show()

def plot_value_list(list,colour,title,save,save_name):
    if len(list) > 1:
        plt.clf()
        plt.plot(np.arange(len(list)), list, color=colour)
        plt.title(title)
        if save == False:
            plt.show()
        elif save == True:
            plt.savefig(save_name+".pdf")
            
def compare_note_order(note1, note2):
    """
    Compare two notes by firstly onset and then pitch.
    """
    if note1.start < note2.start:
        return -1
    elif note1.start == note2.start:
        if note1.pitch < note2.pitch:
            return -1
        elif note1.pitch == note2.pitch:
            return 0
        else:
            return 1
    else:
        return 1
    

def get_note_sequence_from_midi(midi_file):
    """
    Get note sequence from midi file.
    Note sequence is in a list of (pitch, onset, duration, velocity) tuples, in np.array.
    """
    midi_data = pm.PrettyMIDI(str(Path(midi_file)))
    note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
    note_sequence = sorted(note_sequence, key=cmp_to_key(compare_note_order))
    # conver to numpy array
    note_sequence = np.array([(note.pitch, note.start, note.end-note.start, note.velocity) \
                                    for note in note_sequence])
    return note_sequence

def get_annotations_from_annot_file(annot_file):
    """
    Get annotations from annotation file in ASAP dataset.
    annotatioins in a dict of {
        beats: list of beat times,
        downbeats: list of downbeat times,
        time_signatures: list of (time, numerator, denominator) tuples,
        key_signatures: list of (time, key_number) tuples
    }, all in np.array.
    """
    annot_data = pd.read_csv(str(Path(annot_file)), header=None, sep='\t')

    beats, downbeats, key_signatures, time_signatures = [], [], [], []
    for i, row in annot_data.iterrows():
        a = row[2].split(',')
        # beats
        beats.append(row[0])
        # downbeats
        if a[0] == 'db':
            downbeats.append(row[0])
        # time_signatures
        if len(a) >= 2 and a[1] != '':
            numerator, denominator = a[1].split('/')
            time_signatures.append((row[0], int(numerator), int(denominator)))
        # key_signatures
        if len(a) == 3 and a[2] != '':
            key_signatures.append((row[0], keySharps2Number[int(a[2])]))

    # save as annotation dict
    annotations = {
        'beats': np.array(beats),
        'downbeats': np.array(downbeats),
        'time_signatures': np.array(time_signatures),
        'key_signatures': np.array(key_signatures),
        'onsets_musical': None,
        'note_value': None,
        'hands': None,
    }
    return annotations
            
            
def get_note_sequence_and_annotations_from_midi(midi_file):
    """
    Get beat sequence and annotations from midi file.
    Note sequence is in a list of (pitch, onset, duration, velocity) tuples, in np.array.
    annotations in a dict of {
        beats: list of beat times,
        downbeats: list of downbeat times,
        time_signatures: list of (time, numerator, denominator) tuples,
        key_signatures: list of (time, key_number) tuples,
        onsets_musical: list of onsets in musical time for each note (within a beat),
        note_value: list of note values (in beats),
        hands: list of hand part for each note (0: left, 1: right)
    """
    midi_data = pm.PrettyMIDI(str(Path(midi_file)))

    # note sequence and hands
    if len(midi_data.instruments) == 2:
        # two hand parts
        note_sequence_with_hand = []
        for hand, inst in enumerate(midi_data.instruments):
            for note in inst.notes:
                note_sequence_with_hand.append((note, hand))

        def compare_note_with_hand(x, y):
            return compare_note_order(x[0], y[0])
        note_sequence_with_hand = sorted(note_sequence_with_hand, key=cmp_to_key(compare_note_with_hand))

        note_sequence, hands = [], []
        for note, hand in note_sequence_with_hand:
            note_sequence.append(note)
            hands.append(hand)
    else:
        # ignore data with other numbers of hand parts
        note_sequence = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data.instruments])
        note_sequence = sorted(note_sequence, key=cmp_to_key(compare_note_order))
        hands = None

    # beats
    beats = midi_data.get_beats()
    # downbeats
    downbeats = midi_data.get_downbeats()
    # time_signatures
    time_signatures = [(t.time, t.numerator, t.denominator) for t in midi_data.time_signature_changes]
    # key_signatures
    key_signatures = [(k.time, k.key_number) for k in \
                        midi_data.key_signature_changes]
    # onsets_musical and note_values
    def time2pos(t):
        # convert time to position in musical time within a beat (unit: beat, range: 0-1)
        # after checking, we confirmed that beats[0] is always 0
        idx = np.where(beats - t <= tolerance)[0][-1]
        if idx+1 < len(beats):
            base = midi_data.time_to_tick(beats[idx+1]) - midi_data.time_to_tick(beats[idx])
        else:
            base = midi_data.time_to_tick(beats[-1]) - midi_data.time_to_tick(beats[-2])
        return (midi_data.time_to_tick(t) - midi_data.time_to_tick(beats[idx])) / base

    def times2note_value(start, end):
        # convert start and end times to note value (unit: beat, range: 0-4)
        idx = np.where(beats - start <= tolerance)[0][-1]
        if idx+1 < len(beats):
            base = midi_data.time_to_tick(beats[idx+1]) - midi_data.time_to_tick(beats[idx])
        else:
            base = midi_data.time_to_tick(beats[-1]) - midi_data.time_to_tick(beats[-2])
        return (midi_data.time_to_tick(end) - midi_data.time_to_tick(start)) / base

    # get onsets_musical and note_values
    # filter out small negative values (they are usually caused by errors in time_to_tick convertion)
    onsets_musical = [min(1, max(0, time2pos(note.start))) for note in note_sequence]  # in range 0-1
    note_values = [max(0, times2note_value(note.start, note.end)) for note in note_sequence]

    # conver to Tensor
    note_sequence = np.array([[note.pitch, note.start, note.end-note.start, note.velocity] \
                                    for note in note_sequence])
    # save as annotation dict
    annotations = {
        'beats': np.array(beats),
        'downbeats': np.array(downbeats),
        'time_signatures': np.array(time_signatures),
        'key_signatures': np.array(key_signatures),
        'onsets_musical': np.array(onsets_musical),
        'note_value': np.array(note_values),
        'hands': np.array(hands) if hands is not None else None,
    }
    return note_sequence, annotations