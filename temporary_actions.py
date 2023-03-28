import os

#piano_files = list()
#for (dirpath, dirnames, filenames) in os.walk("."):
#    piano_files += [os.path.join(dirpath, file) for file in filenames if file[-13:] == "pianoroll.npy"]
#piano_files = sorted(piano_files)
#print("Number of piano_files:",len(piano_files))

#for elem in piano_files:
#    os.remove(elem)

import pretty_midi as pm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from beat_tracking.data_loading import get_midi_filelist

cpm_list = get_midi_filelist(["cpm"])
amaps_list = get_midi_filelist(["amaps"])
asap_list = get_midi_filelist(["asap"])

for i in range(1):
    midi_file = cpm_list[0][i]
    print(midi_file)
    midi_data = pm.PrettyMIDI(str(Path(midi_file)))
    beats = midi_data.get_beats()
    estimate = midi_data.estimate_beat_start()

    pr = np.load(midi_file[:-4]+"_pianoroll.npy")
    pr_pm = np.load(midi_file[:-4]+"_pianoroll_pm.npy")

    print(beats[:10])
    print("Estimate",estimate)

    fig, ax = plt.subplots(1, figsize=(16, 4))
    ax.imshow(pr.T, origin="lower", cmap='gray', interpolation='nearest', aspect='auto')
    plt.vlines(beats*100,ymin=0,ymax=88)
    plt.title(midi_file)
    plt.savefig("pianorolls/test.png")

    fig, ax = plt.subplots(1, figsize=(16, 4))
    ax.imshow(pr_pm.T, origin="lower", cmap='gray', interpolation='nearest', aspect='auto')
    plt.vlines(beats*100,ymin=0,ymax=88)
    plt.title(midi_file)
    plt.savefig("pianorolls/test_pm.png")