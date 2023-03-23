import os
from beat_tracking.data_loading import get_midi_filelist
import partitura as pt
import numpy as np
import pretty_midi

# Import all_datasets here:
all_datasets = get_midi_filelist(["ASAP","CPM","AMAPS"])

for i, entry in enumerate(all_datasets[0]):
    pianoroll_path = entry[:-4]+"_pianoroll.npy"
    if os.path.exists(pianoroll_path) == False or (os.path.exists(pianoroll_path) == True and np.load(pianoroll_path).dtype == "int64"):
        performedpart = pt.load_performance_midi(entry)
        pr = pt.utils.compute_pianoroll(performedpart,remove_silence=False,piano_range=True,time_div=100,time_unit="second")
        pr = pr.toarray().T
        pr = pr.astype("int32")
        np.save(pianoroll_path,pr)
        if i%10 == 0:
            print(i, entry)
print("Partitura pianorolls - done.")


for i, entry in enumerate(all_datasets[0]):
    pianoroll_path = entry[:-4]+"_pianoroll_pm.npy"
    if os.path.exists(pianoroll_path) == False or (os.path.exists(pianoroll_path) == True and np.load(pianoroll_path).dtype == "int64"):
        file = pretty_midi.PrettyMIDI(entry)
        pr = file.get_piano_roll(fs=100)
        pr = pr.T
        pr = pr.astype("int32")
        np.save(pianoroll_path,pr)
        if i%10 == 0:
            print(i, entry)
print("PrettyMIDI pianorolls - done.")