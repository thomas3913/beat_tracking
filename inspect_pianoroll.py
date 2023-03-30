import os
import partitura as pt
import pretty_midi
import warnings
warnings.filterwarnings("ignore")

# Path to MIDI files:
midi_file_1 = "SOLOM02.mid" #Path in the dataset: "data/asap-dataset/Bach/Prelude/bwv_866/SOLOM02.mid"
midi_file_2 = "MAPS_MUS-chpn_op25_e4_ENSTDkAm.mid" #Path in the dataset: "data/A-MAPS_1.2/MAPS_MUS-chpn_op25_e4_ENSTDkAm.mid"
midi_file_3 = "chpn_op66.mid" #Path in the dataset: "data/Piano-MIDI/midis/Chopin/chpn_op66.mid"


for i, entry in enumerate([midi_file_1,midi_file_2,midi_file_3]):

    # Partitura:
    performedpart = pt.load_performance_midi(entry)
    pr = pt.utils.compute_pianoroll(performedpart,remove_silence=False,piano_range=True,time_div=100,time_unit="sec")

    # Pretty_midi:
    file = pretty_midi.PrettyMIDI(entry)
    pr_pm = file.get_piano_roll(fs=100)

    print("File",entry,"\nPartitura pianoroll shape:",pr.shape,"\nPretty_midi pianoroll shape:",pr_pm.shape,"\n")



