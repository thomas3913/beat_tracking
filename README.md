# beat_tracking
beat_tracking repository

Example run commands:

python evaluate_pm2s.py --results_dir "results" --dataset "asap"

python evaluate_ISMIR.py --results_dir "results" --dataset "all" --pianorolls "pretty_midi" --only_beats True

python train_pm2s.py --dataset "all" --epochs 10 --only_beats True

python train_ISMIR.py --dataset "asap" --epochs 20 --pianorolls "pretty_midi" --only_beats True --stepsize 20

Values for dataset:
"asap" "amaps" "cpm" "all"

Values for mode:
"ismir" "pm2s"

Values for pianorolls: "partitura" "pretty_midi"

Values for only_beats: "False" "True"

Stepsize datermines the number of epochs before the learning rate is adjusted by a factor of 0.1

Remarks:
"evaluate_ISMIR.py" not working yet