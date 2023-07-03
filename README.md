# beat_tracking
beat_tracking repository

Example run commands:

EVALUATION

python evaluate_pm2s.py --dataset "all"

python evaluate_ISMIR.py --dataset "all"

python evaluate_audio_pr.py --mode "audio" --dataset "all"

python evaluate_audio_pr.py --mode "pianorolls" --dataset "all"

TRAINING

python train_pm2s.py --dataset "all" --epochs 50

python train_pm2s.py --dataset "all" --epochs 50 --full_train

python train_ISMIR.py --dataset "all" --epochs 50 --stepsize 15

python train_ISMIR.py --dataset "all" --epochs 50 --stepsize 15 --full_train

python train_audio_pr.py --mode "audio" --dataset "all" --epochs 100 --stepsize 50

python train_audio_pr.py --mode "pianorolls" --dataset "all" --epochs 100 --stepsize 50

Values for dataset:
"asap" "amaps" "cpm" "all"

Stepsize datermines the number of epochs before the learning rate is adjusted by a factor of 0.1

full_train uses the whole specified dataset as the training set