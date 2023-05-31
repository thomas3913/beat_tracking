# beat_tracking
beat_tracking repository

Example run commands:

python evaluate_pm2s.py --dataset "all"

python evaluate_ISMIR.py --dataset "all"

python train_pm2s.py --dataset "all" --epochs 50 --full_train False

python train_ISMIR.py --dataset "all" --epochs 50 --stepsize 15 --full_train False

Values for dataset:
"asap" "amaps" "cpm" "all"

Stepsize datermines the number of epochs before the learning rate is adjusted by a factor of 0.1