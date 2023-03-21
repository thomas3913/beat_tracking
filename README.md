# beat_tracking
beat_tracking repository

Example run commands:

python evaluate_pm2s.py --results_dir "results" --dataset "asap" --mode "pm2s"

python train_ISMIR.py --checkpoints_dir "models_ISMIR/checkpoints/" --figures_dir "models_ISMIR/figures/" --dataset "asap" --mode "ismir"

Values for dataset:
"asap" "amaps" "cpm" "all"

Values for mode:
"ismir" "pm2s"

Remarks:
"evaluate_ISMIR.py" not working yet