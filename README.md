# beat_tracking
beat_tracking

Example run command:

python train_ISMIR.py --checkpoints_dir "models_ISMIR/checkpoints/" --figures_dir "models_ISMIR/figures/" --dataset "asap" --mode "ismir" --validate_every 2000

Values for dataset:
"asap" "amaps" "cpm" "all"

Values for mode:
"ismir" "pm2s"

The parameter "validate_every" also defines how often checkpoints will be created.