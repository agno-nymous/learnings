from configs.base import TrainingConfig

config = TrainingConfig(
    max_steps=10,
    train_subset=100,
    eval_subset=20,
    eval_steps=5,
    output_dir="/runpod_volume/checkpoints/quick-val",
)
