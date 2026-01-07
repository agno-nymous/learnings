from configs.base import TrainingConfig

config = TrainingConfig(
    num_train_epochs=1,
    train_subset=100,
    eval_subset=20,
    output_dir="/runpod_volume/checkpoints/quick-val",
)
