from configs.base import TrainingConfig

config = TrainingConfig(
    # Explicit overrides (rest inherited from base)
    num_train_epochs=3,
    eval_steps=50,
)
