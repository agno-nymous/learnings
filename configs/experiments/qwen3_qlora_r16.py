from configs.base import TrainingConfig

config = TrainingConfig(
    # Explicit overrides (rest inherited from base)
    max_steps=500,
    eval_steps=50,
)
