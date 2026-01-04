from configs.base import TrainingConfig

config = TrainingConfig(
    # PaddleOCR-VL specific config
    model_name="unsloth/PaddleOCR-VL-bnb-4bit",
    
    # Training params (same as before)
    max_steps=500,
    eval_steps=50,
)
