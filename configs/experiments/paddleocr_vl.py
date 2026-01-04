from configs.base import TrainingConfig

config = TrainingConfig(
    # PaddleOCR-VL specific config (from official notebook)
    model_name="unsloth/PaddleOCR-VL",
    r=64,  # PaddleOCR uses r=64
    lora_alpha=64,
)
