#!/usr/bin/env python3
"""Inference script for finetuned PaddleOCR-VL models.

Usage:
    python training/inference.py --model checkpoints/paddleocr-qlora-... --image path/to/image.png
    python training/inference.py --model checkpoints/paddleocr-qlora-... --eval  # Run on eval set
    python training/inference.py --model checkpoints/paddleocr-qlora-... --eval --wandb --wandb-project my-project
"""

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402
import wandb  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from unsloth import FastVisionModel  # noqa: E402

from core.config import OCR_INSTRUCTION  # noqa: E402

# Alias for backward compatibility
DEFAULT_INSTRUCTION = OCR_INSTRUCTION


def load_model(model_path: str) -> tuple:
    """Load finetuned model and processor.

    Args:
        model_path: Path to finetuned model checkpoint.

    Returns:
        Tuple of (model, processor).
    """
    base_model = "unsloth/PaddleOCR-VL"

    print(f"Loading model from {model_path}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,
    )
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    FastVisionModel.for_inference(model)
    print("Model loaded successfully!")

    return model, processor


def run_inference(
    model,
    processor,
    image: Image.Image,
    instruction: str = DEFAULT_INSTRUCTION,
    max_new_tokens: int = 4096,
) -> str:
    """Run inference on a single image.

    Args:
        model: The finetuned model.
        processor: The model processor.
        image: PIL Image to process.
        instruction: OCR instruction prompt.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text output.
    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    text_prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(image, text_prompt, add_special_tokens=False, return_tensors="pt").to(
        model.device
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )
    return processor.tokenizer.decode(output[0], skip_special_tokens=True)


def log_to_wandb(
    images: list,
    predictions: list[str],
    references: list[str],
    table_name: str = "eval_predictions",
) -> None:
    """Log predictions vs references to W&B as a table.

    Args:
        images: List of PIL Images.
        predictions: List of model predictions.
        references: List of reference answers.
        table_name: Name for the W&B table.
    """
    columns = ["image", "prediction", "reference", "pred_length", "ref_length"]
    table = wandb.Table(columns=columns)

    for img, pred, ref in zip(images, predictions, references, strict=True):
        # Convert PIL image to W&B Image
        wandb_img = wandb.Image(img)
        table.add_data(wandb_img, pred, ref, len(pred), len(ref))

    wandb.log({table_name: table})
    print(f"Logged {len(predictions)} samples to W&B table: {table_name}")


def run_on_eval_set(
    model,
    processor,
    num_samples: int = 5,
    log_wandb: bool = False,
    wandb_project: str = None,
    wandb_run_name: str = None,
) -> None:
    """Run inference on samples from the eval dataset.

    Args:
        model: The finetuned model.
        processor: The model processor.
        num_samples: Number of eval samples to process.
        log_wandb: Whether to log results to W&B.
        wandb_project: W&B project name (required if log_wandb=True).
        wandb_run_name: Optional W&B run name.
    """
    from datasets import load_dataset

    print("\nLoading eval dataset...")
    dataset = load_dataset("wrath/well-log-headers-ocr", split="eval")
    print(f"Eval set: {len(dataset)} samples")

    num_samples = min(num_samples, len(dataset))
    print(f"Running inference on {num_samples} samples...\n")

    # Initialize W&B if requested
    if log_wandb:
        if not wandb_project:
            print("Error: --wandb-project required when using --wandb")
            sys.exit(1)
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or "post-training-eval",
            job_type="evaluation",
        )
        print(f"W&B initialized: {wandb_project}")

    images = []
    predictions = []
    references = []

    for i in range(num_samples):
        sample = dataset[i]
        # Decode base64 image
        img_data = base64.b64decode(sample["image_base64"])
        image = Image.open(BytesIO(img_data)).convert("RGB")

        print(f"{'='*60}")
        print(f"Sample {i + 1}/{num_samples}")
        print(f"{'='*60}")

        result = run_inference(model, processor, image)
        ref = sample["answer"]

        # Store for W&B logging
        images.append(image)
        predictions.append(result)
        references.append(ref)

        print("\n📄 PREDICTION:")
        print(result[:1000] + "..." if len(result) > 1000 else result)

        print("\n✅ REFERENCE:")
        print(ref[:1000] + "..." if len(ref) > 1000 else ref)

        print("\n")

    # Log to W&B if enabled
    if log_wandb:
        log_to_wandb(images, predictions, references)
        wandb.finish()
        print("W&B logging complete!")


def main():
    parser = argparse.ArgumentParser(description="Run inference with finetuned model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to finetuned model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file for inference",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run inference on eval dataset samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of eval samples to process (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save result (optional)",
    )

    args = parser.parse_args()

    if not args.image and not args.eval:
        print("Error: Must specify either --image or --eval")
        sys.exit(1)

    model, processor = load_model(args.model)

    if args.eval:
        run_on_eval_set(model, processor, args.num_samples)
    else:
        # Single image inference
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        image = Image.open(image_path).convert("RGB")
        print(f"\nProcessing: {image_path}")

        result = run_inference(model, processor, image)

        print("\n📄 OUTPUT:")
        print(result)

        if args.output:
            Path(args.output).write_text(result)
            print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
