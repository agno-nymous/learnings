#!/usr/bin/env python3
"""Inference script for finetuned PaddleOCR-VL models.

Usage:
    python training/inference.py --model checkpoints/paddleocr-qlora-... --image path/to/image.png
    python training/inference.py --model checkpoints/paddleocr-qlora-... --eval  # Run on eval set
"""

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from unsloth import FastVisionModel

# Default OCR instruction
DEFAULT_INSTRUCTION = """Convert the following document to markdown.
Return only the markdown with no explanation text. Do not include delimiters like ```markdown or ```html.

RULES:
- You must include all information on the page. Do not exclude headers, footers, or subtext.
- Return tables in an HTML format.
- Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable.
- Prefer using ☐ and ☑ for check boxes."""


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


def run_on_eval_set(model, processor, num_samples: int = 5) -> None:
    """Run inference on samples from the eval dataset.

    Args:
        model: The finetuned model.
        processor: The model processor.
        num_samples: Number of eval samples to process.
    """
    from datasets import load_dataset

    print("\nLoading eval dataset...")
    dataset = load_dataset("wrath/well-log-headers-ocr", split="eval")
    print(f"Eval set: {len(dataset)} samples")

    num_samples = min(num_samples, len(dataset))
    print(f"Running inference on {num_samples} samples...\n")

    for i in range(num_samples):
        sample = dataset[i]
        # Decode base64 image
        img_data = base64.b64decode(sample["image_base64"])
        image = Image.open(BytesIO(img_data)).convert("RGB")

        print(f"{'='*60}")
        print(f"Sample {i + 1}/{num_samples}")
        print(f"{'='*60}")

        result = run_inference(model, processor, image)

        print("\n📄 PREDICTION:")
        print(result[:1000] + "..." if len(result) > 1000 else result)

        print("\n✅ REFERENCE:")
        ref = sample["answer"]
        print(ref[:1000] + "..." if len(ref) > 1000 else ref)

        print("\n")


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
