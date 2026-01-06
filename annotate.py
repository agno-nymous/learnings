#!/usr/bin/env python3
"""
Unified OCR annotation CLI.

Usage:
    # Use dataset registry
    python annotate.py --dataset welllog-train
    python annotate.py --dataset olmocr-train --mode batch

    # Or explicit paths (backwards compatible)
    python annotate.py -i ./cropped_headers -o ./dataset.jsonl
"""

import argparse
from pathlib import Path

from annotator.factory import AnnotatorFactory
from core.config import DATASETS, DEFAULT_MODEL, GEMINI_MODELS, get_dataset
from core.file_discovery import find_images
from core.jsonl_utils import create_clean_dataset, get_successful_filenames


def select_model_interactive() -> str:
    """Prompt user to select a model."""
    print("\n" + "=" * 60)
    print("SELECT A MODEL")
    print("=" * 60)
    for key, (model_name, description) in GEMINI_MODELS.items():
        print(f"  [{key}] {description}")
        print(f"      → {model_name}")
    print()

    while True:
        choice = input("Enter choice (1/2/3) or press Enter for default [2]: ").strip()
        if choice == "":
            choice = "2"
        if choice in GEMINI_MODELS:
            model_name, _ = GEMINI_MODELS[choice]
            print(f"\n✓ Selected: {model_name}\n")
            return model_name
        print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Run the OCR annotation pipeline."""
    parser = argparse.ArgumentParser(
        description="OCR annotation using Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Datasets: {', '.join(DATASETS.keys())}

Examples:
  python annotate.py --dataset welllog-train
  python annotate.py --dataset olmocr-train --mode batch
  python annotate.py -i ./images -o ./output.jsonl
        """,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=list(DATASETS.keys()),
        help="Dataset to process (uses registry paths)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["realtime", "batch"],
        default="realtime",
        help="OCR mode: 'realtime' (default) or 'batch' (50%% cheaper)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--select-model", action="store_true", help="Interactively select model")
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=None,
        help="Directory containing images (overrides --dataset)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output JSONL file (overrides --dataset)"
    )
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers")
    parser.add_argument(
        "--limit", "-n", type=int, default=None, help="Limit number of images to process"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between batch job status checks (batch mode only)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Process all files (ignore already processed)"
    )
    args = parser.parse_args()

    # Resolve input_dir and output from dataset or explicit args
    if args.dataset:
        ds = get_dataset(args.dataset)
        input_dir = args.input_dir or ds["images_dir"]
        output = args.output or ds["output"]
    elif args.input_dir and args.output:
        input_dir = args.input_dir
        output = args.output
    else:
        parser.error("Either --dataset or both --input-dir and --output required")

    # Model selection
    model = select_model_interactive() if args.select_model else args.model

    # Create annotator via factory
    annotator = AnnotatorFactory.create(
        mode=args.mode,
        model=model,
        workers=args.workers,
        poll_interval=args.poll_interval,
    )

    # Find images to process
    images = find_images(input_dir)

    if not images:
        print(f"No images found in {input_dir}")
        return

    # Filter already processed (resume by default)
    if not args.no_resume and output.exists():
        processed = get_successful_filenames(output)
        print(f"Resuming: {len(processed)} files already processed")
        images = [f for f in images if f.name not in processed]

    # Apply limit
    if args.limit:
        images = images[: args.limit]

    if not images:
        print("All files already processed!")
        return

    # Run annotation
    print(f"\nDataset: {args.dataset or 'custom'}")
    print(f"Using: {annotator.name}")
    print(f"Processing: {len(images)} images")
    print(f"Output: {output}")
    print(f"Workers: {args.workers}")
    print()

    result = annotator.annotate(images, output)

    print(f"\nDone! Success: {result['success']}, Errors: {result['errors']}")

    # Create clean dataset for Unsloth
    clean_output = output.with_suffix(".clean.jsonl")
    count = create_clean_dataset(output, clean_output)
    print(f"Clean training dataset: {clean_output} ({count} entries)")


if __name__ == "__main__":
    main()
