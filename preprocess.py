#!/usr/bin/env python3
"""
Preprocessing CLI for image preparation pipeline.

Usage:
    python preprocess.py download --limit 500
    python preprocess.py rotate
    python preprocess.py crop
    python preprocess.py all  # Run all steps
    python preprocess.py olmocr --limit 50  # Download olmOCR dataset
    python preprocess.py split-welllog  # Split welllog into train/eval
"""

import argparse
from pathlib import Path

from preprocessing.downloader import download_scans
from preprocessing.rotator import rotate_tiffs
from preprocessing.cropper import crop_headers
from core.config import DEFAULT_DOWNLOADS_DIR, DEFAULT_INPUT_DIR


def cmd_download(args):
    """Download elog scan files."""
    download_scans(
        input_file=args.input_file,
        output_dir=args.output_dir,
        limit=args.limit,
        workers=args.workers,
    )


def cmd_rotate(args):
    """Rotate TIFF files."""
    rotate_tiffs(
        input_dir=args.input_dir,
        workers=args.workers,
    )


def cmd_crop(args):
    """Crop well log headers."""
    crop_headers(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=args.workers,
    )


def cmd_all(args):
    """Run all preprocessing steps."""
    print("=" * 60)
    print("STEP 1: Download")
    print("=" * 60)
    download_scans(
        input_file=args.input_file,
        output_dir=args.downloads_dir,
        limit=args.limit,
        workers=args.workers,
    )
    
    print("\n" + "=" * 60)
    print("STEP 2: Rotate")
    print("=" * 60)
    rotate_tiffs(
        input_dir=args.downloads_dir,
        workers=args.workers,
    )
    
    print("\n" + "=" * 60)
    print("STEP 3: Crop")
    print("=" * 60)
    crop_headers(
        input_dir=args.downloads_dir,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)


def cmd_olmocr(args):
    """Download olmOCR dataset from HuggingFace."""
    from preprocessing.olmocr_downloader import download_olmocr
    
    for split in ["train", "eval"]:
        print(f"\n{'='*60}")
        print(f"Downloading {split} split")
        print(f"{'='*60}")
        download_olmocr(
            split=split,
            limit_per_subset=args.limit,
            min_words=args.min_words,
        )


def cmd_split_welllog(args):
    """Split welllog headers into train/eval."""
    from preprocessing.splitter import split_welllog
    split_welllog(source_dir=args.source_dir, train_ratio=args.train_ratio)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline for OCR datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download command
    p_download = subparsers.add_parser("download", help="Download elog scan files")
    p_download.add_argument("--input-file", type=Path, default=Path("./ks_elog_scans.txt"))
    p_download.add_argument("--output-dir", type=Path, default=DEFAULT_DOWNLOADS_DIR)
    p_download.add_argument("--limit", "-n", type=int, default=500)
    p_download.add_argument("--workers", "-w", type=int, default=4)
    p_download.set_defaults(func=cmd_download)
    
    # Rotate command
    p_rotate = subparsers.add_parser("rotate", help="Rotate TIFF files 90° CCW")
    p_rotate.add_argument("--input-dir", type=Path, default=DEFAULT_DOWNLOADS_DIR)
    p_rotate.add_argument("--workers", "-w", type=int, default=4)
    p_rotate.set_defaults(func=cmd_rotate)
    
    # Crop command
    p_crop = subparsers.add_parser("crop", help="Crop well log headers")
    p_crop.add_argument("--input-dir", type=Path, default=DEFAULT_DOWNLOADS_DIR)
    p_crop.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p_crop.add_argument("--workers", "-w", type=int, default=4)
    p_crop.set_defaults(func=cmd_crop)
    
    # All command (welllog pipeline)
    p_all = subparsers.add_parser("all", help="Run all welllog preprocessing steps")
    p_all.add_argument("--input-file", type=Path, default=Path("./ks_elog_scans.txt"))
    p_all.add_argument("--downloads-dir", type=Path, default=DEFAULT_DOWNLOADS_DIR)
    p_all.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p_all.add_argument("--limit", "-n", type=int, default=500)
    p_all.add_argument("--workers", "-w", type=int, default=4)
    p_all.set_defaults(func=cmd_all)
    
    # olmOCR download command
    p_olmocr = subparsers.add_parser("olmocr", help="Download olmOCR dataset from HuggingFace")
    p_olmocr.add_argument("--limit", "-n", type=int, default=50, help="Images per subset per split")
    p_olmocr.add_argument("--min-words", type=int, default=100, help="Min words to include page")
    p_olmocr.set_defaults(func=cmd_olmocr)
    
    # Split welllog command
    p_split = subparsers.add_parser("split-welllog", help="Split welllog into train/eval")
    p_split.add_argument("--source-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p_split.add_argument("--train-ratio", type=float, default=0.8)
    p_split.set_defaults(func=cmd_split_welllog)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()

