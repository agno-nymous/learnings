"""
olmOCR dataset downloader.

Downloads from HuggingFace, streams PDFs, renders specific pages to PNG.
Filters to pages with >100 words of text.
"""
from pathlib import Path
from io import BytesIO
import requests
import hashlib

from PIL import Image

SUBSETS = ["00_documents", "02_loc_transcripts", "03_national_archives"]
MIN_WORDS = 100


def render_pdf_page(pdf_bytes: bytes, page_number: int) -> Image.Image:
    """Render a specific page from PDF bytes to PIL Image."""
    import fitz  # PyMuPDF
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # page_number is 1-indexed in the dataset
    page_idx = page_number - 1
    if page_idx < 0 or page_idx >= len(doc):
        raise ValueError(f"Page {page_number} out of range (doc has {len(doc)} pages)")
    
    page = doc[page_idx]
    # Render at 2x for better quality
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def download_pdf(url: str) -> bytes:
    """Download PDF from URL."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def download_olmocr(
    split: str = "train",
    limit_per_subset: int = 50,
    output_dir: Path = None,
    min_words: int = MIN_WORDS,
):
    """
    Download olmOCR dataset, render PDF pages to PNG.
    
    Args:
        split: 'train' or 'eval'
        limit_per_subset: Number of images per subset
        output_dir: Where to save PNGs
        min_words: Minimum words in ground truth to include
    """
    from datasets import load_dataset
    
    if output_dir is None:
        output_dir = Path(f"./datasets/olmocr/{split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = 0
    for subset in SUBSETS:
        # Check existing files for this subset (resume logic)
        existing = list(output_dir.glob(f"{subset}_*.png"))
        existing_count = len(existing)
        
        if existing_count >= limit_per_subset:
            print(f"\n{subset}: Already have {existing_count}/{limit_per_subset} - skipping")
            total += existing_count
            continue
        
        need = limit_per_subset - existing_count
        print(f"\n{'='*60}")
        print(f"Downloading {subset} ({split}) - need {need} more (have {existing_count})")
        print(f"{'='*60}")
        
        ds = load_dataset(
            "allenai/olmOCR-mix-1025",
            subset,
            split=split,
            streaming=True,
        )
        
        count = 0
        skipped = 0
        for row in ds:
            # Handle None text values
            text = row.get("text") or row.get("natural_text") or ""
            word_count = len(text.split()) if text else 0
            
            if word_count < min_words:
                skipped += 1
                continue
            
            # Generate filename first to check if already exists
            url_hash = hashlib.md5(row["url"].encode()).hexdigest()[:8]
            filename = f"{subset}_{url_hash}_p{row['page_number']:04d}.png"
            
            if (output_dir / filename).exists():
                count += 1  # Count toward limit but don't re-download
                if count >= need:
                    break
                continue
            
            try:
                pdf_bytes = download_pdf(row["url"])
                img = render_pdf_page(pdf_bytes, row["page_number"])
                img.save(output_dir / filename)
                
                count += 1
                total += 1
                print(f"  [{existing_count + count}/{limit_per_subset}] {filename} ({word_count} words)")
                
                if count >= need:
                    break
                    
            except Exception as e:
                print(f"  Error processing {row['url']}: {e}")
                continue
        
        print(f"  Downloaded: {count} | Skipped (sparse): {skipped}")
    
    print(f"\n{'='*60}")
    print(f"DONE: {total} total images in {output_dir}")
    print(f"{'='*60}")
    return total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download olmOCR dataset")
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--limit", "-n", type=int, default=50, help="Images per subset")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    args = parser.parse_args()
    
    download_olmocr(
        split=args.split,
        limit_per_subset=args.limit,
        min_words=args.min_words,
    )
