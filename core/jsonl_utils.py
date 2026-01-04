"""JSONL file operations for OCR dataset."""
import json
from pathlib import Path
from typing import Iterator, Set, Dict, Any


def read_entries(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield entries from a JSONL file.
    
    Skips malformed lines silently.
    """
    if not path.exists():
        return
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def get_successful_filenames(path: Path) -> Set[str]:
    """Get set of filenames with status='success'.
    
    Used to resume processing from existing output.
    """
    return {
        entry["filename"]
        for entry in read_entries(path)
        if entry.get("status") == "success"
    }


def append_entry(path: Path, entry: Dict[str, Any]) -> None:
    """Append a single entry to JSONL file."""
    with open(path, "a", encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


def create_clean_dataset(input_path: Path, output_path: Path) -> int:
    """Create clean dataset for Unsloth fine-tuning.
    
    Removes status/model columns, keeps only successful entries.
    Output format: {filename, instruction, answer}
    
    Returns number of entries written.
    """
    count = 0
    with open(output_path, "w", encoding='utf-8') as f:
        for entry in read_entries(input_path):
            if entry.get("status") == "success":
                clean = {
                    "filename": entry["filename"],
                    "instruction": entry["instruction"],
                    "answer": entry["answer"],
                }
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
                count += 1
    return count
