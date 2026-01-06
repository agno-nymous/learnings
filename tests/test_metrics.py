"""Tests for OCR metrics: Character Error Rate (CER) and Word Error Rate (WER)."""

from training.metrics import compute_cer, compute_wer


def test_cer_perfect_match():
    """Test CER returns 0 for identical strings."""
    assert compute_cer("hello", "hello") == 0.0


def test_cer_all_wrong():
    """Test CER returns 1 for completely different strings."""
    assert compute_cer("abc", "xyz") == 1.0


def test_cer_partial_match():
    """Test CER with partial character match.

    "hello" -> "hallo" = 1 substitution / 5 chars = 0.2
    """
    cer = compute_cer("hello", "hallo")
    assert abs(cer - 0.2) < 0.01


def test_wer_perfect_match():
    """Test WER returns 0 for identical text."""
    assert compute_wer("hello world", "hello world") == 0.0


def test_wer_substitution():
    """Test WER with word substitution.

    "hello world" -> "hallo world" = 1 error / 2 words = 0.5
    """
    wer = compute_wer("hello world", "hallo world")
    assert abs(wer - 0.5) < 0.01


def test_wer_insertion():
    """Test WER with word insertion."""
    wer = compute_wer("hello world", "hello big world")
    assert abs(wer - 0.5) < 0.01  # 1 insertion / 2 words


def test_wer_deletion():
    """Test WER with word deletion."""
    wer = compute_wer("hello big world", "hello world")
    assert abs(wer - 0.333) < 0.01  # 1 deletion / 3 words
