"""Character and Word Error Rate computation for OCR evaluation."""

import editdistance


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER).

    CER = (substitutions + insertions + deletions) / reference_length

    Args:
        reference: Ground truth text.
        hypothesis: Predicted text.

    Returns:
        CER as float between 0 and 1.
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    distance = editdistance.eval(reference, hypothesis)
    return distance / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER).

    WER = (substitutions + insertions + deletions) / reference_word_count

    Args:
        reference: Ground truth text.
        hypothesis: Predicted text.

    Returns:
        WER as float between 0 and 1.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    distance = editdistance.eval(ref_words, hyp_words)
    return distance / len(ref_words)
