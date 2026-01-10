"""Evaluation logging callback for training.

Runs inference on eval samples and logs predictions vs references to W&B.
"""

import contextlib
import logging
import random
from typing import Any

import torch
import wandb
from transformers import TrainerCallback

from core.config import OCR_INSTRUCTION
from training.eval_utils import log_high_loss_samples

logger = logging.getLogger(__name__)


class EvalWithHighLossLoggingCallback(TrainerCallback):
    """Run evaluation with high-loss sample logging to W&B.

    After each evaluation, runs actual model inference on sampled eval data
    and logs predictions vs references to W&B for visual inspection.

    Attributes:
        eval_dataset: The evaluation dataset to sample from.
        processor: The model processor for inference.
        top_k: Number of samples to log.
    """

    def __init__(self, eval_dataset: Any, processor: Any, top_k: int = 20) -> None:
        """Initialize the callback.

        Args:
            eval_dataset: The evaluation dataset to sample from.
            processor: The model processor for inference.
            top_k: Number of samples to log.
        """
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.top_k = top_k

    def _run_inference(self, model: Any, image: Any) -> str:
        """Run inference on a single image.

        Args:
            model: The model to use for inference.
            image: PIL Image to process.

        Returns:
            Generated text output.
        """
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": OCR_INSTRUCTION}],
            }
        ]
        text_prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            image, text_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=2048,
                use_cache=True,
                temperature=1.5,
                min_p=0.1,
            )
        return self.processor.tokenizer.decode(output[0], skip_special_tokens=True)

    def on_evaluate(
        self,
        _args: Any,
        state: Any,
        _control: Any,
        _metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """After evaluation, run inference and log samples to W&B.

        Args:
            _args: Training arguments (unused).
            state: Trainer state with global_step.
            _control: Trainer control (unused).
            _metrics: Evaluation metrics (unused).
            **kwargs: Additional arguments including model.
        """
        # Lazy import to avoid dependency at module level
        try:
            from unsloth import FastVisionModel
        except ImportError:
            logger.warning("Unsloth not available, skipping eval logging")
            return

        if not wandb.run:
            return

        model = kwargs.get("model")
        if model is None:
            return

        logger.info(f"Running inference on {self.top_k} eval samples for W&B logging...")

        try:
            # Sample eval set
            sample_size = min(self.top_k, len(self.eval_dataset))
            indices = random.sample(range(len(self.eval_dataset)), sample_size)
            samples = [self.eval_dataset[i] for i in indices]

            # Switch to inference mode
            FastVisionModel.for_inference(model)

            predictions = []
            references = []
            losses = []

            for i, sample in enumerate(samples):
                img = sample["images"][0]
                # Extract reference from messages (assistant content)
                ref = ""
                for msg in sample.get("messages", []):
                    if msg.get("role") == "assistant":
                        for content in msg.get("content", []):
                            if content.get("type") == "text":
                                ref = content.get("text", "")
                                break

                # Run actual inference
                pred = self._run_inference(model, img)
                predictions.append(pred)
                references.append(ref)
                losses.append(1.0)  # Placeholder loss (we don't track per-sample loss)

                if (i + 1) % 5 == 0:
                    logger.info(f"  Processed {i + 1}/{sample_size} samples")

            # Switch back to training mode
            FastVisionModel.for_training(model)

            log_high_loss_samples(
                samples=samples,
                predictions=predictions,
                references=references,
                losses=losses,
                top_k=self.top_k,
                table_name=f"high_loss_samples_step_{state.global_step}",
            )
            logger.info(f"Logged {sample_size} samples with predictions to W&B")

        except Exception as e:
            logger.warning(f"Failed to log high-loss samples: {e}")
            # Ensure we return to training mode even on error
            with contextlib.suppress(Exception):
                FastVisionModel.for_training(model)
