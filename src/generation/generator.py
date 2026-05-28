"""
MusicGenerator — inference and music generation from trained models.

Supports:
  - Temperature sampling (randomness control)
  - Top-k sampling (quality control)
  - Priming with existing sequences
  - Saving generated music as MIDI files
"""

import os
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.data.midi_processor import MidiProcessor

logger = logging.getLogger(__name__)


class MusicGenerator:
    """Generate music from a trained model."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Optional[MidiProcessor] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.processor = processor or MidiProcessor()

    @torch.no_grad()
    def generate(
        self,
        max_notes: int = 500,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        prime_tokens: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> List[int]:
        """Generate a sequence of note tokens.

        Args:
            max_notes: Maximum number of tokens to generate
            temperature: Sampling temperature (>1 more random, <1 more deterministic)
            top_k: If set, only sample from top-k logits
            prime_tokens: Optional list of priming tokens to start with
            verbose: Log generation progress

        Returns:
            List of generated token IDs (excluding special tokens for output)
        """
        if prime_tokens is None:
            prime_tokens = [self.processor.SOS_TOKEN]

        generated = list(prime_tokens)
        input_seq = torch.tensor([prime_tokens], dtype=torch.long, device=self.device)

        # LSTM stateful tracking
        hidden = None
        is_lstm = hasattr(self.model, "init_hidden")

        if verbose:
            logger.info(
                "Generating %d notes (temp=%.2f, top_k=%s, prime_len=%d)",
                max_notes, temperature, top_k, len(prime_tokens),
            )

        for step in range(max_notes):
            if is_lstm:
                if hidden is None:
                    hidden = self.model.init_hidden(1, self.device)
                logits, hidden = self.model(input_seq, hidden)
                logits = logits[:, -1, :]  # last token only
                # Detach hidden
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                # Transformer: use full sequence
                logits = self.model(input_seq)
                logits = logits[:, -1, :]  # last token only

            # Sample next token
            next_token = self._sample(logits.squeeze(0), temperature, top_k)

            # Stop if EOS
            if next_token == self.processor.EOS_TOKEN:
                if verbose:
                    logger.info("EOS token reached at step %d", step)
                break

            generated.append(next_token)

            # Prepare next input
            input_seq = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

            if verbose and (step + 1) % 100 == 0:
                logger.info("Generated %d / %d tokens", step + 1, max_notes)

        # Strip special tokens for output
        output = [
            t for t in generated
            if t not in (self.processor.PAD_TOKEN, self.processor.SOS_TOKEN, self.processor.EOS_TOKEN)
        ]

        if verbose:
            logger.info("Generation complete: %d total → %d note tokens", len(generated), len(output))

        return output

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> int:
        """Sample a token from logits with temperature + optional top-k."""
        logits = logits / max(temperature, 1e-8)

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            threshold = values[-1]
            logits[logits < threshold] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def generate_to_midi(
        self,
        output_path: str,
        max_notes: int = 500,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        prime_tokens: Optional[List[int]] = None,
        tempo: float = 120.0,
    ) -> str:
        """Generate music and save as MIDI file."""
        tokens = self.generate(
            max_notes=max_notes,
            temperature=temperature,
            top_k=top_k,
            prime_tokens=prime_tokens,
        )
        self.processor.save_midi(tokens, output_path, tempo=tempo)
        return output_path
