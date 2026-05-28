"""
MusicDataset — PyTorch Dataset for MIDI note sequences.

Loads pre-processed token sequences from disk and provides
batches of (input_seq, target_seq) for language-model-style training.
"""

import os
import glob
import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .midi_processor import MidiProcessor

logger = logging.getLogger(__name__)


class MusicDataset(Dataset):
    """PyTorch Dataset that yields (input_seq, target_seq) pairs."""

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 100,
        stride: Optional[int] = None,
        processor: Optional[MidiProcessor] = None,
    ):
        """
        Args:
            data_dir: Directory containing .npy token sequence files
            sequence_length: Number of tokens per training sample
            stride: Step size when sliding window over long sequences (default: seq_len)
            processor: MIDI processor (used for vocab size reference)
        """
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.processor = processor or MidiProcessor()

        # Find all pre-processed token files
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

        self.token_files = npy_files + txt_files
        if not self.token_files:
            logger.warning("No token files found in %s", data_dir)

        # Pre-compute indices for each window across all files
        self.indices: List[Tuple[int, int, int]] = []  # (file_idx, start, end)
        self._file_lengths: List[int] = []

        for file_idx, file_path in enumerate(self.token_files):
            seq = self._load_sequence(file_path)
            length = len(seq)
            self._file_lengths.append(length)

            if length < sequence_length + 1:
                continue  # skip sequences too short to make at least one pair

            for start in range(0, length - sequence_length, self.stride):
                self.indices.append((file_idx, start, start + sequence_length + 1))

        logger.info(
            "Loaded %d token files, created %d samples (seq_len=%d)",
            len(self.token_files),
            len(self.indices),
            sequence_length,
        )

    def _load_sequence(self, path: str) -> List[int]:
        """Load a token sequence from .npy or .txt."""
        if path.endswith(".npy"):
            arr = np.load(path)
            return arr.tolist() if arr.ndim > 0 else []
        else:  # .txt — one token per line
            with open(path) as f:
                return [int(line.strip()) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, start, end = self.indices[idx]
        path = self.token_files[file_idx]
        seq = self._load_sequence(path)
        chunk = seq[start:end]

        # Pad if short (shouldn't happen with pre-computed indices, but be safe)
        if len(chunk) < self.sequence_length + 1:
            pad_len = self.sequence_length + 1 - len(chunk)
            chunk = chunk + [self.processor.PAD_TOKEN] * pad_len

        input_seq = torch.tensor(chunk[:-1], dtype=torch.long)
        target_seq = torch.tensor(chunk[1:], dtype=torch.long)

        return input_seq, target_seq

    @property
    def vocab_size(self) -> int:
        return self.processor.VOCAB_SIZE

    @staticmethod
    def preprocess_midi_files(
        midi_dir: str,
        output_dir: str,
        processor: Optional[MidiProcessor] = None,
    ) -> None:
        """Batch convert MIDI files to token sequences stored as .npy files."""
        processor = processor or MidiProcessor()
        os.makedirs(output_dir, exist_ok=True)

        midi_files = MidiProcessor.collect_midi_files(midi_dir)
        if not midi_files:
            logger.warning("No MIDI files found in %s", midi_dir)
            return

        saved = 0
        for midi_path in midi_files:
            try:
                notes = processor.midi_to_notes(midi_path)
                if len(notes) < 5:
                    continue  # skip files with too few notes
                tokens = processor.notes_to_sequence(notes)
                base = os.path.splitext(os.path.basename(midi_path))[0]
                out_path = os.path.join(output_dir, f"{base}.npy")
                np.save(out_path, np.array(tokens, dtype=np.int32))
                saved += 1
            except Exception as e:
                logger.debug("Skipping %s: %s", midi_path, e)

        logger.info(
            "Pre-processed %d / %d MIDI files → %s",
            saved, len(midi_files), output_dir,
        )
