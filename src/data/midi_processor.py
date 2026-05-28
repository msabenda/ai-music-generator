"""
MIDI Processor — converts MIDI files into note sequences and back.

Handles:
  - Reading .mid files via pretty_midi
  - Encoding notes as integer tokens (pitch, duration, velocity combined)
  - Decoding token sequences back to MIDI
"""

import os
import glob
import logging
from typing import List, Optional, Tuple

import numpy as np
import pretty_midi

logger = logging.getLogger(__name__)


class MidiProcessor:
    """Convert MIDI files to/from token sequences for model training."""

    # Special tokens
    PAD_TOKEN = 128
    SOS_TOKEN = 129  # Start of sequence
    EOS_TOKEN = 130  # End of sequence
    VOCAB_SIZE = 131  # 0-127 pitches + 3 special tokens

    def __init__(self, note_range: Tuple[int, int] = (21, 108)):
        """
        Args:
            note_range: (min_pitch, max_pitch) — default piano range
        """
        self.min_pitch, self.max_pitch = note_range

    def midi_to_notes(self, midi_path: str) -> np.ndarray:
        """Extract note array from a MIDI file.

        Returns:
            Nx3 array: [pitch, start_time, end_time] (sorted by start_time)
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", midi_path, e)
            return np.empty((0, 3))

        notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue  # skip drums for now
            for note in instrument.notes:
                if self.min_pitch <= note.pitch <= self.max_pitch:
                    notes.append([note.pitch, note.start, note.end])

        if not notes:
            return np.empty((0, 3))

        arr = np.array(notes, dtype=np.float32)
        # Sort by start time
        arr = arr[arr[:, 1].argsort()]
        return arr

    def notes_to_sequence(self, notes: np.ndarray) -> List[int]:
        """Convert note array to a flat token sequence.

        For each note: [pitch, duration_quantized, velocity_quantized]
        (Simplified: just pitch tokens for basic models)
        """
        tokens = [self.SOS_TOKEN]

        for i in range(len(notes)):
            pitch = int(notes[i, 0])
            # Clamp to valid range
            pitch = max(0, min(127, pitch))
            tokens.append(pitch)

        tokens.append(self.EOS_TOKEN)
        return tokens

    def sequence_to_midi(
        self,
        tokens: List[int],
        tempo: float = 120.0,
        note_duration: float = 0.25,
    ) -> pretty_midi.PrettyMIDI:
        """Convert a token sequence back to a MIDI object.

        Special tokens (PAD/SOS/EOS) are skipped.
        Each pitch token becomes a quarter-note at the current time step.
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=0)  # Acoustic grand piano

        current_time = 0.0
        step_duration = 60.0 / tempo * note_duration  # beats to seconds

        for token in tokens:
            if token in (self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN):
                continue
            if not (0 <= token <= 127):
                continue

            note = pretty_midi.Note(
                velocity=80,
                pitch=token,
                start=current_time,
                end=current_time + step_duration,
            )
            piano.notes.append(note)
            current_time += step_duration

        midi.instruments.append(piano)
        return midi

    def save_midi(
        self,
        tokens: List[int],
        output_path: str,
        tempo: float = 120.0,
    ) -> None:
        """Save token sequence as a MIDI file."""
        midi = self.sequence_to_midi(tokens, tempo=tempo)
        midi.write(output_path)
        logger.info("Saved MIDI to %s (%d tokens)", output_path, len(tokens))

    @staticmethod
    def collect_midi_files(directory: str) -> List[str]:
        """Recursively find all .mid / .midi files."""
        patterns = ["**/*.mid", "**/*.midi"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
        return sorted(files)
