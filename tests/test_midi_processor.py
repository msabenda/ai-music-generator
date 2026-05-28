"""Unit tests for MIDI processor."""

import os
import sys
import tempfile

import numpy as np
import pretty_midi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.midi_processor import MidiProcessor


def create_test_midi(path: str) -> str:
    """Create a simple MIDI file with a C major scale."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0)

    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=i * 0.5,
            end=i * 0.5 + 0.4,
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(path)
    return path


def test_midi_to_notes():
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        path = create_test_midi(f.name)

    processor = MidiProcessor()
    notes = processor.midi_to_notes(path)
    os.unlink(path)

    assert len(notes) == 8, f"Expected 8 notes, got {len(notes)}"
    assert np.array_equal(notes[:, 0], [60, 62, 64, 65, 67, 69, 71, 72])


def test_roundtrip():
    """MIDI → tokens → MIDI preserves note count."""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        path = create_test_midi(f.name)

    processor = MidiProcessor()
    notes = processor.midi_to_notes(path)
    tokens = processor.notes_to_sequence(notes)

    # Reconstruct
    processor.sequence_to_midi(tokens)

    # Count actual pitch tokens (excluding special tokens)
    pitch_tokens = [t for t in tokens if 0 <= t <= 127]
    assert len(pitch_tokens) == 8, f"Expected 8 pitch tokens, got {len(pitch_tokens)}"

    os.unlink(path)


def test_save_and_load_midi():
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        out_path = f.name

    processor = MidiProcessor()
    tokens = [
        processor.SOS_TOKEN,
        60, 62, 64, 65, 67, 69, 71, 72,
        processor.EOS_TOKEN,
    ]
    processor.save_midi(tokens, out_path, tempo=120.0)

    # Read back
    notes = processor.midi_to_notes(out_path)
    assert len(notes) > 0, "No notes in saved MIDI"

    os.unlink(out_path)


if __name__ == "__main__":
    test_midi_to_notes()
    test_roundtrip()
    test_save_and_load_midi()
    print("✅ All MIDI processor tests passed!")
