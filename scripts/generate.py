#!/usr/bin/env python3
"""
Generate music from a trained model checkpoint.
Usage:
    python scripts/generate.py --checkpoint outputs/checkpoints/best.pt --length 300
    python scripts/generate.py --checkpoint outputs/checkpoints/best.pt --temp 1.2 --topk 40
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.data.midi_processor import MidiProcessor
from src.models import LSTMMusicModel, TransformerMusicModel
from src.generation.generator import MusicGenerator
from src.utils.logging_setup import setup_logging


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load model architecture and weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", {}).get("model", {})
    history = checkpoint.get("history", {})
    epoch = checkpoint.get("epoch", 0)

    model_name = model_config.get("name", "lstm")
    vocab_size = model_config.get("vocab_size", 131)

    if model_name == "lstm":
        model = LSTMMusicModel(
            vocab_size=vocab_size,
            embedding_dim=model_config.get("embedding_dim", 64),
            hidden_dim=model_config.get("hidden_dim", 256),
            num_layers=model_config.get("num_layers", 2),
            dropout=model_config.get("dropout", 0.3),
        )
    elif model_name == "transformer":
        model = TransformerMusicModel(
            vocab_size=vocab_size,
            embedding_dim=model_config.get("embedding_dim", 64),
            num_heads=model_config.get("num_heads", 4),
            num_layers=model_config.get("num_layers", 3),
            dropout=model_config.get("dropout", 0.1),
            max_seq_len=model_config.get("max_seq_len", 512),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, epoch, history


def main():
    parser = argparse.ArgumentParser(description="Generate music from a trained model")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output MIDI file path",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=300,
        help="Number of notes to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Top-k sampling (only sample from top k logits)",
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=120.0,
        help="MIDI tempo in BPM",
    )
    parser.add_argument(
        "--prime",
        nargs="+",
        type=int,
        default=None,
        help="Priming pitch tokens (e.g., --prime 60 64 67 72)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    setup_logging("INFO")

    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(__file__))

    if not os.path.isfile(args.checkpoint):
        checkpoint_path = os.path.join(project_root, args.checkpoint)
        if not os.path.isfile(checkpoint_path):
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
    else:
        checkpoint_path = args.checkpoint

    # Load model
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    model, epoch, history = load_model_from_checkpoint(checkpoint_path, args.device)

    best_val = None
    if history and "val_loss" in history and history["val_loss"]:
        best_val = min(history["val_loss"])
    val_str = f", best val loss: {best_val:.4f}" if best_val else ""
    print(f"🧠 Model: {type(model).__name__} (epoch {epoch}{val_str})")

    # Generator
    processor = MidiProcessor()
    generator = MusicGenerator(model, processor=processor, device=args.device)

    # Prime tokens
    prime_tokens = None
    if args.prime:
        prime_tokens = [processor.SOS_TOKEN] + args.prime
        print(f"🎹 Priming with: {args.prime}")

    # Generate
    print(f"\n🎵 Generating {args.length} notes (temp={args.temp}, topk={args.topk})...")

    # Output path
    if args.output is None:
        output_dir = os.path.join(project_root, "outputs", "generated")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"generated_temp{args.temp}_len{args.length}.mid",
        )
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    result = generator.generate_to_midi(
        output_path=output_path,
        max_notes=args.length,
        temperature=args.temp,
        top_k=args.topk,
        prime_tokens=prime_tokens,
        tempo=args.tempo,
    )

    print(f"\n✅ Saved to: {result}")
    print("🎧 Open with any MIDI player or DAW to hear it!")


if __name__ == "__main__":
    main()
