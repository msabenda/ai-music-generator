#!/usr/bin/env python3
"""
Train a music generation model.
Usage:
    python scripts/train.py --config configs/lstm_default.yaml
    python scripts/train.py --config configs/transformer_default.yaml --epochs 50
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import MusicDataset
from src.data.midi_processor import MidiProcessor
from src.models import LSTMMusicModel, TransformerMusicModel
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train a music generation model")
    parser.add_argument(
        "--config",
        default="configs/lstm_default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Override device",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    # Load config
    project_root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)
    print(f"📋 Loaded config: {args.config}")

    # Override
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    device = args.device or config["training"].get("device", "cpu")
    print(f"⚙️  Device: {device}")

    # Load pre-processed data
    processed_dir = config["data"]["processed_dir"]
    if not os.path.isdir(processed_dir):
        print(f"❌ Processed data not found at: {processed_dir}")
        print("   Run `python scripts/download_dataset.py` first.")
        sys.exit(1)

    print(f"📂 Loading data from: {processed_dir}")

    seq_len = config["data"].get("sequence_length", 100)
    processor = MidiProcessor()
    full_dataset = MusicDataset(
        processed_dir,
        sequence_length=seq_len,
        processor=processor,
    )

    if len(full_dataset) == 0:
        print("❌ No training samples created. Check your MIDI files.")
        sys.exit(1)

    # Split
    train_ratio = config["data"].get("train_split", 0.85)
    val_ratio = config["data"].get("val_split", 0.10)
    test_ratio = config["data"].get("test_split", 0.05)

    total = len(full_dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    batch_size = config["data"].get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # keep at 0 for safety
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"📊 Dataset: {total} samples "
        f"(train={train_size}, val={val_size}, test={test_size})"
    )

    # Build model
    model_config = config.get("model", {})
    model_name = model_config.get("name", "lstm")
    vocab_size = full_dataset.vocab_size

    print(f"🧠 Building {model_name.upper()} model...")

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

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    lr = config["training"].get("learning_rate", 0.001)
    weight_decay = config["training"].get("weight_decay", 1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
        output_dir=config["output_dir"],
        device=device,
    )

    # Train
    num_epochs = config["training"].get("epochs", 100)
    print(f"\n🚀 Training for {num_epochs} epochs...\n")
    history = trainer.train(num_epochs)

    # Print final results
    best_val = min(history["val_loss"]) if history["val_loss"] else "N/A"
    print(f"\n✅ Training complete!")
    print(f"   Best validation loss: {best_val:.4f}" if isinstance(best_val, float) else f"   Best validation loss: {best_val}")
    print(f"   Checkpoints: {trainer.checkpoint_dir}")
    print(f"   History:     {trainer.log_dir}")


if __name__ == "__main__":
    main()
