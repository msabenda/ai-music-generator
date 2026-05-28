#!/usr/bin/env python3
"""
Download and prepare a music dataset.
Usage:
    python scripts/download_dataset.py --dataset maestro
    python scripts/download_dataset.py --dataset lakh
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.download import download_dataset
from src.data.dataset import MusicDataset
from src.data.midi_processor import MidiProcessor
from src.utils.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Download and prepare a music dataset")
    parser.add_argument(
        "--dataset",
        default="maestro",
        choices=["maestro", "lakh"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already extracted",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only pre-process existing MIDI files",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, args.data_dir)

    if not args.skip_download:
        print(f"📥 Downloading '{args.dataset}' dataset...")
        extracted_dir = download_dataset(args.dataset, data_dir, force=args.force)
        midi_dir = extracted_dir
    else:
        midi_dir = os.path.join(data_dir, "raw", args.dataset)
        if not os.path.isdir(midi_dir):
            print(f"⚠️  MIDI directory not found: {midi_dir}")
            print("   Run without --skip-download first, or place MIDI files in:")
            print(f"   {midi_dir}")
            return

    print(f"🔍 Processing MIDI files from: {midi_dir}")
    processed_dir = os.path.join(data_dir, "processed", args.dataset)

    processor = MidiProcessor()
    MusicDataset.preprocess_midi_files(midi_dir, processed_dir, processor)

    # Show stats
    npy_files = [
        f for f in os.listdir(processed_dir) if f.endswith(".npy")
    ]
    print(f"\n✅ Done! Processed {len(npy_files)} files.")
    print(f"   Processed data: {processed_dir}")
    print(f"   Raw data:       {midi_dir}")


if __name__ == "__main__":
    main()
