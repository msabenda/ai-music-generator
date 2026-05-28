# 🎵 Music Generator

An AI-powered music generation system. Generates symbolic (MIDI) music using deep learning.

## Architecture

```
music-generator/
├── data/               # Datasets (raw & processed)
├── src/                # Source code
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # Model architectures
│   ├── training/       # Training loops & configs
│   ├── generation/     # Inference & music generation
│   └── utils/          # Shared utilities
├── configs/            # Configuration files
├── outputs/            # Generated music & checkpoints
├── notebooks/          # Exploration & analysis
├── scripts/            # Run scripts
└── tests/              # Unit tests
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download & prepare a dataset
python scripts/download_dataset.py --dataset maestro

# Train (CPU-friendly by default)
python scripts/train.py --config configs/lstm_default.yaml

# Generate music
python scripts/generate.py --checkpoint outputs/checkpoints/best.pt --length 200
```

## Datasets

- **MAESTRO** — Piano performances (MIDI + audio)
- **Lakh MIDI** — Large MIDI corpus
- **Custom** — Dump your own MIDI files into `data/raw/`

## Models

| Model | Description | CPU Friendly |
|-------|-------------|--------------|
| LSTM  | Note-by-note prediction | ✅ |
| Transformer | Attention-based (small) | ⚠️ (slow on CPU) |
