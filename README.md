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

## Setup

### Prerequisites

- **Python 3.10+** (tested on 3.13)
- **pip** (Python package manager)
- GNU/Linux, macOS, or **Windows**

### Installation (Virtual Environment)

It's recommended to use a virtual environment to isolate dependencies.

#### Linux / macOS
```bash
# Navigate to the project
cd music-generator

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (PowerShell / CMD)
```powershell
# Navigate to the project
cd music-generator

# Create virtual environment
python -m venv venv

# Activate it (PowerShell)
venv\Scripts\Activate.ps1

# Or if using Command Prompt
venv\Scripts\activate.bat

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> If you get a `torch` error on Windows about CUDA, install the CPU-only version explicitly:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

#### Windows (WSL — Windows Subsystem for Linux)
```bash
# Follow the Linux instructions above; it's the same WSL environment
cd music-generator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch, pretty_midi; print('OK:', torch.__version__, pretty_midi.__version__)"
```

---

## Quick Start

```bash
# Make sure your venv is activated first!
# Linux/macOS: source venv/bin/activate
# Windows: venv\Scripts\Activate.ps1

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
