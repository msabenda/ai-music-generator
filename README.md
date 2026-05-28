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

## How to Run

Each step assumes your virtual environment is **activated**:

| OS | Activate Command |
|-----|-----------------|
| Linux / macOS | `source venv/bin/activate` |
| Windows PowerShell | `venv\Scripts\Activate.ps1` |
| Windows CMD | `venv\Scripts\activate.bat` |

---

### Step 1 — Download a Dataset

You need MIDI data to train the model. Choose one:

```bash
# MAESTRO (classical piano, ~1 GB)
python scripts/download_dataset.py --dataset maestro

# Lakh MIDI (many genres, larger)
python scripts/download_dataset.py --dataset lakh
```

Or use your own MIDI files — just dump `.mid` files into `data/raw/` and run:
```bash
python scripts/download_dataset.py --dataset custom --skip-download
```

**Output:** MIDI files are converted into token sequences in `data/processed/`.

---

### Step 2 — Train a Model

Train the LSTM model (works on CPU, no GPU needed):

```bash
# Default LSTM config — good starting point
python scripts/train.py --config configs/lstm_default.yaml

# Train with more epochs (better quality)
python scripts/train.py --config configs/lstm_default.yaml --epochs 200
```

**Options you can tweak:**
| Flag | What it does |
|------|-------------|
| `--config` | Path to YAML config (LSTM or Transformer) |
| `--epochs` | Override training epochs from config |
| `--device` | `cpu` (default) or `cuda` if you have NVIDIA GPU |

**Output:** Model checkpoints saved to `outputs/checkpoints/` and logs to `outputs/logs/`.

---

### Step 3 — Generate Music

Once the model is trained, generate MIDI:

```bash
# Generate 300 notes with the best checkpoint
python scripts/generate.py --checkpoint outputs/checkpoints/best.pt --length 300

# More creative (higher temperature, more randomness)
python scripts/generate.py --checkpoint outputs/checkpoints/best.pt --temp 1.2 --topk 40

# More structured (lower temperature, less randomness)
python scripts/generate.py --checkpoint outputs/checkpoints/best.pt --temp 0.7 --topk 10

# Generate a longer piece
python scripts/generate.py --checkpoint outputs/checkpoints/best.pt --length 500
```

**Options:**
| Flag | Default | What it does |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to a `.pt` checkpoint file |
| `--length` | 300 | Number of notes to generate |
| `--temp` | 1.0 | Sampling temperature (0.5=stable, 1.5=wild) |
| `--topk` | None (all) | Only consider top-K most likely next notes |
| `--tempo` | 120 | MIDI tempo in BPM |
| `--output` | auto | Custom output `.mid` file path |
| `--device` | cpu | `cpu` or `cuda` |

**Output:** `.mid` file saved to `outputs/generated/`. Open with any MIDI player (VLC, Windows Media Player, GarageBand, FL Studio, etc.)

---

### Full Pipeline Example (5 minutes)

```bash
# 1. Activate venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\Activate.ps1      # Windows

# 2. Get data
python scripts/download_dataset.py --dataset maestro

# 3. Train (adjust epochs if you want faster testing)
python scripts/train.py --config configs/lstm_default.yaml --epochs 50

# 4. Generate
python scripts/generate.py --checkpoint outputs/checkpoints/best.pt --length 400 --temp 1.0

# 5. Play the result — open outputs/generated/generated_temp1.0_len400.mid
```

---

## Datasets

- **MAESTRO** — ~1,200 piano performances (classical, MIDI)
- **Lakh MIDI** — ~45,000 MIDI files (many genres)
- **Custom** — Dump your own MIDI files into `data/raw/`

## Models

| Model | Description | CPU Friendly | Notes |
|-------|-------------|--------------|-------|
| LSTM | Note-by-note prediction with memory | ✅ Fast & light | Start here |
| Transformer | Attention-based with causal masking | ⚠️ Slower | Better with more data |

## Tips

- **Training too slow?** Reduce `hidden_dim` in the config or decrease `sequence_length`
- **Generated music sounds random?** Lower `temperature` (0.6–0.8) and use `--topk 15`
- **Generated music is too repetitive?** Raise `temperature` (1.1–1.3)
- **Want to train faster for testing?** Use `--epochs 20` with `--length 100` in generation
- **Output is just noise?** Train for more epochs or add more MIDI files to the dataset
