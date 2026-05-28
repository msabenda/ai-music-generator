"""
Dataset download utilities.

Handles downloading and extracting common symbolic music datasets.
Uses 'requests' if available, falls back to built-in 'urllib' otherwise.
"""

import os
import zipfile
import tarfile
import logging
from typing import Optional, Callable

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Try to use requests (better UX); fall back to urllib (built-in)
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    import urllib.request
    import urllib.error


# Dataset URLs
DATASET_SOURCES = {
    "maestro": {
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "type": "zip",
        "description": "MAESTRO v3.0.0 (MIDI only)",
    },
    "lakh": {
        "url": "https://huggingface.co/datasets/music-community/lakh-midi/resolve/main/lakh-midi-subset.zip",
        "type": "zip",
        "description": "Lakh MIDI subset",
    },
}


def download_file(url: str, dest: str, chunk_size: int = 8192) -> None:
    """Download a file with progress bar.

    Uses 'requests' if available, otherwise falls back to 'urllib'.
    """
    logger.info("Downloading %s → %s", url, dest)

    if _HAS_REQUESTS:
        _download_with_requests(url, dest, chunk_size)
    else:
        _download_with_urllib(url, dest, chunk_size)

    size_mb = os.path.getsize(dest) / 1e6
    logger.info("Downloaded %s (%.1f MB)", dest, size_mb)


def _download_with_requests(url: str, dest: str, chunk_size: int) -> None:
    """Download with requests library (streaming + progress bar)."""
    import requests
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=os.path.basename(dest),
        total=total,
        unit="B",
        unit_scale=True,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def _download_with_urllib(url: str, dest: str, chunk_size: int) -> None:
    """Download with built-in urllib (no external dependency)."""
    import urllib.request

    def reporthook(block_count: int, block_size: int, total_size: int) -> None:
        if pbar is None:
            return
        downloaded = block_count * block_size
        pbar.total = total_size if total_size > 0 else None
        pbar.update(downloaded - pbar.n)

    pbar = tqdm(desc=os.path.basename(dest), unit="B", unit_scale=True)
    try:
        urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    finally:
        pbar.close()


def extract_zip(zip_path: str, extract_dir: str) -> None:
    """Extract a zip archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    logger.info("Extracted %s → %s", zip_path, extract_dir)


def extract_tar(tar_path: str, extract_dir: str) -> None:
    """Extract a tar.gz archive."""
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)
    logger.info("Extracted %s → %s", tar_path, extract_dir)


def download_dataset(
    dataset_name: str,
    data_dir: str = "data",
    force: bool = False,
) -> str:
    """Download and extract a dataset.

    Args:
        dataset_name: Key in DATASET_SOURCES ("maestro" or "lakh")
        data_dir: Root data directory
        force: Re-download even if extracted directory exists

    Returns:
        Path to extracted MIDI files
    """
    # Handle "custom" — just preprocess whatever is in raw dir
    if dataset_name == "custom":
        custom_dir = os.path.join(data_dir, "raw", "custom")
        if not os.path.isdir(custom_dir):
            raise FileNotFoundError(
                f"Custom MIDI directory not found: {custom_dir}\n"
                f"Place your .mid files in {custom_dir} first."
            )
        return custom_dir

    if dataset_name not in DATASET_SOURCES:
        available = list(DATASET_SOURCES.keys()) + ["custom"]
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    source = DATASET_SOURCES[dataset_name]
    raw_dir = os.path.join(data_dir, "raw", dataset_name)
    extracted_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(raw_dir, exist_ok=True)

    # Check if already extracted
    if not force and os.path.isdir(extracted_dir) and os.listdir(extracted_dir):
        logger.info("Dataset '%s' already extracted at %s", dataset_name, extracted_dir)
        return extracted_dir

    # Download
    archive_name = os.path.basename(source["url"].split("?")[0]) or f"{dataset_name}.zip"
    archive_path = os.path.join(raw_dir, archive_name)

    if not os.path.isfile(archive_path) or force:
        download_file(source["url"], archive_path)

    # Extract
    if source["type"] == "zip":
        extract_zip(archive_path, raw_dir)
    elif source["type"] == "tar":
        extract_tar(archive_path, raw_dir)

    # Find the directory that actually has MIDI files
    found = _find_midi_root(raw_dir)
    if found and found != extracted_dir:
        # On Windows, os.symlink may fail (needs admin / developer mode)
        # Use a junction or just return the actual path
        try:
            if not os.path.exists(extracted_dir):
                os.symlink(found, extracted_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Symlinks not available (e.g., Windows without developer mode)
            pass

    return found or extracted_dir


def _find_midi_root(directory: str) -> Optional[str]:
    """Walk extracted directory to find where MIDI files live."""
    from .midi_processor import MidiProcessor

    try:
        items = sorted(os.listdir(directory))
    except PermissionError:
        return None

    midi_dirs = []
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            files = MidiProcessor.collect_midi_files(item_path)
            if files:
                midi_dirs.append((item_path, len(files)))

    if midi_dirs:
        best = max(midi_dirs, key=lambda x: x[1])
        return best[0]

    files = MidiProcessor.collect_midi_files(directory)
    if files:
        return directory

    return None
