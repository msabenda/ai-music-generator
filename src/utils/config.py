"""Configuration loading utilities."""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with config values
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths relative to the config file
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    # Default values if not in config
    config.setdefault("data", {})
    config.setdefault("training", {})
    config.setdefault("model", {})

    # Resolve data paths relative to project root
    if "data" in config and isinstance(config["data"], dict):
        dataset_name = config["data"].get("dataset", "maestro")
        config["data"]["processed_dir"] = os.path.join(
            project_root, "data", "processed", dataset_name
        )
        config["data"]["raw_dir"] = os.path.join(
            project_root, "data", "raw", dataset_name
        )

    # Output paths
    config["output_dir"] = os.path.join(project_root, "outputs")

    return config
