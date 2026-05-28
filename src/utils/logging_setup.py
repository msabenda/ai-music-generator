"""Logging configuration."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with nice formatting."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
