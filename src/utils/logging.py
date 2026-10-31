"""Shared logging configuration.

All experiment and analysis scripts call get_logger(__name__) instead of
using print(), so that stdout capture in results/raw/*.txt (see README.md's
reproducibility instructions) captures structured, timestamped output.
"""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def _configure_root_logger(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A logging.Logger writing timestamped, leveled lines to stdout.
    """
    _configure_root_logger()
    return logging.getLogger(name)
