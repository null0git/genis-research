"""YAML configuration loading helpers.

Keeps config loading in one place so every script resolves paths the same
way (relative to the repository root, not to the current working
directory) and fails loudly on a missing file rather than silently using
defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def get_repo_root() -> Path:
    """Return the repository root (the directory containing this src/ tree)."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file, resolving relative paths against the repo root.

    Args:
        path: Absolute path, or a path relative to the repository root
            (e.g. "configs/data.yaml").

    Returns:
        The parsed YAML content as a dict.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    p = Path(path)
    if not p.is_absolute():
        p = get_repo_root() / p
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {p}. Paths are resolved relative to the "
            f"repository root ({get_repo_root()}) unless already absolute."
        )
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(experiment_config_path: str | Path) -> Dict[str, Any]:
    """Load an experiment config and inline its referenced data/model configs.

    An experiment YAML (e.g. configs/experiments/e01_*.yaml) references a
    data_config path and one or more model config paths as strings. This
    helper loads those referenced files too, so downstream code gets a
    single, fully-resolved dict rather than having to chase paths itself.

    Args:
        experiment_config_path: Path to the experiment YAML.

    Returns:
        The experiment config dict, with "data_config" replaced by the
        loaded data config dict, and each entry in "models" replaced by the
        loaded model config dict (list order preserved).
    """
    cfg = load_yaml(experiment_config_path)

    if "data_config" in cfg:
        cfg["data_config"] = load_yaml(cfg["data_config"])

    if "models" in cfg:
        cfg["models"] = [load_yaml(m) for m in cfg["models"]]

    return cfg
