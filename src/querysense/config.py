from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed config as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML file is empty.
    """
    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def load_project_configs() -> dict[str, Any]:
    """Load all main project configs into a single dictionary."""
    config_files = {
        "base": "configs/base.yaml",
        "training": "configs/training.yaml",
        "retrieval": "configs/retrieval.yaml",
        "api": "configs/api.yaml",
    }

    return {name: load_yaml_config(path) for name, path in config_files.items()}


def get_project_root() -> Path:
    """Return the absolute project root path."""
    return PROJECT_ROOT