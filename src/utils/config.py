"""Configuration utilities for evaluation."""

import os
import yaml
from typing import Dict, Any


def _get_project_root() -> str:
    """Return the project root directory (images_generation_evaluations)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Recursively merge two configuration dictionaries."""
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary."""
    project_root = _get_project_root()
    default_config_path = os.path.join(project_root, "config", "default.yaml")
    return load_config(default_config_path)
