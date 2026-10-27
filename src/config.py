"""Load the study configuration (config/config.yaml)."""

from pathlib import Path

import yaml

# Repository root: src/config.py -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path=CONFIG_PATH):
    """Return config.yaml as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(relative):
    """Turn a path from the config (relative to the repo root) into an absolute Path."""
    return REPO_ROOT / relative
