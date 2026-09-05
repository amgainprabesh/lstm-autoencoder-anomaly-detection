"""
Utility functions for Time-Series Anomaly Detection.
"""
import os
import random
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for python random, numpy, and torch for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration dictionary to YAML file.

    Args:
        config (dict): Configuration dictionary.
        config_path (str): Destination path.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def get_device() -> torch.device:
    """
    Detect and return the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs(dirs: List[str]) -> None:
    """
    Ensure that a list of directory paths exist, creating them if necessary.

    Args:
        dirs (list[str]): List of directory paths.
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, filepath: str) -> None:
    """
    Save serializable data to JSON file.

    Args:
        data (Any): Data to save.
        filepath (str): Target file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Any:
    """
    Load JSON file data.

    Args:
        filepath (str): Path to JSON file.

    Returns:
        Any: Loaded data.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
