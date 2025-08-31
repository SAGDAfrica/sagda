# WORK IN PROGRESS
from __future__ import annotations
import os, json
import pandas as pd
from typing import Tuple, Dict, Any

# Data Registry
_DATA_REGISTRY = {
    # Example: "toy_wheat_ma": "/path/to/toy.csv"
}

def list_datasets() -> Dict[str, str]:
    """Return the registry mapping of dataset name -> path or URL (if remote)."""
    return dict(_DATA_REGISTRY)

def register_dataset(name: str, path: str) -> None:
    _DATA_REGISTRY[name] = path

def load_dataset(name: str, target_col: str | None = None) -> Tuple[pd.DataFrame, Any, Dict]:
    """Load a dataset by name from the registry.
    Returns (X, y, meta). If target_col is None, returns (df, None, meta).
    """
    if name not in _DATA_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found. Use register_dataset(name, path).")
    path = _DATA_REGISTRY[name]
    if path.startswith(("http://", "https://")):
        raise ValueError("Remote URLs are not downloaded automatically. Please download locally and register the local path.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    meta = {"name": name, "path": path, "columns": list(df.columns)}
    if target_col and target_col in df.columns:
        y = df[target_col].values
        X = df.drop(columns=[target_col])
        return X, y, meta
    return df, None, meta
