"""Typed configuration loading for RouteSentinel.

All constants live in ``config.yaml``; this module only parses and validates.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class DataPaths(BaseModel):
    """Input dataset locations."""

    shipments: str
    notes: str


class OutputPaths(BaseModel):
    """Output artifact locations."""

    output_csv: str
    findings_db: str
    token_cost_log: str


class Settings(BaseModel):
    """Full RouteSentinel configuration (mirrors config.yaml)."""

    own_threshold: float
    peer_threshold: float
    min_history: int
    max_history: int
    absorption_weeks: int
    grace_days: int
    k_retrieval: int
    rrf_k: int
    model: str
    temperature: float
    top_k: int
    seed: int
    data: DataPaths
    outputs: OutputPaths


def load_config(path: str | Path = "config.yaml") -> Settings:
    """Load and validate settings from a YAML file."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Settings(**raw)
