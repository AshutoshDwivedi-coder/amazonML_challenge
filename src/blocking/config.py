# Blocking configuration defaults and YAML loader.
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BlockingConfig:
    # Frequency caps (block skipped when key freq exceeds cap)
    max_name_frequency: int = 200
    max_address_frequency: int = 200
    max_name_address_frequency: int = 500
    max_postal_frequency: int = 5000
    max_postal_name_prefix_frequency: int = 200
    max_postal_house_frequency: int = 200
    max_house_name_frequency: int = 200
    max_prefix_frequency: int = 500
    max_token_frequency: int = 5000
    max_rare_token_block_size: int = 200
    max_address_token_block_size: int = 200
    max_ngram_block_size: int = 100

    # Rare token thresholds
    rare_token_min_frequency: int = 2
    rare_token_max_frequency: int = 50
    rare_token_min_length: int = 4
    max_rare_tokens_per_record: int = 3

    # Prefix / n-gram / fuzzy
    name_prefix_lengths: list[int] = field(default_factory=lambda: [4, 6, 8])
    char_ngram_sizes: list[int] = field(default_factory=lambda: [3, 4, 5])
    char_ngram_top_k: int = 30
    char_ngram_min_shared: int = 2
    rapidfuzz_enabled: bool = True
    rapidfuzz_top_k: int = 20
    rapidfuzz_score_cutoff: int = 85
    rapidfuzz_only_unresolved: bool = True
    rapidfuzz_max_pool: int = 500

    # Processing
    enabled_blocks: list[str] = field(
        default_factory=lambda: [
            "exact_name",
            "exact_address",
            "exact_name_address",
            "postal",
            "house_number_name",
            "rare_name_token",
            "rare_address_token",
            "name_prefix",
            "address_token",
            "char_ngram",
            "rapidfuzz",
        ]
    )
    chunk_size: int = 200_000
    work_dir: str = "outputs/work"
    random_seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockingConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BlockingConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> BlockingConfig:
    if path is None:
        return BlockingConfig()
    return BlockingConfig.from_yaml(path)
