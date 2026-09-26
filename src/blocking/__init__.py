"""Ashutosh blocking / candidate-generation package.

Public API
----------
- ``CandidateGenerator`` — class interface
- ``generate_candidates(...)`` — functional interface for Sudhanshu
- ``load_config`` / ``BlockingConfig`` — configuration
"""

from .candidate_generator import CandidateGenerator, generate_candidates
from .config import BlockingConfig, load_config

__all__ = [
    "BlockingConfig",
    "CandidateGenerator",
    "generate_candidates",
    "load_config",
]
