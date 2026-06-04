"""
CrabQuant Loops — Autonomous optimization loops.

Each loop is a self-contained program (program.md + feature_map.yaml + sandbox helpers)
that runs the existing CrabQuant refinement pipeline with a specific objective.

Current loops:
  - diversity-explorer: Quality-Diversity optimization for portfolio coverage.
  - sharpe-optimizer: Single-metric Sharpe ratio optimization for near-miss strategies.
"""

from __future__ import annotations
