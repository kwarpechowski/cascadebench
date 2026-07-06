# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""cascadebench — a controlled, leak-aware adversarial testbed for graph-based phishing
and lateral movement detectors.

Example:
    import cascadebench as cb
    g = cb.Graph.synthetic(800)
    sc = cb.build_scenario(g, cb.CascadeStrategy(fanout=8, spread=2), seed=0)
    res = cb.evaluate(sc, cb.get_panel(), seed=0)

Components: Graph (graph.py) | Attacker/Scenario (attack.py) | Detector panel (detect.py) |
leak-aware Evaluator (evaluate.py). Extensible: a new detector = a Detector subclass.
"""
from .graph import Graph, load, in_rhythm, rhythm_bucket, REAL_GRAPHS, N_BUCKETS
from .attack import CascadeStrategy, Scenario, cascade, benign_traffic, build_scenario
from .detect import (Detector, OneHop, COMPA, Hopper, HandContext, StaticGNN, TemporalGNN,
                     TemporalGNNAttn, AnomalyForest, PANEL, get_panel)
from .evaluate import (evaluate, aggregate, victim_split, shuffle_time, recall_at_fpr, FPRS)
from .search import (attack_search, defense_search, arms_race, transferability,
                     evaluate_strategy, SPACE)

__version__ = "0.1.0"
__all__ = [
    "Graph", "load", "in_rhythm", "rhythm_bucket", "REAL_GRAPHS", "N_BUCKETS",
    "CascadeStrategy", "Scenario", "cascade", "benign_traffic", "build_scenario",
    "Detector", "OneHop", "COMPA", "Hopper", "HandContext", "StaticGNN", "TemporalGNN",
    "TemporalGNNAttn", "AnomalyForest", "PANEL", "get_panel",
    "evaluate", "aggregate", "victim_split", "shuffle_time", "recall_at_fpr", "FPRS",
    "attack_search", "defense_search", "arms_race", "evaluate_strategy", "SPACE",
]
