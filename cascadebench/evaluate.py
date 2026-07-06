# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""cascadebench.evaluate — leak-aware evaluation (the testbed's methodological core).

Built-in principles (a hard lesson from Publications 1-2): victim/organization-level split,
operational metrics (Recall@low-FPR), causal control (time shuffling), and rhythm-leak
(off-hours) auditing. This is what sets the testbed apart from existing benchmarks.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .attack import Scenario, Event

FPRS = [0.001, 0.005, 0.01, 0.05]


def recall_at_fpr(y, s, t: float = 0.01) -> float:
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y, s)
    m = fpr <= t
    return float(tpr[m].max()) if m.any() else 0.0


def victim_split(events: List[Event], seed: int, frac: float = 1 / 3):
    vics = sorted({e[1] for e in events})
    rng = np.random.default_rng(seed)
    te_v = set(np.array(vics)[rng.permutation(len(vics))[:max(1, int(len(vics) * frac))]])
    te = np.array([e[1] in te_v for e in events]); return ~te, te


def shuffle_time(sc: Scenario, seed: int) -> Scenario:
    """Causal control: same events/features, random bucket order -> destroys structure."""
    rng = np.random.default_rng(1000 + seed)
    ev = [(s, v, int(rng.integers(sc.graph.n_buckets)), l) for (s, v, _b, l) in sc.events]
    return Scenario(sc.graph, ev, sc.n_attacks, sc.infected_frac, sc.strategy)


def evaluate(sc: Scenario, detectors: List, seed: int, fprs=FPRS) -> Dict[str, dict]:
    """Train+evaluate each detector on the scenario (victim-level split). Returns metrics."""
    tr, te = victim_split(sc.events, seed)
    yte = np.array([e[3] for e in sc.events], dtype=np.float32)[te]
    out = {}
    for det in detectors:
        sc_score = det.fit_score(sc, tr, te, seed)
        out[det.name] = {"auc": float(roc_auc_score(yte, sc_score)),
                         **{f"recall@{f}": recall_at_fpr(yte, sc_score, f) for f in fprs}}
    return out


def aggregate(per_seed: List[Dict[str, dict]]) -> Dict[str, dict]:
    """Average metrics across seeds."""
    keys = per_seed[0]
    agg = {}
    for det in keys:
        agg[det] = {m: float(np.mean([ps[det][m] for ps in per_seed])) for m in keys[det]}
    return agg
