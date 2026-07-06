# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""cascadebench.features — feature extraction from the event stream (parametrized by the graph).

Shared across detectors: 1-hop features, cascade context (burst/recency/propagation), volume
features (COMPA), node structural features. Rhythm computed via graph.in_rhythm (real or synthetic).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import List

import numpy as np

from .graph import Graph, in_rhythm

WINDOW = 2


def index_traffic(events):
    recv = defaultdict(set); sent = defaultdict(set)
    for s, v, b, _l in events:
        recv[b].add(v); sent[b].add(s)
    return recv, sent


def onehop(graph: Graph, events: List, seed: int) -> np.ndarray:
    X = []
    for s, v, b, _l in events:
        cs = graph.neighbors(v)
        X.append([1.0 if s in cs else 0.0,
                  1.0 if in_rhythm(graph, s, b, seed) else 0.0,
                  float(len(cs)), float(len(graph.neighbors(s)))])
    return np.asarray(X, np.float32)


def context(graph: Graph, events: List, seed: int) -> np.ndarray:
    recv, sent = index_traffic(events)
    X = []
    for s, v, b, _l in events:
        cs = graph.neighbors(v); deg_v = max(1, len(cs))
        f1 = [1.0 if s in cs else 0.0, 1.0 if in_rhythm(graph, s, b, seed) else 0.0,
              float(len(cs)), float(len(graph.neighbors(s)))]
        recent = set()
        for bb in range(max(0, b - WINDOW), b + 1):
            recent |= recv.get(bb, set())
        burst = len((cs - {s}) & recent) / deg_v
        recency = 0.0
        for bb in range(max(0, b - WINDOW), b):
            if s in recv.get(bb, set()):
                recency = 1.0; break
        prop = len((cs - {s}) & {x for bb in range(max(0, b - WINDOW), b + 1)
                                 for x in sent.get(bb, set())}) / deg_v
        X.append(f1 + [burst, recency, prop])
    return np.asarray(X, np.float32)


def compa(events: List) -> np.ndarray:
    pair = Counter((s, v) for s, v, _b, _l in events)
    sb = Counter((s, b) for s, _v, b, _l in events)
    st = Counter(s for s, _v, _b, _l in events)
    X = []
    for s, v, b, _l in events:
        X.append([1.0 / (1.0 + pair[(s, v)]), 1.0 / (1.0 + sb[(s, b)]), float(st[s])])
    return np.asarray(X, np.float32)


def hopper(graph: Graph, events: List, seed: int) -> np.ndarray:
    """Features in the spirit of Hopper (Ho et al. 2021): \emph{movement path} detection. An event
    is suspicious when the sender was recently REACHED (causal precursor --- part of the chain),
    MOVES with burst to a RARE receiver and WITHOUT a benign explanation (outside rhythm). Adapted
    to graph+time data --- email content (URL/lure from the original) is unavailable, so we encode
    the structural-temporal core of the Hopper signature."""
    recv, sent = index_traffic(events)
    dest_pop = Counter(v for _s, v, _b, _l in events)          # popularity of v as a receiver
    out_win = defaultdict(set)                                  # (s,b) -> set of targets in window (movement burst)
    for s, v, b, _l in events:
        for bb in range(b, b + WINDOW + 1):
            out_win[(s, bb)].add(v)
    X = []
    for s, v, b, _l in events:
        precursor = 0.0                                        # whether s received something in [b-W, b-1] (reached)
        for bb in range(max(0, b - WINDOW), b):
            if s in recv.get(bb, set()):
                precursor = 1.0; break
        fan = float(len(out_win.get((s, b), ())))              # sender's movement burst in the window
        rare_dest = 1.0 / (1.0 + dest_pop[v])                  # rare receiver (Hopper: rare target)
        unexplained = 0.0 if in_rhythm(graph, s, b, seed) else 1.0   # no benign explanation (outside rhythm)
        hop_sig = precursor * unexplained                      # signature: chain movement without explanation
        X.append([precursor, fan, rare_dest, unexplained, hop_sig])
    return np.asarray(X, np.float32)


def node_features(graph: Graph, events: List) -> np.ndarray:
    import torch
    n = len(graph); idx = graph.index
    indeg = Counter(v for _s, v, _b, _l in events)
    outdeg = Counter(s for s, _v, _b, _l in events)
    X = torch.zeros(n, 3)
    for u, i in idx.items():
        X[i, 0] = len(graph.neighbors(u)); X[i, 1] = indeg.get(u, 0); X[i, 2] = outdeg.get(u, 0)
    X = torch.log1p(X)
    return (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp(min=1e-6)
