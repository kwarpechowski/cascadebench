# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""cascadebench.attack — attacker and scenarios (composable evasion strategies).

Threat model: lateral phishing as a propagation CASCADE. The attacker strategy separates the
REACH knobs (fan-out K) from the EVASION knobs (time spreading g, rhythm mimicry, fabrication),
which enables a Pareto front and adaptive attacks in problem-space (Pierazzi et al.).

Event: a tuple (sender, receiver, bucket, label). Scenario = attacks + leak-aware benign traffic.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .graph import Graph, rhythm_bucket

Event = Tuple[str, str, int, int]


@dataclass
class CascadeStrategy:
    """Attacker strategy. fanout = REACH; spread/mimicry/fabrication = EVASION."""
    n_cascades: int = 30
    max_hops: int = 4
    p_infect: float = 0.6
    fanout: int = 8                 # K — contacts infected per carrier (reach)
    spread: int = 1                 # g — bucket gap between sends (temporal evasion)
    mimicry: float = 0.0            # fraction of sends in the sender's rhythm (rhythm evasion)
    fabrication: float = 0.0        # fraction of fabricated edges (placeholder/extension)

    def label(self) -> str:
        return (f"K{self.fanout}-g{self.spread}"
                + (f"-m{self.mimicry}" if self.mimicry else "")
                + (f"-f{self.fabrication}" if self.fabrication else ""))


@dataclass
class Scenario:
    graph: Graph
    events: List[Event]
    n_attacks: int
    infected_frac: float
    strategy: CascadeStrategy = field(default=None)


def cascade(graph: Graph, strat: CascadeStrategy, seed: int) -> Tuple[List[Event], int]:
    """Generates attack events (label 1) per the strategy. Returns (events, #infected)."""
    rng = np.random.default_rng(seed)
    nodes = graph.nodes
    nb = graph.n_buckets
    events: List[Event] = []
    infected_all = set()
    seeds0 = [nodes[i] for i in rng.permutation(len(nodes))[:strat.n_cascades]]
    for ci, s0 in enumerate(seeds0):
        t0 = int(rng.integers(0, nb))
        infected = {s0}; frontier = [(s0, t0)]
        for _hop in range(strat.max_hops):
            nxt = []
            for (u, tu) in frontier:
                real = sorted(graph.neighbors(u)); rng.shuffle(real); real = [v for v in real if v not in infected]
                sent = 0; attempts = 0
                while sent < strat.fanout and attempts < strat.fanout * 4 + 4:
                    attempts += 1
                    # EVASION via fabrication: sending to a NON-contact (an OSINT-spoofed contact,
                    # outside the known graph) instead of a real neighbor.
                    if strat.fabrication and rng.random() < strat.fabrication:
                        v = nodes[int(rng.integers(len(nodes)))]
                        if v == u or v in infected or v in graph.neighbors(u):
                            continue
                    elif real:
                        v = real.pop()
                    elif strat.fabrication:
                        continue            # no real ones left — try fabrication on the next iteration
                    else:
                        break
                    bucket = (tu + 1 + sent * strat.spread) % nb
                    if strat.mimicry and rng.random() < strat.mimicry:
                        bucket = rhythm_bucket(graph, u, f"{ci}:{u}:{v}", seed)
                    events.append((u, v, bucket, 1)); sent += 1
                    if rng.random() < strat.p_infect:
                        infected.add(v); nxt.append((v, bucket))
            frontier = nxt
            if not frontier:
                break
        infected_all |= infected
    return events, len(infected_all)


def benign_traffic(graph: Graph, n_events: int, seed: int, off_hours: float = 0.0,
                   matched: bool = True) -> List[Event]:
    """Legitimate traffic (label 0). matched=True: random contact edges (same distribution as the
    attack) — removes the degree/identity confound. off_hours: fraction outside rhythm (leak-aware)."""
    rng = np.random.default_rng(seed + 777)
    nb = graph.n_buckets
    edges = graph.edges()
    out: List[Event] = []
    for _ in range(n_events):
        u, w = edges[int(rng.integers(len(edges)))]
        b = int(rng.integers(nb))
        # off-hours: with probability off_hours, place benign traffic OUTSIDE the sender's rhythm
        # (rhythm-leak control)
        if off_hours and rng.random() < off_hours:
            act = graph.active_buckets(u)
            if act:
                off = sorted(set(range(nb)) - act) or [b]
                b = off[int(rng.integers(len(off)))]
        out.append((u, w, b, 0))
    return out


def build_scenario(graph: Graph, strat: CascadeStrategy, seed: int,
                   benign_ratio: int = 3, off_hours: float = 0.0) -> Scenario:
    """Full scenario: attacks (cascade) + leak-aware benign traffic, shuffled."""
    atk, n_inf = cascade(graph, strat, seed)
    ben = benign_traffic(graph, benign_ratio * max(1, len(atk)), seed, off_hours=off_hours)
    ev = atk + ben
    np.random.default_rng(seed + 13).shuffle(ev)
    return Scenario(graph, ev, len(atk), n_inf / max(1, len(graph)), strat)
