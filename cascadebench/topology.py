# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""cascadebench.topology — generators for graph families with different structural properties
+ graph statistics. Serves the predictivity probe (does the detector ranking depend on topology).

Families (different degree skew / clustering):
  * ER  — Erdos-Renyi (low skew, low clustering)
  * BA  — Barabasi-Albert (high skew, scale-free)
  * WS  — Watts-Strogatz (high clustering, small-world)
Every node gets a rhythm 'home bucket' -> edges are timestamped (temporal detectors get a signal).
Deterministic generators (seeded). Write edgelists 'u v ts' for loading via Graph.from_edgelist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .graph import Graph, N_BUCKETS, _BUCKET_SECONDS


# ----------------------------- graph statistics -----------------------------
def gini(values) -> float:
    """Gini coefficient of a distribution (degree skew). 0 = equal, ~1 = extremely skewed."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def clustering(graph: Graph) -> float:
    """Global clustering coefficient (transitivity) — fraction of closed triangles."""
    nbr = graph.nbr
    triangles = 0
    triples = 0
    for u in graph.nodes:
        ns = list(nbr.get(u, ()))
        d = len(ns)
        if d < 2:
            continue
        triples += d * (d - 1) // 2
        ns_set = set(ns)
        for i in range(d):
            ni = nbr.get(ns[i], ())
            for j in range(i + 1, d):
                if ns[j] in ni:
                    triangles += 1
    return float(triangles / triples) if triples else 0.0


def graph_stats(graph: Graph) -> Dict[str, float]:
    """Returns key structural graph statistics (candidate predictors of rank transfer)."""
    degs = [len(graph.nbr.get(u, ())) for u in graph.nodes]
    n = len(graph.nodes)
    m = sum(degs) / 2
    return {
        "n": float(n),
        "mean_deg": float(np.mean(degs)) if degs else 0.0,
        "deg_gini": gini(degs),
        "clustering": clustering(graph),
        "density": float(2 * m / (n * (n - 1))) if n > 1 else 0.0,
    }


# ----------------------------- family generators -----------------------------
def _write_edgelist(path: Path, edges, rng) -> Path:
    """Writes edges with timestamps. Every node has a rhythm 'home bucket'.
    Each edge is emitted 2x in the sender's home bucket (count>=2 => rhythm in from_edgelist)."""
    nodes = sorted({u for e in edges for u in e[:2]})
    home = {u: int(rng.integers(0, N_BUCKETS)) for u in nodes}
    lines = []
    for u, v in edges:
        for src in (u, v):                       # both sides get a rhythm
            ts = home[src] * _BUCKET_SECONDS + 100
            lines.append(f"{src} {v if src == u else u} {ts}")
            lines.append(f"{src} {v if src == u else u} {ts + 1}")
    path.write_text("\n".join(lines))
    return path


def er_edgelist(path, n=600, mean_deg=8, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    m = int(n * mean_deg / 2)
    edges = set()
    while len(edges) < m:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            edges.add((min(a, b), max(a, b)))
    return _write_edgelist(Path(path), edges, rng)


def ba_edgelist(path, n=600, m_attach=4, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    targets = list(range(m_attach))
    repeated = list(range(m_attach))             # list for preferential attachment
    edges = set()
    for new in range(m_attach, n):
        chosen = set()
        while len(chosen) < m_attach:
            chosen.add(int(repeated[int(rng.integers(0, len(repeated)))]))
        for t in chosen:
            edges.add((min(new, t), max(new, t)))
        repeated.extend(chosen); repeated.extend([new] * m_attach)
    return _write_edgelist(Path(path), edges, rng)


def ws_edgelist(path, n=600, k=8, beta=0.3, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    edges = set()
    half = k // 2
    for u in range(n):
        for j in range(1, half + 1):
            v = (u + j) % n
            if rng.random() < beta:              # rewiring
                w = int(rng.integers(0, n))
                if w != u:
                    v = w
            if u != v:
                edges.add((min(u, v), max(u, v)))
    return _write_edgelist(Path(path), edges, rng)


def sbm_edgelist(path, n=600, k=6, mean_deg=8, f_intra=0.8, seed=0) -> Path:
    """Stochastic Block Model with CONTROLLABLE modularity. k equal-size communities;
    fraction f_intra of edges falls WITHIN a community (high f_intra -> high modularity).
    Node labels = int (community = node // (n//k)) — allows the partition to be reconstructed."""
    rng = np.random.default_rng(seed)
    size = n // k
    comm = {i: min(i // size, k - 1) for i in range(n)}
    members = {c: [i for i in range(n) if comm[i] == c] for c in range(k)}
    m = int(n * mean_deg / 2)
    edges = set()
    tries = 0
    while len(edges) < m and tries < m * 10:
        tries += 1
        if rng.random() < f_intra:                      # edge within a community
            c = int(rng.integers(k)); grp = members[c]
            a, b = grp[int(rng.integers(len(grp)))], grp[int(rng.integers(len(grp)))]
        else:                                            # edge between communities
            a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b:
            edges.add((min(a, b), max(a, b)))
    return _write_edgelist(Path(path), edges, rng)


def planted_clique_edgelist(path, n=600, k=6, p_clique=0.5, inter=0.02, seed=0) -> Path:
    """Planted-partition with DENSE (clique-like) communities: every pair within a community is
    connected with probability p_clique (high -> high clustering AND modularity), plus sparse edges
    between communities (probability inter). High p_clique mirrors the 'department' structure of the org generator.
    Labels = int (community = node // (n//k))."""
    rng = np.random.default_rng(seed)
    size = n // k
    comm = {i: min(i // size, k - 1) for i in range(n)}
    members = {c: [i for i in range(n) if comm[i] == c] for c in range(k)}
    edges = set()
    for c in range(k):                                   # dense core within a community
        grp = members[c]
        for ii in range(len(grp)):
            for jj in range(ii + 1, len(grp)):
                if rng.random() < p_clique:
                    edges.add((grp[ii], grp[jj]))
    n_inter = int(inter * n * (k - 1) * size / 2)        # sparse edges between communities
    for _ in range(max(1, n_inter)):
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b and comm[a] != comm[b]:
            edges.add((min(a, b), max(a, b)))
    return _write_edgelist(Path(path), edges, rng)


def modularity(graph: Graph, k: int, n: int) -> float:
    """Newman modularity for a block partition (community = int(node) // (n//k))."""
    size = max(1, n // k)
    m = sum(len(graph.nbr[u]) for u in graph.nodes) / 2
    if m == 0:
        return 0.0
    deg = {u: len(graph.nbr[u]) for u in graph.nodes}
    def comm(u):
        return min(int(u) // size, k - 1)
    e_in = {}; a = {}
    for u in graph.nodes:
        cu = comm(u)
        a[cu] = a.get(cu, 0) + deg[u]
        for v in graph.nbr[u]:
            if comm(v) == cu:
                e_in[cu] = e_in.get(cu, 0) + 1            # counts both sides -> /2 below
    q = 0.0
    for c in set(comm(u) for u in graph.nodes):
        q += (e_in.get(c, 0) / 2) / m - (a.get(c, 0) / (2 * m)) ** 2
    return float(q)


def topology_pool(tmpdir, seed: int = 0, core_cap: int = 1500):
    """Diverse topology pool: synthetic org (ref) + ER/BA/WS (different skews) + real SNAP graphs.
    Returns a list of (name, Graph, kind) where kind in {'org','rand','real'} (for sensitivity analysis)."""
    tmp = Path(tmpdir); tmp.mkdir(parents=True, exist_ok=True)
    pool = [
        ("org-400", Graph.synthetic(400), "org"),
        ("org-800", Graph.synthetic(800), "org"),
        ("ER-k6", Graph.from_edgelist(er_edgelist(tmp / "er6.txt", 600, 6, seed), name="ER-k6"), "rand"),
        ("ER-k12", Graph.from_edgelist(er_edgelist(tmp / "er12.txt", 600, 12, seed), name="ER-k12"), "rand"),
        ("BA-m2", Graph.from_edgelist(ba_edgelist(tmp / "ba2.txt", 600, 2, seed), name="BA-m2"), "rand"),
        ("BA-m5", Graph.from_edgelist(ba_edgelist(tmp / "ba5.txt", 600, 5, seed), name="BA-m5"), "rand"),
        ("WS-b03", Graph.from_edgelist(ws_edgelist(tmp / "ws.txt", 600, 8, 0.3, seed), name="WS-b03"), "rand"),
    ]
    # real topologies (if downloaded)
    from .graph import REAL_GRAPHS
    for name, p in REAL_GRAPHS.items():
        if Path(p).exists():
            pool.append((name, Graph.from_edgelist(p, min_deg=2, core_cap=core_cap, name=name), "real"))
    return pool
