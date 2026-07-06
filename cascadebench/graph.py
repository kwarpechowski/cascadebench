# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""cascadebench.graph — communication graph abstraction (synthetic + real topologies).

A single `Graph` class unifies:
  * a procedural synthetic graph (organizations, arbitrary N) — controllable, no LLM,
  * real topologies from timestamped edge lists (SNAP: email-Eu-core, CollegeMsg),
  * the real Enron graph from headers (optionally).
The graph exposes adjacency and — when available — the REAL sender activity rhythm (from time).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent

N_BUCKETS = 28                      # 7 days x 4 blocks of 6h (weekly rhythm)
_BUCKET_SECONDS = 21600


class Graph:
    """Contact graph. `nbr`: node -> set of contacts. `rhythm`: node -> active buckets (optional)."""

    def __init__(self, nbr: dict, rhythm: Optional[dict] = None, name: str = "graph",
                 n_buckets: int = N_BUCKETS):
        self.nbr = {u: set(vs) for u, vs in nbr.items() if vs}
        self.rhythm = rhythm or {}
        self.name = name
        self.n_buckets = n_buckets
        self.nodes = sorted(self.nbr)
        self.index = {u: i for i, u in enumerate(self.nodes)}

    def __len__(self) -> int:
        return len(self.nodes)

    def neighbors(self, u) -> set:
        return self.nbr.get(u, set())

    def edges(self) -> list:
        return [(u, w) for u in self.nbr for w in self.nbr[u]]

    def active_buckets(self, u) -> Optional[set]:
        return self.rhythm.get(u)

    # ----------------------------- constructors ------------------------------
    @classmethod
    def synthetic(cls, n_twins: int = 1600) -> "Graph":
        """Procedural organization graph (controllable, arbitrary N, no LLM)."""
        import sys
        sys.path.insert(0, str(_CODE))
        from data.org_graph import build_structural, contacts
        nodes = build_structural(n_twins)
        nbr = defaultdict(set)
        for tid, node in nodes.items():
            for c in contacts(node):
                nbr[tid].add(c); nbr[c].add(tid)
        return cls(nbr, name=f"synthetic-{n_twins}")

    @classmethod
    def from_edgelist(cls, path, min_deg: int = 2, core_cap: int = 2000,
                      name: Optional[str] = None) -> "Graph":
        """Real topology from an edge list 'src dst [timestamp]'. Derives a real rhythm when a
        timestamp is present. Trims to the active core (degree >= min_deg, top-K)."""
        path = Path(path)
        nbr_full = defaultdict(set)
        sent = defaultdict(list)
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) < 2:
                continue
            u, v = p[0], p[1]
            if u == v:
                continue
            nbr_full[u].add(v); nbr_full[v].add(u)
            if len(p) >= 3:
                try:
                    sent[u].append((int(p[2]) // _BUCKET_SECONDS) % N_BUCKETS)
                except ValueError:
                    pass
        deg = {a: len(s) for a, s in nbr_full.items() if len(s) >= min_deg}
        core = set(sorted(deg, key=deg.get, reverse=True)[:core_cap])
        nbr = {a: (nbr_full[a] & core) for a in core}
        rhythm = {}
        for a in core:
            if sent.get(a):
                cnt = Counter(sent[a])
                rhythm[a] = {b for b, c in cnt.items() if c >= 2}
        return cls(nbr, rhythm, name=name or path.stem)

    @classmethod
    def enron(cls, max_users: int = 40, core_cap: int = 2000) -> "Graph":
        """Real Enron contact graph from From->To headers + rhythm from Date."""
        import sys
        sys.path.insert(0, str(_CODE / "experiments"))
        import exp_enron_multiplex as EM
        EM.MAX_USERS = max_users
        EM.ENRON = _CODE.parents[1] / "personalized-phishing-defense/code/data/enron/maildir"
        sent, active, contact, _corecip, _domain = EM.parse()
        nbr_full = defaultdict(set)
        for s, v in sent.items():
            for r, _b in v:
                nbr_full[s].add(r); nbr_full[r].add(s)
        deg = {a: len(s) for a, s in nbr_full.items() if len(s) >= 2}
        core = set(sorted(deg, key=deg.get, reverse=True)[:core_cap])
        nbr = {a: (nbr_full[a] & core) for a in core}
        rhythm = {a: active[a] for a in core if a in active}
        return cls(nbr, rhythm, name="enron")


def _synth_rhythm():
    import sys
    sys.path.insert(0, str(_CODE))
    from graph.build_temporal_overlay import is_consistent, in_bucket
    return is_consistent, in_bucket


def in_rhythm(graph: "Graph", node, bucket: int, seed: int) -> bool:
    """Whether `node` is in rhythm at `bucket`. Real rhythm if available, else synthetic."""
    act = graph.active_buckets(node)
    if act is not None:
        return bucket in act
    is_consistent, _ = _synth_rhythm()
    return bool(is_consistent(node, bucket, seed))


def rhythm_bucket(graph: "Graph", node, salt: str, seed: int) -> int:
    """Returns a bucket IN `node`'s rhythm (for mimicry). Real rhythm if available, else synthetic."""
    act = graph.active_buckets(node)
    if act:
        a = sorted(act)
        return a[int.from_bytes(salt.encode()[:4] or b"\0", "little") % len(a)]
    _, in_bucket = _synth_rhythm()
    return int(in_bucket(node, salt, seed))


# registry of standard testbed graphs
REAL_GRAPHS = {
    "email-Eu-core": _CODE / "data" / "realgraphs" / "eu.txt",
    "CollegeMsg": _CODE / "data" / "realgraphs" / "college.txt",
    "sx-mathoverflow": _CODE / "data" / "realgraphs" / "sx-mathoverflow.txt",
    "sx-askubuntu": _CODE / "data" / "realgraphs" / "sx-askubuntu.txt",
    "sx-superuser": _CODE / "data" / "realgraphs" / "sx-superuser.txt",
}


def load(spec: str) -> Graph:
    """Factory: 'synthetic[:N]' | 'enron' | a real-graph name | path to an edgelist."""
    if spec.startswith("synthetic"):
        n = int(spec.split(":")[1]) if ":" in spec else 1600
        return Graph.synthetic(n)
    if spec == "enron":
        return Graph.enron()
    if spec in REAL_GRAPHS:
        return Graph.from_edgelist(REAL_GRAPHS[spec], name=spec)
    return Graph.from_edgelist(spec)
