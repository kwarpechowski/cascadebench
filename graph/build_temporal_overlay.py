# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""Synthetic TEMPORAL overlay — per-twin activity rhythm (Phase B1, for H8b).

The static multiplex (contact + OSINT) is BLIND to a compromised in-graph account: a
taken-over known contact passes every static layer (it really is a known sender). The
signal is *dynamic* — the malicious email arrives OFF the sender's usual rhythm. This
overlay gives each twin a deterministic activity profile (a subset of time buckets it is
normally active in) so an experiment can:
  * place a benign email IN the sender's active buckets (temporally consistent), and
  * place a compromised email OFF them (the attacker operates off-pattern),
and test whether a temporal-consistency feature recovers the AUC~0.51 blind spot.

Also exposes a CO-ACTIVITY layer (twins sharing active buckets) as a multiplex layer.

Deterministic (hash-seeded; no random/datetime). Pure stdlib. Mirrors the controlled
design of build_osint_overlay (tunable, ground-truth, ethics-free).
"""

from __future__ import annotations

import csv
import hashlib
from itertools import combinations
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
RESULTS = CODE / "results"

N_BUCKETS = 20            # e.g. 20 work time-of-week buckets
ACTIVE_PER_TWIN = 6       # each twin is normally active in 6 of them
COACTIVITY_MIN = 3        # twins sharing >=3 active buckets get a co-activity edge


def _h(*p: str) -> int:
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def active_buckets(twin: str, seed: int = 0) -> set[int]:
    """Deterministic set of time buckets a twin is normally active in."""
    out: set[int] = set()
    i = 0
    while len(out) < ACTIVE_PER_TWIN:
        out.add(_h(str(seed), twin, str(i)) % N_BUCKETS)
        i += 1
    return out


def is_consistent(twin: str, bucket: int, seed: int = 0) -> bool:
    """Does an email at `bucket` fit the sender's normal rhythm?"""
    return bucket in active_buckets(twin, seed)


def off_bucket(twin: str, salt: str, seed: int = 0) -> int:
    """Pick a bucket the twin is NOT normally active in (attacker off-pattern)."""
    act = active_buckets(twin, seed)
    for i in range(N_BUCKETS * 3):
        b = _h("off", str(seed), twin, salt, str(i)) % N_BUCKETS
        if b not in act:
            return b
    return (max(act) + 1) % N_BUCKETS


def in_bucket(twin: str, salt: str, seed: int = 0) -> int:
    act = sorted(active_buckets(twin, seed))
    return act[_h("in", str(seed), twin, salt) % len(act)]


def coactivity_edges(twins: list[str], seed: int = 0) -> set[tuple[str, str]]:
    buckets = {t: active_buckets(t, seed) for t in twins}
    edges: set[tuple[str, str]] = set()
    for a, b in combinations(sorted(twins), 2):
        if len(buckets[a] & buckets[b]) >= COACTIVITY_MIN:
            edges.add((a, b))
    return edges


def _twins() -> list[str]:
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        return sorted(r["twin_id"] for r in csv.DictReader(f))


def main() -> None:
    (RESULTS / "layers").mkdir(parents=True, exist_ok=True)
    twins = _twins()
    edges = coactivity_edges(twins)
    with (RESULTS / "layers" / "temporal_coactivity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["twin_a", "twin_b", "weight"])
        for a, b in sorted(edges):
            w.writerow([a, b, 0.5])
    deg = 2 * len(edges) / len(twins) if twins else 0.0
    print(f"[temporal] {len(twins)} twins | {N_BUCKETS} buckets, {ACTIVE_PER_TWIN} active/twin "
          f"| co-activity edges={len(edges)} (deg {deg:.1f})")


if __name__ == "__main__":
    main()
