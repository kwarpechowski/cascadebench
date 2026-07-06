# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""Controlled synthetic OSINT overlay — tunable public-footprint layers (H9 design).

The LLM-generated C4/C5 attributes proved too thin/correlated to test the multiplex
hypothesis (the structural layers contact/org/domain collapse at >0.9 Jaccard, and
oss/publication layers are near-empty). Per the pre-registered H9 design, we instead
*assign* each twin a SYNTHETIC public-footprint over a taxonomy of OSINT communities
with KNOWN, TUNABLE properties — so the resulting layers have controllable density,
cross-org overlap and fabrication rate. This both (a) gives independent layers to fuse
(de-correlating the multiplex) and (b) realises the H9a-d sweeps directly.

Mechanism (deterministic, hash-seeded — no random/datetime, reproducible):
  For each twin, for each OSINT category, draw ``memberships`` community ids:
    * with prob ``p_cross`` the community is GLOBAL (shared across orgs)  -> cross-org edge
      else it is ORG-LOCAL (only same-org twins)                          -> intra-org edge
    * with prob ``fabrication_rate`` the membership is FABRICATED (spurious) -> a fake tie an
      attacker plants to look connected (flagged so experiments can sweep H9d crossover).
  Two twins sharing a community get an edge in that category's layer; an edge is
  ``genuine`` iff BOTH memberships are genuine.

Categories (confidence weight = H9c hard-vs-soft):
  osint_mailing_list  w=0.5   |  osint_oss_project w=0.7 (hard)  |  osint_affiliation w=0.6

Outputs (under results/layers/ and results/osint_overlay/):
  osint_<cat>.csv             : twin_a, twin_b, weight, genuine   (edge list per layer)
  osint_overlay/membership.csv: twin_id, category, community, genuine
  osint_overlay/overlay_stats.csv

API for experiments: ``build_overlay(seed, p_cross, fabrication_rate)`` returns
``{category: {(a,b): genuine_bool}}`` so #GP-EXP-6 can sweep p_cross / fabrication_rate.

Pure stdlib; deterministic given (seed, twin id).
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from itertools import combinations
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
RESULTS = CODE / "results"
LAYERS_DIR = RESULTS / "layers"
OVERLAY_DIR = RESULTS / "osint_overlay"

# category -> (n_global_communities, n_local_per_org, memberships_per_twin, weight)
# Seven NAMED OSINT types with varying density/weight; the selection experiment
# (exp_layer_selection) shows which are essential and which are dispensable
# (e.g. the weak social layer, cf. Dewan).
CATEGORIES = {
    "osint_mailing_list":  (20, 3, 2, 0.5),   # shared mailing list / discussion group
    "osint_oss_project":   (25, 2, 2, 0.7),   # repository co-authorship (hard fact)
    "osint_affiliation":   (15, 2, 2, 0.6),   # conference / association
    "osint_vendor":        (18, 2, 2, 0.6),   # shared SaaS vendor / system
    "osint_certification": (22, 1, 2, 0.4),   # shared professional certification
    "osint_event":         (16, 2, 2, 0.5),   # shared event participation
    "osint_social":        (30, 1, 3, 0.3),   # loose social tie (WEAK, Dewan-negative)
}
P_CROSS_DEFAULT = 0.5          # prob a membership is in a global (cross-org) community
FABRICATION_DEFAULT = 0.0      # fraction of fabricated memberships (swept for H9d)


def _h(*p: str) -> int:
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _self_table() -> dict[str, str]:
    org: dict[str, str] = {}
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            org[r["twin_id"]] = r["org"]
    return org


def _memberships(twin: str, org: str, cat: str, spec, seed: int,
                 p_cross: float, fab: float) -> list[tuple[str, bool]]:
    n_glob, n_loc, mem, _w = spec
    out: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for slot in range(mem):
        h = _h(str(seed), twin, cat, str(slot))
        genuine = ((h % 1000) >= int(fab * 1000))
        if ((h // 7) % 1000) < int(p_cross * 1000):
            comm = f"{cat}:G{h % n_glob}"                 # global -> cross-org possible
        else:
            comm = f"{cat}:{org}:L{h % n_loc}"            # org-local
        if comm not in seen:
            seen.add(comm)
            out.append((comm, genuine))
    return out


def build_overlay(seed: int = 0, p_cross: float = P_CROSS_DEFAULT,
                  fabrication_rate: float = FABRICATION_DEFAULT):
    """Return ({category: {(a,b): genuine}}, membership_rows)."""
    org = _self_table()
    twins = sorted(org)
    # category -> community -> [(twin, genuine)]
    comm_members: dict[str, dict[str, list[tuple[str, bool]]]] = {c: defaultdict(list) for c in CATEGORIES}
    membership_rows: list[tuple[str, str, str, int]] = []
    for t in twins:
        for cat, spec in CATEGORIES.items():
            for comm, genuine in _memberships(t, org[t], cat, spec, seed, p_cross, fabrication_rate):
                comm_members[cat][comm].append((t, genuine))
                membership_rows.append((t, cat, comm, int(genuine)))

    layers: dict[str, dict[tuple[str, str], bool]] = {}
    for cat in CATEGORIES:
        edges: dict[tuple[str, str], bool] = {}
        for members in comm_members[cat].values():
            for (a, ga), (b, gb) in combinations(sorted(members), 2):
                if a == b:
                    continue
                key = (a, b)
                gen = ga and gb
                edges[key] = edges.get(key, False) or gen   # genuine if any shared community is genuine
        layers[cat] = edges
    return layers, membership_rows


def _cross_org_frac(edges, org) -> float:
    if not edges:
        return 0.0
    return sum(1 for a, b in edges if org[a] != org[b]) / len(edges)


def main() -> None:
    LAYERS_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    org = _self_table()
    n = len(org)
    layers, membership_rows = build_overlay()

    with (OVERLAY_DIR / "membership.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["twin_id", "category", "community", "genuine"])
        w.writerows(membership_rows)

    for cat, edges in layers.items():
        weight = CATEGORIES[cat][3]
        with (LAYERS_DIR / f"{cat}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["twin_a", "twin_b", "weight", "genuine"])
            for (a, b), gen in sorted(edges.items()):
                w.writerow([a, b, weight, int(gen)])

    with (OVERLAY_DIR / "overlay_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["category", "n_edges", "mean_degree", "cross_org_frac", "genuine_frac"])
        for cat, edges in layers.items():
            gen = sum(1 for v in edges.values() if v)
            w.writerow([cat, len(edges), round(2 * len(edges) / n, 2),
                        round(_cross_org_frac(edges, org), 2),
                        round(gen / len(edges), 2) if edges else 0.0])

    print(f"[osint-overlay] {n} twins | p_cross={P_CROSS_DEFAULT} fab={FABRICATION_DEFAULT}")
    print(f"{'category':20s} {'edges':>6s} {'deg':>6s} {'cross-org':>9s} {'genuine':>8s}")
    for cat, edges in layers.items():
        gen = (sum(1 for v in edges.values() if v) / len(edges)) if edges else 0.0
        print(f"{cat:20s} {len(edges):6d} {2*len(edges)/n:6.1f} {_cross_org_frac(edges, org):9.2f} {gen:8.2f}")
    # quick cross-org overlap demo: how many cross-org edges these layers add over contact
    print("\n[note] cross-org OSINT edges are the new-sender BRIDGES (H9a). "
          "Sweep p_cross / fabrication_rate via build_overlay(...) in #GP-EXP-6.")


if __name__ == "__main__":
    main()
