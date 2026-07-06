# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""Build the MULTIPLEX layers of the phishing graph (Phase A — no LLM, attribute layers).

Reads the 150 synthetic digital twins (``data/twins/*.json``) + the base contact graph
(``results/twin_network.csv``) and emits each relational LAYER as a weighted edge list
over the SAME node set (twin ids). Each layer is a "small graph"; overlaid they form the
multiplex on which #GP-EXP-6 / H8 / H9 are tested.

Layers (Phase A, buildable from existing data — see tech.md feasibility table):
  contact            : base sender<->recipient graph (twin_network.csv)            w=1.0
  org_hierarchy      : same company + manager/report edges (C1/C2)                  w=0.9
  shared_domain      : accounts sharing an email domain (C5/twin_self)             w=0.8
  osint_oss          : shared OSS project (C4.oss_projects) — hard OSINT fact       w=0.7
  osint_publication  : shared publication (C4.publications)                         w=0.7
  osint_conference   : co-attended conference (C4.conferences)                      w=0.6
  osint_platform     : shared platform/handle/website (C5)                          w=0.4
  interest_sim       : overlapping interests (C4.interests, Jaccard) — SOFT/weak    w=0.3

The OSINT layers (osint_*) are the *graph-constructor* edges of H9: a single shared
public fact instantiates a (often cross-org) edge that connects an otherwise-isolated
sender. Weights encode construction confidence (hard facts > soft interests), per H9c.

Outputs (under ``results/layers/``):
  <layer>.csv          : twin_a, twin_b, weight  (undirected, a<b)
  multiplex_edges.csv  : twin_a, twin_b, layer, weight  (all layers stacked)
  layer_stats.csv      : layer, n_edges, mean_degree, cross_org_frac
  layer_overlap.csv    : Jaccard edge-set overlap between every pair of layers
                         (pre-registered caveat: measure correlation BEFORE claiming gains)

Pure stdlib (json/csv/itertools/collections); deterministic; no LLM, no network.
Controls for the causal test live here too: ``shuffle_layer`` (edges to random twins)
and ``inject_fabricated`` (adversarial OSINT edges for H9d) — used by experiments.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent          # .../code
TWINS_DIR = CODE / "data" / "twins"
RESULTS = CODE / "results"
LAYERS_DIR = RESULTS / "layers"

# Per-layer construction-confidence weight (H9c: hard facts > soft interests).
LAYER_WEIGHTS = {
    "contact": 1.0, "org_hierarchy": 0.9, "shared_domain": 0.8,
    "osint_conference": 0.6, "osint_certification": 0.6, "osint_skill": 0.5,
    "osint_routine": 0.4, "osint_platform": 0.4, "osint_event": 0.4,
    "osint_pretext": 0.3, "interest_sim": 0.3,
    # too sparse (osint_oss=3, osint_publication=8 edges) or degenerate
    # (tech_stack/likely_senders ~ near-complete) -> outside the active set.
    "osint_oss": 0.7, "osint_publication": 0.7,
}
# Generic skills/tools shared by almost everyone -> degenerate near-complete subgraphs.
GENERIC_SKILLS = {"microsoft 365", "microsoft office", "office 365", "excel",
                  "microsoft excel", "powerpoint", "word", "outlook", "email",
                  "communication", "teamwork", "leadership", "project management"}
INTEREST_JACCARD_MIN = 0.34   # soft layer: connect only on real interest overlap

# Generic values that connect almost everyone -> degenerate, near-complete subgraphs
# (the first build showed protonmail/gmail and "LinkedIn"/"X" create meaningless edges).
# Filtering them is essential so a layer reflects a SPECIFIC shared fact, not ubiquity.
FREEMAIL = {"gmail.com", "googlemail.com", "yahoo.com", "hotmail.com", "outlook.com",
            "aol.com", "protonmail.com", "proton.me", "icloud.com", "gmx.com", "mail.com",
            "live.com", "msn.com", "yandex.com", "zoho.com", "fastmail.com"}
GENERIC_PLATFORMS = {"linkedin", "x (twitter)", "x", "twitter", "facebook", "instagram",
                     "youtube", "tiktok", "reddit", "bloomberg", "slack", "github",
                     "medium", "github.com", "google scholar", "researchgate", "orcid"}


def _norm(s) -> str:
    return " ".join(str(s).lower().split())


def _h(*p: str) -> int:
    return int(hashlib.sha256("::".join(p).encode()).hexdigest(), 16)


def _load_twins() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(TWINS_DIR.glob("*.json"))}


def _self_table() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["twin_id"]] = r
    return out


def _shared_attr_edges(twins: dict, code: str, field: str,
                       exclude: set[str] | None = None) -> set[tuple[str, str]]:
    """Edges between twins sharing >=1 value of c{code}.{field} (set incidence).

    ``exclude`` drops generic/ubiquitous values (e.g. "linkedin") that would otherwise
    wire a near-complete subgraph rather than reflect a SPECIFIC shared fact."""
    exclude = exclude or set()
    by_val: dict[str, set[str]] = defaultdict(set)
    for tid, d in twins.items():
        vals = d.get(code, {}).get(field) or []
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            nv = _norm(v)
            if nv and nv not in exclude:
                by_val[nv].add(tid)
    edges: set[tuple[str, str]] = set()
    for members in by_val.values():
        for a, b in combinations(sorted(members), 2):
            edges.add((a, b))
    return edges


def _domain_edges(self_tbl: dict, twins: dict) -> set[tuple[str, str]]:
    by_dom: dict[str, set[str]] = defaultdict(set)
    for tid, s in self_tbl.items():
        if s["domain"].lower() not in FREEMAIL:
            by_dom[s["domain"].lower()].add(tid)
        for e in twins.get(tid, {}).get("c5", {}).get("emails", []) or []:
            if "@" in e:
                dom = e.split("@")[-1].lower()
                if dom not in FREEMAIL:                # personal freemail != shared org
                    by_dom[dom].add(tid)
    edges: set[tuple[str, str]] = set()
    for members in by_dom.values():
        for a, b in combinations(sorted(members), 2):
            edges.add((a, b))
    return edges


def _org_hierarchy_edges(self_tbl: dict) -> set[tuple[str, str]]:
    """Same-company clique (small office); contact graph already encodes most of this."""
    by_org: dict[str, set[str]] = defaultdict(set)
    for tid, s in self_tbl.items():
        by_org[s["org"]].add(tid)
    edges: set[tuple[str, str]] = set()
    for members in by_org.values():
        for a, b in combinations(sorted(members), 2):
            edges.add((a, b))
    return edges


def _interest_edges(twins: dict) -> set[tuple[str, str]]:
    sets = {t: {_norm(x) for x in (d.get("c4", {}).get("interests") or [])}
            for t, d in twins.items()}
    edges: set[tuple[str, str]] = set()
    for a, b in combinations(sorted(sets), 2):
        sa, sb = sets[a], sets[b]
        if not sa or not sb:
            continue
        j = len(sa & sb) / len(sa | sb)
        if j >= INTEREST_JACCARD_MIN:
            edges.add((a, b))
    return edges


def _contact_edges() -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    self_addr = {}
    with (RESULTS / "twin_self.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            self_addr[r["address"]] = r["twin_id"]
    with (RESULTS / "twin_network.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = self_addr.get(r["contact_address"])
            if b:
                a, b2 = sorted((r["twin_id"], b))
                if a != b2:
                    edges.add((a, b2))
    return edges


def build_layers() -> dict[str, set[tuple[str, str]]]:
    twins, self_tbl = _load_twins(), _self_table()
    return {
        "contact": _contact_edges(),
        "org_hierarchy": _org_hierarchy_edges(self_tbl),
        "shared_domain": _domain_edges(self_tbl, twins),
        "osint_oss": _shared_attr_edges(twins, "c4", "oss_projects"),
        "osint_publication": _shared_attr_edges(twins, "c4", "publications"),
        "osint_conference": _shared_attr_edges(twins, "c4", "conferences"),
        "osint_certification": _shared_attr_edges(twins, "c3", "certifications"),
        "osint_skill": _shared_attr_edges(twins, "c3", "skills", GENERIC_SKILLS),
        "osint_routine": _shared_attr_edges(twins, "c8", "routine_communications"),
        "osint_event": _shared_attr_edges(twins, "c6", "upcoming_events"),
        "osint_pretext": _shared_attr_edges(twins, "c8", "plausible_pretexts"),
        "osint_platform": (_shared_attr_edges(twins, "c5", "platforms", GENERIC_PLATFORMS)
                           | _shared_attr_edges(twins, "c5", "social_handles")
                           | _shared_attr_edges(twins, "c5", "websites", GENERIC_PLATFORMS)),
        "interest_sim": _interest_edges(twins),
    }


# ---- controls used by experiments (#GP-EXP-6 / H9) -------------------------- #
def shuffle_layer(edges: set[tuple[str, str]], nodes: list[str], seed: int) -> set[tuple[str, str]]:
    """Re-wire a layer's edges onto RANDOM node pairs (causal control: must destroy gain)."""
    out: set[tuple[str, str]] = set()
    n = len(nodes)
    for i, _ in enumerate(sorted(edges)):
        h = _h(str(seed), str(i))
        a, b = nodes[h % n], nodes[(h // n) % n]
        if a != b:
            out.add(tuple(sorted((a, b))))
    return out


def inject_fabricated(edges: set[tuple[str, str]], nodes: list[str], rate: float,
                      seed: int) -> set[tuple[str, str]]:
    """Add a `rate` fraction of FAKE edges (H9d adversarial crossover sweep)."""
    out = set(edges)
    k = int(rate * len(edges))
    n = len(nodes)
    for i in range(k):
        h = _h("fab", str(seed), str(i))
        a, b = nodes[h % n], nodes[(h // n) % n]
        if a != b:
            out.add(tuple(sorted((a, b))))
    return out


def _cross_org_frac(edges: set[tuple[str, str]], self_tbl: dict) -> float:
    if not edges:
        return 0.0
    x = sum(1 for a, b in edges if self_tbl[a]["org"] != self_tbl[b]["org"])
    return x / len(edges)


def main() -> None:
    LAYERS_DIR.mkdir(parents=True, exist_ok=True)
    self_tbl = _self_table()
    nodes = sorted(self_tbl)
    layers = build_layers()

    # per-layer csv + stacked multiplex
    with (LAYERS_DIR / "multiplex_edges.csv").open("w", newline="", encoding="utf-8") as mf:
        mw = csv.writer(mf); mw.writerow(["twin_a", "twin_b", "layer", "weight"])
        for name, edges in layers.items():
            w = LAYER_WEIGHTS[name]
            with (LAYERS_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                lw = csv.writer(f); lw.writerow(["twin_a", "twin_b", "weight"])
                for a, b in sorted(edges):
                    lw.writerow([a, b, w]); mw.writerow([a, b, name, w])

    # stats
    deg = lambda E: (2 * len(E) / len(nodes)) if nodes else 0.0
    with (LAYERS_DIR / "layer_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["layer", "n_edges", "mean_degree", "cross_org_frac", "weight"])
        for name, edges in layers.items():
            w.writerow([name, len(edges), round(deg(edges), 2),
                        round(_cross_org_frac(edges, self_tbl), 2), LAYER_WEIGHTS[name]])

    # pairwise Jaccard overlap (pre-registered caveat)
    names = list(layers)
    with (LAYERS_DIR / "layer_overlap.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["layer_a", "layer_b", "jaccard_edges"])
        for a, b in combinations(names, 2):
            ea, eb = layers[a], layers[b]
            j = len(ea & eb) / len(ea | eb) if (ea or eb) else 0.0
            w.writerow([a, b, round(j, 3)])

    print(f"[layers] {len(nodes)} nodes, {len(layers)} layers -> {LAYERS_DIR}")
    print(f"{'layer':18s} {'edges':>6s} {'deg':>6s} {'cross-org':>9s}")
    for name, edges in layers.items():
        print(f"{name:18s} {len(edges):6d} {deg(edges):6.1f} {_cross_org_frac(edges, self_tbl):9.2f}")
    # flag heavily-overlapping pairs (redundant layers)
    hi = [(a, b, len(layers[a] & layers[b]) / len(layers[a] | layers[b]))
          for a, b in combinations(names, 2)
          if (layers[a] or layers[b]) and len(layers[a] & layers[b]) / len(layers[a] | layers[b]) > 0.5]
    if hi:
        print("\n[caveat] highly-overlapping layer pairs (>0.5 Jaccard) — little independent signal:")
        for a, b, j in sorted(hi, key=lambda x: -x[2]):
            print(f"  {a} ~ {b}: {j:.2f}")


if __name__ == "__main__":
    main()
