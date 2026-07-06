#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""Generates P3 figures from cb_*.csv results -> latex_p3/figures/ (PDF).

Run from the code/ directory:  ../../venv/Scripts/python cascadebench/plot_cb.py
Reads:  results/cb_*.csv
Writes:  ../latex_p3/figures/*.pdf
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# DejaVu Sans (bundled with matplotlib) has Polish diacritics; fonttype 42 embeds TrueType in the PDF.
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.unicode_minus": False})

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGS = HERE.parents[1] / "latex_p3" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# detector order and colors (consistent across figures)
DETS = ["1-hop", "COMPA", "hopper", "anomaly-forest", "GCN-static",
        "hand-context", "temporal-GNN", "temporal-GNN-attention"]
COLOR = {
    "1-hop": "#9e9e9e", "COMPA": "#d62728", "hopper": "#bcbd22", "anomaly-forest": "#8c564b",
    "GCN-static": "#1f77b4", "hand-context": "#2ca02c",
    "temporal-GNN": "#ff7f0e", "temporal-GNN-attention": "#9467bd",
}
MARK = {
    "1-hop": "o", "COMPA": "s", "hopper": "X", "anomaly-forest": "^", "GCN-static": "D",
    "hand-context": "v", "temporal-GNN": "P", "temporal-GNN-attention": "*",
}
# Display names: we don't attribute narrow reimplementations to full systems (P3 review #18).
DISP = {"COMPA": "volumetric", "hopper": "path-based"}
def _disp(name): return DISP.get(name, name)


def _read(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_robustness():
    rows = _read("cb_robustness_curves.csv")
    reach = [float(r["reach"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for d in DETS:
        if d not in rows[0]:
            continue
        ys = [float(r[d]) for r in rows]
        ax.plot(reach, ys, marker=MARK[d], color=COLOR[d], label=_disp(d),
                linewidth=1.8, markersize=7)
    ax.set_xlabel("Attack reach [% of nodes]  (fan-out 2→8)")
    ax.set_ylabel("Detection (AUC)")
    ax.set_ylim(0.50, 0.88)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower center")
    ax.set_title("Detector robustness vs. attacker reach")
    fig.tight_layout()
    out = FIGS / "robustness_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_panel():
    rows = _read("cb_panel.csv")
    strat = [r["strategy"] for r in rows]
    dets = [c for c in rows[0] if c not in ("strategy", "reach")]
    x = range(len(strat))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    n = len(dets)
    w = 0.8 / n
    for i, d in enumerate(dets):
        ys = [float(r[d]) for r in rows]
        xs = [xi + (i - n / 2) * w + w / 2 for xi in x]
        ax.bar(xs, ys, width=w, color=COLOR.get(d, "#555"), label=_disp(d))
    ax.set_xticks(list(x))
    ax.set_xticklabels(["naive\n(reach 94%)", "spread-out\n(reach 94%)", "low reach\n(54%)"])
    ax.set_ylabel("Detection (AUC)")
    ax.set_ylim(0.4, 0.9)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title("Detector panel vs. attacker strategy")
    fig.tight_layout()
    out = FIGS / "panel.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_validate_real():
    rows = _read("cb_validate_real.csv")
    graphs = [r["graph"] for r in rows]
    dets = [c for c in rows[0] if c != "graph"]
    x = range(len(graphs))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    n = len(dets)
    w = 0.8 / n
    for i, d in enumerate(dets):
        ys = [float(r[d]) for r in rows]
        xs = [xi + (i - n / 2) * w + w / 2 for xi in x]
        ax.bar(xs, ys, width=w, color=COLOR.get(d, "#555"), label=_disp(d))
    ax.set_xticks(list(x))
    ax.set_xticklabels([g.replace("synthetic:600", "synthetic") for g in graphs])
    ax.set_ylabel("Detection (AUC)")
    ax.set_ylim(0.4, 0.9)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title("Synthetic vs. real topologies: rankings do not transfer")
    fig.tight_layout()
    out = FIGS / "validate_real.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_predictivity():
    """Scatter: structural distance vs. detector rank transfer (predictivity probe core)."""
    fp = RESULTS / "cb_predictivity_transfer.csv"
    if not fp.exists():
        print("skip predictivity (no CSV)")
        return
    rows = _read("cb_predictivity_transfer.csv")
    xs = [float(r["struct_dist"]) for r in rows]
    ys = [float(r["rank_transfer"]) for r in rows]
    labels = [r["graph"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(xs, ys, s=70, color="#9467bd", zorder=3, edgecolor="k", linewidth=0.5)
    for x, y, lb in zip(xs, ys, labels):
        ax.annotate(lb, (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    # trend line
    if len(xs) >= 2:
        import numpy as np
        a, b = np.polyfit(xs, ys, 1)
        xx = [min(xs), max(xs)]
        ax.plot(xx, [a * x + b for x in xx], "--", color="#d62728", lw=1.5,
                label=f"trend (slope {a:.2f})")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("Structural distance from the training graph\n(degree skew + clustering + density)")
    ax.set_ylabel("Detector rank transfer (Spearman ρ)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Predictivity probe: rank transfer decreases with structural distance")
    fig.tight_layout()
    out = FIGS / "predictivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_predictivity_gap():
    """Bar: synthetic->real transfer gap per detector (MAIN probe result).
    Negative = detector inflated by synthetic; positive/zero = transfers faithfully."""
    fp = RESULTS / "cb_predictivity_gap.csv"
    if not fp.exists():
        print("skip predictivity_gap (no CSV)")
        return
    rows = _read("cb_predictivity_gap.csv")
    labels = [r["detector"] for r in rows]
    gaps = [float(r["gap"]) for r in rows]
    cis = [float(r.get("gap_ci95", 0) or 0) for r in rows]
    cols = [COLOR.get(l, "#555") for l in labels]
    order = sorted(range(len(gaps)), key=lambda i: gaps[i])
    labels = [labels[i] for i in order]; gaps = [gaps[i] for i in order]
    cis = [cis[i] for i in order]; cols = [cols[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(range(len(gaps)), gaps, xerr=cis, color=cols, edgecolor="k", linewidth=0.5,
            error_kw=dict(ecolor="#333", capsize=3, lw=1))
    ax.set_yticks(range(len(gaps))); ax.set_yticklabels([_disp(l) for l in labels], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("AUC transfer gap (real − synthetic),  whiskers = 95% CI")
    ax.set_title("Synthetic→real transfer gap per detector (20 seeds)")
    fig.tight_layout()
    out = FIGS / "predictivity_gap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_clustering():
    """Scatter of graph clustering vs. detector AUC (M2): GCN/volumetric increase with clustering (inflation),
    temporal ones stay flat (robust). Mechanism of the headline finding."""
    fp = RESULTS / "cb_predictivity.csv"
    if not fp.exists():
        print("skip clustering (no CSV)")
        return
    rows = _read("cb_predictivity.csv")
    clust = [float(r["clustering"]) for r in rows]
    show = ["GCN-static", "COMPA", "temporal-GNN", "temporal-GNN-attention"]
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for d in show:
        if d not in rows[0]:
            continue
        ys = [float(r[d]) for r in rows]
        ax.scatter(clust, ys, s=45, color=COLOR.get(d, "#555"), marker=MARK.get(d, "o"),
                   edgecolor="k", linewidth=0.4, zorder=3, label=_disp(d))
        if len(set(clust)) >= 2:
            a, b = np.polyfit(clust, ys, 1)
            xx = np.array([min(clust), max(clust)])
            ax.plot(xx, a * xx + b, "--", color=COLOR.get(d, "#555"), lw=1.2, alpha=0.7)
    ax.set_xlabel("Training graph clustering")
    ax.set_ylabel("Detector AUC")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Inflation ∝ clustering: GCN/volumetric increase, temporal stays flat")
    fig.tight_layout()
    out = FIGS / "clustering.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_fabrication():
    """(D) Detection vs. edge fabrication fraction per detector."""
    fp = RESULTS / "cb_fabrication.csv"
    if not fp.exists():
        print("skip fabrication (no CSV)"); return
    rows = _read("cb_fabrication.csv")
    xs = [float(r["fabrication"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for d in DETS:
        if d not in rows[0]:
            continue
        ax.plot(xs, [float(r[d]) for r in rows], marker=MARK[d], color=COLOR[d], label=_disp(d), lw=1.6)
    ax.set_xlabel("Edge fabrication fraction (OSINT-spoofed contacts)")
    ax.set_ylabel("Detection (AUC)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Edge-fabricating attack vs. detector panel")
    fig.tight_layout()
    out = FIGS / "fabrication.pdf"; fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def fig_mechanism():
    """(C) AUC vs. modularity Q, split by family: SBM (low clustering) and cliques (high clustering)."""
    fp = RESULTS / "cb_mechanism.csv"
    if not fp.exists():
        print("skip mechanism (no CSV)"); return
    rows = _read("cb_mechanism.csv")
    show = ["GCN-static", "temporal-GNN", "COMPA"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for fam, mk, ls in (("SBM", "o", "-"), ("clique", "s", "--")):
        fr = [r for r in rows if r["family"] == fam]
        if not fr:
            continue
        Q = [float(r["modularity_Q"]) for r in fr]
        for d in show:
            if d not in fr[0]:
                continue
            ax.plot(Q, [float(r[d]) for r in fr], marker=mk, ls=ls, color=COLOR.get(d, "#555"),
                    lw=1.5, label=f"{d} [{fam}]")
    ax.set_xlabel("Modularity Q  (SBM: low clustering; cliques: high)")
    ax.set_ylabel("Detection (AUC)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=6, ncol=2)
    ax.set_title("Mechanism: GCN responds to clique structure, not modularity alone")
    fig.tight_layout()
    out = FIGS / "mechanism.pdf"; fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_robustness()
    fig_panel()
    fig_validate_real()
    fig_predictivity()
    fig_predictivity_gap()
    fig_clustering()
    fig_fabrication()
    fig_mechanism()
    print("DONE")
