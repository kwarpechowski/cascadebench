# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""Regenerate the figures embedded in the top-level README from the bundled result CSVs.

Usage (from the repository root):
    python docs/make_readme_figures.py

Reads results/{cb_headline_ci,cb_robustness_curves,exp_cascade_enron}.csv and writes
docs/figures/*.png. Pure matplotlib; no model training or GPU required.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# The bundled CSVs retain the original (Polish) detector/regime labels; map to English
# for display only. The code and paper use the English names throughout.
DET = {
    "1-hop": "1-hop", "COMPA": "COMPA", "hopper": "hopper",
    "anomaly-forest": "anomaly-forest", "GCN-statyczny": "GCN-static",
    "reczny-kontekst": "hand-context", "temporalny-GNN": "temporal-GNN",
    "temporalny-GNN-uwaga": "temporal-GNN-attn",
}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def fig_panel_by_regime():
    rows = list(csv.DictReader(open(os.path.join(RES, "cb_headline_ci.csv"), encoding="utf-8")))
    dets = list(DET)
    naive = {r["detektor"]: float(r["auc_mean"]) for r in rows if r["rezim"] == "naiwny"}
    stealth = {r["detektor"]: float(r["auc_mean"]) for r in rows if r["rezim"] == "skryty"}
    x = np.arange(len(dets)); w = 0.4
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - w / 2, [naive[d] for d in dets], w, label="naive attacker", color="#4C78A8")
    ax.bar(x + w / 2, [stealth[d] for d in dets], w, label="stealthy attacker", color="#E45756")
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([DET[d] for d in dets], rotation=30, ha="right")
    ax.set_ylabel("AUC"); ax.set_ylim(0.5, 0.92)
    ax.set_title("Detector panel AUC by attack regime (no single detector dominates)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(os.path.join(OUT, "panel_by_regime.png")); plt.close(fig)


def fig_reach_frontier():
    rc = list(csv.DictReader(open(os.path.join(RES, "cb_robustness_curves.csv"), encoding="utf-8")))
    reach = np.array([float(r["zasieg"]) for r in rc]); order = np.argsort(reach)
    reach = reach[order]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for d in ["temporalny-GNN-uwaga", "GCN-statyczny", "anomaly-forest", "hopper", "1-hop"]:
        y = np.array([float(r[d]) for r in rc])[order]
        ax.plot(reach, y, marker="o", ms=4, label=DET[d])
    ax.set_xlabel("attack reach (fraction of nodes infected)  — narrow/stealthy -> wide/loud")
    ax.set_ylabel("detection AUC")
    ax.set_title("Reach-evasion frontier: best detector depends on reach")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(os.path.join(OUT, "reach_frontier.png")); plt.close(fig)


def fig_leak_audit():
    en = {r["model"]: r for r in csv.DictReader(
        open(os.path.join(RES, "exp_cascade_enron.csv"), encoding="utf-8"))}
    groups = [("temporal-GNN", "gnn_temporal"),
              ("temporal-GNN\n(time-shuffled)", "gnn_temporal_shuf"),
              ("1-hop", "tab_1hop")]
    auc = [float(en[k]["auc"]) for _, k in groups]
    rec = [float(en[k]["recall_fpr1"]) for _, k in groups]
    x = np.arange(len(groups)); col = ["#54A24B", "#B0B0B0", "#4C78A8"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4))
    a1.bar(x, auc, color=col); a1.axhline(0.5, ls="--", c="gray", lw=0.8)
    a1.set_xticks(x); a1.set_xticklabels([g[0] for g in groups], fontsize=8)
    a1.set_ylabel("AUC"); a1.set_ylim(0.4, 0.9); a1.set_title("AUC")
    a2.bar(x, rec, color=col)
    a2.set_xticks(x); a2.set_xticklabels([g[0] for g in groups], fontsize=8)
    a2.set_ylabel("Recall@FPR=1%"); a2.set_title("Recall@FPR=1%")
    fig.suptitle("Leak audit (Enron): time-shuffle removes part - not all - of the temporal advantage")
    fig.savefig(os.path.join(OUT, "leak_audit.png")); plt.close(fig)


if __name__ == "__main__":
    fig_panel_by_regime()
    fig_reach_frontier()
    fig_leak_audit()
    print("wrote:", sorted(os.listdir(OUT)))
