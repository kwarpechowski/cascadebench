# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""Basic testbed tests — sanity check (run: python -m cascadebench.tests.test_smoke)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cascadebench as cb  # noqa: E402


def test_graph_synthetic():
    g = cb.Graph.synthetic(200)
    assert len(g) > 100
    assert all(g.neighbors(u) for u in g.nodes[:5])
    print("ok: graph_synthetic")


def test_scenario_labels():
    g = cb.Graph.synthetic(200)
    sc = cb.build_scenario(g, cb.CascadeStrategy(fanout=6, n_cascades=10), 0)
    ys = [e[3] for e in sc.events]
    assert set(ys) == {0, 1} and sc.n_attacks > 0
    assert 0.0 < sc.infected_frac <= 1.0
    print("ok: scenario_labels")


def test_panel_runs():
    g = cb.Graph.synthetic(200)
    sc = cb.build_scenario(g, cb.CascadeStrategy(fanout=6, n_cascades=10), 0)
    res = cb.evaluate(sc, cb.get_panel(), 0)
    assert set(d.name for d in cb.get_panel()) <= set(res)
    for m in res.values():
        assert 0.0 <= m["auc"] <= 1.0
    print("ok: panel_runs")


def test_shuffle_control():
    g = cb.Graph.synthetic(200)
    sc = cb.build_scenario(g, cb.CascadeStrategy(fanout=6, n_cascades=10), 0)
    sh = cb.shuffle_time(sc, 0)
    assert len(sh.events) == len(sc.events)
    print("ok: shuffle_control")


if __name__ == "__main__":
    test_graph_synthetic(); test_scenario_labels(); test_panel_runs(); test_shuffle_control()
    print("ALL TESTS OK")
