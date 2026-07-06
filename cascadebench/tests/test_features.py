# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""Feature tests exercising the main CascadeBench features through the public API only.

These tests drive the documented workflow: build a graph -> build a scenario ->
evaluate the detector panel -> check leak-aware controls (time shuffle). They use small
synthetic graphs to stay fast on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cascadebench as cb  # noqa: E402


def test_synthetic_graph_has_nodes_and_edges():
    g = cb.Graph.synthetic(200)
    assert len(g) > 0
    assert len(g.nodes) == len(g)
    assert len(g.edges()) > 0


def test_build_scenario_has_events_attacks_labels():
    g = cb.Graph.synthetic(200)
    sc = cb.build_scenario(g, cb.CascadeStrategy(fanout=5, spread=1), seed=0)
    assert len(sc.events) > 0
    assert sc.n_attacks >= 1
    labels = {e[3] for e in sc.events}
    assert labels == {0, 1}


def test_evaluate_panel_metrics_in_range():
    g = cb.Graph.synthetic(200)
    sc = cb.build_scenario(g, cb.CascadeStrategy(fanout=5, spread=1), seed=0)
    res = cb.evaluate(sc, cb.get_panel(), seed=0)
    assert len(res) > 0
    for name, m in res.items():
        assert 0.0 <= m["auc"] <= 1.0, name
        assert "recall@0.01" in m
        assert 0.0 <= m["recall@0.01"] <= 1.0, name


def test_evaluate_is_deterministic_for_same_seed():
    g = cb.Graph.synthetic(200)
    sc = cb.build_scenario(g, cb.CascadeStrategy(fanout=5, spread=1), seed=0)
    r1 = cb.evaluate(sc, cb.get_panel(), seed=0)
    r2 = cb.evaluate(sc, cb.get_panel(), seed=0)
    assert set(r1) == set(r2)
    # The gradient-boosted detectors (fixed random_state, seeded features) are exactly
    # reproducible; assert bitwise determinism on them through the public API.
    for name in ("COMPA", "1-hop"):
        assert r1[name]["auc"] == r2[name]["auc"], name
        assert r1[name]["recall@0.01"] == r2[name]["recall@0.01"], name
    # Every detector (incl. the torch GNNs, which are not bitwise-stable on CPU) must be
    # reproducible up to a small numerical tolerance.
    for name in r1:
        assert abs(r1[name]["auc"] - r2[name]["auc"]) < 1e-2, name


def test_shuffle_time_returns_scenario_and_changes_buckets():
    g = cb.Graph.synthetic(200)
    sc = cb.build_scenario(g, cb.CascadeStrategy(fanout=5, spread=1), seed=0)
    sh = cb.shuffle_time(sc, seed=0)
    assert isinstance(sh, cb.Scenario)
    assert len(sh.events) == len(sc.events)
    # the time-shuffle control reassigns buckets, so the bucket sequence must differ
    orig_buckets = [e[2] for e in sc.events]
    shuf_buckets = [e[2] for e in sh.events]
    assert orig_buckets != shuf_buckets
