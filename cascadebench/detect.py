# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""cascadebench.detect — detector families sharing a common interface.

Detector.fit_score(scenario, train_mask, test_mask, seed) -> scores on the test events.
Families: 1-hop (local), COMPA (per-account volume), context (hand-crafted burst), static GNN,
temporal GNN. Easy to add a new detector: subclass with fit_score. This is the testbed's backbone
(extensibility).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier

from . import features as F
from . import models as M
from .attack import Scenario

EPOCHS = 100


class Detector:
    name = "base"

    def fit_score(self, sc: Scenario, tr: np.ndarray, te: np.ndarray, seed: int) -> np.ndarray:
        raise NotImplementedError


class _LGBMDetector(Detector):
    def _X(self, sc, seed):
        raise NotImplementedError

    def fit_score(self, sc, tr, te, seed):
        X = self._X(sc, seed)
        y = np.array([e[3] for e in sc.events], dtype=np.float32)
        clf = LGBMClassifier(random_state=42, n_estimators=200, verbose=-1).fit(X[tr], y[tr])
        return clf.predict_proba(X[te])[:, 1]


class OneHop(_LGBMDetector):
    name = "1-hop"
    def _X(self, sc, seed): return F.onehop(sc.graph, sc.events, seed)


class COMPA(_LGBMDetector):
    name = "COMPA"
    def _X(self, sc, seed): return F.compa(sc.events)


class HandContext(_LGBMDetector):
    name = "hand-context"
    def _X(self, sc, seed): return F.context(sc.graph, sc.events, seed)


class Hopper(_LGBMDetector):
    """Detector in the spirit of Hopper (Ho et al. 2021) --- movement path: causal precursor
    + burst + rare destination + no benign explanation. Adapted to graph+time (no email content)."""
    name = "hopper"
    def _X(self, sc, seed): return F.hopper(sc.graph, sc.events, seed)


class _TorchDetector(Detector):
    def _tensors(self, sc, seed):
        g = sc.graph; n = len(g); idx = g.index
        X1 = F.onehop(g, sc.events, seed)
        Xn = F.node_features(g, sc.events)
        s_idx = torch.tensor([idx[e[0]] for e in sc.events])
        v_idx = torch.tensor([idx[e[1]] for e in sc.events])
        ef = torch.tensor(X1, dtype=torch.float32)
        y = torch.tensor([e[3] for e in sc.events], dtype=torch.float32)
        return g, n, Xn, s_idx, v_idx, ef, y


class StaticGNN(_TorchDetector):
    name = "GCN-static"

    def fit_score(self, sc, tr, te, seed):
        torch.manual_seed(seed)
        g, n, Xn, s_idx, v_idx, ef, y = self._tensors(sc, seed)
        A = M.norm_adj(g, n)
        model = M.StaticGNN(Xn.shape[1], ef.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        lf = nn.BCEWithLogitsLoss(); trt = torch.tensor(tr)
        for _ in range(EPOCHS):
            model.train(); opt.zero_grad()
            lf(model(A, Xn, s_idx, v_idx, ef)[trt], y[trt]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(A, Xn, s_idx, v_idx, ef))[torch.tensor(te)].numpy()


class TemporalGNN(_TorchDetector):
    name = "temporal-GNN"

    def _make(self, node_dim, edge_dim):
        return M.TemporalGNN(node_dim, edge_dim)

    def fit_score(self, sc, tr, te, seed):
        torch.manual_seed(seed)
        g, n, Xn, s_idx, v_idx, ef, y = self._tensors(sc, seed)
        pb = M.per_bucket(g, sc.events)
        model = self._make(Xn.shape[1], ef.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        lf = nn.BCEWithLogitsLoss(); trt = torch.tensor(tr)
        for _ in range(EPOCHS):
            model.train(); opt.zero_grad()
            lf(model(Xn, pb, n, ef, s_idx, v_idx)[trt], y[trt]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(Xn, pb, n, ef, s_idx, v_idx))[torch.tensor(te)].numpy()


class TemporalGNNAttn(TemporalGNN):
    name = "temporal-GNN-attention"

    def _make(self, node_dim, edge_dim):
        return M.TGATLite(node_dim, edge_dim)


class AnomalyForest(Detector):
    """Unsupervised: IsolationForest on context features, trained on benign, score = anomaly."""
    name = "anomaly-forest"

    def fit_score(self, sc, tr, te, seed):
        from sklearn.ensemble import IsolationForest
        X = F.context(sc.graph, sc.events, seed)
        y = np.array([e[3] for e in sc.events])
        ben_tr = tr & (y == 0)
        clf = IsolationForest(random_state=42, n_estimators=150, contamination="auto")
        clf.fit(X[ben_tr] if ben_tr.sum() > 10 else X[tr])
        return -clf.decision_function(X[te])      # higher = more anomalous = attack


# standard testbed panel (8 detector families, including Hopper-style SOTA)
PANEL = [OneHop, COMPA, Hopper, AnomalyForest, StaticGNN, HandContext, TemporalGNN, TemporalGNNAttn]


def get_panel():
    return [d() for d in PANEL]
