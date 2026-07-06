# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""cascadebench.stats — statistical rigor for reported numbers.

Addresses the statistical review: bootstrap-CI instead of single means, Student's t quantile (not 1.96)
for small n, Holm correction for multiple comparisons, Cliff's delta effect size, paired test.
Deterministic (explicit seed) — no Math.random/Date.now (consistency with the rest of the package).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Dict
import numpy as np


def mean_ci(values: Sequence[float], alpha: float = 0.05, method: str = "bootstrap",
            n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi). method='bootstrap' (percentile) or 't' (Student's t quantile).
    Bootstrap is robust to small n and assumes no normality; 't' is a faster alternative."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, m, m
    if method == "t":
        from scipy.stats import t
        se = x.std(ddof=1) / np.sqrt(n)
        q = t.ppf(1 - alpha / 2, df=n - 1)
        return m, m - q * se, m + q * se
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return m, float(lo), float(hi)


def fmt_ci(values: Sequence[float], prec: int = 3, **kw) -> str:
    """Format 'mean [lo, hi]' for report/LaTeX."""
    m, lo, hi = mean_ci(values, **kw)
    return f"{m:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Tuple[float, str]:
    """Cliff's delta — non-parametric effect size (a vs b). Thresholds: |d|<0.147 negligible,
    <0.33 small, <0.474 medium, otherwise large (Romano et al.)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return 0.0, "n/a"
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    d = (gt - lt) / (len(a) * len(b))
    ad = abs(d)
    mag = "negligible" if ad < 0.147 else "small" if ad < 0.33 else "medium" if ad < 0.474 else "large"
    return float(d), mag


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Paired Wilcoxon test (same seeds). Returns (statistic, p). NOTE: for n<6, the two-sided p
    has a FLOOR (>0.05 may be unreachable) — report this explicitly."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.allclose(a, b):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(a, b)
        return float(stat), float(p)
    except ValueError:
        return float("nan"), 1.0


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Tuple[float, bool]]:
    """Holm-Bonferroni correction for multiple comparisons. Returns name -> (p_adjusted, significant)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Tuple[float, bool]] = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        p_adj = min(1.0, max(prev, (m - i) * p))   # monotonicity
        prev = p_adj
        out[name] = (p_adj, p_adj < alpha)
    return out


def min_two_sided_wilcoxon_p(n: int) -> float:
    """Smallest achievable two-sided p of the signed-rank test for n pairs (power floor).
    E.g. n=5 -> 0.0625 (>0.05 UNREACHABLE). Used for an honest power report."""
    return 2.0 / (2 ** n) * 1.0 if n <= 0 else 2.0 / (2 ** n)
