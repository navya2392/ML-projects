"""Algebraic circle fitting with a radius prior, for sub-pixel arc refinement.

A truncated blocker gives you an arc, not a disk. Centroid-based estimates of
its centre are biased inboard by exactly the amount that was clipped, which is
the single largest source of position error at the die edge. Fitting a circle
to the *arc* removes that bias entirely -- the arc knows where its centre is
even when most of the disk is missing.
"""

from __future__ import annotations

import numpy as np


def kasa_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares algebraic circle fit (Kasa). Fast, unbiased enough once
    RANSAC has removed outliers, and stable for arcs down to ~90 degrees."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mx, my = float(x.mean()), float(y.mean())
    u, v = x - mx, y - my
    Suu, Svv, Suv = float((u * u).sum()), float((v * v).sum()), float((u * v).sum())
    Suuu, Svvv = float((u**3).sum()), float((v**3).sum())
    Suvv, Svuu = float((u * v * v).sum()), float((v * u * u).sum())
    A = np.array([[Suu, Suv], [Suv, Svv]], float)
    b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu], float)
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan")
    r = float(np.sqrt(c @ c + (Suu + Svv) / n))
    return mx + float(c[0]), my + float(c[1]), r


def ransac_circle(
    x: np.ndarray,
    y: np.ndarray,
    *,
    r_prior: float,
    r_tol: float = 0.35,
    inlier_tol: float = 2.5,
    iters: int = 120,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, np.ndarray] | None:
    """RANSAC circle fit constrained to ``r_prior * (1 +- r_tol)``.

    The radius prior is what makes this work on short arcs: unconstrained
    three-point fits on a 60-degree arc are wildly unstable, but rejecting any
    hypothesis whose radius is off-nominal throws those away for free.
    """
    rng = rng or np.random.default_rng(0)
    n = len(x)
    if n < 6:
        return None
    lo, hi = r_prior * (1 - r_tol), r_prior * (1 + r_tol)
    best = None
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        cx, cy, r = kasa_fit(x[idx], y[idx])
        if not np.isfinite(r) or not (lo <= r <= hi):
            continue
        d = np.abs(np.hypot(x - cx, y - cy) - r)
        inl = d < inlier_tol
        c = int(inl.sum())
        if c > best_count:
            best_count, best = c, inl
    if best is None or best_count < 6:
        return None
    cx, cy, r = kasa_fit(x[best], y[best])
    if not np.isfinite(r) or not (lo <= r <= hi):
        return None
    return cx, cy, r, best


def arc_support(
    cx: float, cy: float, r: float, x: np.ndarray, y: np.ndarray, tol: float = 3.0
) -> float:
    """Fraction of the circle's circumference covered by supporting points.

    Reported per detection as a confidence signal, and used to reject ring
    artifacts, which produce points at many radii but rarely a coherent arc.
    """
    if len(x) == 0:
        return 0.0
    d = np.hypot(x - cx, y - cy)
    on = np.abs(d - r) < tol
    if on.sum() < 3:
        return 0.0
    th = np.arctan2(y[on] - cy, x[on] - cx)
    bins = np.zeros(72, bool)
    bins[((th + np.pi) / (2 * np.pi) * 72).astype(int) % 72] = True
    return float(bins.sum() / 72.0)
