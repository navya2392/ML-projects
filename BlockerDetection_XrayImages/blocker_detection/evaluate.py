"""Greedy one-to-one matching and metrics, split by how truncated a blocker is.

Aggregate precision/recall hides the failure that matters here. The interesting
number is recall on blockers that are cut by the die edge, so it is reported
separately.
"""

from __future__ import annotations

import numpy as np


def match(gt, dets, tol_factor: float = 0.6):
    """Greedy nearest-first matching. Returns (pairs, unmatched_gt, unmatched_det)."""
    pairs = []
    used_d = set()
    cost = []
    for gi, g in enumerate(gt):
        for di, d in enumerate(dets):
            dist = float(np.hypot(g.cx - d.cx, g.cy - d.cy))
            if dist <= tol_factor * g.r:
                cost.append((dist, gi, di))
    cost.sort()
    used_g = set()
    for dist, gi, di in cost:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        pairs.append((gi, di, dist))
    return (
        pairs,
        [i for i in range(len(gt)) if i not in used_g],
        [i for i in range(len(dets)) if i not in used_d],
    )


def report(gt, dets, edge_threshold: float = 0.90) -> dict:
    pairs, miss, fp = match(gt, dets)
    tp = len(pairs)
    prec = tp / max(len(dets), 1)
    rec = tp / max(len(gt), 1)
    edge_idx = {i for i, g in enumerate(gt) if g.visible_fraction < edge_threshold}
    edge_tp = sum(1 for gi, _, _ in pairs if gi in edge_idx)
    whole_idx = set(range(len(gt))) - edge_idx
    whole_tp = sum(1 for gi, _, _ in pairs if gi in whole_idx)
    errs = [d for _, _, d in pairs]
    return {
        "n_gt": len(gt),
        "n_det": len(dets),
        "tp": tp,
        "fp": len(fp),
        "fn": len(miss),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "n_edge_gt": len(edge_idx),
        "edge_recall": edge_tp / max(len(edge_idx), 1),
        "n_whole_gt": len(whole_idx),
        "whole_recall": whole_tp / max(len(whole_idx), 1),
        "median_centre_error_px": float(np.median(errs)) if errs else float("nan"),
    }
