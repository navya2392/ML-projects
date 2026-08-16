"""Align the expected blocker constellation to the actual frame.

Expected coordinates come from a recipe, in nominal part space. The part in the
frame is shifted, slightly rotated and slightly scaled relative to that. If the
alignment is off by even a third of a radius, an ROI centred on the nominal
coordinate straddles the beam edge and the site "cannot be found" -- which is a
registration failure being reported as a missing blocker.

The constellation itself is the best possible fiducial: the blockers that *are*
unambiguous pin down the transform, and the transform then predicts the
ambiguous ones to sub-pixel accuracy. This is why the hard edge cases get
easier, not harder, once you use the known layout.
"""

from __future__ import annotations

import numpy as np


def similarity_from_pairs(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity (rotation + uniform scale + translation).

    Deliberately not a full affine or homography: the physical part cannot
    shear, and allowing it to would let a handful of mismatches distort the
    prediction for the very sites we most need to get right.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var = (s0**2).sum()
    if var < 1e-9:
        return np.eye(2), mu_d - mu_s
    H = s0.T @ d0
    U, S, Vt = np.linalg.svd(H)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    scale = S.sum() / var
    A = scale * R
    return A, mu_d - A @ mu_s


def apply(A: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (np.asarray(pts, float) @ A.T) + t


def ransac_align(
    expected: np.ndarray,
    observed: np.ndarray,
    *,
    radius: float,
    iters: int = 3000,
    tol_factor: float = 0.45,
    rng: np.random.Generator | None = None,
):
    """Align expected sites to blind detections without known correspondences.

    This is the bootstrap that bounded local search cannot provide. A local
    search around a nominal coordinate can only recover a misregistration
    smaller than its own window; beyond that it locks onto the wrong feature or
    onto nothing, and every downstream stage inherits the error. Matching the
    two point sets is correspondence-free, so it recovers an arbitrarily large
    shift, and it is why blind detection remains necessary even when the
    expected layout is known.

    Two-point samples suffice: a similarity has four degrees of freedom, and the
    pairwise distance ratio screens out most bad samples before any fit.

    Returns ``(A, t, n_inliers)``, falling back to identity when the evidence is
    weak -- a confidently wrong transform moves every ROI off target and
    manufactures exactly the failures this is meant to remove.
    """
    rng = rng or np.random.default_rng(0)
    E = np.asarray(expected, float)
    O = np.asarray(observed, float)
    if len(E) < 3 or len(O) < 3:
        return np.eye(2), np.zeros(2), 0

    tol = tol_factor * radius
    best = (np.eye(2), np.zeros(2), -1)

    ei = rng.integers(0, len(E), size=(iters, 2))
    oi = rng.integers(0, len(O), size=(iters, 2))
    for (i, j), (k, l) in zip(ei, oi):
        if i == j or k == l:
            continue
        de = np.linalg.norm(E[i] - E[j])
        do = np.linalg.norm(O[k] - O[l])
        if de < 2.5 * radius or do < 2.5 * radius:
            continue
        if abs(de - do) > 0.12 * de:  # scale must be close to 1
            continue
        A, t = similarity_from_pairs(E[[i, j]], O[[k, l]])
        P = E @ A.T + t
        d = np.linalg.norm(P[:, None, :] - O[None, :, :], axis=2)
        n = int((d.min(axis=1) < tol).sum())
        if n > best[2]:
            best = (A, t, n)
            if n == len(E):
                break

    A, t, n = best
    if n < max(4, int(0.35 * len(E))):
        return np.eye(2), np.zeros(2), 0

    for _ in range(3):  # re-fit on inlier correspondences
        P = E @ A.T + t
        d = np.linalg.norm(P[:, None, :] - O[None, :, :], axis=2)
        nn = d.argmin(axis=1)
        inl = d.min(axis=1) < tol
        if inl.sum() < 4:
            break
        A, t = similarity_from_pairs(E[inl], O[nn[inl]])
    return A, t, int(inl.sum())


def verify_frame(image, expected_xy, radius=None, *, params=None, detect_params=None):
    """Full known-site verification: blind detect -> align -> verify.

    This is the entry point to use. The two modes are not alternatives; they
    compose. Blind detection supplies the registration bootstrap, and the known
    layout supplies the per-site decision that blind detection alone cannot make
    at a truncated site.
    """
    from .pipeline import Params, detect, ideal_die, segment_die
    from .verify import verify

    exp = np.asarray(expected_xy, float).reshape(-1, 2)
    dets, info = detect(image, detect_params or Params())
    r = float(radius) if radius else float(info["radius"])

    obs = np.array([[d.cx, d.cy] for d in dets], float)
    A, t, n_inl = ransac_align(exp, obs, radius=r)
    aligned = exp @ A.T + t

    beam_ideal = ideal_die(segment_die(np.asarray(image, np.float32)), r, 1.9)
    res, vinfo = verify(image, aligned, r, params=params, beam_ideal=beam_ideal)
    vinfo.update({"bootstrap_inliers": n_inl, "n_blind_detections": len(dets),
                  "aligned_xy": aligned, "detections": dets})
    return res, vinfo
