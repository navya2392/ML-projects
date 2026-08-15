"""Fast Radial Symmetry Transform (Loy & Zelinsky, ECCV 2002).

Why this and not Hough: FRST votes once per gradient pixel into a single
accumulator cell at a *known* radius, instead of smearing a vote over a whole
radius range. When the target radius is known to within ~15% -- which it is
here, blockers are near-constant size -- FRST is both cheaper and far less
prone to the spurious peaks that Hough produces on ring artifacts and on the
straight die boundary.

Crucially for this problem, the vote is per-pixel and additive: a blocker that
only shows a 40% arc still deposits 40% of the votes at the correct centre, and
that centre may lie *outside* the die. Nothing about the transform assumes the
blob is whole.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def frst(
    image: np.ndarray,
    radii,
    *,
    alpha: float = 2.0,
    grad_percentile: float = 80.0,
    polarity: str = "bright",
    valid: np.ndarray | None = None,
    pad: int = 0,
) -> np.ndarray:
    """Radial symmetry response.

    Parameters
    ----------
    image
        Float image. Pass the closing residual (blobs bright), not the raw
        frame -- the residual has no gradient along the die boundary.
    radii
        Iterable of radii to vote at. Averaged into one response map.
    polarity
        ``"bright"`` finds bright blobs, ``"dark"`` finds dark ones.
    valid
        Optional bool mask of pixels allowed to *vote*. Use it to knock out
        gradients that sit on the die silhouette, which otherwise vote for a
        phantom centre one radius inboard of every edge.
    pad
        Grow the accumulator by this many pixels on each side so that centres
        lying outside the image are still representable. The returned map is
        cropped back to ``image.shape``.
    """
    img = image.astype(np.float32)
    gy, gx = np.gradient(ndi.gaussian_filter(img, 1.0))
    mag = np.hypot(gx, gy)

    if valid is not None:
        mag = mag * valid.astype(np.float32)

    nz = mag > 0
    if not nz.any():
        return np.zeros_like(img)
    thr = np.percentile(mag[nz], grad_percentile)
    sel = mag > max(thr, 1e-8)
    if not sel.any():
        return np.zeros_like(img)

    ys, xs = np.nonzero(sel)
    m = mag[ys, xs]
    ux = gx[ys, xs] / m
    uy = gy[ys, xs] / m
    if polarity == "bright":
        # For a bright blob the gradient points *inward*, toward the centre.
        sgn = 1.0
    else:
        sgn = -1.0

    h, w = img.shape
    H, W = h + 2 * pad, w + 2 * pad
    out = np.zeros((H, W), np.float32)

    for n in radii:
        n = float(n)
        px = np.rint(xs + sgn * ux * n).astype(np.int64) + pad
        py = np.rint(ys + sgn * uy * n).astype(np.int64) + pad
        keep = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px, py, mm = px[keep], py[keep], m[keep]

        # np.bincount on flattened indices is ~50x faster than np.add.at here.
        flat = py * W + px
        O = np.bincount(flat, minlength=H * W).astype(np.float32).reshape(H, W)
        M = np.bincount(flat, weights=mm, minlength=H * W).astype(np.float32).reshape(H, W)

        kappa = max(9.9 if n > 1 else 8.0, 1.0)
        O_t = np.clip(O / kappa, 0.0, 1.0)
        F = (O_t**alpha) * (M / kappa)
        # Blur by ~half the radius: the vote cloud for a real circle has that
        # much spread once you account for gradient-angle noise.
        out += ndi.gaussian_filter(F, 0.35 * n)

    out /= max(len(list(radii)), 1)
    return out[pad : pad + h, pad : pad + w]
