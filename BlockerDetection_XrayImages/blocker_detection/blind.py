"""Blind blocker detection for real ISP_NF frames.

End-to-end, no prior on positions or count:

    calibrate scale  ->  occupancy field  ->  matched filter  ->  peaks
      ->  joint least-squares refit per overlapping cluster  ->  accept/reject

Every threshold is expressed in units of the auto-calibrated radius or in
occupancy (a physical 0-to-1 quantity), so nothing here depends on the frame's
exposure, its internal brightness gradient, or its blur.

Acceptance is on **occupancy**, not on shape. A blocker clipped by the beam edge
has no closed contour and fails any circularity or area test, but the beam
pixels it does cover are just as opaque as a whole blocker's -- so occupancy
measured over valid pixels only is unchanged by truncation. That is the whole
reason edge blockers survive this pipeline and do not survive a contour-based
one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from .scale import Calibration, _disc, calibrate


@dataclass
class Blob:
    cx: float
    cy: float
    r: float
    alpha: float  # 1 = fully opaque disc, 0 = bare beam
    sigma: float  # uncertainty on alpha
    visible_fraction: float  # fraction of the disc inside the beam footprint
    edge: bool  # clipped by the beam boundary

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlindParams:
    alpha_accept: float = 0.55  # occupancy required to call it a blocker
    alpha_seed: float = 0.35  # looser gate for proposing a candidate
    nms_factor: float = 0.75  # peak spacing, in radii
    min_valid_fraction: float = 0.15  # too little beam left to judge
    edge_threshold: float = 0.92  # below this, report as edge-clipped
    support_frac: float = 0.12  # min valid support for a peak to be considered
    k_sigma: float = 2.0  # confidence margin on acceptance
    max_splits: int = 2  # rounds of merged-blob splitting
    f_split: float = 150.0  # residual F required to admit an extra source
    use_bites: bool = False  # silhouette-bite channel; OFF -- see note below
    bite_min_depth: float = 0.30  # min boundary excursion, in radii
    bite_shape_lo: float = 0.55  # bite half-width vs the width a disc would cut
    bite_shape_hi: float = 1.70
    bite_dedupe: float = 1.30  # drop a bite already explained by a peak, in radii


def detect_blind(image: np.ndarray, params: BlindParams | None = None,
                 cal: Calibration | None = None):
    """Detect all blockers in a frame. Returns ``(blobs, info)``."""
    p = params or BlindParams()
    img = np.asarray(image, np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    cal = cal or calibrate(img)
    r, psf = cal.radius, cal.psf
    y, valid = cal.shortfall, cal.beam_ideal

    D = _disc(r, psf)
    v = valid.astype(np.float32)
    num = fftconvolve(y * v, D, mode="same").astype(np.float32)
    den = fftconvolve(v, D.astype(np.float64) ** 2, mode="same").astype(np.float32)
    alpha = num / np.maximum(den, 1e-6)
    snr = num / np.sqrt(np.maximum(den, 1e-6))

    den_full = float((D * D).sum())
    support = valid & (den > p.support_frac * den_full)

    # Peaks of the matched-filter SNR, spaced so a touching pair still yields two.
    sep = max(3, int(round(p.nms_factor * r)))
    mx = ndi.maximum_filter(snr, size=2 * sep + 1)
    pk = (snr == mx) & support & (alpha > p.alpha_seed)
    ys, xs = np.nonzero(pk)
    if len(ys) == 0:
        return [], {"calibration": cal, "n_candidates": 0}

    order = np.argsort(-snr[ys, xs])
    centres = np.stack([xs[order], ys[order]], axis=1).astype(float)

    # Separate channel for blockers that sit mostly *outside* the beam. Their
    # centre is beyond the boundary, so only a sliver of the disc lands on valid
    # pixels and the occupancy gate rejects them even though the occupancy
    # itself is right (measured on a real missed case: alpha 0.99, but 10% of
    # the disc inside the footprint). Requiring a fraction of the disc to fall
    # inside a footprint that is itself truncated is circular. What is *not*
    # ambiguous is that beam material is missing: the difference between the
    # fitted beam outline and the observed silhouette is the bite, and a bite is
    # a blocker by definition.
    bites = _edge_profile_bites(cal.beam, r, p) if p.use_bites else np.empty((0, 2))
    if len(bites) and len(centres):
        # A blocker cut by the edge but still mostly inside the beam is found by
        # both channels, at slightly different centres (the bite estimate sits
        # further out). Deduplicate generously, or the same blocker is counted
        # twice.
        d = np.linalg.norm(bites[:, None, :] - centres[None, :, :], axis=2)
        bites = bites[d.min(axis=1) > p.bite_dedupe * r]
    if len(bites):
        centres = np.vstack([centres, bites]) if len(centres) else bites

    # Touching blockers merge into one elongated blob with a *single* matched
    # filter maximum, so no amount of peak spacing recovers the second one --
    # measured, tightening the spacing from 0.75r to 0.50r changed nothing. The
    # partner has to be found as structure the one-source model cannot explain.
    for _ in range(p.max_splits):
        extra = _split_pass(y, valid, centres, r, psf, p)
        if not len(extra):
            break
        centres = np.vstack([centres, extra])

    alphas, sigmas = _joint(y, valid, centres, r, psf)

    bite_set = {tuple(np.round(b, 3)) for b in bites} if len(bites) else set()

    blobs: list[Blob] = []
    for (cx, cy), a, sg in zip(centres, alphas, sigmas):
        from_bite = tuple(np.round([cx, cy], 3)) in bite_set
        if not np.isfinite(a) and not from_bite:
            continue
        vf = _valid_fraction(cx, cy, r, valid)
        if from_bite:
            # The silhouette itself is the evidence: beam material is missing in
            # a shape only a disc of this radius produces. No occupancy gate is
            # applied, because the occupancy field is structurally blind here.
            a = a if np.isfinite(a) else float("nan")
        else:
            if vf < p.min_valid_fraction:
                continue
            if a - p.k_sigma * max(sg, 0.02) < p.alpha_accept:
                continue
        blobs.append(Blob(float(cx), float(cy), float(r), float(a), float(sg),
                          float(vf), bool(vf < p.edge_threshold)))

    blobs.sort(key=lambda b: -b.alpha)
    kept: list[Blob] = []
    for b in blobs:
        if all(np.hypot(b.cx - k.cx, b.cy - k.cy) >= p.nms_factor * r for k in kept):
            kept.append(b)
    return kept, {"calibration": cal, "n_candidates": len(centres),
                  "radius": r, "psf": psf}


# --------------------------------------------------------------------------


def _edge_profile_bites(beam, r, p):
    """Find blockers from inward excursions of the beam boundary.

    DISABLED BY DEFAULT (``use_bites``), and the reason is worth recording.

    The idea is sound and the geometry below is right, but the instrument is
    not: on the real frames the beam boundary is heavily blurred, so a blocker
    that visibly eats ~60 px into the beam moves the *thresholded silhouette* by
    only 7 px. A soft bite barely registers in a hard outline. Measured on the
    real set, this channel found none of the confirmed edge bites and did add
    spurious detections along a frame whose top edge is irregular -- the wrong
    trade entirely for a system whose failure mode is false alarms.

    Left in, off, because with ground-truth coordinates it can be evaluated
    properly, and because the deep-bite case it targets is real and still
    unsolved. It should be driven off a sub-pixel boundary estimate (an
    edge-response fit rather than an Otsu silhouette) before being trusted.

    Differencing the silhouette against a fitted rectangle does locate bites,
    but it also picks up the beam's natural bowing and its corner rounding, and
    those merge with the real bites into components many times a disc in area --
    on a real frame one such component was 11x a disc, which is a false positive
    waiting to happen. Comparing the boundary to a *smoothed version of itself*
    separates the two cleanly: bowing is low frequency and survives the smoothing,
    a blocker bite is a localised excursion and does not.

    Geometry then gives the centre directly. A disc of radius r whose centre lies
    a distance (r - d) outside a straight edge cuts a bite of depth d and
    half-width sqrt(r^2 - (r-d)^2), so depth and width over-determine the disc
    and their consistency is itself the acceptance test -- no occupancy needed.
    That matters because the occupancy field cannot see these bites at all: the
    grey-scale closing that builds it cannot bridge a gap wider than its own
    structuring element.
    """
    H, W = beam.shape
    win = int(max(5, round(8 * r))) | 1
    out = []

    for side in ("top", "bottom", "left", "right"):
        arr = beam if side in ("top", "bottom") else beam.T
        n_along = arr.shape[1]
        first = np.full(n_along, np.nan)
        for u in range(n_along):
            col = np.nonzero(arr[:, u])[0]
            if len(col) == 0:
                continue
            first[u] = col[0] if side in ("top", "left") else col[-1]
        ok = np.isfinite(first)
        if ok.sum() < win:
            continue
        prof = np.copy(first)
        prof[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], first[ok])
        base = ndi.median_filter(prof, size=win, mode="nearest")
        dev = (prof - base) if side in ("top", "left") else (base - prof)
        dev[~ok] = 0.0

        mask = dev > p.bite_min_depth * r
        lab, n = ndi.label(mask)
        for i in range(1, n + 1):
            idx = np.nonzero(lab == i)[0]
            d = float(dev[idx].max())
            half_w = 0.5 * len(idx)
            expect = np.sqrt(max(r * r - (r - min(d, r)) ** 2, 1.0))
            # A genuine disc bite has width and depth in the relation above;
            # a scratch or a step in the beam edge does not.
            if not (p.bite_shape_lo * expect <= half_w <= p.bite_shape_hi * expect):
                continue
            u = float(idx[int(np.argmax(dev[idx]))])
            edge = float(base[int(u)])
            offset = r - min(d, r)
            if side == "top":
                c = (u, edge - offset)
            elif side == "bottom":
                c = (u, edge + offset)
            elif side == "left":
                c = (edge - offset, u)
            else:
                c = (edge + offset, u)
            out.append(c)
    return np.array(out, float) if out else np.empty((0, 2))


def _split_pass(y, valid, centres, r, psf, p):
    """Propose one extra source per cluster where the residual demands one.

    Fit the cluster, look at what is left over, and admit a new source only if
    it removes a large, statistically significant part of that residual *and*
    is itself opaque. Both conditions matter: the F-test alone will happily
    endorse a source that explains a sliver of noise.
    """
    D = _disc(r, psf)
    half = D.shape[0] // 2
    h, w = y.shape
    added = []
    for grp in _clusters(centres, 2.0 * r + 3.0 * psf):
        cs = centres[grp]
        x0 = int(max(0, np.floor(cs[:, 0].min()) - half - r))
        x1 = int(min(w, np.ceil(cs[:, 0].max()) + half + r))
        y0 = int(max(0, np.floor(cs[:, 1].min()) - half - r))
        y1 = int(min(h, np.ceil(cs[:, 1].max()) + half + r))
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        vm = valid[y0:y1, x0:x1]
        if vm.sum() < 20:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        X = np.stack([_tmpl(xx, yy, cx, cy, r, psf)[vm] for cx, cy in cs], axis=1)
        target = y[y0:y1, x0:x1][vm]
        sol, *_ = np.linalg.lstsq(X, target, rcond=None)
        res = target - X @ sol
        rss = float(res @ res)
        dof = max(len(target) - len(grp) - 1, 1)

        rimg = np.zeros((y1 - y0, x1 - x0), np.float32)
        rimg[vm] = res
        rimg = ndi.gaussian_filter(rimg, max(1.0, 0.35 * r))
        for cx, cy in cs:
            rimg[np.hypot(xx - cx, yy - cy) < 0.7 * r] = -np.inf
        rimg[~vm] = -np.inf
        if not np.isfinite(rimg).any():
            continue
        jy, jx = np.unravel_index(int(rimg.argmax()), rimg.shape)
        ncx, ncy = float(x0 + jx), float(y0 + jy)

        X2 = np.concatenate([X, _tmpl(xx, yy, ncx, ncy, r, psf)[vm][:, None]], axis=1)
        sol2, *_ = np.linalg.lstsq(X2, target, rcond=None)
        res2 = target - X2 @ sol2
        rss2 = float(res2 @ res2)
        f = (rss - rss2) / max(rss2 / dof, 1e-12)
        # Require *every* source in the split cluster to remain opaque, not
        # just the new one. Splitting a single blocker in two yields a pair of
        # half-strength discs that each pass an "is the new source real" test
        # while jointly destroying a detection that was correct before the
        # split -- measured, this turned a clean alpha of 0.94 into 0.60 and
        # then into nothing.
        if f > p.f_split and float(np.min(sol2)) > p.alpha_accept:
            added.append((ncx, ncy))
    return np.array(added, float) if added else np.empty((0, 2))


def _tmpl(xx, yy, cx, cy, r, psf):
    d = np.hypot(xx - cx, yy - cy)
    t = np.clip(r + 0.5 - d, 0.0, 1.0).astype(np.float32)
    return ndi.gaussian_filter(t, psf) if psf > 0 else t


def _clusters(centres, reach):
    n = len(centres)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(centres[i] - centres[j])) < reach:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    g = {}
    for i in range(n):
        g.setdefault(find(i), []).append(i)
    return list(g.values())


def _joint(y, valid, centres, r, psf):
    """Fit overlapping candidates together so neither steals the other's pixels."""
    n = len(centres)
    alphas = np.full(n, np.nan)
    sigmas = np.full(n, np.inf)
    D = _disc(r, psf)
    half = D.shape[0] // 2
    h, w = y.shape

    for grp in _clusters(centres, 2.0 * r + 3.0 * psf):
        cs = centres[grp]
        x0 = int(max(0, np.floor(cs[:, 0].min()) - half - 1))
        x1 = int(min(w, np.ceil(cs[:, 0].max()) + half + 2))
        y0 = int(max(0, np.floor(cs[:, 1].min()) - half - 1))
        y1 = int(min(h, np.ceil(cs[:, 1].max()) + half + 2))
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        vm = valid[y0:y1, x0:x1]
        if vm.sum() < 12:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        cols = []
        for cx, cy in cs:
            d = np.hypot(xx - cx, yy - cy)
            t = np.clip(r + 0.5 - d, 0.0, 1.0).astype(np.float32)
            cols.append((ndi.gaussian_filter(t, psf) if psf > 0 else t)[vm])
        X = np.stack(cols, axis=1)
        target = y[y0:y1, x0:x1][vm]
        XtX = X.T @ X
        ridge = 1e-6 * float(np.trace(XtX)) / max(len(grp), 1)
        try:
            inv = np.linalg.inv(XtX + ridge * np.eye(len(grp)))
        except np.linalg.LinAlgError:
            continue
        a = inv @ (X.T @ target)
        res = target - X @ a
        noise = float(1.4826 * np.median(np.abs(res - np.median(res))) + 1e-4)
        var = noise * noise * np.diag(inv)
        for k, i in enumerate(grp):
            alphas[i] = float(a[k])
            sigmas[i] = float(np.sqrt(max(var[k], 0.0)))
    return alphas, sigmas


def _valid_fraction(cx, cy, r, valid):
    h, w = valid.shape
    x0, x1 = int(max(0, cx - r - 1)), int(min(w, cx + r + 2))
    y0, y1 = int(max(0, cy - r - 1)), int(min(h, cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return float((d & valid[y0:y1, x0:x1]).sum() / max(np.pi * r * r, 1.0))
