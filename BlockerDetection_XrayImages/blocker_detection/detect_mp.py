"""Occupancy-based refinement of blind detections (and a pure-pursuit variant).

The FRST + watershed detector's residual errors were almost entirely heavily
overlapping pairs -- centres closer than ~1.2 radii, where the two discs merge
into one elongated blob. Peak-picking cannot fix that, because there is only one
peak to pick; the blob has to be *explained* rather than segmented.

Matching pursuit does exactly that. Repeatedly take the strongest matched-filter
response, record it as a source, subtract that source's contribution, and
continue on the residual. A merged pair is resolved by the second pass finding
what the first pass left behind. This is the CLEAN algorithm from radio
astronomy, and merged point sources are precisely the problem it was built for.

Two details make it work here rather than merely run:

* Detection uses matched-filter SNR (num/sqrt(den)), while amplitude uses
  occupancy (num/den). Picking peaks by occupancy would reward positions with a
  sliver of valid support, where a single dark pixel scores 1.0.
* Deflation subtracts the template *autocorrelation*, not the template. The
  response map is already correlated with the template, so removing a source
  from it means removing its imprint on that map.

The greedy pass is deliberately loose; a joint least-squares refit at the end
cleans up amplitudes, which is what actually decides acceptance.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .pipeline import Detection, Params, bootstrap_radius, ideal_die, segment_die
from .verify import (
    VerifyParams,
    _alpha_maps,
    _joint_fit,
    disc_template,
    estimate_psf,
    shortfall_field,
)


def detect_mp(
    image: np.ndarray,
    *,
    radius: float | None = None,
    alpha_stop: float = 0.45,
    max_sources: int = 200,
    nms_factor: float = 0.55,
    close_factor: float = 1.9,
    min_valid_fraction: float = 0.15,
    alpha_keep: float = 0.50,
):
    """Detect blockers with no prior on their positions.

    Returns ``(detections, info)`` matching :func:`pipeline.detect`.
    """
    img = np.asarray(image, np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    beam = segment_die(img)
    r = float(radius) if radius else bootstrap_radius(img, beam)
    beam_ideal = ideal_die(beam, r, close_factor)
    # Refine the radius on the residual, where edge bites are visible too.
    if radius is None:
        y0 = shortfall_field(img, beam_ideal, r, close_factor)
        r = _refine_radius(y0, beam_ideal, r) or r
        beam_ideal = ideal_die(beam, r, close_factor)

    y = shortfall_field(img, beam_ideal, r, close_factor)

    vp = VerifyParams()
    seeds = _coarse_peaks(y, beam_ideal, r)
    psf = estimate_psf(y, beam_ideal, r, seeds, vp.psf_grid)

    D = disc_template(r, psf)
    num, den = _alpha_maps(y, beam_ideal, D)
    # Imprint of one source on the response map: the template autocorrelation.
    AC = fftconvolve(D, D[::-1, ::-1], mode="full").astype(np.float32)
    ah = AC.shape[0] // 2

    h, w = y.shape
    den_max = float(den.max()) if den.size else 1.0
    support = den > max(0.12 * den_max, 1e-6)
    picked: list[tuple[float, float]] = []
    work = num.copy()
    sep = max(2, int(round(nms_factor * r)))

    for _ in range(max_sources):
        snr = np.where(support, work / np.sqrt(np.maximum(den, 1e-6)), -np.inf)
        iy, ix = np.unravel_index(int(snr.argmax()), snr.shape)
        a = float(work[iy, ix] / max(den[iy, ix], 1e-6))
        if not np.isfinite(a) or a < alpha_stop:
            break
        picked.append((float(ix), float(iy)))
        _deflate(work, AC, ah, ix, iy, a)
        # Block the immediate neighbourhood so the next pick is a different
        # source rather than a leftover shoulder of this one.
        y0b, y1b = max(0, iy - sep), min(h, iy + sep + 1)
        x0b, x1b = max(0, ix - sep), min(w, ix + sep + 1)
        support[y0b:y1b, x0b:x1b] = False

    if not picked:
        return [], {"radius": r, "psf": psf, "beam_ideal": beam_ideal}

    centres = np.array(picked, float)
    alphas, sigmas, rho, _f = _joint_fit(y, beam_ideal, centres, r, psf, vp)

    dets: list[Detection] = []
    for (cx, cy), a, sg in zip(centres, alphas, sigmas):
        if not np.isfinite(a) or a < alpha_keep:
            continue
        vf = _valid_frac(cx, cy, r, beam_ideal)
        if vf < min_valid_fraction:
            continue
        dets.append(Detection(float(cx), float(cy), float(r), float(min(a, 1.5)),
                              float(a), 0.0, float(vf), "matching-pursuit"))

    dets.sort(key=lambda d: -d.score)
    kept: list[Detection] = []
    for d in dets:
        if all(np.hypot(d.cx - k.cx, d.cy - k.cy) >= nms_factor * r for k in kept):
            kept.append(d)
    return kept, {"radius": r, "psf": psf, "beam_ideal": beam_ideal,
                  "n_pursuit": len(picked)}


# --------------------------------------------------------------------------


def _deflate(work, AC, ah, ix, iy, a):
    h, w = work.shape
    y0, y1 = max(0, iy - ah), min(h, iy + ah + 1)
    x0, x1 = max(0, ix - ah), min(w, ix + ah + 1)
    work[y0:y1, x0:x1] -= a * AC[y0 - iy + ah : y1 - iy + ah,
                                 x0 - ix + ah : x1 - ix + ah]


def _coarse_peaks(y, valid, r, limit=25):
    """Cheap seeds for the PSF search: local maxima of the smoothed field."""
    from scipy import ndimage as ndi

    sm = ndi.gaussian_filter(y, max(1.0, 0.3 * r))
    mx = ndi.maximum_filter(sm, size=int(max(3, 1.5 * r)))
    pk = (sm == mx) & valid & (sm > 0.4)
    ys, xs = np.nonzero(pk)
    if len(ys) == 0:
        return np.empty((0, 2))
    order = np.argsort(-sm[ys, xs])[:limit]
    return np.stack([xs[order], ys[order]], axis=1).astype(float)


def _refine_radius(y, valid, r0):
    """Pick the radius whose template best explains the field, near r0."""
    best, best_score = None, -np.inf
    for f in (0.8, 0.9, 1.0, 1.1, 1.2):
        r = r0 * f
        D = disc_template(r, max(1.0, 0.15 * r))
        num, den = _alpha_maps(y, valid, D)
        snr = num / np.sqrt(np.maximum(den, 1e-6))
        score = float(np.percentile(snr[valid], 99.5)) if valid.any() else -np.inf
        if score > best_score:
            best, best_score = r, score
    return best


def _valid_frac(cx, cy, r, valid):
    h, w = valid.shape
    x0, x1 = int(max(0, cx - r - 1)), int(min(w, cx + r + 2))
    y0, y1 = int(max(0, cy - r - 1)), int(min(h, cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return float((d & valid[y0:y1, x0:x1]).sum() / max(np.pi * r * r, 1.0))


# --------------------------------------------------------------------------
# Refinement: the form of this machinery that actually beats the baseline.
# --------------------------------------------------------------------------


def refine_detections(image, dets, *, radius=None, close_factor=1.9,
                      alpha_keep=0.30, f_add=60.0, max_extra=2,
                      min_valid_fraction=0.15):
    """Prune and complete a candidate list using the occupancy model.

    Pure matching pursuit run from scratch over-picks badly -- every shoulder of
    a real blocker and every mottle excursion clears a fixed amplitude
    threshold, and the greedy pass has no notion of how many sources the data
    actually supports. Run over an existing candidate list instead, the same
    model does the two things peak-picking cannot:

    * **Prune.** A candidate whose fitted occupancy is low is not a blocker,
      whatever its symmetry response said. Ring artifacts die here.
    * **Complete.** Within a cluster, test whether adding one more source at the
      largest residual peak significantly reduces the residual sum of squares.
      That is the merged-pair case: the second blocker announces itself as
      structure the one-source model cannot explain.

    Measured honestly, this buys precision and not recall: on the blind
    benchmark it moves 0.970/0.962 to 0.962/0.975 with edge recall still 1.000.
    The add-one test almost never fires at any threshold tried, so merged pairs
    remain blind detection's real limit. That limit is not hidden in
    verification mode -- the same pairs surface there as ``rho`` entangled sites
    routed to REVIEW, and the recipe already knows two blockers are present.
    """
    img = np.asarray(image, np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    beam = segment_die(img)
    r = float(radius) if radius else (float(np.median([d.r for d in dets])) if dets
                                      else bootstrap_radius(img, beam))
    beam_ideal = ideal_die(beam, r, close_factor)
    y = shortfall_field(img, beam_ideal, r, close_factor)

    vp = VerifyParams()
    centres = np.array([[d.cx, d.cy] for d in dets], float) if dets else np.empty((0, 2))
    if len(centres) == 0:
        return [], {"radius": r}
    psf = estimate_psf(y, beam_ideal, r, centres, vp.psf_grid)

    for _ in range(max_extra):
        extra = _add_one_pass(y, beam_ideal, centres, r, psf, vp, f_add, alpha_keep)
        if not len(extra):
            break
        centres = np.vstack([centres, extra])

    alphas, sigmas, rho, _f = _joint_fit(y, beam_ideal, centres, r, psf, vp)
    out = []
    for (cx, cy), a in zip(centres, alphas):
        if not np.isfinite(a) or a < alpha_keep:
            continue
        vf = _valid_frac(cx, cy, r, beam_ideal)
        if vf < min_valid_fraction:
            continue
        out.append(Detection(float(cx), float(cy), float(r), float(min(a, 1.5)),
                             float(a), 0.0, float(vf), "occupancy-refined"))
    out.sort(key=lambda d: -d.score)
    kept = []
    for d in out:
        if all(np.hypot(d.cx - k.cx, d.cy - k.cy) >= 0.55 * r for k in kept):
            kept.append(d)
    return kept, {"radius": r, "psf": psf, "beam_ideal": beam_ideal,
                  "n_added": len(centres) - len(dets)}


def _add_one_pass(y, valid, centres, r, psf, vp, f_add, alpha_keep):
    """Propose at most one extra source per cluster, where the residual demands it."""
    from scipy import ndimage as ndi

    D = disc_template(r, psf)
    half = D.shape[0] // 2
    h, w = y.shape
    added = []
    for grp in _clusters_local(centres, 2.0 * r + 3.0 * psf):
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
        cols = [_tmpl(xx, yy, cx, cy, r, psf)[vm] for cx, cy in cs]
        X = np.stack(cols, axis=1)
        target = y[y0:y1, x0:x1][vm]
        sol, *_ = np.linalg.lstsq(X, target, rcond=None)
        res = target - X @ sol
        rss = float(res @ res)
        dof = max(len(target) - len(grp) - 1, 1)

        resid_img = np.zeros((y1 - y0, x1 - x0), np.float32)
        resid_img[vm] = res
        resid_img = ndi.gaussian_filter(resid_img, max(1.0, 0.3 * r))
        # Do not re-propose a source on top of an existing one.
        for cx, cy in cs:
            d = np.hypot(xx - cx, yy - cy)
            resid_img[d < 0.55 * r] = -np.inf
        resid_img[~vm] = -np.inf
        if not np.isfinite(resid_img).any():
            continue
        jy, jx = np.unravel_index(int(resid_img.argmax()), resid_img.shape)
        ncx, ncy = float(x0 + jx), float(y0 + jy)

        X2 = np.concatenate([X, _tmpl(xx, yy, ncx, ncy, r, psf)[vm][:, None]], axis=1)
        sol2, *_ = np.linalg.lstsq(X2, target, rcond=None)
        res2 = target - X2 @ sol2
        rss2 = float(res2 @ res2)
        f = (rss - rss2) / max(rss2 / dof, 1e-12)
        if f > f_add and sol2[-1] > alpha_keep:
            added.append((ncx, ncy))
    return np.array(added, float) if added else np.empty((0, 2))


def _tmpl(xx, yy, cx, cy, r, psf):
    from scipy import ndimage as ndi

    d = np.hypot(xx - cx, yy - cy)
    t = np.clip(r + 0.5 - d, 0.0, 1.0).astype(np.float32)
    return ndi.gaussian_filter(t, psf) if psf > 0 else t


def _clusters_local(centres, reach):
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
