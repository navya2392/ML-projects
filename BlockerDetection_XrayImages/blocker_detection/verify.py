"""Verification at known blocker sites -- the mode that matches the real task.

The customer's system already knows where every blocker should be. Its failure
is not localisation, it is the *decision*: at a beam-edge site it cannot find a
full circle, so it reports "not found" even though a human can plainly see the
blocker. Every frame in the bad set has the blocker present. The whole problem
is therefore a per-site binary decision that must not fall over when the disc is
truncated.

The estimator is a matched filter for **occupancy**:

    observed_shortfall(x) ~= alpha * template(x)         for x in valid pixels

where ``template`` is a disc of the known radius blurred by the frame's PSF,
``alpha`` is how much of that disc is actually present (1 = full blocker,
0 = bare beam), and ``valid`` is the reconstructed beam footprint. Least squares
gives

    alpha_hat = sum(y*D) / sum(D*D),   sigma_alpha = noise / sqrt(sum(D*D))

Both sums run **only over valid pixels**. That single restriction is what makes
truncation a non-issue: a blocker with 35% of its disc inside the beam is
measured over that 35%, and alpha is unbiased -- just noisier. There is no
circularity test, no area gate and no "must find a full circle" anywhere in the
decision, because those are exactly the tests truncation breaks.

The noisiness is not discarded, it is the product. ``sigma_alpha`` grows as the
valid area shrinks, so a site with too little beam left to judge yields a wide
interval and is routed to REVIEW rather than guessed at. That is how a recall
guarantee is actually obtained: never force a call the evidence cannot support.

Two things beyond the estimator turned out to matter more than the estimator
itself, both learned from measuring rather than reasoning:

1. **Local re-centring dominates.** Every residual false alarm in testing was a
   *fully visible* blocker whose ROI sat 1.3-1.7 radii off target -- recipe
   coordinates misregistered, not truncation. Searching a bounded window around
   the nominal site fixes those completely. Beware the obvious trap: maximising
   alpha over offsets biases it upward, which could mask a genuinely absent
   blocker, so the window is kept below the minimum blocker separation and each
   claimed blob is checked for uniqueness.

2. **The background must come from a blocker-free model**, not from a local
   annulus. An annulus around a site can contain a neighbouring blocker, which
   drags the reference level down and makes a present blocker look shallow. The
   grey-closing reference is blocker-free by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from .pipeline import _grey_close, ideal_die, segment_die
from .register import similarity_from_pairs

PRESENT, MISSING, REVIEW = "present", "missing", "review"


@dataclass
class SiteResult:
    index: int
    cx: float  # refined centre actually used for the decision
    cy: float
    nominal_cx: float  # where the recipe said to look
    nominal_cy: float
    r: float
    alpha: float  # 1 = full blocker, 0 = bare beam
    sigma: float  # 1-sigma uncertainty on alpha
    shift: float  # px between nominal and refined centre
    valid_fraction: float  # fraction of the disc inside the beam footprint
    verdict: str
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerifyParams:
    alpha_present: float = 0.60  # occupancy consistent with a blocker being there
    alpha_missing: float = 0.25  # occupancy consistent with bare beam
    k_sigma: float = 3.0  # confidence margin, in sigmas
    sigma_floor: float = 0.03  # systematic model error dominates below this
    rho_entangled: float = 0.50  # template correlation above which sites merge
    f_needed: float = 25.0  # drop-one F above which a site is demonstrably needed
    min_valid_fraction: float = 0.12  # below this there is nothing to judge
    search_radius: float = 0.85  # re-centring window, in radii
    max_residual_ok: float = 0.45  # departure from the consensus fit -> REVIEW
    dup_distance: float = 0.60  # two sites landing this close, in radii, -> REVIEW
    close_factor: float = 1.9
    psf_grid: tuple = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 13.0)


# --------------------------------------------------------------------------
# shortfall field and template
# --------------------------------------------------------------------------


def shortfall_field(image: np.ndarray, beam_ideal: np.ndarray, radius: float,
                    close_factor: float):
    """Normalised occupancy field: 0 = bare beam, 1 = fully opaque.

    Uses the grey-closing reconstruction as the blocker-free beam level, so the
    normalisation is immune to the bright rim, the illumination gradient and to
    neighbouring blockers -- all of which corrupt a local-annulus reference.
    """
    ref = _grey_close(image, radius * close_factor)
    y = np.clip(ref - image, 0.0, None) / np.clip(ref, 1e-3, None)
    y = np.clip(y, 0.0, 1.5).astype(np.float32)
    y[~beam_ideal] = 0.0
    return y


def disc_template(radius: float, sigma: float) -> np.ndarray:
    half = int(np.ceil(radius + 3 * sigma)) + 2
    size = 2 * half + 1
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.hypot(xx - half, yy - half)
    disc = np.clip(radius + 0.5 - d, 0.0, 1.0).astype(np.float32)
    return ndi.gaussian_filter(disc, sigma) if sigma > 0 else disc


def _alpha_maps(y: np.ndarray, valid: np.ndarray, D: np.ndarray):
    """Masked least-squares occupancy at *every* offset, in two convolutions.

    num(x) = sum_u y(x+u) D(u) restricted to valid pixels
    den(x) = sum_u valid(x+u) D(u)^2
    alpha  = num / den

    Computing it densely rather than per-candidate is what makes the bounded
    re-centring search affordable: one pair of convolutions serves every site
    and every offset.
    """
    # FFT convolution, not spatial: the template is tens of pixels across, so
    # the direct form costs ~10^10 operations per frame and dominates runtime.
    v = valid.astype(np.float32)
    num = fftconvolve(y * v, D, mode="same")
    den = fftconvolve(v, D.astype(np.float64) ** 2, mode="same")
    return num.astype(np.float32), den.astype(np.float32)


def estimate_psf(y, valid, radius, seeds, grid):
    """Choose the PSF width whose template best explains confident sites.

    The frames span crisp to heavily smoothed. alpha is only comparable across
    frames if the template matches the frame's actual blur, so blur is measured
    per frame rather than assumed.
    """
    best, best_score = grid[len(grid) // 2], -np.inf
    h, w = y.shape
    for sigma in grid:
        D = disc_template(radius, sigma)
        half = D.shape[0] // 2
        # Evaluated on small crops around each site rather than via full-image
        # maps: the search runs once per candidate blur, and only the site
        # neighbourhoods are ever read.
        vals = []
        for cx, cy in seeds:
            ix, iy = int(round(cx)), int(round(cy))
            x0, x1 = max(0, ix - half), min(w, ix + half + 1)
            y0, y1 = max(0, iy - half), min(h, iy + half + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            Dm = D[y0 - iy + half : y1 - iy + half, x0 - ix + half : x1 - ix + half]
            vm = valid[y0:y1, x0:x1]
            den_i = float(((Dm * Dm) * vm).sum())
            if den_i < 1e-3:
                continue
            vals.append(float((y[y0:y1, x0:x1] * Dm * vm).sum()) / den_i)
        if not vals:
            continue
        # A template that matches the true blur puts occupancy near 1 at real
        # blockers; one that is too sharp or too broad dilutes it.
        score = -abs(float(np.median(vals)) - 1.0)
        if score > best_score:
            best, best_score = sigma, score
    return best


# --------------------------------------------------------------------------



def _clusters(centres, reach):
    """Single-linkage grouping of sites whose templates overlap."""
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
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _joint_fit(y, valid, centres, radius, psf, p):
    """Fit all overlapping sites in a cluster simultaneously.

    Touching blockers are the other half of the problem: two sites 1.2 radii
    apart share pixels, and fitting them independently lets one site's blocker
    explain the other site's evidence -- which is exactly how a genuinely absent
    blocker could be scored present. Solving the cluster as one linear system
    attributes the shared pixels correctly.

    The covariance carries the payoff. When two templates overlap so heavily
    that their occupancies are not separately identifiable, (X'X)^-1 blows up
    and both sites abstain, instead of the pair being silently split by an
    arbitrary tie-break.
    """
    n = len(centres)
    alphas = np.full(n, np.nan)
    sigmas = np.full(n, np.inf)
    rho = np.zeros(n)  # max template correlation with a cluster neighbour
    fstat = np.full(n, np.inf)  # drop-one significance
    D = disc_template(radius, psf)
    half = D.shape[0] // 2
    h, w = y.shape

    for grp in _clusters(centres, 2.0 * radius + 3.0 * psf):
        cs = centres[grp]
        x0 = int(max(0, np.floor(cs[:, 0].min()) - half - 1))
        x1 = int(min(w, np.ceil(cs[:, 0].max()) + half + 2))
        y0 = int(max(0, np.floor(cs[:, 1].min()) - half - 1))
        y1 = int(min(h, np.ceil(cs[:, 1].max()) + half + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        vm = valid[y0:y1, x0:x1]
        if vm.sum() < 12:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        cols = []
        for cx, cy in cs:
            d = np.hypot(xx - cx, yy - cy)
            t = np.clip(radius + 0.5 - d, 0.0, 1.0).astype(np.float32)
            if psf > 0:
                t = ndi.gaussian_filter(t, psf)
            cols.append(t[vm])
        X = np.stack(cols, axis=1)
        target = y[y0:y1, x0:x1][vm]
        XtX = X.T @ X
        # Tiny ridge: keeps the solve stable without masking true degeneracy,
        # which still shows up as a large variance rather than a silent guess.
        ridge = 1e-6 * float(np.trace(XtX)) / max(len(grp), 1)
        try:
            inv = np.linalg.inv(XtX + ridge * np.eye(len(grp)))
        except np.linalg.LinAlgError:
            continue
        a = inv @ (X.T @ target)
        res = target - X @ a
        rss_full = float(res @ res)
        noise = float(1.4826 * np.median(np.abs(res - np.median(res))) + 1e-4)
        var = noise * noise * np.diag(inv)
        dof = max(len(target) - len(grp), 1)

        # Column correlation measures how entangled a site is with its
        # neighbours. Above ~0.5 the individual occupancies stop being
        # separately identifiable and the split between them is arbitrary.
        norms = np.linalg.norm(X, axis=0) + 1e-9
        C = np.abs((X.T @ X) / np.outer(norms, norms))
        np.fill_diagonal(C, 0.0)

        for k, i in enumerate(grp):
            alphas[i] = float(a[k])
            sigmas[i] = float(np.sqrt(max(var[k], 0.0)))
            rho[i] = float(C[k].max()) if len(grp) > 1 else 0.0
            # Drop-one test: refit without this site and see whether the data
            # still explains itself. If removing the column barely hurts, this
            # site's blocker is not *needed* -- which is evidence of absence
            # only when it is also not merely masked by an overlapping
            # neighbour. That distinction is what the rho flag carries.
            if len(grp) > 1:
                Xr = np.delete(X, k, axis=1)
                sol, *_ = np.linalg.lstsq(Xr, target, rcond=None)
                rr = target - Xr @ sol
                rss_r = float(rr @ rr)
                fstat[i] = (rss_r - rss_full) / max(rss_full / dof, 1e-12)
            else:
                fstat[i] = np.inf
    return alphas, sigmas, rho, fstat


def _search_windows(pts, radius, p):
    """Per-site search radius, capped at half the distance to the nearest site.

    A window that reaches past the midpoint to a neighbour lets two sites in a
    touching pair swap targets, and a swap is the one failure that produces a
    *confident* wrong answer rather than an uncertain one. The recipe already
    states the spacing, so the cap costs nothing.
    """
    n = len(pts)
    win = np.full(n, p.search_radius * radius)
    if n > 1:
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        win = np.minimum(win, 0.45 * d.min(axis=1))
    return np.maximum(win, 1.0)


def _recentre_all(num, den, pts, radius, p, windows=None):
    """Bounded matched-filter re-centring for every site; returns refined centres."""
    h, w = num.shape
    if windows is None:
        windows = _search_windows(np.asarray(pts, float), radius, p)
    den_full = float(den.max()) if den.size else 1.0
    out = []
    for (cx0, cy0), wr in zip(pts, windows):
        win = max(1, int(round(wr)))
        ix, iy = int(round(cx0)), int(round(cy0))
        x0, x1 = max(0, ix - win), min(w, ix + win + 1)
        y0, y1 = max(0, iy - win), min(h, iy + win + 1)
        if x1 <= x0 or y1 <= y0:
            out.append((cx0, cy0))
            continue
        sub_d = den[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        msk = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= win * win
        usable = msk & (sub_d > 0.15 * den_full)
        if not usable.any():
            usable = msk & (sub_d > 0.02 * den_full)
        if not usable.any():
            out.append((cx0, cy0))
            continue
        snr = num[y0:y1, x0:x1] / np.sqrt(np.maximum(sub_d, 1e-6))
        jy, jx = np.unravel_index(int(np.where(usable, snr, -np.inf).argmax()), snr.shape)
        # Parabolic sub-pixel peak. Integer-pixel centres inject up to half a
        # pixel of error each, which propagates straight into the consensus
        # residual that decides whether a site gets flagged.
        dx = dy = 0.0
        if 0 < jx < snr.shape[1] - 1:
            a0, b0, c0 = snr[jy, jx - 1], snr[jy, jx], snr[jy, jx + 1]
            den_x = a0 - 2 * b0 + c0
            if abs(den_x) > 1e-9:
                dx = float(np.clip(0.5 * (a0 - c0) / den_x, -1, 1))
        if 0 < jy < snr.shape[0] - 1:
            a0, b0, c0 = snr[jy - 1, jx], snr[jy, jx], snr[jy + 1, jx]
            den_y = a0 - 2 * b0 + c0
            if abs(den_y) > 1e-9:
                dy = float(np.clip(0.5 * (a0 - c0) / den_y, -1, 1))
        out.append((float(x0 + jx) + dx, float(y0 + jy) + dy))
    return np.array(out, float)


def verify(
    image: np.ndarray,
    expected_xy: np.ndarray,
    radius: float,
    *,
    params: VerifyParams | None = None,
    beam_ideal: np.ndarray | None = None,
) -> tuple[list[SiteResult], dict]:
    """Decide present / missing / review at each expected site.

    ``expected_xy`` must already be in pixel coordinates for this frame; use
    :mod:`blocker_detection.register` to bring recipe coordinates into it.
    """
    p = params or VerifyParams()
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    if beam_ideal is None:
        beam_ideal = ideal_die(segment_die(img), radius, p.close_factor)

    pts = np.asarray(expected_xy, float).reshape(-1, 2)
    y = shortfall_field(img, beam_ideal, radius, p.close_factor)

    # PSF has to be measured at the *right* pixels. Estimating it at nominal
    # coordinates that are misregistered by a radius makes every template look
    # too sharp, and the fit compensates by inflating alpha. So: localise with a
    # provisional template, then re-estimate blur on the corrected centres.
    sigma = estimate_psf(y, beam_ideal, radius, pts, p.psf_grid)
    windows = _search_windows(pts, radius, p)
    seeds = pts
    for _ in range(2):
        D = disc_template(radius, sigma)
        num, den = _alpha_maps(y, beam_ideal, D)
        seeds = _recentre_all(num, den, pts, radius, p, windows)
        new_sigma = estimate_psf(y, beam_ideal, radius, seeds, p.psf_grid)
        if new_sigma == sigma:
            break
        sigma = new_sigma
    D = disc_template(radius, sigma)
    num, den = _alpha_maps(y, beam_ideal, D)
    alpha_map = num / np.maximum(den, 1e-6)

    seeds = _recentre_all(num, den, pts, radius, p, windows)

    # ---- consensus registration ----
    # A site that moves *with* its neighbours has corrected a global
    # misregistration and is trustworthy. A site that moves *against* them has
    # locked onto something else. Only the second is suspicious, so the flag is
    # on the residual from consensus, never on raw displacement.
    seed_alpha = np.array([
        alpha_map[int(np.clip(round(cy), 0, img.shape[0] - 1)),
                  int(np.clip(round(cx), 0, img.shape[1] - 1))]
        for cx, cy in seeds
    ])
    conf = seed_alpha > 0.55
    predicted = seeds.copy()
    consensus_rms = float("nan")
    if conf.sum() >= 4:
        A, t = similarity_from_pairs(pts[conf], seeds[conf])
        for _ in range(3):
            pred = pts @ A.T + t
            res = np.linalg.norm(seeds - pred, axis=1)
            keep = conf & (res < max(0.4 * radius, 2.5 * float(np.median(res[conf])) + 1e-6))
            if keep.sum() < 4:
                break
            A, t = similarity_from_pairs(pts[keep], seeds[keep])
        predicted = pts @ A.T + t
        consensus_rms = float(np.sqrt(np.mean(
            np.linalg.norm(seeds[conf] - predicted[conf], axis=1) ** 2)))
    residuals = np.linalg.norm(seeds - predicted, axis=1)

    # ---- final positions ----
    # Take positions from the consensus transform, not from independent local
    # search. The recipe knows two dumbbell sites are 1.2 radii apart; the
    # transform preserves that, whereas independent argmax lets both collapse
    # onto the same blob. Only a tight nudge is allowed on top, too small to
    # merge neighbours.
    tight = replace(p, search_radius=0.22)
    final = _recentre_all(num, den, predicted, radius, tight,
                          np.minimum(windows, 0.22 * radius))

    # A site is suspicious when it disagrees with its neighbours by more than
    # this frame's own registration precision -- not by a fixed distance. On a
    # frame where registration is inherently noisy, a fixed threshold flags
    # every site; on a frame where it is tight, it misses the real outlier.
    res_thr = float(np.clip(3.0 * consensus_rms if np.isfinite(consensus_rms) else np.inf,
                            p.max_residual_ok * radius, 0.95 * radius))

    alphas, sigmas, rho, fstat = _joint_fit(y, beam_ideal, final, radius, sigma, p)

    results: list[SiteResult] = []
    for i, (cx, cy) in enumerate(final):
        cx0, cy0 = pts[i]
        a, sig = alphas[i], sigmas[i]
        vfrac = _valid_fraction(cx, cy, radius, beam_ideal)
        shift = float(np.hypot(cx - cx0, cy - cy0))

        if not np.isfinite(a):
            v, why = REVIEW, "no usable beam pixels at this site"
        elif vfrac < p.min_valid_fraction:
            v, why = REVIEW, f"only {vfrac:.0%} of the disc lies inside the beam"
        elif residuals[i] > res_thr:
            v, why = REVIEW, (f"moved {residuals[i] / radius:.2f}r against the consensus "
                              f"registration; check alignment at this site")
        else:
            # Decide by *consistency with a hypothesis*, not by which side of a
            # line the estimate falls on. An occupancy of 0.27 +- 0.001 is not a
            # missing blocker -- it is confidently neither present nor absent,
            # i.e. the model does not describe this site, and the only safe
            # answer is to ask a human. Reading it as "below threshold" is
            # precisely the reasoning that turns a model failure into a false
            # alarm.
            sig = max(sig, p.sigma_floor)
            lo, hi = a - p.k_sigma * sig, a + p.k_sigma * sig
            could_be_present = hi > p.alpha_present
            could_be_absent = lo < p.alpha_missing

            if rho[i] > p.rho_entangled:
                # Heavily overlapping neighbour: how the shared pixels are split
                # between the two sites is arbitrary, so the individual
                # occupancy means little. Only the drop-one test is meaningful,
                # and MISSING is withheld entirely -- "absent" and "hidden under
                # the neighbour" are indistinguishable here, and guessing
                # between them is exactly how a false alarm gets manufactured.
                # The drop-one F-statistic alone is not enough: with the very
                # low residual noise of a clean frame its denominator collapses
                # and it reports overwhelming significance for a site whose
                # fitted occupancy is essentially zero. Significance says the
                # column matters; only alpha says a blocker is there. Both are
                # required, and MISSING stays unavailable, because "absent" and
                # "hidden under the neighbour" are indistinguishable here.
                if fstat[i] > p.f_needed and could_be_present:
                    v, why = PRESENT, ""
                else:
                    v, why = REVIEW, (f"overlaps a neighbouring blocker "
                                      f"(rho={rho[i]:.2f}); occupancy not separable")
            elif could_be_present and not could_be_absent:
                v, why = PRESENT, ""
            elif could_be_absent and not could_be_present:
                v, why = MISSING, "occupancy consistent with bare beam"
            elif could_be_present and could_be_absent:
                v, why = REVIEW, f"alpha={a:.2f}+-{sig:.2f} cannot separate the hypotheses"
            else:
                v, why = REVIEW, (f"alpha={a:.2f}+-{sig:.2f} fits neither a full blocker "
                                  f"nor bare beam; model misfit at this site")

        results.append(SiteResult(i, float(cx), float(cy), float(cx0), float(cy0),
                                  radius, float(a), float(sig), shift, float(vfrac),
                                  v, why))

    return results, {"psf_sigma": sigma, "beam_ideal": beam_ideal, "radius": radius,
                     "shortfall": y, "consensus_rms": consensus_rms,
                     "registration_shift": float(np.median(np.linalg.norm(predicted - pts, axis=1)))}


def _valid_fraction(cx, cy, r, valid) -> float:
    h, w = valid.shape
    x0, x1 = int(max(0, cx - r - 1)), int(min(w, cx + r + 2))
    y0, y1 = int(max(0, cy - r - 1)), int(min(h, cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return float((d & valid[y0:y1, x0:x1]).sum() / max(np.pi * r * r, 1.0))


def _sigma_alpha(y, valid, D, cx, cy, alpha, dsupport) -> float:
    """Residual-based uncertainty, so noisy or barely-supported sites abstain."""
    half = D.shape[0] // 2
    h, w = y.shape
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - half), min(w, ix + half + 1)
    y0, y1 = max(0, iy - half), min(h, iy + half + 1)
    if x1 <= x0 or y1 <= y0 or dsupport <= 1e-6:
        return float("inf")
    sub = y[y0:y1, x0:x1]
    vm = valid[y0:y1, x0:x1]
    Dm = D[y0 - iy + half : y1 - iy + half, x0 - ix + half : x1 - ix + half]
    if vm.sum() < 12:
        return float("inf")
    res = (sub - alpha * Dm)[vm]
    noise = float(1.4826 * np.median(np.abs(res - np.median(res))) + 1e-4)
    return noise / np.sqrt(max(dsupport, 1e-6))
