"""Scale-space calibration of blocker radius, beam footprint and PSF.

The absolute-threshold bootstrap that worked on synthetic frames fails outright
on the real ISP_NF images. Those frames carry a strong internal brightness
gradient -- the beam runs about 0.30 in the middle and 0.75 near the rim -- so
Otsu applied inside the beam splits *the beam itself* into dark-centre and
bright-rim rather than separating blockers from beam. Measured on a real frame
it returned a threshold of 0.470 against a beam median of 0.392, called 46% of
the image "holes", and produced a 4.8 px radius for ~28 px blockers.

Nothing about an absolute grey level is trustworthy here. Two things are:

1. **Local shortfall.** Comparing each pixel to a grey-scale closing with a disc
   larger than any plausible blocker removes the gradient by construction, and
   the same operation folds beam-edge bites into the field (see pipeline.py).

2. **Scale selection by occupancy crossing.** Fit occupancy alpha against a disc
   template of radius r. While r is smaller than the true blocker the template
   sits entirely inside the dark core and alpha stays at ~1; once r exceeds it
   the template starts averaging in bright beam and alpha falls away. The true
   radius is therefore the largest r at which alpha is still 1 -- the knee, not
   a peak.

   Matched-filter SNR, the more obvious criterion, does *not* work here: it rose
   monotonically on every real frame and pinned the estimate to whatever cap the
   search was given (20 of 23 frames returned r_max). Occupancy is the quantity
   with a physical zero crossing. On a frame whose blockers measure 32-34 px by
   direct radial profiling, the crossing lands at 33.

Both are scale-free and gradient-free, which is what the real data demands.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from skimage import filters, measure

from .pipeline import _grey_close, ideal_die


@dataclass
class Calibration:
    radius: float
    psf: float
    beam: np.ndarray  # observed beam footprint, holes filled
    beam_ideal: np.ndarray  # with edge bites closed back in
    shortfall: np.ndarray  # occupancy field, 0 = bare beam, 1 = opaque
    rect_ideal: np.ndarray  # geometric beam outline, for edge-bite recovery
    curve: list  # (radius, occupancy, snr) per scale, for diagnostics


def segment_beam(image: np.ndarray, min_frac: float = 0.15) -> np.ndarray:
    """Beam footprint: bright region, largest component, holes filled.

    Otsu on the *whole* frame is well posed -- beam against black exterior is a
    genuinely bimodal split -- unlike Otsu inside the beam, which is not.
    """
    sm = ndi.gaussian_filter(image, 3.0)
    # The frame is trimodal -- black exterior, beam interior, bright rim -- so
    # plain two-class Otsu often splits beam from rim instead of exterior from
    # beam, which is what produced beam areas ranging 0.22 to 0.74 across the
    # real set. Three classes puts the first threshold where it belongs.
    try:
        thr = float(filters.threshold_multiotsu(sm, classes=3)[0])
    except Exception:
        thr = float(filters.threshold_otsu(sm))

    def largest(m):
        lab = measure.label(m)
        if lab.max() == 0:
            return np.zeros_like(m)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        return ndi.binary_fill_holes(lab == int(sizes.argmax()))

    mask = largest(sm > thr)
    if mask.mean() < min_frac:
        mask = largest(sm > 0.5 * thr)
    return mask


def _shortfall(image, ref_radius, beam_mask):
    ref = _grey_close(image, ref_radius)
    y = np.clip(ref - image, 0.0, None) / np.clip(ref, 1e-3, None)
    y = np.clip(y, 0.0, 1.5).astype(np.float32)
    y[~beam_mask] = 0.0
    return y


def _disc(r, sigma):
    half = int(np.ceil(r + 3 * sigma)) + 2
    n = 2 * half + 1
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.hypot(xx - half, yy - half)
    t = np.clip(r + 0.5 - d, 0.0, 1.0).astype(np.float32)
    return ndi.gaussian_filter(t, sigma) if sigma > 0 else t


def _peak_alpha(alpha, snr, beam, r, n_peaks):
    """Median occupancy at the strongest well-separated responses at this scale."""
    sep = max(3, int(round(2.0 * r)))
    mx = ndi.maximum_filter(snr, size=2 * sep + 1)
    pk = (snr == mx) & beam
    ys, xs = np.nonzero(pk)
    if len(ys) == 0:
        return float(np.percentile(alpha[beam], 99.9))
    order = np.argsort(-snr[ys, xs])[:n_peaks]
    return float(np.median(alpha[ys[order], xs[order]]))


def beam_footprint(beam: np.ndarray, radius: float, close_factor: float = 1.9):
    """Reconstruct the beam's un-bitten footprint, corners included.

    Morphological closing recovers a blocker bite taken out of a straight edge,
    because that bite is a concavity. It cannot recover one taken out of a
    *corner*: removing material at a convex corner just makes the corner look
    more rounded, and closing has no way to tell that from genuine rounding. On
    the real frames this silently deleted corner blockers -- they never even
    became candidates, because they fell outside the valid mask entirely.

    Geometry supplies what morphology cannot. The beam is a rounded rectangle,
    so its minimum-area enclosing rectangle is unaffected by bites (a bite only
    removes material, never extends the outline), and the corner rounding can be
    read off whichever corners are intact -- taking the tightest of the four,
    since a bitten corner can only look more cut than a clean one.

    The result is intersected with a bounded dilation of the observed beam, so a
    misfit rectangle can never introduce a large region of true exterior, where
    black sky would read as a perfectly opaque blocker.
    """
    u8 = beam.astype(np.uint8)
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    closed = ideal_die(beam, radius, close_factor)
    if not cnts:
        return closed, closed
    cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    if w < 10 or h < 10:
        return closed, closed

    box = cv2.boxPoints(((cx, cy), (w, h), ang))
    # Corner rounding: distance from each rectangle corner to the nearest beam
    # pixel. Intact corners give the true rounding; a bitten corner is larger,
    # so the minimum is the robust estimate.
    dist_out = cv2.distanceTransform(1 - u8, cv2.DIST_L2, 5)
    H, W = beam.shape
    gaps = []
    for bx, by in box:
        ix, iy = int(np.clip(bx, 0, W - 1)), int(np.clip(by, 0, H - 1))
        gaps.append(float(dist_out[iy, ix]))
    gap = float(np.min(gaps))
    corner = float(np.clip(gap / 0.4142, 0.0, 0.25 * min(w, h)))  # gap = c*(sqrt2-1)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.deg2rad(ang)
    xr = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    yr = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    dx = np.abs(xr) - (w / 2 - corner)
    dy = np.abs(yr) - (h / 2 - corner)
    sd = (np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
          + np.minimum(np.maximum(dx, dy), 0) - corner)
    rect_ideal = sd <= 0

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(close_factor * radius) + 1,) * 2)
    reachable = cv2.dilate(u8, k).astype(bool)
    return (closed | (rect_ideal & reachable)), (rect_ideal & reachable)


def calibrate(
    image: np.ndarray,
    *,
    r_min: float = 6.0,
    r_max: float = 60.0,
    n_steps: int = 22,
    close_factor: float = 2.2,
    psf_frac: float = 0.16,
    alpha_cross: float = 1.0,
    alpha_pct: float = 99.9,
    n_peaks: int = 10,
) -> Calibration:
    """Estimate blocker radius, PSF and beam geometry from the frame alone."""
    img = np.asarray(image, np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    beam = segment_beam(img)

    radii = np.geomspace(r_min, r_max, n_steps)
    v = beam.astype(np.float32)
    curve = []
    for r in radii:
        # The background reference is rebuilt at *each* scale. With one fixed
        # oversized closing, structure larger than a blocker survives in the
        # shortfall field and a big template feeds on it, so the score rises
        # monotonically and the search pins to whatever cap it is given -- which
        # is exactly what happened (20 of 23 real frames returned r_max). Tying
        # the reference to the scale under test makes this a genuine band-pass
        # and restores the peak at the true radius.
        y = _shortfall(img, close_factor * r, beam)
        D = _disc(r, max(1.0, psf_frac * r))
        num = fftconvolve(y * v, D, mode="same")
        den = fftconvolve(v, D.astype(np.float64) ** 2, mode="same")
        snr = num / np.sqrt(np.maximum(den, 1e-6))
        # 99.9th percentile, not the max: robust to a single hot artifact while
        # still tracking the strongest genuine sources.
        alpha = num / np.maximum(den, 1e-6)
        # Median occupancy over several well-separated peaks, not one extreme
        # percentile. A single merged cluster of blockers keeps reading alpha>=1
        # for templates far larger than any individual blocker, and on a global
        # percentile that one blob sets the radius for the whole frame -- which
        # is what produced the 40-60 px outliers. Requiring the *typical* source
        # to agree removes them.
        curve.append((float(r), _peak_alpha(alpha, snr, beam, r, n_peaks),
                      float(np.percentile(snr[beam], 99.9))))

    a_curve = np.array([c[1] for c in curve])
    lr = np.log(radii)
    # Largest radius still consistent with a fully dark core, found by linear
    # interpolation of the crossing in log-radius.
    above = np.nonzero(a_curve >= alpha_cross)[0]
    if len(above) == 0:
        r_best = float(radii[int(a_curve.argmax())])
    else:
        k = int(above[-1])
        if k >= len(radii) - 1:
            r_best = float(radii[-1])
        else:
            a0, a1 = a_curve[k], a_curve[k + 1]
            f = (a0 - alpha_cross) / max(a0 - a1, 1e-9)
            r_best = float(np.exp(lr[k] + np.clip(f, 0.0, 1.0) * (lr[k + 1] - lr[k])))
    r_best = float(np.clip(r_best, r_min, r_max))

    psf = max(1.0, psf_frac * r_best)
    beam_ideal, rect_ideal = beam_footprint(beam, r_best, 1.9)
    # The reference closing must exceed the largest *cluster*, not the largest
    # single blocker. Two touching blockers span about 2.5 radii, so a 1.9r
    # element cannot bridge them and the closing absorbs part of the pair into
    # the background it is supposed to represent -- which depresses occupancy
    # exactly for merged pairs and pushed them under the acceptance threshold.
    y_final = _shortfall(img, 2.6 * r_best, beam_ideal)
    return Calibration(r_best, psf, beam, beam_ideal, y_final, rect_ideal, curve)
