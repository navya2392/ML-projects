"""Blocker detection pipeline.

The design turns on one observation: a blocker that touches the die edge is not
a dark blob at all. It is a *bite taken out of the die silhouette*, and its
pixels are indistinguishable from the black field surrounding the part. Any
detector built on "find dark round things" is structurally unable to see it,
because there is nothing locally distinguishing it from the exterior.

So the pipeline never looks for dark things. It builds a blocker-free reference
of the die by grey-scale closing, and looks at where the real frame falls short
of that reference. Closing with a disk larger than a blocker fills interior
blockers and edge bites alike, so both classes show up in one residual map with
the same polarity, the same normalisation, and no special-casing.

Stages
------
1. Die silhouette, and the *ideal* silhouette recovered by closing the bites.
2. Bootstrap the blocker radius from interior holes (it differs per frame).
3. Grey-scale closing residual, contrast-normalised -> scale-free blob map.
4. FRST voting for centres (arcs vote correctly; truncation is not special).
5. Watershed on the residual as a second candidate source (recall insurance).
6. RANSAC arc fitting per candidate -> unbiased sub-pixel centre and radius.
7. Scoring on depth / arc support / radius consistency, then NMS.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation

from .circlefit import arc_support, ransac_circle
from .frst import frst


@dataclass
class Detection:
    cx: float
    cy: float
    r: float
    score: float
    depth: float
    arc: float
    visible_fraction: float
    source: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Params:
    close_factor: float = 1.9  # closing SE radius, in units of blocker radius
    depth_min: float = 0.42  # min normalised darkness at the candidate
    arc_min: float = 0.20  # min fraction of circumference supported (~72 deg)
    radius_tol: float = 0.35  # accepted deviation from the frame's median radius
    nms_factor: float = 0.60  # suppression distance, in units of radius
    frst_percentile: float = 78.0
    boundary_guard: int = 3  # px of die-silhouette gradient to ignore
    min_visible: float = 0.15  # discard blockers with almost nothing showing


# --------------------------------------------------------------------------
# stage 1-3: geometry, radius calibration, residual
# --------------------------------------------------------------------------


def _grey_close(img: np.ndarray, radius: float) -> np.ndarray:
    """Grey-scale closing, computed at reduced resolution for speed.

    Only a smooth blocker-free reference is needed, so the sub-pixel accuracy
    lost to the resample is irrelevant, and it turns an O(k^2) full-resolution
    morphology into something that runs in tens of milliseconds.
    """
    ds = max(1, int(np.ceil(radius / 20.0)))
    small = (
        cv2.resize(img, None, fx=1.0 / ds, fy=1.0 / ds, interpolation=cv2.INTER_AREA)
        if ds > 1
        else img
    )
    rr = max(1, int(round(radius / ds)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, k, borderType=cv2.BORDER_REPLICATE)
    if ds > 1:
        closed = cv2.resize(
            closed, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return closed


def segment_die(image: np.ndarray) -> np.ndarray:
    """Observed die silhouette: bright part, largest component, holes filled.

    Holes are filled deliberately -- interior blockers become part of the die,
    so that the only concavities left in the silhouette are the edge bites.
    """
    sm = ndi.gaussian_filter(image, 2.0)
    thr = filters.threshold_otsu(sm)
    mask = sm > thr
    lab = measure.label(mask)
    if lab.max() == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    mask = lab == int(sizes.argmax())
    return ndi.binary_fill_holes(mask)


def bootstrap_radius(image: np.ndarray, die_filled: np.ndarray) -> float:
    """Estimate the frame's blocker radius from unambiguous interior holes.

    The three supplied frames differ in magnification and blur, so the radius
    is a per-frame quantity. Measuring it rather than hard-coding it is what
    lets one parameter set cover all of them.
    """
    sm = ndi.gaussian_filter(image, 1.5)
    thr = filters.threshold_otsu(sm[die_filled]) if die_filled.any() else 0.5
    holes = die_filled & (sm < thr)
    holes = morphology.remove_small_objects(holes, min_size=25)
    lab = measure.label(holes)
    radii = []
    for p in measure.regionprops(lab):
        if p.area < 40 or p.solidity < 0.85:
            continue  # skip merged dumbbells; singles are enough to calibrate
        radii.append(np.sqrt(p.area / np.pi))
    if not radii:
        return max(6.0, 0.02 * min(image.shape))
    return float(np.median(radii))


def ideal_die(die_filled: np.ndarray, radius: float, factor: float) -> np.ndarray:
    """Die silhouette with the edge bites closed back in.

    Binary closing with a disk larger than a blocker fills every concavity a
    blocker can produce while leaving the genuine (convex) corner rounding
    untouched -- which is exactly the discrimination we need.
    """
    rr = max(3, int(round(radius * factor)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    u8 = die_filled.astype(np.uint8)
    return cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k, borderType=cv2.BORDER_CONSTANT,
                            borderValue=0).astype(bool)


def residual_map(
    image: np.ndarray, die_ideal: np.ndarray, radius: float, factor: float
) -> np.ndarray:
    """Contrast-normalised shortfall against the blocker-free reference.

    Dividing by the reference removes the bright rim, the left/right
    illumination gradient and the frame-to-frame exposure difference in one
    step, so a fixed threshold means the same thing on every frame.
    """
    ref = _grey_close(image, radius * factor)
    resid = np.clip(ref - image, 0.0, None) / np.clip(ref, 1e-3, None)
    resid = np.clip(resid, 0.0, 1.0).astype(np.float32)
    resid[~die_ideal] = 0.0
    return ndi.gaussian_filter(resid, max(1.0, 0.12 * radius))


# --------------------------------------------------------------------------
# stage 4-5: candidate generation
# --------------------------------------------------------------------------


def _boundary_guard(die_ideal: np.ndarray, width: int) -> np.ndarray:
    """Pixels whose gradient comes from the die silhouette, not from a blocker.

    Without this the straight cut where a bite meets the silhouette votes for a
    phantom centre one radius inboard of every edge -- the classic false-alarm
    ring you get around the border of a part.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    u8 = die_ideal.astype(np.uint8)
    er = cv2.erode(u8, k).astype(bool)
    di = cv2.dilate(u8, k).astype(bool)
    return ~(di & ~er)


def candidates_frst(resid, die_ideal, radius, p: Params):
    valid = _boundary_guard(die_ideal, p.boundary_guard)
    resp = frst(
        resid,
        [0.85 * radius, radius, 1.15 * radius],
        polarity="bright",
        grad_percentile=p.frst_percentile,
        valid=valid,
        pad=int(2 * radius),
    )
    if resp.max() <= 0:
        return np.empty((0, 2)), resp
    pk = feature.peak_local_max(
        resp,
        min_distance=max(3, int(round(p.nms_factor * radius))),
        threshold_rel=0.08,
    )
    return pk[:, ::-1].astype(float), resp  # -> (x, y)


def candidates_watershed(resid, die_ideal, radius, p: Params):
    """Second candidate source: split the residual blobs geometrically.

    FRST occasionally misses a heavily blurred blocker whose gradient ring is
    weak. Distance-transform watershed catches those, and splits touching
    dumbbells into separate basins by construction.
    """
    binm = (resid > p.depth_min) & die_ideal
    binm = morphology.remove_small_objects(binm, min_size=int(0.25 * np.pi * radius**2))
    if not binm.any():
        return np.empty((0, 2))
    dist = ndi.distance_transform_edt(binm)
    seeds = morphology.h_maxima(dist, max(1.0, 0.30 * radius))
    markers = measure.label(seeds)
    if markers.max() == 0:
        markers = measure.label(binm)
    ws = segmentation.watershed(-dist, markers, mask=binm)
    out = []
    for pr in measure.regionprops(ws):
        if pr.area < 0.20 * np.pi * radius**2:
            continue
        cy, cx = pr.centroid
        out.append((cx, cy))
    return np.array(out, float) if out else np.empty((0, 2))


# --------------------------------------------------------------------------
# stage 6-7: refinement and scoring
# --------------------------------------------------------------------------


def _edge_points(resid: np.ndarray, die_ideal: np.ndarray, p: Params):
    guard = _boundary_guard(die_ideal, p.boundary_guard + 1)
    g = filters.sobel(resid) * guard
    thr = np.percentile(g[g > 0], 90) if (g > 0).any() else 1.0
    ys, xs = np.nonzero(g > thr)
    return xs.astype(float), ys.astype(float)


def _visible_fraction(cx, cy, r, die_ideal) -> float:
    h, w = die_ideal.shape
    x0, x1 = int(max(0, cx - r - 1)), int(min(w, cx + r + 2))
    y0, y1 = int(max(0, cy - r - 1)), int(min(h, cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    total = np.pi * r * r
    return float((d & die_ideal[y0:y1, x0:x1]).sum() / max(total, 1.0))


def _depth(cx, cy, r, resid, die_ideal) -> float:
    """Median residual over the visible part of the disc.

    Measured only over pixels that exist. A blocker showing 30% of its area is
    just as dark over that 30% as a whole one is -- averaging over the missing
    part would penalise exactly the detections we are trying to keep.
    """
    h, w = resid.shape
    for frac in (0.55, 0.85):
        rr = frac * r
        x0, x1 = int(max(0, cx - rr - 1)), int(min(w, cx + rr + 2))
        y0, y1 = int(max(0, cy - rr - 1)), int(min(h, cy + rr + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = ((xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr) & die_ideal[y0:y1, x0:x1]
        if m.sum() >= 12:
            return float(np.median(resid[y0:y1, x0:x1][m]))
    return 0.0


def detect(image: np.ndarray, params: Params | None = None, *, debug: bool = False):
    """Detect blockers. Returns ``(detections, info)``."""
    p = params or Params()
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    die_filled = segment_die(img)
    r0 = bootstrap_radius(img, die_filled)
    die_id = ideal_die(die_filled, r0, p.close_factor)
    resid = residual_map(img, die_id, r0, p.close_factor)

    # Re-calibrate on the residual now that the bites are visible too, then
    # rebuild once. One iteration is enough; it converges immediately.
    binm = morphology.remove_small_objects((resid > p.depth_min) & die_id, min_size=25)
    lab = measure.label(binm)
    radii = [
        np.sqrt(pr.area / np.pi)
        for pr in measure.regionprops(lab)
        if pr.area > 40 and pr.solidity > 0.88
    ]
    if radii:
        r0 = float(np.median(radii))
        die_id = ideal_die(die_filled, r0, p.close_factor)
        resid = residual_map(img, die_id, r0, p.close_factor)

    c_frst, resp = candidates_frst(resid, die_id, r0, p)
    c_ws = candidates_watershed(resid, die_id, r0, p)
    cands = [(c, "frst") for c in c_frst] + [(c, "watershed") for c in c_ws]

    ex, ey = _edge_points(resid, die_id, p)
    rng = np.random.default_rng(0)

    raw: list[Detection] = []
    for (cx, cy), src in cands:
        # Refine against the arc: unbiased even when most of the disc is gone.
        sel = (np.abs(ex - cx) < 1.8 * r0) & (np.abs(ey - cy) < 1.8 * r0)
        px, py = ex[sel], ey[sel]
        d = np.hypot(px - cx, py - cy)
        keep = (d > 0.45 * r0) & (d < 1.7 * r0)
        fit = ransac_circle(
            px[keep], py[keep], r_prior=r0, r_tol=p.radius_tol,
            inlier_tol=max(2.0, 0.10 * r0), rng=rng,
        )
        if fit is not None:
            fx, fy, fr, inl = fit
            if np.hypot(fx - cx, fy - cy) < 0.7 * r0:
                cx, cy, fr_used = fx, fy, fr
                arc = arc_support(cx, cy, fr, px[keep], py[keep], tol=max(3.0, 0.12 * r0))
            else:
                fr_used, arc = r0, arc_support(cx, cy, r0, px[keep], py[keep], tol=max(3.0, 0.12 * r0))
        else:
            fr_used = r0
            arc = arc_support(cx, cy, r0, px[keep], py[keep], tol=max(3.0, 0.12 * r0))

        vis = _visible_fraction(cx, cy, fr_used, die_id)
        dep = _depth(cx, cy, fr_used, resid, die_id)
        if vis < p.min_visible or dep < p.depth_min:
            continue
        if abs(fr_used - r0) / r0 > p.radius_tol:
            continue
        # Arc support is only demanded where an arc can exist. A blocker with
        # 25% of itself showing physically cannot present a full circumference.
        if arc < p.arc_min * min(1.0, vis / 0.6 + 0.25):
            continue
        score = float(dep * (0.55 + 0.45 * min(arc / 0.5, 1.0)))
        raw.append(Detection(float(cx), float(cy), float(fr_used), score, float(dep), float(arc), float(vis), src))

    dets = _nms(raw, p.nms_factor * r0)

    info = {"radius": r0, "n_candidates": len(cands), "die_ideal": die_id}
    if debug:
        info.update({"residual": resid, "frst": resp, "die_filled": die_filled})
    return dets, info


def _nms(dets: list[Detection], min_dist: float) -> list[Detection]:
    out: list[Detection] = []
    for d in sorted(dets, key=lambda z: -z.score):
        if all(np.hypot(d.cx - k.cx, d.cy - k.cy) >= min_dist for k in out):
            out.append(d)
    return out
