"""Synthetic generator that mimics the ISP_NF X-ray frames.

The point of this module is not art -- it is ground truth. The real frames have
no labels, so every design decision in the detector is validated against
synthetic frames that reproduce the properties that actually break detectors:

  * a bright rounded-rectangle die on a black field,
  * per-frame blur (the real frames range from crisp to heavily smoothed),
  * near-black blockers of *almost* constant radius, with a per-frame radius,
  * touching / overlapping blocker pairs that merge into dumbbells,
  * blockers straddling the die boundary, which merge into the black exterior,
  * a bright rim just inside the die edge, low-frequency mottle, ring artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Blocker:
    cx: float
    cy: float
    r: float
    visible_fraction: float = 1.0  # fraction of the disk area inside the die


@dataclass
class Frame:
    image: np.ndarray  # float32 in [0, 1]
    blockers: list[Blocker] = field(default_factory=list)
    die_polygon: np.ndarray | None = None
    blur_sigma: float = 0.0
    nominal_radius: float = 0.0


def _rounded_rect_mask(shape, cx, cy, half_w, half_h, corner, angle_deg):
    """Signed-distance style rounded rectangle, optionally rotated."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    t = np.deg2rad(angle_deg)
    x = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    y = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    dx = np.abs(x) - (half_w - corner)
    dy = np.abs(y) - (half_h - corner)
    dx_c = np.maximum(dx, 0.0)
    dy_c = np.maximum(dy, 0.0)
    dist = np.hypot(dx_c, dy_c) + np.minimum(np.maximum(dx, dy), 0.0) - corner
    return dist <= 0.0


def _disk_field(shape, cx, cy, r):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    return np.hypot(xx - cx, yy - cy) <= r


def _place_blockers(rng, die_mask, n, r0, r_jitter, n_pairs, n_edge):
    """Sample blocker centres: interior, touching pairs, and edge-straddling."""
    h, w = die_mask.shape
    ys, xs = np.nonzero(die_mask)
    # Distance from every die pixel to the die boundary -- used to place
    # edge blockers at a controlled amount of truncation.
    dist = ndi.distance_transform_edt(die_mask)

    out: list[Blocker] = []

    def ok(cx, cy, r, min_gap):
        for b in out:
            if np.hypot(b.cx - cx, b.cy - cy) < min_gap * (b.r + r):
                return False
        return True

    # 1. interior singles
    interior = np.nonzero(dist > 2.5 * r0)
    tries = 0
    while len([b for b in out]) < n and tries < 4000:
        tries += 1
        i = rng.integers(len(interior[0]))
        cy, cx = float(interior[0][i]), float(interior[1][i])
        r = r0 * (1.0 + r_jitter * rng.standard_normal())
        if ok(cx, cy, r, 1.6):
            out.append(Blocker(cx, cy, r))

    # 2. touching pairs (the dumbbells)
    for _ in range(n_pairs):
        for _ in range(200):
            i = rng.integers(len(interior[0]))
            cy, cx = float(interior[0][i]), float(interior[1][i])
            r = r0 * (1.0 + r_jitter * rng.standard_normal())
            if not ok(cx, cy, r, 2.2):
                continue
            th = rng.uniform(0, 2 * np.pi)
            sep = rng.uniform(1.05, 1.6) * r  # centres closer than 2r -> merged
            cx2, cy2 = cx + sep * np.cos(th), cy + sep * np.sin(th)
            if not die_mask[int(round(cy2)) % h, int(round(cx2)) % w]:
                continue
            out.append(Blocker(cx, cy, r))
            out.append(Blocker(cx2, cy2, r * (1.0 + r_jitter * rng.standard_normal())))
            break

    # 3. edge-straddling: centre within +-0.9r of the boundary
    band = np.nonzero((dist > 0) & (dist < 0.9 * r0))
    for _ in range(n_edge):
        for _ in range(200):
            i = rng.integers(len(band[0]))
            cy, cx = float(band[0][i]), float(band[1][i])
            r = r0 * (1.0 + r_jitter * rng.standard_normal())
            # push a random amount further out, sometimes past the boundary
            if ok(cx, cy, r, 1.6):
                out.append(Blocker(cx, cy, r))
                break

    # measure true visible fraction
    for b in out:
        d = _disk_field(die_mask.shape, b.cx, b.cy, b.r)
        area = d.sum()
        b.visible_fraction = float((d & die_mask).sum() / max(area, 1))
    return [b for b in out if b.visible_fraction > 0.12]


def make_frame(
    size: int = 1360,
    blur_sigma: float = 4.0,
    nominal_radius: float = 22.0,
    n_interior: int = 14,
    n_pairs: int = 2,
    n_edge: int = 5,
    radius_jitter: float = 0.07,
    noise: float = 0.018,
    die_gray: float = 0.72,
    blocker_gray: float = 0.03,
    rim_gain: float = 0.16,
    ring_artifacts: int = 6,
    seed: int = 0,
) -> Frame:
    rng = np.random.default_rng(seed)
    shape = (size, size)

    half = size * 0.42
    die = _rounded_rect_mask(
        shape,
        size / 2 + rng.uniform(-8, 8),
        size / 2 + rng.uniform(-8, 8),
        half,
        half * rng.uniform(0.97, 1.03),
        corner=size * 0.025,
        angle_deg=rng.uniform(-0.8, 0.8),
    )

    img = np.zeros(shape, np.float32)
    img[die] = die_gray

    # low-frequency mottle inside the die
    mottle = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), 28)
    mottle /= max(mottle.std(), 1e-6)
    img[die] += 0.035 * mottle[die]

    # bright rim just inside the die boundary (reconstruction edge enhancement)
    d_in = ndi.distance_transform_edt(die).astype(np.float32)
    rim = np.exp(-((d_in / (size * 0.012)) ** 2)) * die
    img += rim_gain * rim
    # the real frames are brighter down the right/left edges -- add a gradient
    gx = np.linspace(-1, 1, size, dtype=np.float32)[None, :]
    img[die] += (0.05 * np.abs(gx) ** 2 * die)[die]

    blockers = _place_blockers(
        rng, die, n_interior, nominal_radius, radius_jitter, n_pairs, n_edge
    )
    for b in blockers:
        m = _disk_field(shape, b.cx, b.cy, b.r) & die
        img[m] = blocker_gray

    # faint concentric ring artifacts -- classic Hough false positives
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for _ in range(ring_artifacts):
        i = rng.integers(die.sum())
        ys, xs = np.nonzero(die)
        cy, cx = float(ys[i]), float(xs[i])
        rr = np.hypot(xx - cx, yy - cy)
        env = np.exp(-((rr / rng.uniform(30, 70)) ** 2))
        img += 0.012 * np.cos(rr / rng.uniform(2.0, 4.0)) * env * die

    img = ndi.gaussian_filter(img, blur_sigma)
    img = img + noise * rng.standard_normal(shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    return Frame(
        image=img,
        blockers=blockers,
        blur_sigma=blur_sigma,
        nominal_radius=nominal_radius,
    )


#: Three presets chosen to bracket the supplied frames (crisp / heavily
#: smoothed / medium), including their different apparent blocker sizes.
PRESETS = {
    "Q15B_sharp": dict(blur_sigma=3.0, nominal_radius=21.0, n_interior=15, seed=15),
    "Q31B_blurry": dict(blur_sigma=9.0, nominal_radius=33.0, n_interior=17, seed=31),
    "Q14B_medium": dict(blur_sigma=5.0, nominal_radius=25.0, n_interior=18, seed=14),
}
