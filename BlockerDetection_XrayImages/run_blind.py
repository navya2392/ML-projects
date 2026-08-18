"""Blind blocker-detection workflow: point it at a folder of frames.

    python run_blind.py data/data                 # CSV summary to stdout
    python run_blind.py data/data --out blind_out # + annotated PNG per frame

No expected coordinates, no per-lot tuning, no prior on how many blockers there
are. Writes one row per detected blocker (frame, x, y, r, occupancy, visible
fraction, edge flag) plus a per-frame summary.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

from blocker_detection.blind import BlindParams, detect_blind


def load(path):
    return np.asarray(Image.open(path).convert("L"), np.float32) / 255.0


def annotate(img, blobs, cal, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6), dpi=100)
    for ax, base, name in ((axes[0], img, "frame"),
                           (axes[1], cal.shortfall, "occupancy field")):
        ax.imshow(base, cmap="gray", vmin=0, vmax=1)
        for b in blobs:
            c = "#f0883e" if b.edge else "#58a6ff"
            ax.add_patch(Circle((b.cx, b.cy), b.r, fill=False, ec=c, lw=1.5))
            ax.plot([b.cx], [b.cy], "+", color=c, ms=5)
        ax.set_title(name, fontsize=9)
        ax.set_axis_off()
    n_edge = sum(b.edge for b in blobs)
    fig.suptitle(f"{title}   n={len(blobs)}  edge={n_edge}  r={cal.radius:.1f}px",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder searched recursively for images")
    ap.add_argument("--out", default=None, help="directory for annotated PNGs")
    ap.add_argument("--csv", default=None, help="write per-blocker rows here")
    args = ap.parse_args()

    files = sorted(sum((glob.glob(os.path.join(args.root, "**", e), recursive=True)
                        for e in ("*.jpg", "*.png", "*.tif", "*.tiff")), []))
    if not files:
        sys.exit(f"no images under {args.root}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    rows = []
    for f in files:
        img = load(f)
        t = time.perf_counter()
        blobs, info = detect_blind(img, BlindParams())
        dt = time.perf_counter() - t
        cal = info["calibration"]
        label = os.path.relpath(f, args.root)
        n_edge = sum(b.edge for b in blobs)
        print(f"{label:58s} n={len(blobs):3d} edge={n_edge:2d} "
              f"r={cal.radius:5.1f} {dt:4.1f}s")
        for b in blobs:
            rows.append(dict(frame=label, **b.as_dict()))
        if args.out:
            annotate(img, blobs, cal, os.path.join(
                args.out, label.replace(os.sep, "__").rsplit(".", 1)[0] + ".png"), label)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
