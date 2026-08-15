"""Run the detector over synthetic frames matched to the three supplied images.

    python run_demo.py            # metrics only
    python run_demo.py --save out # also write annotated PNGs
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from blocker_detection import PRESETS, Params, detect, make_frame
from blocker_detection.evaluate import report


def annotate(img, gt, dets, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    for g in gt:
        ax.add_patch(Circle((g.cx, g.cy), g.r, fill=False, ec="#3fb950", lw=1.6, ls="--"))
    for d in dets:
        ec = "#f0883e" if d.visible_fraction < 0.9 else "#58a6ff"
        ax.add_patch(Circle((d.cx, d.cy), d.r, fill=False, ec=ec, lw=1.6))
        ax.plot([d.cx], [d.cy], "+", color=ec, ms=6)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None, help="directory for annotated PNGs")
    args = ap.parse_args()
    if args.save:
        import os

        os.makedirs(args.save, exist_ok=True)

    totals = []
    for name, cfg in PRESETS.items():
        frame = make_frame(**cfg)
        t0 = time.perf_counter()
        dets, info = detect(frame.image, Params())
        dt = time.perf_counter() - t0
        rep = report(frame.blockers, dets)
        rep["frame"] = name
        rep["seconds"] = round(dt, 2)
        rep["radius_true"] = round(frame.nominal_radius, 1)
        rep["radius_est"] = round(info["radius"], 1)
        totals.append(rep)
        print(json.dumps(rep, indent=2, default=float))
        if args.save:
            annotate(frame.image, frame.blockers, dets, f"{args.save}/{name}.png")

    print("\n=== aggregate ===")
    for k in ("recall", "precision", "edge_recall", "whole_recall"):
        print(f"{k:>14}: {np.mean([t[k] for t in totals]):.3f}")
    print(f"{'centre err px':>14}: {np.mean([t['median_centre_error_px'] for t in totals]):.2f}")


if __name__ == "__main__":
    main()
