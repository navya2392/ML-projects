"""Run blind detection over the real ISP_NF frames and render annotated output."""
import glob, json, os, sys, time
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from blocker_detection.pipeline import Params, detect

def load(p):
    return np.asarray(Image.open(p).convert("L"), np.float32) / 255.0

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "real_out"
    os.makedirs(out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob("data/data/*/*.jpg")):
        cls = f.split("/")[-2]; name = os.path.basename(f)[:-4]
        img = load(f); t = time.perf_counter()
        dets, info = detect(img, Params())
        dt = time.perf_counter() - t
        edge = [d for d in dets if d.visible_fraction < 0.92]
        rows.append(dict(cls=cls, name=name, n=len(dets), n_edge=len(edge),
                         r=round(info["radius"], 1), sec=round(dt, 1)))
        print(json.dumps(rows[-1]))
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        for d in dets:
            c = "#f0883e" if d.visible_fraction < 0.92 else "#58a6ff"
            ax.add_patch(Circle((d.cx, d.cy), d.r, fill=False, ec=c, lw=1.6))
            ax.plot([d.cx], [d.cy], "+", color=c, ms=5)
        ax.set_title(f"{cls}/{name}  n={len(dets)} edge={len(edge)} r={info['radius']:.0f}",
                     fontsize=8)
        ax.set_axis_off()
        fig.savefig(f"{out}/{cls}__{name}.png", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"\n{len(rows)} frames; radius range "
          f"{min(r['r'] for r in rows)}-{max(r['r'] for r in rows)} px; "
          f"blockers/frame {min(r['n'] for r in rows)}-{max(r['n'] for r in rows)}")

main()
