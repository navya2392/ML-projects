"""Large-sample verification sweep -- the number that decides whether this ships.

Reports false alarms (present blocker called missing, the customer's actual
failure), escapes (ablated blocker called present, the safety failure), and the
REVIEW rate, which is the price paid for making no unsupported call.
"""

from __future__ import annotations

import collections
import sys

import numpy as np

from blocker_detection import synth
from blocker_detection.register import verify_frame
from blocker_detection.verify import MISSING, PRESENT, REVIEW


def sweep(presets, label, show_reasons=True):
    fa = esc = rev = npres = nabs = nedge = revedge = 0
    reasons = collections.Counter()
    fa_detail, esc_detail = [], []
    for name, cfg in presets.items():
        f = synth.make_frame(**cfg)
        st = f.expected
        exp = np.array([[q.cx, q.cy] for q in st], float)
        r = float(np.median([q.r for q in st]))
        res, _ = verify_frame(f.image, exp, r)
        for i, q in enumerate(st):
            edge = q.present and q.visible_fraction < 0.92
            npres += q.present
            nabs += not q.present
            nedge += edge
            v = res[i].verdict
            if q.present and v == MISSING:
                fa += 1
                fa_detail.append((name, i, res[i].alpha, q.visible_fraction))
            if (not q.present) and v == PRESENT:
                esc += 1
                esc_detail.append((name, i, res[i].alpha))
            if v == REVIEW:
                rev += 1
                revedge += edge
                reasons[res[i].reason.split(";")[0][:46]] += 1
    n = npres + nabs
    print(f"\n### {label}: {len(presets)} frames, {n} sites "
          f"({npres} present / {nabs} ablated / {nedge} edge-truncated)")
    print(f"    FALSE ALARMS {fa:3d}   ESCAPES {esc:3d}   "
          f"REVIEW {rev}/{n} = {rev / n:.1%} ({revedge} edge)")
    if show_reasons:
        for k, v in reasons.most_common(6):
            print(f"        {v:4d}  {k}")
    for d in fa_detail:
        print(f"        FA  {d}")
    for d in esc_detail:
        print(f"        ESC {d}")
    return fa, esc, rev, n


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    big = {f"{n}_s{k}": dict(cfg, seed=cfg["seed"] + 7919 * k)
           for k in range(reps) for n, cfg in synth.VERIFY_PRESETS.items()}
    a = sweep(big, f"MAIN ({len(big)} frames)")
    b = sweep(synth.STRESS_PRESETS, "STRESS (abusive misregistration)")
    tf, te = a[0] + b[0], a[1] + b[1]
    tr, tn = a[2] + b[2], a[3] + b[3]
    print("\n" + "=" * 64)
    print(f"TOTAL sites {tn}   false alarms {tf}   escapes {te}   "
          f"review {tr / tn:.1%}")
    print("VERDICT:", "clean on both axes" if tf == 0 and te == 0 else "NOT CLEAN")


if __name__ == "__main__":
    main()
