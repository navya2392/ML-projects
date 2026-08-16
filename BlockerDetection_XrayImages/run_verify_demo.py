"""Verification-mode evaluation: known sites, three-way decision.

Reports the two numbers that matter for the real system:
  * recall on blockers that ARE present  -> must be 1.000 (no false alarms)
  * alert precision on ablated sites     -> proves it still catches real misses

plus the REVIEW rate, which is the honest price of the recall guarantee.
"""

from __future__ import annotations

import json

import numpy as np

from blocker_detection import synth
from blocker_detection.pipeline import Params, detect, ideal_die, segment_die
from blocker_detection.register import apply, ransac_align
from blocker_detection.verify import MISSING, PRESENT, REVIEW, VerifyParams, verify


def run(preset_name, cfg, use_registration=True):
    frame = synth.make_frame(**cfg)
    img = frame.image
    sites = frame.expected
    exp = np.array([[s.cx, s.cy] for s in sites], float)
    r_nom = float(np.median([s.r for s in sites]))

    beam = segment_die(img)
    beam_ideal = ideal_die(beam, r_nom, 1.9)


    res, info = verify(img, exp, r_nom, params=VerifyParams(), beam_ideal=beam_ideal)

    present_idx = [i for i, s in enumerate(sites) if s.present]
    absent_idx = [i for i, s in enumerate(sites) if not s.present]
    edge_idx = [i for i, s in enumerate(sites) if s.visible_fraction < 0.92 and s.present]

    false_alarm = [i for i in present_idx if res[i].verdict == MISSING]
    escaped = [i for i in absent_idx if res[i].verdict == PRESENT]
    review = [i for i, r in enumerate(res) if r.verdict == REVIEW]

    return {
        "frame": preset_name,
        "n_sites": len(sites),
        "n_present": len(present_idx),
        "n_ablated": len(absent_idx),
        "n_edge_present": len(edge_idx),
        "false_alarms": len(false_alarm),
        "escapes": len(escaped),
        "review": len(review),
        "review_edge": sum(1 for i in review if i in edge_idx),
        "psf_sigma": info["psf_sigma"],
        "reg_shift_px": round(info["registration_shift"], 1),
        "consensus_rms_px": round(info["consensus_rms"], 2),
        "alpha_present_med": float(np.median([res[i].alpha for i in present_idx])),
        "alpha_absent_med": float(np.median([res[i].alpha for i in absent_idx])) if absent_idx else float("nan"),
        "alpha_edge_med": float(np.median([res[i].alpha for i in edge_idx])) if edge_idx else float("nan"),
    }


def main():
    rows = [run(n, c) for n, c in synth.VERIFY_PRESETS.items()]
    for r in rows:
        print(json.dumps(r, default=float))

    tot = lambda k: sum(r[k] for r in rows)
    n_p, n_a = tot("n_present"), tot("n_ablated")
    fa, esc, rev = tot("false_alarms"), tot("escapes"), tot("review")
    n_edge = tot("n_edge_present")
    print("\n" + "=" * 62)
    print(f"sites={tot('n_sites')}  present={n_p}  ablated={n_a}  edge-present={n_edge}")
    print(f"FALSE ALARMS (present called missing) : {fa}   -> recall {1 - fa / n_p:.4f}")
    print(f"ESCAPES      (ablated called present) : {esc}  -> "
          f"{'clean' if esc == 0 else 'MISSED A REAL DEFECT'}")
    print(f"REVIEW rate                            : {rev}/{tot('n_sites')} = {rev / tot('n_sites'):.1%}"
          f"  ({tot('review_edge')} of them edge sites)")
    print(f"median alpha  present={np.mean([r['alpha_present_med'] for r in rows]):.3f}  "
          f"edge={np.nanmean([r['alpha_edge_med'] for r in rows]):.3f}  "
          f"ablated={np.nanmean([r['alpha_absent_med'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
