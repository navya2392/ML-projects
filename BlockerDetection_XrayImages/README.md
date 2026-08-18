# Blocker detection and verification in X-ray beam images

Finding near-constant-size dark discs ("blockers") in radiographic frames,
**including blockers clipped by the edge of the beam** — the case that makes an
existing production algorithm report "blocker not found" when a human can see
the blocker perfectly well.

```
python run_demo.py               # blind detection metrics
python run_demo.py --save out    # + annotated PNGs
python run_verify_demo.py        # known-site verification, per-frame
python run_sweep.py 4            # full statistical sweep (the number that matters)
```

---

## 1. Two different problems

| | Blind detection | **Known-site verification** |
|---|---|---|
| Question | "where are all the blockers?" | "is a blocker at *this* coordinate?" |
| Search space | ~1.8M pixels | N small ROIs |
| Failure cost | miss / false blob | **false alarm to an operator** |
| Matches the real system | no | **yes** |

The production system already knows where every blocker should be. Its failure
is not localisation — it is the *decision* at a known site: it cannot find a
full circle at the beam edge, so it reports "not found". **Every frame in the
bad set has the blocker present.** So verification is the mode to optimise, and
100% recall means *zero false alarms*.

**Use the known coordinates. But keep blind detection — the two compose.**
Blind detection is not an alternative here, it is the *registration bootstrap*:
a local search around a nominal coordinate can only recover a misregistration
smaller than its own search window, while point-set matching is
correspondence-free and recovers an arbitrarily large shift. Removing it
reintroduced 53 false alarms (§4). Blind detection finds the constellation; the
recipe decides each site.

> ⚠️ One caveat about the bad set: because every blocker in it is present, that
> set **cannot distinguish a good algorithm from one that always answers
> "found it"**. It measures the false-alarm axis only. Negative controls are
> required — see §5.

---

## 2. Why the edge case is worse than "partial circle"

Outside the beam is black. A blocker is black. A blocker touching the beam edge
**merges with the background and stops being an object** — no closed contour, no
enclosed dark region, no local intensity anomaly. Threshold + connected
components, LoG/DoG blob detection, Hough circles and plain template matching are
all *structurally* unable to see it at any parameter setting.

**The fix: never look for dark things.** Grey-scale morphological closing with a
disc larger than a blocker reconstructs the beam as it would look with no
blockers — and because closing also bridges the concavity where a blocker bit
into the outline, interior blockers and edge bites both reappear in one residual
map with the same polarity and the same normalisation. The edge case stops being
a special case.

```
occupancy field  y = (closing(img, disc(1.9·r)) − img) / closing(img, disc(1.9·r))
```

Dividing by the reconstruction cancels the bright rim, the illumination gradient
and the frame-to-frame exposure difference at once, so one fixed threshold means
the same thing on every frame. Binary closing of the silhouette recovers the
un-bitten beam outline: it fills the concave bites while leaving the convex
corner rounding untouched.

---

## 3. The verification decision

At each site, fit **occupancy** by masked least squares:

```
y(x) ≈ α · D(x)          for x in the beam footprint only
α̂ = Σ yD / Σ D²          σ_α = noise / √(Σ D²)
```

`D` is a disc of the known radius blurred by the frame's measured PSF; `α` is
1 for a full blocker and 0 for bare beam. Both sums run **only over valid
pixels** — that single restriction is what makes truncation a non-issue. A
blocker with 35% of its disc inside the beam is measured over that 35%, and α is
unbiased, just noisier. There is no circularity test, no area gate and no "find
a full circle" anywhere in the decision, because those are exactly the tests
truncation breaks.

Measured: median α is **1.02 for interior blockers and 0.99 for edge-truncated
ones** — truncation costs essentially nothing — against **0.05** for a site with
no blocker.

The noise is not discarded, it is the product. σ_α grows as valid area shrinks,
so a site with too little beam left to judge abstains instead of being guessed
at. **That is how a recall guarantee is actually obtained: never force a call
the evidence cannot support.**

### The three-way output

| Verdict | Meaning | Action |
|---|---|---|
| `PRESENT` | α consistent with a blocker, not with bare beam | no alert |
| `MISSING` | α consistent with bare beam, not with a blocker | **alert** |
| `REVIEW` | consistent with both, or with neither | route to a human |

Deciding by *consistency with a hypothesis* rather than by which side of a
threshold α falls on is essential. An occupancy of `0.27 ± 0.001` is not a
missing blocker — it is confidently neither, i.e. the model does not describe
that site. Reading it as "below threshold" is precisely the reasoning that turns
a model failure into a false alarm; it cost one false alarm until fixed.

---

## 4. What actually mattered (measured, not assumed)

Every remaining false alarm during development was diagnosed rather than tuned
away. None of them were caused by truncation.

| Fix | Why | Effect |
|---|---|---|
| **Registration bootstrap from blind detection** | Bounded local search cannot recover a shift larger than its own window | **53 → 0** false alarms |
| **Consensus registration, flag on residual not raw shift** | A site moving *with* its neighbours corrected a global offset and is fine; only a site moving *against* them is suspect | review 54% → 12% |
| **Cap the search window at half the nearest-site spacing** | A wider window lets a touching pair swap targets — the one failure that produces a *confident* wrong answer | removed a class of FA |
| **Joint fit per overlapping cluster** | Fitting neighbours independently lets one blocker explain another's evidence | correct attribution in dumbbells |
| **Withhold `MISSING` for entangled sites (ρ > 0.5)** | "Absent" and "hidden under the neighbour" are indistinguishable when templates are near-collinear | last FA → 0 |
| **Gate the drop-one F-test on α too** | With low residual noise the F denominator collapses and reports overwhelming significance for a site whose α is ~0 | last escape → 0 |
| **Hypothesis-consistency decision rule** | See §3 | removed model-misfit FAs |
| **Sub-pixel parabolic peaks; adaptive residual threshold** | Integer-pixel centres feed straight into the consensus residual | review 17% → 6.5% |
| **FFT convolution; crop-based PSF search** | Large-kernel spatial convolution dominated runtime | 8 s → **0.44 s** per frame |

---

## 5. Results

Synthetic frames reproducing the supplied images (blur from crisp to heavily
smoothed, per-frame radius, bright rim, mottle, ring artifacts, touching pairs,
beam-edge blockers) **plus ablated negative controls** — sites present in the
recipe with the blocker deliberately not rendered, without which the false-alarm
axis is unfalsifiable.

### Verification

| Set | Frames | Sites | Edge | **False alarms** | **Escapes** | Review |
|---|---|---|---|---|---|---|
| Main | 48 | 1232 | 333 | **0** | **0** | 6.8% |
| Stress¹ | 6 | 154 | 36 | **0** | **0** | 3.9% |
| **Held-out²** | 36 | 924 | 263 | **0** | **0** | 8.1% |

¹ abusive misregistration: 2.2·r shift, 1.6° rotation, 1.2% scale error.
² seeds never used during development — the honest number.

*False alarm* = present blocker called missing (the customer's failure).
*Escape* = ablated blocker called present (the safety failure).
**2310 sites, 0 of each.** Only 2 of the 596 edge-truncated sites landed in
REVIEW; truncation is essentially solved.

Runtime ~0.5 s/frame for verification, ~3 s including the blind bootstrap.

### Blind detection

| | recall | precision | edge recall | centre err |
|---|---|---|---|---|
| FRST + watershed | 0.970 | 0.962 | **1.000** | 3.2 px |
| + occupancy refinement | 0.962 | **0.975** | **1.000** | 3.1 px |

Blind edge recall is 1.000. The residual errors are **heavily overlapping pairs**
(centres < 1.2·r apart), and I could not remove them: pure matching pursuit
over-picked badly (221 false positives), and occupancy refinement buys precision
but not recall. This is an honest limit of the blind mode. It is *not* a limit of
the deployed system — those same pairs surface in verification as `rho`-entangled
sites routed to REVIEW, and the recipe already states that two blockers are there.

![annotated output](demo_out/Q31B_blurry.png)

*Blind detection. Green dashed = ground truth, blue = interior, orange =
edge-truncated (centres on and beyond the beam boundary).*

---

## 5b. Real data (23 frames: 11 bad, 11 good, 1 special, 1 raw TIFF)

The blind workflow now runs end to end on the real frames:

```
python run_blind.py data/data --out blind_out --csv blind_detections.csv
```

It finds 17-30 blockers per frame at ~5 s/frame, with occupancy near 1.0 on
accepted detections. **Everything that worked on synthetic data failed on the
real frames**, and the fixes are the substance of this section.

| Broke on real data | Why | Fix |
|---|---|---|
| Otsu-inside-beam radius bootstrap | The beam has a strong internal gradient (0.30 centre, 0.75 rim), so Otsu split *the beam* rather than blockers from beam: threshold 0.470 against beam median 0.392, 46% of the frame called "holes", 4.8 px radius for 28 px blockers | Scale-space calibration on the occupancy field (`scale.py`) |
| Matched-filter SNR for scale selection | Rose monotonically; pinned 20 of 23 frames to the search cap | **Occupancy crossing**: alpha stays ~1 while the template fits inside the core and falls once it exceeds it. Validated against radial profiling -- blockers measure 32-34 px, the crossing returns 31.9 |
| Two-class Otsu beam segmentation | The frame is trimodal (exterior / beam / bright rim); beam area ranged 0.22-0.74 | Three-class Otsu; beam area now 0.67-0.75 |
| Fixed oversized reference closing | Structure larger than a blocker survived, so big templates fed on it | Reference rebuilt at each scale -- a genuine band-pass |
| Global percentile for occupancy | One merged cluster set the radius for the whole frame (40-60 px outliers) | Median over several separated peaks; radius now 20-36.6 px, CV 0.16 |
| Morphological closing for the footprint | Cannot restore a **corner** bite: cutting a convex corner is indistinguishable from more rounding, so corner blockers never became candidates at all | Geometric footprint from the min-area rectangle with rounding read off intact corners |
| Reference closing at 1.9r | A touching pair spans ~2.5r, so the closing absorbed part of the pair into its own background and depressed occupancy exactly for merged pairs | Closing at 2.6r |
| Merged-pair splitting | Split a genuine single blocker into two half-strength discs, destroying a clean 0.94 detection | Require *every* member of a split cluster to stay opaque, not just the new one |

Merged pairs are now resolved on real data (verified on several dumbbells), and
the split threshold is insensitive across F = 80-400.

### What is still unsolved

**Deep edge bites.** One confirmed case (`bad/ISP_NF_N260530-001-002_Q16T`, top
edge) is a blocker eating ~60 px into the beam whose centre lies outside it.
Its occupancy is measured correctly (alpha 0.99) but it is rejected because only
10% of the disc lands inside the reconstructed footprint. A dedicated
silhouette-bite channel was written for exactly this and is **shipped disabled**
(`BlindParams.use_bites=False`): the real boundary is blurred enough that a
60 px visual bite moves the thresholded silhouette by only 7 px, so the channel
missed every confirmed bite while adding spurious detections along one frame
with an irregular top edge. That is the wrong trade for a system whose failure
mode is false alarms. It needs a sub-pixel boundary estimate, not an Otsu
outline.

**Recall is unmeasured.** There are no expected coordinates in the archive, so
none of the counts above can be scored. See §7.

---

## 6. Honest limits

- **100% recall *and* 100% precision simultaneously is not achievable** for a
  forced binary call on genuinely ambiguous data. What is achievable, and what
  is measured above, is 0 errors on both axes *with a ~7% abstain rate*. If
  REVIEW must be lower, it trades against the guarantee — that is the real dial,
  and it is the operator's call, not the algorithm's.
- **These numbers are synthetic.** They validate the design and the failure
  analysis; they are not a claim about your data. The next step is to run the
  real good/bad folders through it.
- The blur, radius, contrast and artifact ranges are my estimates from three
  images. If real frames sit outside them, thresholds will need re-tuning
  (all are expressed in units of the auto-calibrated radius, so they should
  transfer).

---

## 7. What I need from you

1. **The expected blocker coordinates** — still the single highest-value input,
   and now the only thing blocking a real accuracy number. State the frame:
   recipe/CAD units or pixels, and the origin. Without them every count in §5b
   is unscored: I cannot tell a miss from a blocker that was never there.
2. ~~The good/bad image folders~~ — received, 23 frames.
3. **Which specific site failed** in each bad image. With 11 bad frames and no
   labels, I can only confirm the failures I happen to spot by eye.
4. **Negative controls**: any real cases where a blocker was genuinely absent. If
   none exist, they can be synthesised by inpainting a blocker out of a good
   image, and that is worth doing before trusting any precision number.

## 8. If you later want a learned model

Use **centre-heatmap + occupancy regression** (CenterNet-style), not a box or
instance detector. Mask R-CNN and YOLO-seg are trained to predict what is
*visible*, so they systematically shrink and shift clipped objects — the exact
failure being avoided. Feed the beam mask as a second input channel so the
network can tell "outside the beam" from "blocker". `synth.py` already emits
labelled frames with correct truncation geometry, so pretraining needs no
labelling. Wrap it in **conformal prediction** to convert a calibrated
probability into a guaranteed-coverage abstain set — that is the rigorous way to
deliver "100% recall at a stated confidence" rather than asserting it.

Keep the classical pipeline regardless: as the label bootstrapper, the sanity
check, and the fallback for frames unlike anything in training.

---

## 9. Layout

```
blocker_detection/
  synth.py        synthetic frames + ground truth + ablated negative controls
  pipeline.py     blind detection (closing residual -> FRST -> watershed -> RANSAC arc)
  verify.py       known-site occupancy verification, three-way decision
  register.py     constellation alignment + verify_frame() entry point
  detect_mp.py    matching pursuit / occupancy refinement
  frst.py         Fast Radial Symmetry Transform
  circlefit.py    algebraic + RANSAC circle fitting for arcs
  evaluate.py     matching and metrics, split by truncation
run_demo.py       blind detection demo
run_verify_demo.py  per-frame verification demo
run_sweep.py      large-sample sweep (main / stress / held-out)
```

Entry point: `blocker_detection.verify_frame(image, expected_xy, radius)`.

Requires `numpy`, `scipy`, `scikit-image`, `opencv-python-headless`;
`matplotlib` for `--save`.
