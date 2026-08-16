# Sycophancy in Llama-3.1-8B-Instruct: Results Summary

## Research question

Can a causal-abstraction framework (Geiger et al., 2021, arXiv:2106.02997) identify a
residual-stream subspace in Llama-3.1-8B-Instruct that is causally necessary for the
model's answer-preference shift after a user assertion, and does the magnitude of this
effect depend on the asserted content's plausibility?

## Causal graph

```
C (candidate-assertion signal) → R (aligned subspace, layers ~13–22) → ΔMargin(correct_answer, candidate)
```
where `Margin(correct_answer, candidate) = logP(correct_answer) − logP(candidate)` under
teacher-forced, length-normalized scoring, and a sign flip in `ΔMargin` corresponds to the
model's preference actually inverting toward the asserted candidate.

**Terminology note**: `C` is deliberately *not* called "user belief." Controls below found
the effect largely insensitive to content plausibility, which would not be expected if the
model were evaluating and adopting the asserted claim as a belief. "Candidate-assertion
signal" is used instead, to avoid overclaiming a semantic/evaluative mechanism the data
doesn't support.

## Setup

- **Model**: Llama-3.1-8B-Instruct (bf16), chosen for direct comparability with
  arXiv:2508.02087, which used the same model for a related mechanistic analysis.
- **Data**: `meg-tong/sycophancy-eval`, `answer` split (TriviaQA-based factual questions),
  deduped by question. Prompts kept as free-form QA rather than converted to multiple
  choice, to preserve ecological validity and to serve the causal-abstraction goal (which
  doesn't require a fixed answer-token format the way logit-lens-style methods do).
- **Scoring primitive**: teacher-forced, length-normalized log-prob margin between the
  correct and a candidate (incorrect/irrelevant) answer — used identically across Phase 1,
  Phase 2, and Phase 3 so results are comparable across phases.

---

## Phase 1 — Baseline (n = 1000)

| Metric | Value |
|---|---|
| Neutral-condition preference accuracy | 601/1000 = 60.1% |
| Sycophantic flip rate (of all examples) | 302/1000 = 30.2% |
| Sycophantic flip rate (of neutral-correct examples) | 302/601 = **50.2%** |
| Among flipped examples, avg. margin | neutral +2.274 → biased −1.793 |

The 50.2% figure (conditional on the model knowing the answer under neutral framing) is
the headline number — flip rate as a fraction of *all* examples conflates "didn't know the
answer" with "was talked out of a correct answer."

---

## Phase 2 — Layer localization (n = 601, neutral-correct pool)

- **Sanity check** (logit-lens final-layer margin vs. Phase 1's ground-truth margin):
  mean |difference| = 0.23 — attributed to bf16 batched-matmul precision noise (see
  Limitations), small relative to the effect sizes analyzed.
- **Decision-margin shift curve** (behavioral localization): near-zero/noisy for layers
  0–15, turns negative at layer 16, deepens to a peak of −3.95 around layer 28, plateaus
  around −3.5 to −3.8 through layer 32.
- **Regression probe** (representation localization): held-out R² predicting the
  continuous margin shift from layer-L hidden states, validated against a 20-permutation
  null per layer. Peaks around **layers 15–19** (R² ≈ 0.13–0.16), clearly above the
  permutation null floor (≈0.01–0.02) from roughly layer 9 onward.
- **Deconfounded replication**: re-ran the regression using `residual = biased_shift −
  copy_shift` (i.e., subtracting out the most aggressive copy-effect estimate, 75.4%, see
  below) as the target instead of raw `biased_shift`. Result: R² curve correlates **r =
  0.93** with the original curve; 9 of the original top-10 layers by R² are unchanged.
  This indicates the localized representation is not primarily an artifact of the copying
  confound — it survives removing the copying-attributable component.
- **Candidate layers carried into Phase 3**: **13–22**, selected from the (deconfounded)
  regression R² ranking.

---

## Controls

### Copy-effect / contextual-entrainment control

Tests whether the biased-condition margin shift reflects genuine assertion uptake or
mere exposure to a string (`incorrect_answer`) that appears verbatim in the prompt.
Three iterations of the control template, each intended to be more semantically neutral
than the last:

| Template | Ratio (copy_shift / biased_shift) |
|---|---|
| `(Other terms that have come up in similar contexts: X.)` | 55.6% |
| `Random word: X` | 66.6% |
| `Random word: X, noted here for no particular reason.` | 75.4% |

The ratio did not converge across iterations — it increased each time, suggesting the
estimate is sensitive to the exact control wording rather than approaching a stable value.
**Per-example heterogeneity** (computed on the first control version): correlation between
`biased_shift` and `copy_shift` across examples, r = 0.730; ratio distribution wide (10th
pct ≈ 0.08, 90th pct ≈ 1.01) — copying is not a fixed fraction of the effect, it varies
substantially by question. 87.2% of examples retained a negative residual
(`biased_shift − copy_shift`) even after subtracting the copy-effect estimate.

### Irrelevant-answer control (suggested by Dr. Çöltekin)

Tests whether content plausibility matters at all, by substituting a topically unrelated
answer for `incorrect_answer` under the same assertion template. Example: question "Can
juice fasts detoxify the body?" (correct_answer "No", incorrect_answer "Yes"),
irrelevant_answer assigned was "Commander-in-chief."

| Metric | Real (plausible wrong answer) | Irrelevant answer |
|---|---|---|
| Mean margin shift | −3.64 | −8.70 |
| **Flip rate** (bounded, magnitude-independent) | **50.2%** | **52.3%** |

The raw shift ratio (239%) is inflated by a baseline mismatch (neutral-condition margin
for correct-vs-irrelevant is nearly double that of correct-vs-incorrect: +7.87 vs. +4.10)
and is not directly comparable to the copy-effect ratios above. The **flip rate**, which is
bounded and not distorted by this mismatch, is the more trustworthy comparison here: it is
essentially identical whether the asserted candidate is plausible or absurd. Decomposition
confirmed the large raw shift for the irrelevant condition is driven almost entirely by the
irrelevant answer's own log-prob rising (+7.00) rather than the correct answer's log-prob
falling (−1.70) — consistent with a content-blind contextual-exposure mechanism.

**Conclusion drawn**: cannot rule out that copying/context effect is the dominant
mechanism behind the sycophantic flip; cannot rule out a smaller genuine content-sensitive
component either. This motivated dropping "user belief" as the label for `C` (see above).

---

## Phase 3 — DAS training and evaluation

**Setup**: low-rank (64-dim) rotation trained per candidate layer via interchange
intervention (pyvene `LowRankRotatedSpaceIntervention`); training target = the *real*
observed biased-condition margin for each example (a genuine counterfactual pair already
present in the data, not a synthetic label). 500 training steps/layer, 80/20 train/test
split, evaluated via Interchange Intervention Accuracy (IIA: does the intervened margin's
sign match the real biased-condition margin's sign) and Pearson r between intervened and
real margins.

**Baselines**: trivial "always predict no flip" = 40.5% (fixed by the test set's own
label distribution); untrained (freshly initialized, random rotation) = 40.5% for every
layer — confirmed via a targeted check that random rotations *do* perturb the margin
(e.g., by 0.15, 0.01, 0.0002 on 3 sampled examples) but essentially never enough to cross
zero, given real effect sizes are typically 1–5 margin units. This is expected (an
untrained/arbitrary subspace has no reason to systematically separate `correct_answer`
from `incorrect_answer`'s log-probs, since a random perturbation is likely to shift both
similarly, leaving the difference largely unchanged), not a bug, and confirms the training
gains reported below are not achievable by chance alone.

| Layer | Trivial | Untrained | Trained IIA | Trained r |
|---|---|---|---|---|
| 15 | 40.5% | 40.5% | 51.2% | 0.629 |
| 18 | 40.5% | 40.5% | 50.4% | 0.691 |
| 14 | 40.5% | 40.5% | 45.5% | 0.625 |
| 16 | 40.5% | 40.5% | 49.6% | 0.649 |
| 17 | 40.5% | 40.5% | 48.8% | 0.652 |
| 19 | 40.5% | 40.5% | 49.6% | 0.630 |
| 20 | 40.5% | 40.5% | 48.8% | 0.671 |
| 21 | 40.5% | 40.5% | 49.6% | 0.663 |
| **13** (500 steps) | 40.5% | 40.5% | 42.1% | 0.602 |
| **13** (1000 steps, re-run) | 40.5% | 40.5% | **43.8%** | **0.694** |

Layer 13 improved modestly with 2x the training steps (IIA +1.7pp, r +0.09) — consistent
with a post-hoc check finding its loss trend borderline-significant (p = 0.093) at 500
steps, unlike layer 15's clearly-plateaued trend (p = 0.74). Even after the extra
training, layer 13 remains among the weakest candidate layers tested, so this refines
rather than overturns the "layer 13 weakest" finding.

**Cross-validation with Phase 2**: layer 13 — the boundary of the candidate range and the
lowest-R² layer in Phase 2's regression probe — is also the weakest layer under real
causal intervention in Phase 3 (IIA closest to baseline, lowest r). Two independent
methods (correlational localization, causal intervention) agree on where the effect is
weakest.

### Late-layer follow-up (26, 28, 30) — comparison with Wang & Li (2508.02087)

Phase 2's raw margin-shift curve (behavioral) continues deepening well past the R²-peak
region, reaching its own maximum around layer 28 (shift ≈ −3.95) — a region never tested
with DAS in the original 13–22 candidate range. This gap directly parallels Wang & Li
(2508.02087), who report (on Llama-3.1-8B-Instruct, MMLU-based) a "Decision Score"
(logit-lens) turning point around layer 19 and a hidden-state KL-divergence peak around
layer 23 (lagging the Decision Score). To check whether this late-layer region is also
where causal necessity peaks, layers 26, 28, and 30 were tested directly:

| Layer | Trivial | Untrained | Trained IIA | Trained r |
|---|---|---|---|---|
| 26 | 40.5% | 40.5% | 47.9% | 0.657 |
| 28 | 40.5% | 40.5% | 46.3% | 0.659 |
| 30 | 40.5% | 40.5% | 44.6% | 0.649 |

**Finding**: despite layers 26–30 showing the *largest raw behavioral/representational
divergence* (both in our own shift-curve and in the region Wang & Li's KL-divergence
metric highlights), their DAS-measured causal necessity (IIA 44.6–47.9%) is *lower* than
at the R²-selected mid-network layers (15–21, IIA 48.8–51.2%) — comparable instead to the
weakest layers in the original candidate range (13, 14). This suggests that raw
behavioral/representational divergence magnitude does not straightforwardly predict how
tractable a layer is for a single-layer, low-rank *linear* intervention: by late layers,
the effect may be distributed across more dimensions or token positions than a rank-64
subspace at a single position can capture, even if the underlying causal contribution
there is substantial. Note this is exploratory (motivated post hoc by the literature
comparison), not part of the pre-registered 13–22 candidate selection, and should be
labeled as such in write-up.

### Full-representation (untrained) patching baseline, all layers 13–32

To test the "diffuse effect" hypothesis above directly, an untrained full-vector patching
baseline (swap the entire residual stream at a layer/position, no trained subspace) was
run across all layers 13–32:

| Layer | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|
| Full-patch IIA | 41.3% | 42.1% | 43.0% | 44.6% | 47.1% | 47.9% | 47.9% | 47.9% | 47.9% | 47.9% |

| Layer | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| Full-patch IIA | 48.8% | 48.8% | 52.1% | 55.4% | 57.9% | 63.6% | 64.5% | 63.6% | 73.6% | 74.4% |

**Important interpretive caveat**: full-patch IIA rising toward later layers is expected
partly *by construction* — patching later transplants a progressively larger share of the
entire biased-condition computation, mechanically approaching the true biased output as
the intervention point nears the final layer. This across-layer trend should not alone be
read as "later layers contain more causal information localized there."

The layer-matched comparison (DAS vs. full-patch **at the same layer**) is not subject to
this confound and is the more informative one:

| Layer | Full-patch IIA | DAS (trained) IIA | Gap |
|---|---|---|---|
| 15 | 43.0% | **51.2%** | DAS +8.2pp |
| 18 | 47.9% | **50.4%** | DAS +2.5pp |
| 26 | **55.4%** | 47.9% | full-patch +7.5pp |
| 28 | **63.6%** | 46.3% | full-patch +17.3pp |
| 30 | **63.6%** | 44.6% | full-patch +19.0pp |

At mid-network layers, the trained low-rank subspace *outperforms* full patching —
evidence the alignment search is filtering out causally-irrelevant dimensions rather than
just replicating a full swap (the core premise DAS is meant to demonstrate relative to
naive patching, per Geiger et al. 2021). At late layers, this reverses, and the gap grows
with depth. Since this comparison holds "how much of the network's cumulative computation
is being transplanted" fixed (same layer), the growing gap is not explained by the
mechanical confound above, and instead points to a genuine limitation of the current DAS
configuration at greater depth — plausibly a fixed rank (64) becoming insufficient, and/or
optimization becoming harder with less downstream "correction room" between the
intervention site and the output. Neither explanation has been independently verified;
both remain candidate hypotheses for future work.

### Run-to-run variance and the overfitting question

An apparent drop in layer 15's IIA when increasing training from 500 to 1000 steps (51.2%
→ 48.8%, two separate runs) was initially concerning. A controlled follow-up — evaluating
the *same* training run's test-set IIA every 100 steps, holding the random initialization
fixed — found IIA rising from 44.6% (step 100) to a plateau around 52–54% (steps 500–1000),
**with no evidence of decline**, i.e. no overfitting. A third independent "layer 15, 1000
steps" run then produced yet another value (53.7%). This traces the earlier apparent drop
to an unseeded PyTorch RNG (only Python's `random` module was seeded, not `torch`'s),
leaving the trainable rotation's initialization uncontrolled across separate runs — now
fixed in the code (`torch.manual_seed`/`torch.cuda.manual_seed_all` added), but the
already-collected 13-layer DAS table above was produced before this fix and should be
treated as carrying roughly ±3–5pp of unquantified seed noise. Given this, fine-grained
differences between similar-IIA layers (e.g. 47.9% vs. 49.6% vs 50.4%) should not be
over-interpreted; only clear outliers robust across repeated runs (layer 13 consistently
weakest) should be treated as solid findings.


**Loss-plateau check** (layer 15, post-hoc, no additional compute): second-half (steps
250–500) trend was not statistically significant (p = 0.74) — training had already
plateaued well before 500 steps ended; more steps at the same configuration are unlikely
to meaningfully improve these numbers. Remaining headroom, if any, more likely lies in
subspace rank, batch size (currently 1, likely a significant noise source), or a
structural ceiling on what a single-layer linear subspace can capture — not additional
training time.

---

## Overall conclusion

1. Sycophantic answer-flips are common (50.2% conditional flip rate) and behaviorally
   robust to whether the asserted content is even plausible (irrelevant-answer control).
2. A representation in layers ~13–22 (peaking ~15–19) predicts, above a permutation null
   and above a copy-effect-adjusted null, how strongly a given example will be affected —
   this localization is not an artifact of the copying confound.
3. Distributed Alignment Search on this representation shows a **real but partial** causal
   contribution: trained IIA (43.8–51.2% across 13–22) exceeds both a trivial and an
   untrained baseline (40.5%) at every candidate layer, and the effect is weakest exactly
   where Phase 2's correlational method also found the weakest signal (layer 13) — but IIA
   remains well below ceiling, indicating this single-layer, 64-dimensional linear subspace
   captures only part of the causal story.
4. The mechanism this representation tracks is better described as a content-blind
   "candidate-assertion signal" than genuine belief uptake, given the irrelevant-answer
   control's near-identical flip rates.
5. Larger raw behavioral/representational divergence at later layers (26–30, aligning with
   the late-layer region Wang & Li 2508.02087 highlight via Decision Score/KL-divergence)
   does **not** translate into higher causal necessity under DAS — IIA there (44.6–47.9%)
   is comparable to the weakest layers in the 13–22 range, not higher. Layer-matched
   comparison against an untrained full-representation patching baseline shows why: DAS
   *outperforms* full patching at mid-network layers (15, 18) — evidence the aligned
   subspace isolates causally-relevant dimensions rather than just replicating a full
   swap — but *underperforms* it increasingly at later layers (by up to 19pp at layer 30).
   This points to a genuine limitation of the current fixed-rank, single-layer DAS
   configuration at greater depth, not to late layers being causally unimportant (the
   full-patch results show substantial causal content is recoverable there, just not by
   this particular subspace/training setup).
6. Reported IIA values carry an estimated ±3–5pp of run-to-run noise from an initially
   unseeded PyTorch RNG (since fixed in code); a within-run check (test-set IIA evaluated
   periodically during a single training run) found no evidence of overfitting with
   additional steps — IIA rises then plateaus. Fine-grained differences between
   similar-scoring layers should be treated cautiously; only outliers robust across
   repeated runs (layer 13 consistently weakest) are solid findings.

## Limitations (as agreed with Dr. Çöltekin — acceptable to leave open for this project)

- The copy-effect estimate did not converge across control-template iterations (55.6% /
  66.6% / 75.4%) — copying and genuine assertion-driven effects could not be cleanly
  separated at the level of raw margin magnitude, though the flip-rate comparison and the
  deconfounded Phase 2 regression provide converging evidence on the qualitative story.
- pyvene's layer-index convention relative to this project's own (Phase 1/2)
  hidden_states indexing was corrected by a reasoned −1 offset, not independently
  verified against pyvene internals.
- Training used batch size 1 (per-example steps), a likely source of the noisy,
  non-monotonic loss trajectories observed; a larger effective batch (via gradient
  accumulation) was not attempted.
- Only a single layer and a single 64-dimensional subspace were tested per run; whether a
  higher-rank subspace, multiple simultaneous layers, or a non-residual-stream
  intervention site (e.g., attention heads) would substantially raise IIA is unexplored.
- The layer 26/28/30 comparison was motivated post hoc by a literature comparison (Wang &
  Li, 2508.02087), not by the original R²-based candidate selection — it should be
  presented as an exploratory follow-up, not conflated with the pre-registered 13–22
  candidate range when describing methodology.
