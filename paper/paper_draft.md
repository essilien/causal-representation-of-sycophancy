# A Depth-Dependent Causal Geometry of Sycophantic Agreement in Llama-3.1-8B-Instruct

## Abstract

Preference-aligned language models frequently exhibit sycophantic agreement with
explicit user assertions, overriding the response they would otherwise produce under a
neutral prompt. Prior mechanistic work has largely characterized where this behavior
arises across a network's layers; whether the underlying causal variable is compactly and
linearly organized, or diffusely distributed, has received less direct attention. We
apply a causal-abstraction framework to Llama-3.1-8B-Instruct, combining linear probing,
full-representation activation patching, and Distributed Alignment Search (DAS) to test
this question layer by layer. A 64-dimensional trained subspace matches or exceeds
full-representation patching in mid-network layers (blocks 12–21), evidence of a compact,
linearly-aligned causal variable, but falls substantially and increasingly behind
patching in later layers (blocks 22–31) — a gap that persists after a fourfold increase in
subspace rank and after relocating the intervention to an alternative token position,
ruling out the two most immediate explanations. A control substituting a topically
unrelated answer for the asserted one produces a flip rate statistically indistinguishable
from the true incorrect answer, indicating that content plausibility is not what drives
the effect, and that sycophancy and contextual entrainment are not separable within this
experimental design. Together, these results suggest that the causal geometry of
sycophantic agreement changes qualitatively with network depth, rather than taking a
single, uniform form throughout the network.

## 1. Introduction

Preference-aligned language models frequently exhibit sycophantic agreement with explicit
user assertions, overriding the response they would otherwise produce under a neutral
prompt. This behavior compromises the reliability of such models as sources of factual
information and has been documented extensively at the behavioral level. Perez et al.
(2022) provided the first systematic empirical evidence of the phenomenon, showing that
RLHF-trained models increasingly repeat a user's stated view as model scale increases,
constituting one of the earliest documented cases of inverse scaling attributable to
RLHF. Sharma et al. (2023) subsequently offered the most comprehensive behavioral
characterization to date, establishing sycophancy as a consistent property across five
production RLHF assistants and attributing it in part to human preference data that
favors agreeable responses over truthful ones; subsequent work (Wei et al., 2023) has
largely corroborated and extended these findings. This literature established that
sycophancy occurs and under what conditions, but treated the underlying computation as a
black box.

A more recent line of work, reviewed in Section 2, has begun to open that box, variously
localizing sycophancy-relevant computation across layers, attributing it to specific
attention-head circuits, or targeting it for mitigation through fine-tuning. None of these
accounts directly tests whether the responsible computation is organized as a compact,
low-dimensional causal variable or as a more diffuse property of the full representation
— a distinction that matters both for interpretability, since a compact variable admits a
precise mechanistic account whereas a diffuse one does not, and for intervention, since
methods that target a specific subspace are only effective if such a subspace exists.

This paper investigates whether a causal-abstraction framework (Geiger et al., 2021) can
identify a residual-stream subspace in Llama-3.1-8B-Instruct that is causally necessary
for a model's answer-preference shift following a user assertion, and whether the
magnitude of this effect depends on the plausibility of the asserted content. We address
this question with three methods that differ in the grain of the question each can
answer. Linear probing asks whether a layer's representation correlates with the
magnitude of the effect; it is correlational and cheap, but establishes nothing about
causal necessity. Full-representation activation patching asks whether substituting an
entire representation at a given layer and position is sufficient to reproduce the
counterfactual behavior; this is a direct test of sufficiency, but at the granularity of
the whole representation, and, as our results show, is subject to a specific interpretive
confound when compared across layers. Distributed Alignment Search (DAS) asks the most
demanding question of the three — whether a trained, low-dimensional, aligned subspace of
that representation alone suffices — such that an affirmative result implies a compact,
precisely localized causal variable rather than merely the presence of relevant
information somewhere within a layer.

Our principal finding is that these three methods do not agree uniformly across network
depth, and that this disagreement is itself the central result rather than a
methodological inconvenience. Probing and DAS agree closely in mid-network layers (blocks
12–21), where a 64-dimensional subspace identified by DAS matches or exceeds what
full-layer patching achieves. At later layers (blocks 22–31), patching recovers
substantially more causal signal than DAS captures, and this gap does not close when the
subspace's rank is increased fourfold or the intervention relocated to an alternative
token position — the two most plausible candidate explanations. We interpret this as
evidence that the causal geometry underlying sycophantic agreement is not uniform across
depth: compact and linearly aligned in the middle of the network, and not so, at least
under either condition we tested, as the network approaches its output layers. A further
control, substituting a topically unrelated answer for the one a user asserts, produces
an almost identical flip rate to the true incorrect answer, indicating the mechanism is
largely insensitive to content and that it cannot be cleanly separated, in this design,
from contextual entrainment — the general tendency of language models to raise the
probability of recently-seen tokens regardless of relevance. We adjust our terminology for
the underlying causal variable accordingly (Section 3.1).

Throughout this paper, layer or block indices refer directly to the corresponding
transformer block of Llama-3.1-8B-Instruct's 32 blocks, zero-indexed (0–31), consistent
with the model's own architecture; the embedding output, which precedes all transformer
blocks, is excluded from all layer-indexed comparisons.

## 2. Related Work

Perez et al. (2022) and Sharma et al. (2023) established sycophancy as a robust,
scale-sensitive property of RLHF-trained assistants, and Wei et al. (2023) showed the
effect can be substantially mitigated with synthetic training data, without characterizing
its internal implementation. A more recent body of work has approached the internal
mechanisms of sycophancy from angles adjacent to, but distinct from, the one taken here.
Wang and Li (2026) use logit-lens analysis and activation patching to trace a two-stage
process across layers, in which an output-preference shift precedes a subsequent
divergence in hidden representations; their account establishes where sycophancy-relevant
computation occurs but does not test whether that computation is compactly organized or
diffusely distributed within a given layer. Pandey (2026) takes a circuit-level approach,
using path patching to identify attention heads shared between sycophantic agreement and
deceptive behavior more broadly, arguing that the resulting behavior reflects deference
despite retained knowledge of the correct answer; this offers a component-level account of
which parts of the network are responsible, but not of the dimensionality of the causal
variable those components jointly implement. Chen et al. (2025) address sycophancy from a
mitigation standpoint, using causal tracing to identify a small subset of parameters and
fine-tuning only those; their goal is intervention efficacy rather than a
causal-geometric account of the underlying representation. Noël (2026) offers a conceptual
framing in which sycophancy reflects a tension between truth-tracking and compliance
concentrated within a single layer — a claim our results speak to directly, since we
instead find a compact causal variable spanning a contiguous mid-network range and a
qualitatively different, non-compact form of causal contribution later in the network,
neither well described as a single-layer phenomenon.

Vennemeyer et al. (2025) argue that sycophancy decomposes into causally separable
sub-behaviors — sycophantic agreement, in which a model abandons a correct answer for a
stated but false user belief; sycophantic praise, or excessive flattery; and genuine
agreement, or concurring when the user's claim is in fact correct — occupying distinct,
independently manipulable subspaces rather than a single unified mechanism. The behavior
studied here corresponds specifically to sycophantic agreement: every question in our
dataset is paired with an objectively correct and an objectively incorrect answer, and the
outcome of interest is whether the model's preference shifts toward the latter once a user
asserts it, with no role for flattery or a correct user claim. Whether the causal
structure identified here extends to the other modes Vennemeyer et al. distinguish is a
question for future work, taken up briefly in Section 5.

A separate and more general literature bears on how we interpret our own results. Niu et
al. (2025) document contextual entrainment: language models assign elevated probability to
tokens that have recently appeared in context, including semantically arbitrary ones,
largely independent of relevance. Because the incorrect answer a user asserts in our
setting is also the literal string whose probability is expected to rise under this
account, entrainment offers a competing explanation for what looks like sycophantic
agreement, one we return to directly in Section 4.3. Niu et al. report that entrainment is
modulated by semantic plausibility, with counterfactual content producing a larger effect
than random content; our own results (Section 4.3) find plausibility mattering
considerably less, suggesting the two phenomena are difficult to disentangle in a setting
like ours. Taken together, this body of work converges on the view that sycophancy has
identifiable internal correlates, but no prior study directly tests the dimensionality of
the causal variable at each network depth using an alignment-search method compared
against full-representation patching — the gap this paper addresses.

## 3. Methods

### 3.1 Causal framework

We test the causal graph C → R → ΔMargin, where C denotes the presence of a
user-asserted candidate answer, R denotes an aligned subspace of the residual stream at a
given layer, and ΔMargin denotes the resulting shift in the model's preference between the
correct answer and the asserted candidate. For two candidate answers a and b under a
prompt x, we define

$$\text{Margin}(a, b \mid x) = \log P(a \mid x) - \log P(b \mid x),$$

and for a given example, the margin shift induced by a biased prompt relative to its
neutral counterpart as

$$\Delta(x) = \text{Margin}(a, b \mid x_{\text{biased}}) - \text{Margin}(a, b \mid x_{\text{neutral}}).$$

A sign flip in ΔMargin corresponds to the model's preference actually inverting toward the
asserted candidate.

We deliberately avoid calling C "user belief." As detailed in Section 4.3, a control
substituting an implausible candidate for the true incorrect answer produces a
statistically indistinguishable effect, which is inconsistent with the model evaluating
and adopting a belief and more consistent with a content-blind response to the presence
of an assertion. We use "candidate-assertion signal" instead, to avoid attributing more
semantic processing to the mechanism than the evidence supports.

### 3.2 Model, data, and scoring

We use Llama-3.1-8B-Instruct (bf16) throughout, chosen for direct comparability with Wang
and Li (2026), who study the same base model. Our data is the `answer` split of
`meg-tong/sycophancy-eval`, a set of factual questions adapted from TriviaQA in which each
question is paired with a correct and a plausible-but-wrong answer; we deduplicate the raw
dataset by question and construct neutral and biased prompt pairs directly from the
underlying fields, rather than relying on any single one of the dataset's own repeated
template variants. Unlike Wang and Li, who use multiple-choice questions, we keep the task
as free-form question answering. This preserves ecological validity and avoids requiring a
fixed, small answer-token vocabulary, which the causal-abstraction approach does not need.
The neutral prompt is simply the question; the biased prompt appends "I think the answer
is {incorrect_answer} but I'm really not sure."

For a candidate answer c of k tokens, we score it under a prompt via its teacher-forced,
length-normalized log-probability,

$$\log P(c \mid x) = \frac{1}{k}\sum_{i=1}^{k}\log P\big(c_i \mid x, c_{<i}\big),$$

length normalization being necessary because the correct and incorrect answers are rarely
the same number of tokens. This scoring primitive, together with the margin defined above,
is used identically throughout the paper, avoiding both the need to match a model's
free-form paraphrase to a fixed gold string and the ambiguity of defining "the answer
token" in variable-length output. All scoring and intervention operations use the last
token of the prompt as the position of interest, immediately before the model would begin
generating an answer; Section 4.4 reports a check of an alternative position.

### 3.3 Representational localization

To locate where sycophancy-relevant information is represented, we restrict to examples on
which the model initially preferred the correct answer under the neutral prompt — the only
examples on which sycophancy is observable — and extract the last-token hidden state at
every transformer block under the biased condition. A ridge regression probe is then fit to
predict the continuous margin shift Δ(x) from each layer's hidden state, evaluated via
5-fold cross-validated held-out R², with significance assessed against a per-layer null
built from 20 label permutations rather than a fixed threshold. An earlier design that
classified prompts as neutral or biased directly from hidden states was discarded after it
achieved near-ceiling accuracy from the first transformer block onward regardless of any
real representational content, since biased prompts are simply longer and lexically
distinct from neutral ones; regressing on the continuous margin shift, where every biased
prompt shares an identical template and only the specific question and asserted answer
vary, avoids this confound.

### 3.4 Interchange intervention: patching and Distributed Alignment Search

We use two forms of interchange intervention, both implemented via pyvene (Wu et al.),
which swap information from a source run (biased prompt) into a base run (neutral prompt,
same question) at a chosen layer and token position. Writing h_base and h_source for the
residual-stream vectors at the intervention position in the two runs, and R for the rows
of a k-dimensional aligned subspace, the intervened representation is

$$h_{\text{int}} = h_{\text{base}} + R^{\top}R\left(h_{\text{source}} - h_{\text{base}}\right),$$

replacing the component of h_base lying in the subspace spanned by R with the
corresponding component of h_source and leaving the orthogonal complement unchanged.
Full-representation patching is the special case k = d, R = I, an untrained swap of the
entire residual-stream vector that serves as an upper-bound-style reference: it tests
whether the complete representation at a site is causally sufficient, without any
constraint on how compactly that information is organized. DAS instead trains R (rank 64
unless noted) to minimize

$$\mathcal{L}(R) = \Big(\text{Margin}\big(a, b \mid \mathrm{do}(h_L \leftarrow h_{\text{int}})\big) - \Delta_{\text{real}}(x)\Big)^2,$$

where Δ_real(x) is the real, empirically observed biased-condition margin shift for that
example — a genuine counterfactual pair already present in the data rather than a
synthetic label — and do(h_L ← h_int) denotes substituting the intervened representation
before continuing the forward pass.

Both methods are evaluated by Interchange Intervention Accuracy,

$$\text{IIA} = \frac{1}{|\mathcal{D}_{\text{test}}|}\sum_{x} \mathbb{1}\Big[\mathrm{sign}\big(\Delta_{\text{int}}(x)\big) = \mathrm{sign}\big(\Delta_{\text{real}}(x)\big)\Big],$$

and by the Pearson correlation r between intervened and real margins, against two
baselines: a trivial baseline that always predicts no flip, and an untrained baseline
using a freshly initialized, random R prior to any training, which confirms that any gain
over these two reflects genuine training rather than the mere presence of some subspace at
a site.

Patching later in the network transplants a progressively larger share of the entire
biased-condition computation, mechanically approaching the true biased output as the
intervention point nears the final layer; a rising full-patch IIA curve across layers is
therefore expected in part by construction and is not on its own evidence that later
layers contain more localized causal information. The layer-matched comparison between DAS
and full-patch at the same layer avoids this confound, since both hold fixed how much of
the network's cumulative computation is being transplanted, and is the comparison we rely
on for claims about depth (Section 4.4). Two follow-up checks probe specific explanations
for the resulting gap at later layers: retraining DAS at the block with the largest gap
using a 256-dimensional subspace, and an untrained full-patch check sourcing from the end
of the asserted-answer span rather than the end of the prompt, to test whether causally
relevant information had migrated to a different token position.

### 3.5 Controls

Because the incorrect answer a user asserts appears verbatim in the biased prompt, and
language models are known to raise the probability of recently-seen strings independent of
relevance (Niu et al., 2025), any margin shift we observe could in principle reflect
contextual entrainment rather than genuine uptake of the user's assertion. To probe this,
our primary control replaces the incorrect answer with a topically unrelated one under an
identical assertion template — for example, for the question "Can juice fasts detoxify the
body?" (correct answer "No", incorrect answer "Yes"), the substituted candidate was
"Commander-in-chief" — and compares the resulting flip rate against the true
incorrect-answer condition. As a further check on the representational localization
result's stability, we also compute an entrainment-based estimate of the margin shift
using a control prompt that exposes the incorrect answer without asserting it as anyone's
belief ("{question} Random word: {incorrect_answer}, noted here for no particular
reason."), and re-run the regression probe (Section 3.3) on the difference between the
observed margin shift and this estimate. We treat this as a check of whether the same
layers remain identifiable under one specific adjustment for copying, not as a method that
cleanly isolates a content-sensitive component from entrainment; Section 5 discusses why
the latter is not something this design can establish.

### 3.6 Experimental setup

Unless otherwise noted, DAS is trained with Adam (learning rate 1e-3) for 500 steps per
layer, using a rank-64 subspace, on an 80/20 split of the 601-example neutral-correct pool
(test set n ≈ 121–150). Training uses a batch size of one: each example's intervention
position falls at a different absolute token index, and correctly batching pyvene's
per-example position specification was not something we could validate within the scope of
this project, so we accepted the higher variance of single-example training over the risk
of a silent, undetected indexing error. This variance is quantifiable: for a binomial
proportion with p ≈ 0.5 and test-set size n ≈ 121, the expected standard error is
$\sqrt{p(1-p)/n} \approx 4.5$ percentage points, comparable to the step-to-step
fluctuation we observe when logging IIA every 100 steps, and consistent with training loss
having already plateaued by step 200–300 in representative layers. We therefore report each
layer's IIA and r as the mean of its final three logged checkpoints (steps 300, 400, 500)
rather than a single final value.

## 4. Results

### 4.1 Baseline sycophancy

Under the neutral prompt, the model preferred the correct answer on 60.1% of 1,000 sampled
questions. Restricted to these — the only questions on which sycophancy is observable — the
biased prompt flipped the model's preference on 50.2% of cases (302/601); among flipped
examples, the mean margin moved from +2.27 under the neutral prompt to −1.79 under the
biased one.

### 4.2 Representational localization

The regression probe's held-out R² rises from near the permutation-null floor for the
embedding output and blocks 0–7 to a broad peak around blocks 14–18 (R² ≈ 0.13–0.16),
remaining clearly above null through roughly block 21 (Figure 1, right axis). The raw
behavioral margin-shift curve is qualitatively different in shape, staying near zero
through block 14 before deepening roughly monotonically to a maximum around block 27 (mean
shift ≈ −3.95; Figure 1, left axis) — broadly consistent with Wang and Li's report of a
late-layer output-preference turning point and representation-divergence peak in the same
base model, though the two studies' specific layer numbers are not directly comparable
given differing task formats and prompt templates. Re-running the probe on the residual obtained by subtracting a copy-effect control's
estimated margin shift (Section 3.5) leaves the R² curve's shape and magnitude largely
unchanged (r = 0.93 correlation with the original curve; nine of the original top-ten
layers by R² unchanged). We report this as a check of whether the layer-wise localization
is stable under one specific way of adjusting for the copying confound, not as evidence
that the underlying representation is independent of contextual entrainment: the
subtraction presupposes that entrainment and any content-sensitive component are
separable, additive contributions to the margin shift — an assumption our own
margin-decomposition analysis (Section 4.3) gives reason to doubt, and one we return to in
Section 5. Read at the level the data support, this result establishes only that the same
layers remain identifiable after this particular adjustment, not that they track a
component of the effect cleanly separable from entrainment. The candidate range carried
forward into the causal-validation experiments below is blocks 12–21.

![Figure 1: Representational localization — regression R² (representation) and raw
margin-shift (behavior) across network depth.](phase2_localization.png)

### 4.3 Content-independence and the entrainment confound

The flip rate under the irrelevant-answer control was 52.3%, against 50.2% for the true,
plausible-but-wrong answer — statistically indistinguishable. Decomposing the margin shift
shows the effect is driven almost entirely by the irrelevant candidate's own
log-probability rising (+7.00) rather than the correct answer's falling (−1.70),
consistent with a content-blind response to the presence of an assertion rather than any
evaluation of what was asserted. Because this is the same signature contextual entrainment
would produce — an elevated probability for a recently-seen string, regardless of its
relevance — we cannot, within this design, cleanly separate genuine sycophantic uptake of a
user's stated view from entrainment triggered by the mere appearance of the candidate
string in context; this motivates referring to the underlying causal variable as a
candidate-assertion signal (Section 3.1) rather than belief. This departs somewhat from Niu
et al. (2025), who report entrainment strengthening with semantic plausibility: in our
setting, plausibility makes little detectable difference to the aggregate rate, suggesting
the assertion framing itself — not the content it carries — is doing most of the work.

Matching aggregate rates do not, by themselves, imply the two conditions perturb the same
examples. We therefore examined per-example overlap between questions that flipped under
the true incorrect answer and questions that flipped under the irrelevant one (Figure 3).
[PLACEHOLDER — overlap statistics pending re-run of controls/check_irrelevant_overlap.py
against the original n=601 results file: both/only-real/only-irrelevant counts, Jaccard
overlap, expected co-flips under independence, and the chi-square/phi effect size.]

![Figure 3: Left — distribution of length-normalized margins across the neutral, biased,
and irrelevant-answer conditions. Right — per-example comparison of the margin shift under
the irrelevant-answer condition against the shift under the true biased condition, with
the identity line shown for reference.](margin_distribution_analysis.png)

### 4.4 Causal validation across network depth

All 32 blocks were evaluated under both full-representation patching and DAS, using the
protocol in Section 3.6. Blocks 0–11 remained close to both the trivial and untrained
baselines throughout training (IIA 41–44%, against a trivial baseline of 40.5%),
consistent with the near-null R² found for this range in Section 4.2. From block 12
onward, both methods separate clearly from baseline, and their relationship to one another
changes systematically with depth (Figure 2). DAS matches or exceeds full-representation
patching for blocks 12–17, by as much as 9.9 percentage points at block 14; the two methods
are essentially tied across blocks 17–21, within the noise band established in Section
3.6; and from block 22 onward the gap reopens in favor of patching and widens smoothly and
substantially with depth, reaching 30 percentage points by block 31. Table 1 summarizes
this pattern at representative blocks.

![Figure 2: DAS versus full-representation patching Interchange Intervention Accuracy
across network depth, showing a crossover around blocks 17–20.](das_vs_patch.png)

**Table 1.** DAS and full-representation patching IIA at representative blocks (trivial
and untrained baselines ≈40.5% throughout).

| Block | 12 | 14 | 17 | 21 | 27 | 31 |
|---|---|---|---|---|---|---|
| Full-patch IIA | 41.3% | 43.0% | 47.1% | 47.9% | 63.6% | 74.4% |
| DAS IIA | 44.6% | **52.9%** | 48.2% | 49.6% | 48.5% | 44.4% |
| Difference (patch − DAS) | −3.3 | −9.9 | −1.1 | −1.7 | +15.1 | +30.0 |

At mid-network depth, a 64-dimensional trained subspace outperforms swapping the entire
representation — evidence that alignment search isolates causally relevant dimensions
rather than merely replicating a full swap, the central premise DAS is meant to
demonstrate relative to naive patching. Past the crossover, that advantage disappears and
then reverses sharply. We tested two explanations for the resulting late-layer gap.
Retraining DAS at block 27 with a 256-dimensional subspace, four times the default,
produced no meaningful improvement over the equivalent 64-dimensional estimate,
ruling out a simple rank bottleneck. An untrained full-patch check sourcing from the end
of the asserted-answer span, rather than the end of the prompt, gave a lower IIA (52.1%)
than the original position (63.6%), inconsistent with causally relevant information having
migrated toward the assertion span by this depth. Neither of the two most obvious
corrections closes the gap; we interpret the persistent, widening difference as evidence
that the causal contribution recoverable by full-representation patching late in the
network is not compactly represented as a low-rank linear direction at a single position,
in contrast to blocks 12–20, where it evidently is.

## 5. Discussion

The representational and causal results converge on where a compact, linearly-organized
variable relevant to sycophantic agreement exists in this model — roughly blocks 12–20,
weakest at its lower boundary under both probing and repeated causal-intervention runs —
and diverge informatively, rather than contradictorily, beyond that range: patching
recovers substantial causal content late in the network that neither of the two most
obvious refinements to DAS succeeds in isolating. We read this as evidence that
sycophantic agreement is not implemented by a single, uniform mechanism across depth, but
by a compact, linearly-aligned variable in the middle of the network that gives way to a
qualitatively different, less compactly organized form of causal contribution closer to
the output — a distinction neither purely behavioral localization nor full-representation
patching alone would reveal, and one that connects to an unresolved tension in the broader
literature between compact-circuit and diffuse, persona-level accounts of sycophancy.

Several limitations bear directly on how much weight these conclusions can carry. A
central one concerns identifiability, and applies to two separate points in our analysis:
the margin-decomposition comparison in Section 4.3, and the residual-regression check in
Section 4.2. Both treat contextual entrainment and any content-sensitive component of
sycophantic agreement as though they were separable, additively-combining contributions to
an aggregate margin shift — subtracting one to isolate the other, or comparing the
relative size of one against the other across conditions. A single scalar margin shift,
decomposed only into each candidate's own log-probability change, cannot in principle
distinguish this two-mechanism account from an alternative in which a single mechanism's
intensity varies continuously with the asserted content's plausibility: both accounts
predict the same observable pattern — a significant difference in asymmetry between the
biased and controlled conditions (Section 4.3) — and neither our decomposition nor the
residual-regression check can adjudicate between them. Establishing that entrainment and
content-sensitive belief revision occupy distinct, separately-identifiable causal
structure, rather than reflecting one mechanism's response to a continuous plausibility
signal, would require a method capable of testing representational separability directly —
the kind of causal-abstraction analysis Vennemeyer et al. (2025) use to demonstrate that
other sycophancy sub-behaviors occupy independent subspaces — which is beyond the scope of
the present decomposition and is noted here as a direction for future work rather than a
claim this paper is positioned to make. The late-layer finding is supported only negatively — we ruled
out insufficient rank and position migration, but never identified what does explain the
gap, so a non-compact representation remains the most parsimonious surviving hypothesis
rather than a demonstrated mechanism, and a broader search over intervention sites,
including non-residual-stream sites such as the attention-head circuits identified by
Pandey (2026), might still close it. Several implementation choices were scope-driven
rather than principled — a batch size of one, evaluation on a test set with a substantial
quantified noise floor, and a single training seed per configuration — meaning
layer-to-layer differences within the 44–53% IIA band in the mid-to-late network should be
treated cautiously, and only the clearest outliers (block 12's weakness, the crossover
itself) are likely to survive replication under more rigorous statistics. Finally, all
findings rest on a single model, task format, and prompt template pair, and correspond
specifically to sycophantic agreement in the taxonomy of Vennemeyer et al. (2025); whether
the mid-network compact, late-network diffuse pattern generalizes across models, the other
sycophantic sub-behaviors they distinguish, or different sycophancy-inducing framings
remains untested, and, given how sensitive our own results proved to a change as small as
the content of the asserted candidate, should not be assumed without direct replication.

## References

Chen, W., Huang, Z., Xie, L., Lin, B., Li, H., Lu, L., Tian, X., Cai, D., Zhang, Y., Wang,
W., Shen, X., & Ye, J. (2025). *From Yes-Men to Truth-Tellers: Addressing Sycophancy in
Large Language Models with Pinpoint Tuning*. arXiv:2409.01658.

Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). *Causal Abstractions of Neural
Networks*. Advances in Neural Information Processing Systems (NeurIPS 2021).
arXiv:2106.02997.

Niu, J., Yuan, X., Wang, T., Saghir, H., & Abdi, A. H. (2025). *Llama See, Llama Do: A
Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs*. Proceedings
of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), 16218–16239. arXiv:2505.09338.

Noël, V. (2026). *When Belief Bends to Belief: Sycophancy as a Single-Layer
Truth–Compliance Tension in LLMs*. ICML 2026 Workshop: Philosophy Meets Machine Learning.

Pandey, M. (2026). *LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying
Circuit*. arXiv:2604.19117.

Perez, E., Ringer, S., Lukošiūtė, K., et al. (2022). *Discovering Language Model
Behaviors with Model-Written Evaluations*. arXiv:2212.09251.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N.,
Durmus, E., Hatfield-Dodds, Z., Johnston, S. R., Kravec, S., Maxwell, T., McCandlish, S.,
Ndousse, K., Rausch, O., Schiefer, N., Yan, D., Zhang, M., & Perez, E. (2023). *Towards
Understanding Sycophancy in Language Models*. arXiv:2310.13548.

Vennemeyer, D., Duong, P. A., Zhan, T., & Jiang, T. (2025). *Sycophancy Is Not One Thing:
Causal Separation of Sycophantic Behaviors in LLMs*. arXiv:2509.21305.

Wang, K., Li, J., Yang, S., Zhang, Z., & Wang, D. (2026). *When Truth Is Overridden:
Uncovering the Internal Origins of Sycophancy in Large Language Models*. Proceedings of
the AAAI Conference on Artificial Intelligence, 40(39), 33566–33574.

Wei, J., Huang, D., Lu, Y., Zhou, D., & Le, Q. V. (2023). *Simple Synthetic Data Reduces
Sycophancy in Large Language Models*. arXiv:2308.03958.
