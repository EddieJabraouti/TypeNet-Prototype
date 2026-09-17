# MED pause experiment on the full Aalto freeze

Same people and synthetic draws as the full-Aalto parent freeze. MED is a length-5 Wiggins kurtosis filter estimated on each person's clean gallery press-to-press, applied to gallery and query, then scored with the same 42-d pause recipe. No generator pause mask. No maximum-entropy density. Selection is among med, augmented, augmented+med, augmented+pause, and augmented+pause+med by 4-way accuracy, then log loss. Chance accuracy is 25%.

Cohort: train 55,732; selection 1,306; calibration 1,034; validation 1,202; test 2,738.

Selection candidates:
- summary (618 features): selection 64.34%, log loss 0.7894; train 66.41% (4-way); parent 64.34%
- med (42 features): selection 41.77%, log loss 1.2002; train 44.99% (4-way)
- augmented (1443 features): selection 66.21%, log loss 0.7527; train 68.64% (4-way); parent 66.21%
- augmented_med (1485 features): selection 66.96%, log loss 0.7442; train 69.07% (4-way)
- augmented_pause (1485 features): selection 67.40%, log loss 0.7449; train 69.06% (4-way); parent 67.40%
- augmented_pause_med (1527 features): selection 67.30%, log loss 0.7435; train 69.14% (4-way)

Selected: **augmented_pause**. Train accuracy 69.06%; selection accuracy 67.40% (4-way, chance 0.25).

## Parent freeze (augmented, no raw-series concat)

- Validation: 64.70%; EER 14.81%. Recall normal 78.5%, mild 55.7%, moderate 50.7%, severe 74.0%.
- Test: 65.75%; EER 13.93%. Recall normal 79.8%, mild 57.6%, moderate 51.5%, severe 74.1%.

## Four-class accuracy versus the 618-feature summary

- **Validation:** 63.14% → **65.45%**; change +2.31 pp (95% participant-bootstrap CI +1.37, +3.31).
  EER 15.86% → 14.57%; ECE 1.85 → 1.37 pp; log loss 0.7982 → 0.7573.
  Recall: normal 78.4%, mild 55.4%, moderate 51.5%, severe 76.5%.
  Versus previous augmented: 64.70% → 65.45%; change +0.75 pp (95% CI +0.06, +1.50).
- **Test:** 64.11% → **66.66%**; change +2.56 pp (95% participant-bootstrap CI +1.90, +3.19).
  EER 15.75% → 13.99%; ECE 1.06 → 1.50 pp; log loss 0.7817 → 0.7348.
  Recall: normal 80.1%, mild 58.7%, moderate 53.1%, severe 74.8%.
  Versus previous augmented: 65.75% → 66.66%; change +0.91 pp (95% CI +0.43, +1.40).

## Pause concat on the same parent

Selected pause representation: augmented_pause. Selection 67.40%; train 69.06% (4-way).
- Validation: 64.70% → 65.45%; change +0.75 pp (95% CI +0.06, +1.50). Recall normal 78.4%, mild 55.4%, moderate 51.5%, severe 76.5%.
- Test: 65.75% → 66.66%; change +0.91 pp (95% CI +0.43, +1.40). Recall normal 80.1%, mild 58.7%, moderate 53.1%, severe 74.8%.

MED descriptors are absolute spikes after an enrollment-only kurtosis deconvolution of press-to-press. They are not a fitted maximum-entropy density. These synthetic classes are not clinically validated severity labels.

## Architecture research after the augmented_pause freeze

The relevant starting point is the **1,485-feature augmented_pause model** and
its existing participant splits. An architecture comparison should use these
same inputs first, so a change in feature extraction is not mistaken for an
architecture improvement. The 222,928 training rows represent 55,732 people
with four related examples each; the 100 query–baseline comparisons within an
example are also dependent.

- **NODE is the closest neural extension.** It jointly learns ensembles of
  differentiable oblivious trees and can compose them into multiple layers.
  This could learn interactions between timing, distribution, and pause
  descriptors. A faithful experiment needs sparse entmax feature selection,
  differentiable routing, data-dependent initialization, and a quantile
  transform fitted on training data only. It is a replacement learner on the
  frozen features, not a continuation of the existing fitted trees.
  [Original paper](https://arxiv.org/abs/1909.06312),
  [author implementation](https://github.com/Qwicen/node).
- **gcForest is a natural forest extension through its cascade.** Each layer
  augments the original features with class probabilities from forests. Its
  multi-grained scanning is inappropriate for arbitrary adjacent columns of
  this engineered vector; use the cascade alone initially. Class-probability
  features must be generated out of fold, grouping all four examples from a
  participant together. Ordinary row-level folds would allow related examples
  from one person into both sides. Cascade depth is selected using validation
  participants. [Original paper](https://www.ijcai.org/proceedings/2017/0497.pdf),
  [author implementation](https://github.com/kingfengji/gcForest).
- **TabNet is plausible, but a lower-priority first experiment.** Its sequential
  sparse attention can choose different subsets of the 1,485 descriptors for
  different examples. That is useful when the relevant anomaly channels vary.
  A faithful implementation includes attentive masks, shared and step-specific
  gated feature transforms, ghost batch normalization, and the mask sparsity
  penalty. Correlated engineered columns make an attention mask insufficient
  evidence of a unique causal explanation.
  [Original paper](https://arxiv.org/abs/1908.07442),
  [author implementation](https://github.com/google-research/google-research/tree/master/tabnet).

These are credible hypotheses, not evidence that any will reach 70% here.
Controlled comparisons of tabular networks, including NODE and TabNet, find
no universally superior learner; strong simple baselines remain necessary.
[Comparative study](https://arxiv.org/abs/2106.11959).

There is an obvious unfinished baseline experiment before introducing a new
architecture: all full-Aalto representation comparisons fixed the classifier
at 600 rounds. The earlier regularization search selected its settings on the
618-feature representation and a smaller cohort. It did not tune the final
1,485-feature representation. The frozen train–validation gap is 3.60 percentage points. A capacity check
can determine whether extra fitting improves validation/test accuracy; the gap
alone does not prove underfitting or establish an attainable accuracy.

Each architecture implementation occupies exactly one source file
(`node.py`, `gcforest.py`, or `tabnet.py`), including its architecture-specific
layers, preprocessing, training, checkpoint handling, and prediction logic.
No new helper modules or architecture subdirectories are needed. Keep the
existing frozen feature definition and participant protocol as the comparison
contract. Per the requested development protocol, use **validation accuracy**
with log loss as the tie-breaker, calibrate using calibration participants, and
evaluate each locked candidate once on test. The historical selection role is
diagnostic only and never rejects an architecture. These benchmark participants
have already been inspected historically; they are not a fresh external test
population. Validation now participates in checkpoint choice and must be
interpreted as a development metric.

### Bounded capacity diagnostic before the protocol change

An in-memory copy of the frozen classifier was continued on the same full
training cohort, changing only the boosting-round count:

- 600 rounds: train 69.06%, historical selection 67.40%, selection log loss 0.7449.
- 900 rounds: train 70.84%, historical selection 67.17%, selection log loss 0.7362.
- 1,200 rounds: train 72.34%, historical selection 67.32%, selection log loss 0.7317.

The initial probe omitted validation/test and retained only its historical
selection-best checkpoint. The fixed 1,200-round model was subsequently
reproduced exactly (including 72.344434% training accuracy), recalibrated using
calibration participants, and evaluated on validation/test:

- **1,200 rounds: train 72.34%, validation 65.81%, test 66.74%.**
- Validation change versus the frozen 600-round model: **+0.35 percentage
  points**, participant-bootstrap 95% CI **[−0.21, +0.87]**.
- Test change: **+0.07 percentage points**, 95% CI **[−0.30, +0.42]**.

Both intervals include zero. Extra rounds improve training accuracy substantially
but do not establish a clear validation/test accuracy gain. The evaluated model
and complete numerical results are retained in
`runs_med/augmented_pause_1200.joblib`. The frozen model and source remain
unchanged, and historical selection scores do not gate the new architectures.

### Implemented architecture configurations

- `node.py`: two densely connected layers, 64 oblivious trees per layer,
  depth four, four outputs per tree, sparse entmax-1.5 choices and routing,
  training-only quantile normalization and data-aware initialization. AdamW
  at learning rate 0.003 replaces the paper's QHAdam/checkpoint averaging.
- `tabnet.py`: three decision steps, decision/attention widths 32/32, two
  shared and two independent GLU transforms, sparsemax masks, ghost batch
  normalization, gamma 1.5 and entropy penalty 0.001. Training-only quantile
  preprocessing is an adaptation for the skewed engineered inputs.
- `gcforest.py`: at most three cascade levels, each with one 64-tree random
  forest and one 64-tree random-feature ExtraTrees forest, maximum depth 16
  and minimum leaf size 20. Three participant-grouped folds generate class
  vectors per level. This is a bounded gcForest-style cascade, not a complete
  reproduction of the paper's larger forests or multi-grained scanning.
  Cross-fitting is performed per level, not nested across the entire cascade;
  the reported training accuracy uses fitted-model inference, not OOF vectors.

Neural runs allow up to 60 epochs with validation patience 12 and batch size
2,048. These are initial controlled configurations, not an exhaustive search
over the model families. Each file owns its architecture and experiment logic;
only raw-window prediction calls the existing frozen feature encoder. Checkpoints
and histories live under the existing `runs_med/` directory.

### Full-cohort architecture benchmark

Accuracy below is reported as **train / validation / test**, at the chosen
checkpoint. Validation chooses the checkpoint; test is evaluated afterward.
Temperature calibration changes confidence, not these predicted classes.

- **Frozen augmented_pause:** **69.06% / 65.45% / 66.66%**.
- **NODE:** **65.58% / 64.52% / 65.02%**. Validation chose epoch 30;
  training stopped after epoch 42. Test change versus the frozen winner is
  **−1.64 percentage points**, participant-bootstrap 95% CI **[−2.26, −1.01]**.
  This compact configuration underperforms the baseline on all three roles;
  its lower training accuracy leaves capacity/optimization as an open question.
- **TabNet:** **66.91% / 65.58% / 66.07%**. Validation chose epoch 30;
  training stopped after epoch 42. Test change is **−0.59 percentage points**,
  participant-bootstrap 95% CI **[−1.23, +0.07]**. The small validation gain
  did not translate into better test accuracy. Its calibrated test log loss
  is 0.7295 versus the frozen winner's 0.7348, which improves probability
  scoring but does not meet the accuracy objective.
- **gcForest:** **72.76% / 61.00% / 62.49%**. Validation chose the third
  cascade level. Test change is **−4.17 percentage points**, participant-bootstrap
  95% CI **[−4.85, −3.49]**. Higher training accuracy accompanies substantially
  worse validation/test performance in this configuration.

None of these initial architecture configurations reaches 70% validation/test
or improves test accuracy over the frozen winner. The 1,200-round capacity
check gives only a small, statistically inconclusive gain. Keep the original
`augmented_pause` freeze. These results do not rule out better configurations
within the architecture families; they do rule out promoting these particular
candidates. All three completed, calibrated studies and their histories are
saved as `runs_med/{node,tabnet,gcforest}_candidate.joblib`.

Verification included entmax/sparsemax gradient checks, synthetic learning
checks for the neural architectures, TabNet inference batch invariance,
participant-disjoint cascade folds, cascade checkpoint round-trip predictions,
source compilation, and frozen artifact/source hash checks. Architecture code
is contained in the three source files; no new helper modules were introduced.


## Generator and recoverability audit

### Scope and decision

The generator contains recoverable signal. The audit does **not** establish
that 66–67% is a Bayes ceiling, that another architecture cannot improve it,
or that the synthetic labels correspond to clinical cognitive stages.
The priority is to validate the generated behavior and observation protocol
against cognitive data before enlarging the architecture search.
The existing generator, feature encoder, participant protocol, and frozen
winner remain unchanged. No new project source files or directories were added.

### Matched diagnostic experiment

Deterministically select 3,000 train, 600 validation, and 1,000 test participants
from their existing, disjoint roles. Within each role, sort IDs by SHA-256 of
`audit:9182026:{role}:{uid}` and retain the first requested number. Each person
contributes all four classes. Reproduce the original synthetic draws with
`rng_for`; preserve the ten clean enrollment and ten query windows.
These are **diagnostic subsets**, not replacement full-cohort benchmarks.
Selection participants are unused. No validation/test parameter search occurs.

All probes use the same fixed HistGradientBoostingClassifier: 600 iterations,
15 leaves, depth 4, minimum leaf size 40, L2 50, learning rate 0.04, seed 9182026,
no early stopping. Minimum leaf size differs from the full-cohort winner because
these fits have 12,000 training examples. These are diagnostic fits of the
existing learner family, not a new architecture. The high training scores show
that these smaller-cohort fits overfit; compare probes on matched held-out people.

Accuracy is **train / validation / test**:

- **Frozen winner evaluated on these subsets: 68.90% / 65.79% / 67.15%.**
- **Existing 1,485 observed features: 93.23% / 63.33% / 64.48%.**
- **Observed features + 48 sequence descriptors: 93.50% / 63.08% / 64.98%.**
- **Privileged clean-twin + trace descriptors (31 features): 90.61% / 84.21% / 85.52%.**
- **Clean-twin timing residuals only (14 features): 88.49% / 84.04% / 84.92%.**
- **Clean-twin hold residuals only (7 features): 79.83% / 76.50% / 77.72%.**
- **Injected-pause traces only (16 features): 78.90% / 75.29% / 75.67%.**

The initial three probes were specified before their results were inspected.
The three component probes were added afterward to distinguish clean-twin
information from generator pause-mask information; they are exploratory diagnostics.

The observable additions measure long-interval run lengths, threshold crossings,
normalized lag correlations, conditional persistence, hold/press coupling, and
post-long-interval recovery. Window means/standard deviations are differenced
against enrollment. All thresholds come from the person's clean enrollment.
They add no generator labels. The existing encoder already includes timing
lag statistics, residual increments, and tail persistence; it is not entirely
order-blind.

Adding these 48 descriptors changes validation by **−0.25 percentage points**
(95% participant-bootstrap CI **[−1.21, +0.67]**) and test by **+0.50 points**
(**[−0.18, +1.18]**). This does not establish a gain. It does not rule out a
sequence model or other temporal representations.

The privileged combined probe improves test by **21.05 points** over the
matched observed-feature probe (95% paired CI **[+19.53, +22.58]**).
Its absolute test interval is **[84.48%, 86.53%]**. The clean-twin-only probe
reaches 84.93% without generator pause masks, showing that access to the exact
unperturbed query explains most of this particular diagnostic gain.
Both probes trivially recognize unchanged normal queries; among the three
impaired classes their correct-class rates are 80.70% and 79.90%, respectively.

These are **not deployable models or upper bounds**. A clean query twin removes
natural session, content, and timing variation that an earlier enrollment
period cannot remove. Privileged traces reveal which delays were injected.
The experiment therefore cannot attribute the entire gap to feature compression
or promise that a raw-sequence model can close it. All confidence intervals
resample participants, with their four classes kept together (1,500 resamples).

### Mechanism sensitivity and repeatability

On the first 200 test participants from that deterministic subset, regenerate
800 original examples. All 1,485 features reproduce the frozen cache **exactly**
(maximum absolute difference 0), as do the model's predictions.
Apply the frozen model without retraining to one-mechanism-at-a-time ablations:

- Original generator: **65.50%** test accuracy.
- No injected pauses: **61.38%**; difference **−4.13 points**
  (95% paired CI **[−6.00, −2.38]**).
- No motor perturbation: **28.13%**; difference **−37.38 points**
  (**[−40.88, −34.00]**).
- Motor noise correlation changed from 0.65 to zero: **61.88%**;
  difference **−3.63 points** (**[−6.13, −1.18]**).

Ablations retain class labels, clean source windows, profiles where applicable,
and random-number consumption. These are distribution-shift sensitivity tests,
not retrained performance ceilings or an additive decomposition of importance.
They show that the frozen classifier relies strongly on the generated motor
channel. Pause signals help, especially severe-class recall (71.5% original
versus 46.5% without injected pauses).

For each of the same source examples, generate three additional independent
realizations. Holding the original impairment profile fixed gives overall
accuracies **65.50%, 66.13%, 65.88%, 66.38%** across the four draws. Nevertheless,
**37.0% of impaired examples change predicted class at least once**; overall
pairwise disagreement is 15.85%. Redrawing profiles as well gives accuracies
**65.50%, 64.88%, 65.00%, 63.75%**, with **76.33% of impaired examples changing
predicted class at least once** and overall pairwise disagreement 35.75%.
The unchanged normal class is excluded from the impaired-example percentages.
This is same-source synthetic repeatability, not repeatability across new real
sessions. Stable aggregate accuracy conceals appreciable individual instability.

### Class overlap and sparsity

The mean of the sampled motor and cognitive burdens equals
`severity_factor × clipped_lognormal_heterogeneity`. The motor/cognitive split
and speed-matched flag add no severity information once that mean is known.
The three factors are 0.25, 0.55, and 1.0; heterogeneity spans 0.35–1.75.
Their burden supports overlap substantially.

An exact posterior classifier for **these burdens alone**, including the
clipping atoms, achieves approximately **86.51% balanced four-class accuracy**
in a simulation of 200,000 profiles per impaired class, or 82.01% among impaired
classes. Normal burden is zero and is recognized exactly. This is a restricted
information experiment, **not a Bayes ceiling for the full generator**: injected
pause probabilities depend on severity separately from the burdens.

Across the 1,000 test participants, the median observation contains **353 valid
transitions**, rather than 500 (shorter windows and terminal-transition exclusion).
Median injected-pause counts are **2 / 6 / 16** for mild/moderate/severe.
**16.1% of mild** and **0.8% of moderate** examples contain no injected pause at all.
Among participants with an injected pause, the median of each person's median
added delay is **97 / 194 / 349 ms**. These are added delays, not complete observed
intervals or a clinically validated definition of a cognitive pause.

### Clinical evidence and measurement mismatch

The local `data/clinical/ad_mci_writing/Participant_level.csv` contains 59 task
rows from **30 participants: 15 controls, 10 MCI, and 5 Alzheimer’s participants**.
Average available tasks within each participant before describing each group;
do not treat task rows as independent patients. One MCI participant lacks the
pause-time proportion and some rate measures. The following medians are
unadjusted descriptive summaries, not fitted causal effects or stage cutoffs:

- Proportion of task time in pauses above 2 seconds: controls **36.35%**,
  MCI **60.40%** (n=9), AD **53.75%**.
- Production-burst duration: controls **9.89 s**, MCI **4.23 s**, AD **4.73 s**.
- Production bursts per minute: controls **4.07**, MCI **5.60** (n=9), AD **6.62**.

The corresponding study measures writing and pauses during picture-description
production, with a 2-second threshold for production bursts and adjustments for
typing speed. Its findings support altered fluency; they do not establish
three ordered impairment tiers or a unique repetitive signature per severity.
[Cognitive Writing Process Characteristics in Alzheimer’s Disease](https://pmc.ncbi.nlm.nih.gov/articles/PMC9311409/).

For context only, within our 200 sampled participants' synthetic windows,
median observed press-to-press time share in intervals above 2 seconds is
**0.00% / 0.00% / 0.84% / 5.08%** for normal/mild/moderate/severe.
This is **not an apples-to-apples clinical comparison**: whole-task versus
within-window measurement, different tasks/cohorts, and interval definitions
matter. In particular, `ordered_blocks` separates sentences and the feature
encoder excludes terminal transitions, so inter-sentence planning gaps are
absent. This mismatch prevents directly calibrating the generator by multiplying
pause rates until its numbers resemble the clinical table.

The generator's documented hold-location priors also borrow from Tappy and
neuroQWERTY. Those are Parkinson’s/motor datasets, not a calibration of cognitive
severity. They can motivate a motor nuisance or comorbidity model, but do not
validate the present mapping from motor burden to cognitive labels.
[neuroQWERTY dataset](https://physionet.org/content/nqmitcsxpd/1.0.0/),
[Tappy study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188226).

### Concrete next revision, constrained by the evidence

1. **Align observation and measurement first.** Choose whether the intended
   deployed task is continuous natural writing or short typing windows. Retain
   sentence/session boundaries and censored intervals for continuous writing;
   measure clinical and synthetic bursts under the same definition. The current
   cached windows cannot reconstruct gaps that were discarded.
2. **Separate cognitive processes from motor nuisance.** Preserve person-specific
   baseline timing, but do not use Parkinson’s-derived motor amplitude as the
   principal definition of cognitive stage. Model ordinary session variation
   in controls as well as affected examples. Fit relationships between motor
   and cognitive effects only where corresponding labels support them.
3. **Fit duration and context, not prescribed class signatures.** A candidate
   revision can represent fluent production and hesitation with explicit dwell
   times, conditioned on word/sentence context. Correction/recovery states need
   relevant measured evidence; unchanged key streams cannot validate generated
   correction frequency. Keep parameters persistent across windows from the
   same simulated session. Do not invent deterministic patterns to improve
   separability.
4. **Use labels the evidence can support.** The available writing cohort supports
   a small control-versus-cognitively-impaired study and exploratory MCI/AD
   comparisons. Mild/moderate/severe remain synthetic stress levels unless
   linked to an appropriate clinical severity measure. Do not relabel the
   existing benchmark or claim a clinically calibrated progression.
5. **Judge the revision by behavioral fit and robustness.** Compare participant
   distributions of pause-time share, burst lengths, context-conditioned pauses,
   session repeatability, and timing variability. Include alternative generator
   settings and repeated draws. Only then compare train/validation/test model
   performance; a synthetic 70% alone is not the acceptance criterion.

No generator parameters were changed by this audit because the measurement
mismatch has to be resolved before the clinical numbers can justify those
changes. The audit narrows the next work to generator/observation validity and
baseline-relative estimation; it gives no positive evidence for prioritizing
more TabNet capacity or these 48 extra descriptors.

### Audit provenance and verification

Scratch code, selected participant IDs, source-file hashes, full confusion
matrices, uncertainty intervals, and raw numerical summaries were written to
`/private/tmp/exp3_generator_audit/` rather than adding files to exp3.
Scripts are `audit.py`, `repeat.py`, and `supplement.py`; results are
`results.json`, `repeat_results.json`, `supplement_results.json`, and
`long_pause_results.json`. Scratch files are temporary; this section records
the persistent methods and conclusions. No diagnostic checkpoint replaces any
saved architecture or frozen classifier.

The audit verified role disjointness, feature/cache/source hashes, finite
features, deterministic draw reproduction, exact feature/prediction agreement
on 800 examples, and unchanged frozen artifacts after execution. Diagnostic
learner fits use training participants only. The historical test population
has already been examined repeatedly and is not a fresh external clinical test.


## Clinical duration generator implementation

### Implementation and scope

Implemented an **opt-in control-versus-impaired clinical duration generator**
in the existing `prototype_net/perturbation_generator/__init__.py`. Its public
API is `ClinicalCalibration`, `ClinicalProfile`, `ClinicalTimingGenerator`,
and `measure_clinical_session`. The legacy `perturb.py` remains byte-identical;
its four synthetic stress classes and all existing exp3 freezes remain intact.
The package entry file owns all new generator logic. No source files or
architecture directories were added, and no legacy integrity checks were bypassed.

The available cohort supports control versus pooled MCI/AD. It does not identify
four clinical severity stages. This mode therefore has two groups, explicitly
separate from the old four-way task; its accuracy is not an improvement on the
66.66% four-way benchmark. Existing exp3 training/inference continues using the
legacy generator unless a caller explicitly chooses the new API.

### What is fitted to clinical measurements

Read `data/clinical/ad_mci_writing/Participant_level.csv`, with optional explicit
clinical training participant IDs. There are **15 usable controls and 14 usable
impaired participants**, contributing 30 and 27 task rows. One of the original
30 participants has both pause-fraction measurements missing; both rows are
excluded with recorded reasons. Clinical donor IDs and the input-file SHA-256
are carried in the calibration object.

Each clinical task supplies mean fluent-burst duration F and pause-time fraction
p. Under an alternating fluent/pause renewal model, mean pause duration is
P = F p / (1 − p). Fit a bivariate hierarchical Gaussian in log F and log(P − 2),
preserving covariance rather than independently sampling the two measurements.
Participant means receive equal weight. Within-participant task residuals
estimate session covariance; estimated finite-session noise is subtracted from
the between-participant covariance, then negative eigenvalues are projected to
zero. The control covariance needs a projection (Frobenius change 0.05290),
showing the uncertainty in estimating these components from only two tasks.
Within-person task variation here is a proxy for session variability, not a
measurement of day-to-day progression.

A profile persists across a person's simulated sessions. Each session adds
within-person variation, then alternates fluent production and pauses above
**2 seconds**. Durations run across the complete input session, before any
window slicing. Mean durations are fitted; individual dwell-time shapes are
not available in the aggregate clinical data. Defaults use exponential fluent
durations and shifted-exponential pause durations. Optional gamma shapes are
exposed solely for sensitivity testing. The maximum-entropy assumption is
explicit and was not selected by classifier accuracy.

Both classes receive the same generation procedure. **Hold times and key
identities are unchanged in both classes.** Existing source gaps above two
seconds are replaced with samples from that source's fluent timing pool before
new planning pauses are generated. This is full pause resynthesis, not additive
noise on affected cases only. Timing identities are reconstructed, preserving
valid negative inter-key times from key overlap. There are no arbitrary
severity-specific motor changes, word multipliers, or correction multipliers.
Those relationships cannot be fitted from the participant aggregate file.

The API requires finite, unpadded whole sessions with nonnegative hold and
press-to-press durations. Inputs without positive fluent intervals are rejected.
It returns ordinary `[events, 5]` features; downstream classifiers need no
profile, group, generator pause mask, or clean query twin.

### Observable measurement convention

`measure_clinical_session` measures first-to-last-key time and pauses strictly
above two seconds. Mean production duration uses only complete interior bursts;
left/right edge durations are returned as censored observations. If there is no
complete burst, the mean is `None`, not a fabricated zero. Pause onsets per
minute are observable and approximate the source's production-burst rate;
finite-task counting and the unavailable raw Inputlog boundary convention
prevent claiming exact equality with the clinical task statistics.

Whole-session source timelines for the experiment were reconstructed from raw
Aalto timestamps, including sentence boundaries, rather than concatenating the
old cached windows. Events with impossible negative hold durations were filtered
before encoding, identically for both generated classes. Original long intervals,
which can contain task/UI waits, are resynthesized by the new generator. These
Aalto timelines are motor carriers, not real observations of cognitive decline
or reproductions of the clinical picture-description task.

### Synthetic train/validation/test assay

Use existing identity-disjoint Aalto roles, deterministically choosing the first
**1,000 train / 300 validation / 600 test** IDs sorted by SHA-256 of
`clinical:9192026:{role}:{uid}`. Generate both groups for each person. The new
calibration uses all 29 usable clinical participants for this **synthetic-only**
assay. Clinical cohort transfer is evaluated separately below with donor exclusion.

A fixed logistic regression (C=1, maximum 500 iterations) observes just two
measurements: log complete-burst duration and logit pause-time fraction. No
hyperparameter or checkpoint search occurs. This is a signal assay, not a new
production model architecture. For insufficient-burst observations, the assay
uses log observation time and a fixed −10 logit placeholder; the generator API
itself reports the missing measurement explicitly. There were 23 / 2 / 8 such
examples in train / validation / test.

- **Train: 78.15% accuracy**, AUROC 0.8619; participant-bootstrap 95% accuracy CI **[76.35%, 79.85%]**.
- **Validation: 79.33% accuracy**, AUROC 0.8837; participant-bootstrap 95% accuracy CI **[76.33%, 82.33%]**.
- **Test: 77.17% accuracy**, AUROC 0.8511; participant-bootstrap 95% accuracy CI **[74.83%, 79.42%]**.

Chance is 50% on this balanced binary task. The result demonstrates observable
signal in the fitted duration process, not four-stage classification or clinical
validation. Bootstrap intervals resample Aalto participants with both examples
kept together (1,500 resamples).

### Behavioral fit and finite observation effects

Short source timelines give generated test medians of fluent-burst duration
**8.11 s / 4.61 s** and pause-time fraction **31.67% / 54.17%** for control /
impaired. Their clinical participant medians are **9.89 s / 4.33 s** and
**36.35% / 57.58%**. Finite-session censoring and the fitted distribution both
contribute to this imperfect match.

A second behavioral check averages two longer, 1,501-event simulated sessions
within each of 500 new profiles per group, closer to the clinical task averaging:

- Controls: median burst **9.54 s** (IQR 7.70–11.83), pause fraction **33.21%**
  (IQR 26.14–41.34%), pause onsets **4.13/min**. Clinical medians are
  **9.89 s / 36.35% / 4.07 bursts per minute**.
- Impaired: median burst **4.82 s** (IQR 3.68–6.06), pause fraction **55.06%**
  (IQR 49.64–60.81%), pause onsets **5.56/min**. Clinical medians are
  **4.33 s / 57.58% / 5.99 bursts per minute**.

These longer runs use a neutral motor carrier resampled from one training
participant's fluent transitions, equally for both classes. They assess the
renewal process, not external clinical realism. Matching cohort summaries does
not establish correct linguistic context, individual pause tails, or disease
progression.

### Duration-shape sensitivity

Keep the classifier, profiles, means, and test source people fixed; change both
gamma dwell shapes from the default 1.0. No retraining or winner selection:

- Shape **0.5** (more variable durations): test **73.42%**, change **−3.75 points**,
  paired 95% CI **[−6.29, −1.17]**.
- Shape **1.0** (default): test **77.17%**.
- Shape **2.0** (less variable durations): test **81.33%**, change **+4.17 points**,
  paired 95% CI **[+1.87, +6.50]**.

The default remains 1.0. These results show material sensitivity to duration
shapes that the source aggregates do not identify; choosing 2.0 because its
accuracy is higher would be unsupported calibration.

### Exploratory transfer to real clinical summaries

Perform **29 leave-one-clinical-participant-out folds**. In each fold, remove
that person's task rows before fitting duration priors. Generate 200 profiles
per class, each with two 1,501-event sessions; fit the same fixed logistic
regression on observable synthetic summaries. Evaluate it on the excluded
person's actual task-averaged burst duration and pause-time fraction. A shared
neutral motor carrier contains no clinical labels. No held-out person is used
in the calibration or classifier fitting for their fold.

- **25/29 correct: 86.21% accuracy**, AUROC **0.9571**.
- Controls: **13/15 correct**; impaired: **12/14 correct**.
- Stratified participant-bootstrap accuracy interval: **72.41–96.55%**.

This is exploratory within-cohort transfer on a small, already-studied clinical
dataset, using aggregate outcomes directly related to the calibration targets.
It is not raw-keystroke clinical validation or an independent external cohort.
The bootstrap conditions on the fitted out-of-fold predictions and does not
capture all calibration/model-selection uncertainty. The encouraging result
supports a statistically meaningful signal in these measurements; it does not
validate a clinical diagnostic or severity-staging system.

### Verification and next use

Verified deterministic seeds, source non-mutation, unchanged hold/key channels,
physical timing identities, strict two-second threshold, censored/missing burst
handling, invalid-input rejection, and exclusion of clinical held-out records.
Changing an excluded participant's CSV measurements did not alter fitted means
or covariances. Legacy deterministic checks pass, frozen artifact/source hashes
match, and all four legacy feature vectors for a validation participant reproduce
the stored 1,485-feature cache exactly.

Use the new mode for explicitly labeled binary clinical-duration experiments.
For further calibration, obtain event-level cognitive typing data with matched
task boundaries to estimate dwell shapes, word context, and longitudinal changes.
The old four-class winner remains available unchanged. Scratch scripts and full
numerical output are under `/private/tmp/exp3_clinical_generator/`; this section
is the persistent report. No production classifier was replaced.


## Personal-baseline binary deviation experiment

**Superseded threshold policy:** this historical run selected its threshold from
validation negatives. The training-only revision below replaces that procedure.
These historical scores do not describe the corrected model.

This is the approved personal-enrollment task. It supersedes the earlier 77.17%
synthetic-versus-synthetic assay for the intended system; the tasks and numbers
are not directly comparable.

### Data, labels, and weighting

- Gallery: ten **untouched** Aalto windows from each user's earlier typing.
- Class 0: ten separate, **untouched** query windows from the later chronological
  half of that same user's sentences. No healthy-profile resynthesis is applied.
- Class 1: three perturbed copies of that query, one cognitive-only, one
  motor-only, and one combined. These are mechanisms, not severity labels.
- All original full-cohort roles are retained: **55,732 train / 1,202 validation /
  2,738 test participants**. Roles are identity-disjoint. No selection-score gate.
- Each person contributes one negative and three positives. Sample weights are
  **2 for the negative and 2/3 for each positive**, so each class receives half
  the total training weight, and positive mechanisms have equal weight.
  Training contains 222,928 rows. Metrics use the same equal class weighting.

The negative label means *no injected deviation*, not a clinical diagnosis of
health. Aalto is the motor/behavioral source; clinical data constrains the
positive interventions. No clinical patient labels are assigned to Aalto users.

The binary encoder reproduces the same **1,485 gallery-relative features** as
the frozen winner. Every untouched vector in each of train, validation, and test
was checked for exact equality against the existing frozen cache. Source files
are re-read; gallery/query arrays are checked for non-mutation. Class 0 data is
not passed through a generator. IDs, generator profiles, pause masks, and clean
query twins are not classifier inputs.

### Positive generator and explicit assumptions

`ClinicalDeviationGenerator` in the existing generator package is an additive,
positive-only adapter. It preserves existing long pauses instead of replacing
them. Cognitive-only perturbations preserve hold times; all mechanisms preserve
key identity and padding. Inter-key and release intervals are reconstructed to
satisfy the original timing identities.

The clinical calibration still uses the 29 usable control/impaired participants
from the local writing CSV. The fitted ratio of control to impaired mean fluent
duration on the log scale is **2.04658**, used as a relative pause-onset multiplier.
For a person's gallery, the baseline hazard is:

`(observed pauses > 2 seconds + 0.5) / observed fluent exposure in seconds`.

The extra event count has Poisson mean
`baseline hazard × (2.04658 − 1) × query fluent exposure`. Cognitive positives
condition on at least one injected event, since their label is an actual injected
intervention. Locations are sampled without replacement, weighted by the
original query's fluent interval duration. Added waits are two seconds plus a
gamma excess whose mean comes from one sampled impaired clinical profile and
session. The profile is shared across the ten windows. No missing time between
windows is invented. The default gamma shape is 1 (shifted exponential).

**Assumptions, not measured clinical mappings:** transfer of the clinical
relative rate to short Aalto windows; the half-event Jeffreys smoothing prior;
the dwell distribution shape; and conditioning positives on an observed
intervention. Conditioning changes the realized count distribution from the
unconditioned clinical-rate model. This is not a simulation of every clinically
impaired person or a fit of absolute clinical task statistics to Aalto.

In test, both cognitive and combined examples have a median of **one** injected
pause (IQR **1–2**). The largest counts are **99** and **102**, respectively,
reflecting extreme personal hazard/exposure estimates. The unconditioned count
mean has median **0.61** (IQR **0.53–1.71**), so the positive-event condition is
material. These tails and the weak-enrollment smoothing remain modeling
limitations; no thresholds were changed in response to their test performance.

Motor perturbations reuse the existing Parkinson's-informed hold/press timing
priors, with the legacy pause component disabled. The three former stress
amplitudes are sampled equally, as uncertainty/stress variation within the
positive class—not severity outputs. Existing long source pauses are not
shortened by the motor branch. The combined branch applies motor changes and
then the clinical cognitive additions. Ordinary negative examples remain raw.

### Model and threshold policy

Use the frozen winner's HistGradientBoostingClassifier settings unchanged:
**15 leaves, depth 4, 200 samples per leaf, L2 50, learning rate 0.04, 600 rounds**,
with early stopping disabled. No architecture change or hyperparameter search.
Synthetic draw seed is 9202026; the classifier retains its original seed 9172026.
The same cohort and frozen feature scale floor are used throughout.

After fitting, select the lowest threshold above the
`floor(0.05 × validation_negative_count) + 1`th largest validation-negative score.
This handles ties conservatively and guarantees at most 5% empirical validation
false positives. Lock the threshold before computing any test scores. The locked
threshold is **0.760856289590732**, with alerts defined by `score >= threshold`.
Scores are not calibrated probabilities of clinical impairment.

### Full-cohort results

**Primary results use the locked alert threshold. Balanced accuracy gives the
untouched and perturbed classes equal weight**, despite storing three positive
rows for every negative. Sensitivity averages the three mechanisms equally.

- **Train: 85.81% balanced accuracy** (95% CI **[85.69, 85.92]**), false positives **2.76%**, sensitivity **74.37%**.
- **Validation: 85.36% balanced accuracy** (95% CI **[84.54, 86.22]**), false positives **4.99%**, sensitivity **75.71%**.
- **Test: 85.42% balanced accuracy** (95% CI **[84.87, 85.98]**), false positives **3.69%**, sensitivity **74.53%**.

Test sensitivity by injected mechanism:

- **Cognitive: 70.01%** (95% CI **[68.30, 71.73]**).
- **Motor: 61.18%** (95% CI **[59.39, 62.82]**).
- **Combined: 92.40%** (95% CI **[91.38, 93.39]**).

Test AUROC is **0.9468**. Test false-positive rate is **3.69%**
(95% CI **[2.99, 4.46]**), representing 101 flagged untouched
queries out of 2,738. This is the cohort rate, not a guarantee for each user.

For completeness, at the conventional 0.5 threshold, balanced accuracy is
**88.56% train / 87.13% validation / 87.90% test**, but test false positives rise
to **10.30%**. That threshold is not the chosen alert policy.

Intervals use 1,000 participant bootstrap resamples, preserving each person's
negative and three positive queries. The validation interval conditions on the
chosen threshold. These are benchmark results on unmodified versus injected
Aalto deviations, not clinical diagnostic accuracy. The historical test cohort
has been inspected in earlier experiments and is not an external untouched test.

### Repeatability and generator sensitivity

The first 200 test participants were specified for the robustness check. The
original draw reproduces the full-run scores exactly. Redraw all positive
profiles/noise with two independent seeds, preserving each person's gallery
and untouched query. Across three draws, **39.5% of positive cases change their
alert decision at least once**. Negative decisions change **0%**. These are
new interventions on the same raw query, not longitudinal patient observations
or repeated noise draws with a fixed impairment profile.

Balanced accuracy at the locked threshold on that same 200-person subset:

- Original draw: **85.42%**, sensitivity **74.83%**.
- Independent draw 1: **87.33%**, sensitivity **78.67%**.
- Independent draw 2: **85.08%**, sensitivity **74.17%**.
- Pause shape 0.5: **83.17%**, sensitivity **70.33%**.
- Pause shape 2: **86.58%**, sensitivity **77.17%**.
- Half added-pause hazard: **82.42%**, sensitivity **68.83%**.
- Double added-pause hazard: **87.33%**, sensitivity **78.67%**.

All variants have the same **4.00%** untouched-query flag rate in this subset.
No variant is promoted or used to retune the classifier/threshold. This small
subset shows sensitivity to unidentifiable generator assumptions and positive
realizations; the result does not justify claiming stable per-user detection.
Motor-only sensitivity is the weakest of the three mechanisms at the chosen
false-positive constraint.

### Artifacts, compatibility, and verification

The runnable model and clinical calibration are saved in
`runs_med/binary_deviation.joblib`; full numerical results and source hashes
are in `runs_med/binary_deviation.json`. Scratch feature caches and execution
log are under `/private/tmp/exp3_binary/`. No source files or project directories
were added. Existing model artifacts and manifests were not overwritten.

`model.py` now contains the original model source verbatim as a readable
`_LEGACY_SOURCE` archive. Its original protocol hash is checked before execution
in an isolated namespace. Legacy source checks refer to that verified archive;
all other legacy artifact checks remain active. The three architecture files
normalize only their explicitly marked compatibility adapter when checking
historical source identity. Their model computation is unchanged. New binary
artifacts hash the full current `model.py`, generator package, legacy generator,
and clinical CSV, and verify the frozen parent model/protocol before prediction.

Verification passed for all three positive mechanisms: deterministic draws,
untouched input preservation, key/padding preservation, existing-pause
preservation, and physical timing identities. Modified archived source is
rejected. Legacy four-class prediction and all three saved architecture
prediction commands still work. Binary prediction requires both gallery and
query, uses the locked threshold, and does not perturb either inference input.

## Binary model leakage, hindsight, and perturbation audit

Audit performed after the personal-baseline binary experiment, on the unchanged
saved model and threshold. This is an exploratory audit, not a new independent
confirmation experiment. No model, generator, feature, or threshold was changed.

**Verdict:** the audited data passes the checked identity, exact-window,
chronology, and training-only normalization tests. Freedom from hindsight bias
cannot be certified because the historical test cohort has repeatedly been
examined. Perturbations have real stochastic and strength variation, but their
clinical coverage is incomplete and subtle deviations are substantially harder
than the headline average suggests. The 85.42% test balanced accuracy remains a
result for the specified synthetic mixture, not a demonstrated real-world rate.

### Leakage and numerical reproduction: passed checks

- Reopened the frozen binary artifact and verified its current source contract,
  parent model, and protocol. Recomputed train/validation/test scores from the
  stored feature caches; all reported operating metrics and participant
  bootstrap intervals reproduced exactly.
- All five protocol roles have zero pairwise participant-ID overlap. The binary
  fit uses only training examples, with no early stopping or test-dependent
  checkpoint selection. Related untouched/perturbed examples remain grouped
  under their source participant.
- Independently re-read **all 59,672 train/validation/test source files** and
  reconstructed **1,193,440 selected windows**. No source hash differed from the
  original availability manifest or binary cache. No exact selected window was
  duplicated within or across these participants. `ordered_blocks`' raw-time
  check passed for every participant: all baseline releases precede all query
  presses across the sentence-half boundary.
- Recomputed the 57-dimensional normalization floor from the **55,732 training
  participants' galleries only**. Maximum absolute difference from the saved
  floor was **0.0**. Personal key centers, percentiles and comparison references
  are estimated from that person's gallery, not their later query.
- Encoder inspection confirmed that labels, participant IDs, generator masks,
  latent profiles and clean query twins are not classifier inputs. Shared
  baseline use across a person's examples is intended enrollment, with that
  person's entire group confined to one role.
- Regenerated the original examples for 300 test participants and reproduced
  their cached prediction scores exactly. Checked input non-mutation, preserved
  key identity and padding, reconstructed timing identities, and unchanged
  cognitive-only hold durations for all three mechanisms.
- A supplementary shuffled-label tree fit using 4,000 training participants and
  all 2,738 test participants produced **AUROC 0.5053** against independently
  shuffled test labels. This single null fit retained 3:1 row prevalence; its
  balanced accuracy was 50%. It is a sanity check, not proof of universal absence
  of leakage, and is not a candidate model.

These checks exclude exact selected-window duplicates, not every possible
near-duplicate, shared typing prompt or identity alias. They establish the
current files' consistency; source hashes alone cannot establish causal
independence or clinical validity.

### Hindsight and validation: unresolved limitations

The binary study locks its threshold before scoring test, and this was verified
by code inspection and exact reconstruction from validation-negative scores.
However, earlier architecture/feature/generator investigations have used this
historical test cohort. Its current score cannot be treated as a pristine,
independent estimate after that history. New audit challenges also use it and
must not be used to choose fixes and then called independent confirmation.

Validation explicitly **fits the alert threshold**. Its 4.99% FPR is therefore
an empirical constraint achieved on threshold-fitting data, not independent
verification of a population FPR below 5%. The published validation bootstrap
holds that fitted threshold fixed and does not incorporate its estimation
uncertainty. Test bootstraps likewise condition on this fitted model, generator
and threshold and omit uncertainty from prior research choices and clinical
calibration.

The same 29 clinical participants calibrate the simulator used for every Aalto
role. This is not direct reuse of held-out Aalto labels. It means the experiment
holds out typing carriers, **not clinical source populations or generator
families**. It cannot validate transfer to unseen clinical patients. A future
clinical evaluation must exclude evaluated clinical participants from generator
calibration too.

### Perturbation diversity: present, with important omissions

Three mechanisms are present: cognitive-only, motor-only and combined. Motor
stress factors are 0.25, 0.55 and 1.0, with profile, session and event noise and
random speed-matching. These are simulation strengths, not validated severity
stages.

A controlled check generated 500 motor draws at each strength using the same
source window. Mean within-window variance of the **log hold-time multiplier**
was **0.00702 / 0.03202 / 0.10639** across increasing strengths. Between-draw
variance of the mean log multiplier was **0.00113 / 0.00431 / 0.01313**. Thus both
noise variance and overall effect vary; the generator does not simply add one
constant shift. The numeric motor priors, fixed AR correlation 0.65 and equal
mixture weights are not clinically fitted severity trajectories or prevalence.

Cognitive pauses also vary in location, count and duration. Ten thousand draws
from the fitted impaired profile/session hierarchy gave mean-pause-duration
5th/50th/95th percentiles of **3.86 / 5.83 / 9.87 seconds**. However:

- Every cognitive-positive example is conditioned on at least one injected
  event, and every added event exceeds two seconds. **56.06%** of the full test
  cognitive examples contain exactly one added pause. Averaging the
  unconditioned Poisson zero probabilities across test gives **40.63%**; the
  positive conditioning materially changes this distribution. Removing that
  conditioning would require an explicit label definition for windows without
  observable intervention, not automatically labelling them changed.
- The clinical onset-rate ratio is the same **2.04658** for every profile. The
  sampled fluent-duration coordinate is discarded by the deviation adapter;
  therefore its fitted relationship with pause duration does not jointly drive
  individual pause rates. Counts vary through personal baseline exposure and
  random sampling, not a sampled clinical relative-rate profile.
- All default pause excesses use the same gamma shape. Different fitted means
  produce different variances, but do not validate the distributional shape,
  temporal clustering, linguistic location or a progression process. Cognitive
  locations are weighted by source interval duration, without an explicit
  linguistic model. Each batch resamples profiles; longitudinal persistence
  across successive batches has not been modeled or validated.
- Baseline smoothing and transfer to short Aalto windows are assumptions.
  Extreme test injections reach 99 cognitive / 102 combined pauses. In the
  300-person challenge, the largest single added cognitive delay was **74.40 s**.
  These tails require independent plausibility checks rather than adjustments
  chosen for classification accuracy.

Leaving one clinical participant out at a time changes the fitted relative-rate
ratio to **1.92282–2.14860**. This checks sensitivity of that scalar only; it does
not establish full generator robustness or independence from this small cohort.

The [clinical writing study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9311409/)
measured two picture-description tasks in 30 participants and reports linguistic
context effects. It supports studying pause/burst behavior, not the exact
additive model used here. The [neuroQWERTY dataset](https://physionet.org/content/nqmitcsxpd/1.0.0/)
supports investigation of Parkinson's motor timing. Neither source alone
validates this mixed generator as longitudinal cognitive deterioration.

### Detection by strength and diagnostic challenges

Using all 2,738 test participants and the unchanged operating threshold:

- Motor-only sensitivity at weak / middle / strong stress was **22.11% / 69.21%
  / 93.50%**, with **927 / 919 / 892** examples respectively.
- Combined sensitivity at those stresses was **84.99% / 93.92% / 98.32%**, with
  **906 / 938 / 894** examples. Added cognitive pauses can mask weak motor
  detection in this aggregate.
- Cognitive sensitivity with exactly one injected pause was **59.74%**, versus
  **83.13%** with multiple pauses.

Additional frozen-model challenges used the first 300 test participants. These
are synthetic diagnostics, not newly calibrated clinical conditions:

- Original cognitive sensitivity: **69.67%**.
- Same injected sites with every cognitive delay multiplied by 0.5: **36.00%**;
  by 0.25: **15.67%**; by 0.1: **4.33%**. Untouched flag rate: **3.67%**.
- Rounding perturbed hold/press timings back to millisecond resolution gave
  unchanged aggregate flag rates for cognitive (**69.67%**), motor (**59.33%**)
  and combined (**93.67%**) conditions. This does not support a gross precision
  artifact explanation on this sample; it is not an exhaustive shortcut test.
- Adding one five-second wait at the first query transition, without any motor
  change, produced a **52.67% flag rate**. This is a simulated interruption
  challenge, not a measured real-world false-positive rate. It shows that a
  brief interruption alone can trigger the deviation detector. Such a flag is
  compatible with generic deviation detection, but cannot distinguish a
  transient distraction from a clinically meaningful persistent change.

### Deployment boundary and next confirmation requirements

The offline loader checks chronology, but the public prediction arrays contain
no participant IDs or timestamps. The caller must enforce same-person baseline
ownership, earlier enrollment, disjoint windows and fixed window construction.
The study takes the last ten windows from the earlier sentence half, not
necessarily the first ten windows a new application would collect. Initial
onboarding and repeated future batches therefore still need an end-to-end
prospective check. Untouched Aalto observations also have no clinical healthy
labels, so no-injection is the only justified class-0 interpretation.

Before claiming the requested protections are established:

1. Freeze model, generator, feature construction and acceptance criteria before
   opening a genuinely new confirmation cohort; use development data for fixes.
2. Evaluate weak changes, alternate perturbation families and clinically held-out
   calibration sources separately. Preserve raw normal-class keystrokes.
3. Evaluate fixed-baseline, repeated-batch behavior and transient interruptions
   on real longitudinal data. Decide persistence/escalation separately from the
   per-batch score.

Scratch evidence and runnable audit scripts are under
`/private/tmp/exp3_binary_audit/`: `audit.py`, `variance.py`, execution log and
JSON results. Commands were `python3 /private/tmp/exp3_binary_audit/audit.py` and
`python3 /private/tmp/exp3_binary_audit/variance.py`; both completed successfully.
No new exp3 files or architectures were introduced by this audit.

## Training-only calibration and automatic generation checks

This revision addresses the prohibition on learning parameters or thresholds
from validation/test observations. The earlier binary threshold policy is
superseded. The original clinical generator and raw class-0 data are unchanged;
the pipeline, data ownership and recurring development checks are corrected.

### Data ownership and execution proof

A deterministic hash ordering of the original 55,732 training IDs, fixed before
scores were computed, assigns **39,012 model-fit / 8,360 development / 8,360
threshold-calibration participants**. The roles are disjoint. No original
selection, calibration, validation or test participant enters these subsets.
The external clinical summary table remains simulator-training evidence; this
run does not claim an independent clinical evaluation.

The normalization floor is recomputed from model-fit galleries only. The tree
is fitted on model-fit examples only, with the existing hyperparameters and
balanced mechanism weights. The threshold is fitted on untouched negatives from
the training-calibration subset only. Development examples contribute neither
to this model fit nor to threshold calibration. Future changes may use this
training-side development pool, never final validation/test observations.

The complete run used file-access guards in the parent and worker processes
that reject all original nontraining raw participant files and validation/test
feature caches. Explicit attempted reads confirmed the guards reject these
paths. The full run completed with `GUARD_PASS`. Protocol-ID metadata is read to
verify separation; heldout observations and scores are not used. No real
validation/test evaluation was executed for this revision.

Regression checks passed for partition disjointness, invariance to replacement
of heldout IDs with other disjoint IDs, threshold handling of ties and very
small samples, rejection of nonfinite threshold input, and preservation of the
archived four-class source hash. A synthetic-fixture test of `binary_evaluate`
confirmed it calls prediction only, does not fit a threshold or write a model,
and refuses to overwrite an existing evaluation report. Fixture metrics are
unit-test outputs, not experimental model performance.

### Corrected artifact and training-side results

The protocol was saved before feature loading and fitting. The new artifact is
`runs_med/binary_deviation_trainonly.joblib`; its `.protocol.json` records the
partition/design and its `.json` records results and generation checks. The
old binary artifact was not overwritten. Because its full-source hash predates
this correction, the current predictor deliberately rejects it. The original
four-class winner continues using its verified archived source.

The new threshold is **0.7060234079421164**, with alerts at score >= threshold.
Its target remains at most 5% empirical false positives on training-calibration
negatives. This target is a fixed policy, not learned from heldout performance.

- Model-fit: **87.12% balanced accuracy**, **3.56% FPR**, **77.79% sensitivity**.
- Training-calibration: **86.58% balanced accuracy**, **5.00% FPR**, **78.17%
  sensitivity**. This is threshold-fitting data, not independent confirmation.
- Training-development: **86.22% balanced accuracy**, **5.33% FPR**, **77.78%
  sensitivity**, **0.9447 AUROC** across all 8,360 development participants.
- Validation/test: **not evaluated** in this revision. Historical scores cannot
  be reused as the new model's performance or as pristine confirmation.

Public prediction passed on a fixture built exclusively from a model-fit
participant: score 0.23270214098416656, below the locked threshold. All score and
source contracts remained intact after the run.

### Automatic checks on every study

`binary_study` now always runs a fixed development suite on 300 training-side
participants, using three independent draws plus pause gamma shapes 0.5 and 2.
For every mechanism it evaluates injected effect multipliers 0.125, 0.25, 0.5
and 1.0. Multipliers attenuate the original timing deltas for diagnosis of
coverage; they are not clinically calibrated severity levels and do not change
the production generator or training labels.

Every check verifies unchanged raw inputs, key identities, padding, timing
identities, finite feature arrays and unchanged cognitive-only holds. Reports
include sensitivity, balanced accuracy, AUROC, motor log-change variance,
press-change RMS quantiles, extreme delays, and a one-split decision tree using
eight absolute/gallery-relative mean/std timing descriptors as a limited
shortcut probe. The stump is trained exclusively on model-fit data. It is one
aggregate diagnostic, not an exhaustive search for trivial predictors.

Each attenuated condition is compared with the corresponding full-strength
condition using 1,000 paired participant bootstraps. A 95% lower bound on the
sensitivity loss above ten percentage points flags a material coverage failure.
This predeclared diagnostic threshold is not a clinical acceptance standard.
Intervals are exploratory and not corrected for multiple comparisons; the
five scenarios also share participants and are not independent cohorts.

### Outcome: weak-change coverage fails

**All 45 attenuated mechanism/scenario combinations were flagged.** The saved
status is `insufficient_weak_change_coverage`, with `clinically_validated=false`.
Completion of fitting does not mean passing this generation/detection review.

Across the three default-shape draws:

- Cognitive sensitivity: **73.33–77.33%** at full strength, **39.00–42.33%** at
  half, **14.33–19.00%** at quarter, **6.67–8.67%** at one-eighth.
- Motor sensitivity: **62.00–65.00%** at full strength, **37.00–40.00%** at half,
  **18.67–21.00%** at quarter, **9.33–10.67%** at one-eighth.
- Combined sensitivity: **93.33–95.00%** at full strength, **65.67–70.33%** at
  half, **31.33–35.00%** at quarter, **12.67–14.33%** at one-eighth.

The unchanged negative batches have **6.33% FPR** in this 300-person subset;
the full development FPR is 5.33%. On draw zero, one-eighth-strength cognitive
AUROC is **0.5362**, compared with **0.9497** at full strength. That low ranking
performance is not explained solely by the choice of alert threshold. It does
not establish an irreducible Bayes limit: training exposure, representation and
natural overlap remain possible contributors.

Motor log-change variance on draw zero increases from **0.000984 / 0.003629 /
0.012899 / 0.046019** across increasing multipliers, confirming the challenge
suite contains different effect variances. The limited stump's full-strength
AUROCs were 0.5000 cognitive, 0.6483 motor and 0.6333 combined, versus the full
model's 0.9497, 0.8886 and 0.9881. Because one shared stump selects only one
statistic, these results cannot rule out other simple cognitive/pause shortcuts.

The result supports the user's concern about incomplete weak-effect coverage.
It does not justify making positives stronger, changing heldout thresholds,
claiming all small perturbations are clinically meaningful, or claiming the
model has covered all possible feature variation. The next model/generator
iteration must address weak-effect training coverage and realistic heterogeneity
using training-side evidence and then rerun this same suite. Clinical calibration
of the weaker effects and longitudinal behavior remains unresolved.

Scratch reproduction scripts, tests, fitting/development feature caches and
execution logs are under `/private/tmp/exp3_binary_trainonly/`. The source
implementation remains in the existing `model.py`; no new architecture or
source file was added under exp3. The README documents the separated `study`,
`evaluate`, and `predict` commands.


## Training coverage versus representation: matched four-way experiment

### Question and controlled design

Test whether weak-effect performance is limited by training exposure, by the
current representation, or both. Use the existing HistGradientBoostingClassifier
with exactly the same 15 leaves, depth 4, 200 samples/leaf, L2 50, learning rate
0.04, 600 iterations, no early stopping and random seed as the corrected model.
No new architecture or source file was added under exp3.

All participants belong to the **original training pool**: 39,012 model-fit,
8,360 threshold-calibration, and 1,500 training-development participants. The
primary development group is indices 300:1800 of the established development
partition; its first 500 participants receive two additional independent draws.
These are development comparisons, not final validation or a new clinical
cohort. Original nontraining participant files and validation/test caches were
blocked by file-access guards throughout the run. The guard rejection tests and
complete guarded run passed. No real validation/test evaluation was performed.

A protocol with participant IDs, feature dimensions, model settings, strengths,
and the experiment source hash was written before generating comparison data.
Four conditions form a two-by-two comparison: current versus richer features,
and original versus broad training strength coverage. The original current-
feature classifier is reused; its complete training feature matrix and training
metrics reproduce exactly. The three other models use the identical fitting
participants and **156,048 rows each**, retaining one untouched negative and
one positive for each of cognitive, motor and combined mechanisms per person.
Negative sample weight is 2; each positive weight is 2/3.

The original curriculum uses multiplier 1 on the existing generated effects.
This does not mean a maximum clinical severity: the original generator already
samples motor stresses and clinical profiles. The broad curriculum assigns
multipliers **0.125, 0.25, 0.5 and 1.0**, with exactly **9,753 participants at
each multiplier for each mechanism**. Both curricula use identical underlying
perturbation draws. Attenuation scales timing changes only, preserving event
locations and counts. These multipliers are diagnostic augmentation strengths,
not clinically calibrated severity labels. Class-0 queries and galleries remain
untouched. No positive multiplier is supplied to the learner.

Each model's threshold is set exclusively from the same 8,360 training-
calibration negatives, targeting 5% empirical FPR. All four achieved exactly
5.00% on those calibration negatives. All thresholds were locked before the
comparison's development scores were computed. Model-fit gallery normalization
is inherited unchanged; the additional features use the person's own gallery
and fixed definitions, not population parameters fitted on development data.

### Additional representation

Append **426 features**, increasing the input from **1,485 to 1,911**. For each
of hold time, press-to-press time and key-adjusted hold, the 142 new values are:

- Differences at 23 pooled timing quantiles, scaled by gallery IQR with a fixed
  0.05 log-unit floor, and empirical CDF differences at those gallery quantiles.
- Three signed/absolute maximum empirical-CDF gaps and a normalized mean
  absolute quantile difference.
- Changes in 5-by-5 timing-rank transition frequencies at lags one and two.
- Changes in high/low rank exceedance rates, adjacent exceedances and run lengths.
- Changes in local mean-rank distributions for blocks of 4, 8 and 16 events.

Ranks are defined using gallery observations. Timing channels retain the
existing millisecond quantization. Temporal calculations never join independent
windows. These features use observed query data and the gallery, without clean
query twins, latent profiles or generator masks. They are still summaries;
this experiment does not test a lossless raw-sequence learner or isolate pooled
statistics from the temporal portion of the added feature block.

Checks passed for zero added features on identical gallery/query data,
deterministic reproduction, and a temporal permutation that changes the
sequence features while preserving pooled-distribution features. Every generated
case was checked for unchanged raw inputs, keys, padding and timing identities.
All 39,012 original-strength fitting vectors matched the existing cached
1,485-dimensional vectors exactly.

### Primary development results

All four models evaluate the same untouched query plus all 12 mechanism/strength
combinations for each of the 1,500 people. Balanced accuracy gives the negative
class half the weight and each of the 12 positive conditions equal weight.
**Weak sensitivity** below averages the 0.125 and 0.25 multipliers across all
three mechanisms. It is sensitivity, not overall classification accuracy.

- **Current features, original curriculum:** weak sensitivity **15.44%**, FPR **4.53%**, balanced accuracy **67.59%**. Threshold 0.706023407942116.
- **Current features, broad curriculum:** weak sensitivity **21.11%**, FPR **5.20%**, balanced accuracy **68.81%**. Threshold 0.713181076423329.
- **Richer features, original curriculum:** weak sensitivity **19.80%**, FPR **4.27%**, balanced accuracy **69.67%**. Threshold 0.701388368459766.
- **Richer features, broad curriculum:** weak sensitivity **31.07%**, FPR **4.80%**, balanced accuracy **72.82%**. Threshold 0.701098247237438.

The 67–73% balanced accuracies above concern this deliberately harder, expanded
strength mixture. They must not be compared directly with the earlier 86%
development score on the original generator mixture or described as validation/
test accuracy.

Prespecified paired contrasts use 1,000 participant bootstrap resamples, keeping
all conditions and predictions together within each person:

- Current features, broad curriculum versus Current features, original curriculum: weak sensitivity **+5.67 percentage points**, 95% CI **[+4.73, +6.46]**; FPR difference +0.67 points, CI [-0.20, +1.53].
- Richer features, broad curriculum versus Current features, broad curriculum: weak sensitivity **+9.96 percentage points**, 95% CI **[+8.87, +10.98]**; FPR difference -0.40 points, CI [-1.20, +0.47].
- Richer features, original curriculum versus Current features, original curriculum: weak sensitivity **+4.36 percentage points**, 95% CI **[+3.59, +5.07]**; FPR difference -0.27 points, CI [-0.80, +0.20].
- Richer features, broad curriculum versus Richer features, original curriculum: weak sensitivity **+11.27 percentage points**, 95% CI **[+10.28, +12.38]**; FPR difference +0.53 points, CI [-0.33, +1.40].

Thus broader training helps with the current representation, and the added
features provide a further substantial gain under broad training. The latter
gain is accompanied by a slightly lower development FPR, not a higher one.
The intervals condition on the fitted models and thresholds; they do not include
refitting, threshold-estimation or clinical-calibration uncertainty, and are
not adjusted for multiple comparisons.

### Mechanism breakdown: strengths 0.125 / 0.25 / 0.5 / 1.0

Numbers below are detection sensitivities on the primary development cohort.

**Current features, original curriculum**

- Cognitive: 6.93% / 16.27% / 42.93% / 75.67%.
- Motor: 8.93% / 18.60% / 37.60% / 64.93%.
- Combined: 10.60% / 31.33% / 68.73% / 94.00%.

**Current features, broad curriculum**

- Cognitive: 8.93% / 20.20% / 43.73% / 68.47%.
- Motor: 13.40% / 23.13% / 40.60% / 65.93%.
- Combined: 18.33% / 42.67% / 74.93% / 93.47%.

**Richer features, original curriculum**

- Cognitive: 6.53% / 16.07% / 42.40% / 74.60%.
- Motor: 14.00% / 25.73% / 45.33% / 70.87%.
- Combined: 16.20% / 40.27% / 75.67% / 95.53%.

**Richer features, broad curriculum**

- Cognitive: 8.00% / 18.40% / 41.47% / 66.40%.
- Motor: 28.67% / 40.60% / 55.80% / 76.20%.
- Combined: 33.47% / 57.27% / 83.07% / 96.00%.


**The representation gain is concentrated in motor and combined changes.**
At quarter strength, motor sensitivity rises from 18.60% in the original model
to 40.60% with broad training and richer features. Under broad training, the
extra features increase quarter-strength motor AUROC from 0.7146 to 0.8132,
providing evidence of improved ranking as well as operating-point sensitivity.

Cognitive-only performance is not solved: quarter-strength sensitivity with
broad/richer training is 18.40%, versus 20.20% with broad/current features.
Original-strength cognitive sensitivity falls from 75.67% in the original model
to 66.40% with broad/richer training. This is a real trade-off hidden by the
pooled improvement. More of the same generic timing features cannot be assumed
to resolve it.

### Repeated perturbation draws

Two additional draws on the same first 500 development participants keep all
models and thresholds frozen. The negative predictions reproduce exactly across
draws; positive profiles and noise are resampled. These are repeated simulator
draws, not independent participant cohorts or longitudinal clinical validation.

**Repeat 1**

- Current features, original curriculum: weak sensitivity 16.07%, FPR 5.80%, balanced accuracy 67.13%.
- Current features, broad curriculum: weak sensitivity 21.73%, FPR 6.20%, balanced accuracy 68.45%.
- Richer features, original curriculum: weak sensitivity 19.77%, FPR 4.80%, balanced accuracy 69.34%.
- Richer features, broad curriculum: weak sensitivity 30.67%, FPR 5.80%, balanced accuracy 72.29%.
- Incremental representation gain under broad training: **+8.93 points**, 95% CI **[+7.13, +10.93]**.

**Repeat 2**

- Current features, original curriculum: weak sensitivity 17.30%, FPR 5.80%, balanced accuracy 67.40%.
- Current features, broad curriculum: weak sensitivity 22.80%, FPR 6.20%, balanced accuracy 69.01%.
- Richer features, original curriculum: weak sensitivity 20.90%, FPR 4.80%, balanced accuracy 69.58%.
- Richer features, broad curriculum: weak sensitivity 32.17%, FPR 5.80%, balanced accuracy 72.44%.
- Incremental representation gain under broad training: **+9.37 points**, 95% CI **[+7.60, +11.33]**.


The repeated draws support the same direction of effect; they do not establish
robustness to different generator families or real clinical changes.

### Training metrics and remaining limits

Training balanced accuracy at each model's own calibration-derived threshold:

- Current features, original curriculum: 87.12%.
- Current features, broad curriculum: 69.34%.
- Richer features, original curriculum: 88.52%.
- Richer features, broad curriculum: 73.26%.

Training scores across curricula concern different effect mixtures. Broad
training deliberately has harder positive examples, so a lower training score
is not itself evidence of a worse detector. The shared development comparisons
above provide the controlled assessment.

The fitting cohort's cognitive injections have median one pause (IQR 1–2);
56.33% contain exactly one. Attenuating a sparse pause is not equivalent to
simulating a persistent, clinically measured pattern of small changes. The
experiment establishes that training exposure and this representation both
contributed to the observed weakness, especially for motor changes. It does
not establish a Bayes limit, validate the clinical realism of weaker injections,
or demonstrate adequate weak-change detection: even the best pooled condition
misses roughly 69% of the defined weak positives.

Cognitive-only behavior is now the main unresolved mechanism in this comparison.
Its representation and generation assumptions require separate investigation;
the aggregate gain does not justify declaring that mechanism solved. The
existing default model was not replaced by this research candidate.

### Artifacts and reproducibility

All four classifiers, frozen thresholds, scores, source hashes, verification
flags and the exact experiment script text are archived in
`runs_med/binary_representation_probe.joblib`. The adjacent `.json` contains
numerical results; `.protocol.json` contains the design fixed before generation.
The existing model and generator source contracts and the parent checkpoint
hash remain unchanged. Reloaded classifiers reproduced regenerated development
case predictions exactly, and baseline training metrics reproduced exactly.

Scratch scripts/caches are in `/private/tmp/exp3_representation_probe/`.
Commands: `python3 /private/tmp/exp3_representation_probe/probe.py`, followed by
`verify.py` in the same scratch directory. The full run completed with
`GUARD_PASS COMPLETE`. No new source file or directory was added under exp3.

## Severity assessment, correction attempts and frozen final evaluation

**Reporting correction:** the headline validation/test figures in this historical
section omit the additional 0.125/0.25/0.5 attenuation used in broad training.
They describe a narrower, easier mixture and must not be paired with the 73.26%
broad-training score as a matched-distribution comparison. The full-distribution
replacement is recorded separately below. Historical figures are retained for
auditability, not as the current headline assessment.

### Scope and outcome

All six planned stages were executed: subgroup diagnosis, training-only
correction, independent training-development comparison, fair mixed assessment,
checkpoint/threshold freeze, and final validation/test evaluation. The selected
model remains the preceding **1,911-feature richer/broad tree**. Correction
attempts did not improve the weakest-group selection criterion. Completing the
experiment does not mean that severity-dependent sensitivity has been solved.

The classifier still predicts untouched versus perturbed queries. Enrollment is
ten untouched earlier Aalto windows; classification uses ten later windows.
Class-0 queries and galleries are not resynthesized. Severity denotes generator
settings, not validated clinical stages or classifier output labels.

### 1. Diagnose levels using training participants

The existing fit cohort contains 39,012 people. Threshold calibration uses 8,360
separate training people. From the remaining training-development pool, 1,500
people were assigned to diagnosis and a disjoint 1,500 to candidate selection.
No validation/test people were available to this development stage.

The current clinical deviation generator has an unstratified cognitive component
and three motor stress settings. Seven positive groups were assessed: cognitive;
motor mild/moderate/severe; combined mild/moderate/severe. The cognitive component
was not assigned invented severity labels. The motor settings use the original
0.25/0.55/1 factors. A random-number wrapper consumes the usual stress draw and
substitutes the requested setting, preserving other draws and the generator
implementation. Each person's underlying draws are matched across stress levels.

At the existing training-calibrated threshold, diagnosis sensitivity was:

- Cognitive: 66.27%.
- Motor mild / moderate / severe: 50.40% / 80.40% / 96.07%.
- Combined mild / moderate / severe: 90.80% / 96.80% / 99.33%.
- Untouched false-positive rate: 3.93%; weighted balanced accuracy: 87.62%.

The ranking is mild, moderate, severe from weakest to strongest. This is an
operating-point sensitivity gap; subgroup counts alone do not explain it.

### 2. Attempt corrections without changing the generator

One tree was refitted with inverse diagnosis-sensitivity loss weights (sensitivity
floor 0.1, multiplier cap 3), normalized to keep positive and negative class mass
equal. Cognitive receives one third of positive assessment mass; each motor or
combined subgroup receives one ninth before reweighting. This gives equal
mechanism mass and equal levels within the mechanisms that have levels.

The training examples, broad strength curriculum (0.125/0.25/0.5/1), 1,911
features, and tree hyperparameters were otherwise unchanged: 15 leaves, depth 4,
200 samples per leaf, L2 50, learning rate 0.04, 600 rounds, no early stopping.
A second correction candidate was a fixed 50/50 score blend of the reweighted
model and the existing richer-feature original-curriculum model. No new
architecture or production source file was introduced.

Each candidate's global threshold was fitted only to the same 8,360 training
calibration negatives, yielding empirical FPR 5%: baseline 0.7010982472374381;
reweighted 0.7047519015120242; blend 0.6881907968224235.

### 3. Compare corrections on separate training-development people

The declared objective was maximum minimum sensitivity over the seven groups.
Eligibility also required, relative to baseline, no more than 1 percentage point
loss in weighted balanced accuracy, 2 points loss in weak-change sensitivity, or
1 point increase in FPR. Weak challenges use 0.125/0.25 attenuation. Ties use
balanced accuracy, then favor baseline.

On the separate 1,500-person selection cohort:

- Baseline: balanced accuracy 86.70%, worst-group sensitivity 51.87%, weak
  sensitivity 30.92%, FPR 5.73%.
- Reweighted: balanced accuracy 86.77%, worst-group sensitivity 51.07%, weak
  sensitivity 30.88%, FPR 6.00%. Eligible, but it lost the primary comparison.
- Fixed blend: balanced accuracy 87.73%, worst-group sensitivity 45.47%, weak
  sensitivity 24.29%, FPR 5.60%. Ineligible because weak sensitivity regressed.

The baseline was retained. These results do not prove reweighting can never
help; this particular correction did not meet the declared objective. A larger
pooled score was not allowed to hide worse detection of difficult positives.

On the actual broad fitting curriculum, training balanced accuracy was **73.26%**
for the retained baseline, 73.30% for reweighted, and 71.26% for the blend.
The retained baseline's training sensitivity was 49.77%, FPR 3.24%. This training
mixture deliberately includes much weaker injections than the full-strength
final benchmark, so those training and final percentages are not like-for-like
measures of generalization.

### 4. Fair mixed assessment

Assessment assigns half of total weight to untouched negatives and half to
positives. Clinical positives have equal mechanism mass and equal motor-level
mass within applicable mechanisms. Standalone subgroup comparisons use the same
untouched negatives. In addition, each final participant contributes one
untouched and one randomly scheduled positive query batch, with balanced level
and mechanism counts (up to rounding); rows are shuffled before inference.
Predictions were verified invariant to row order. Random ordering alone does
not correct subgroup sensitivity, so pooled and subgroup scores are both shown.

### 5. Freeze and leakage safeguards

The model, threshold, source archives and design were hashed and frozen before
final evaluation. Development file-access guards exclude nontraining raw
participants and heldout feature caches. Final evaluation checks the freeze and
prohibits both classifier fitting and threshold estimation. Artifact hashes
remain identical after both final assessments. Raw class-0 predictions match
across the two generator benchmarks. Independent metric arithmetic and random
order checks passed; no threshold was changed after observing heldout results.

The historical validation/test cohorts have been observed in earlier research.
This cycle prevents new fitting from those observations, but cannot erase prior
exposure or establish absence of all historical hindsight bias. Results are
synthetic-deviation benchmark estimates, not independent clinical confirmation.

### 6. Final validation and test results

Validation contains 1,202 people; test contains 2,738. Uncertainty intervals
resample participants, keeping their related query variants together.

**Current clinical-generator benchmark**

- Validation weighted balanced accuracy **86.81%** (95% CI 86.06–87.55%),
  positive sensitivity 80.19%, FPR **6.57%**.
- Test weighted balanced accuracy **87.31%** (95% CI 86.87–87.81%), positive
  sensitivity 79.26%, FPR **4.64%**.
- Randomized mixed accuracy: **87.19% validation / 87.25% test**.
- Cognitive sensitivity: **66.56% validation / 66.33% test**.
- Motor mild / moderate / severe sensitivity: validation **55.82% / 81.20% /
  95.34%**; test **50.91% / 79.95% / 94.74%**.
- Combined mild / moderate / severe sensitivity: validation **92.60% / 97.42% /
  99.67%**; test **91.89% / 97.48% / 99.38%**.
- Weak challenge sensitivity: **33.29% validation / 30.62% test**.

Here cognitive generation has no severity tier. Any aggregate clinical level
comparison shares the same cognitive distribution across levels; its level
labels apply to motor components only.

**Literal original normal/mild/moderate/severe generator benchmark**

To directly answer the original four-class question, a secondary transfer
protocol was declared before its heldout execution. It uses the unchanged
`StructuredTimingPerturber(PerturbationConfig())` and original `observed_windows`
and `rng_for` construction. The classifier and threshold remain frozen and
binary. Original classes 1/2/3 are each tested against untouched class 0, then
mixed with equal severity representation. No retraining or candidate selection
uses this secondary benchmark.

- Validation all-level balanced accuracy **85.50%** (95% CI 84.60–86.36%).
- Test all-level balanced accuracy **86.37%** (95% CI 85.83–86.91%).
- Randomized mixed accuracy: **85.61% validation / 86.60% test**. Because each
  person contributes one negative and one positive, accuracy equals balanced
  accuracy. Severity counts are 401/401/400 and 913/913/912 respectively.
- Mild standalone balanced accuracy: **73.50% validation / 73.65% test**;
  positive sensitivity **53.58% / 51.94%**.
- Moderate standalone balanced accuracy: **87.77% validation / 89.08% test**;
  positive sensitivity **82.11% / 82.80%**.
- Severe standalone balanced accuracy: **95.22% validation / 96.38% test**;
  positive sensitivity **97.00% / 97.41%**.
- Untouched specificity: **93.43% validation / 95.36% test**. FPR is **6.57%**
  (95% CI 5.32–7.99%) and **4.64%** (95% CI 3.87–5.41%) respectively.

Mild remains the weakest group on both cohorts. Although the mixed benchmark
exceeds the requested numerical range, about half of original mild positives
are missed. The current clinical weak challenge misses about two thirds.
Validation also exceeds the 5% training-calibration FPR target. Consequently,
this experiment establishes neither uniform sensitivity across the tested range
nor a guaranteed deployment false-positive rate. The generator's clinical
realism and real-world specificity still require independent data.

### Saved evidence and implementation status

The existing `runs_med/` directory contains:

- `binary_severity_design.json`: declared development design.
- `binary_severity_development.json`: diagnosis, weights, training and selection.
- `binary_severity_final.joblib`: frozen model collection, selected baseline,
  thresholds, calibration and full study/feature source archives.
- `binary_severity_freeze.json`: artifact, source and design hashes.
- `binary_severity_evaluation.json`: final clinical-generator assessment.
- `binary_severity_legacy_protocol.json` and `binary_severity_legacy_transfer.json`:
  secondary original-generator protocol and final results.
- `binary_severity_verification.json`: completed checks and explicit unresolved
  flags for weak-group coverage and validation FPR.

Scratch runners/logs are in `/private/tmp/exp3_severity_study/`; the primary study
source is archived in the frozen artifact. Existing production model/generator
sources were unchanged. The default prediction command still uses the preceding
1,485-feature train-only artifact, not this 1,911-feature research checkpoint.
