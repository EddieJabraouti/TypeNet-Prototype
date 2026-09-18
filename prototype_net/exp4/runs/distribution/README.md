# Exp4: clinical keystroke benchmarks

One modular `model.py`; all generated files live in `runs/`. Reuses only the
TabNet and NODE architecture layers in `prototype_net/failed/exp3`. No prior
checkpoints, synthetic disease labels or clinical generators are reused.

## Completed run — 2026-09-17

All **52 configurations** were independently evaluated on both validation and
test: four model families (XGBoost, random forest, TabNet, NODE), 48 model
configurations plus four dummy references. There are 17 PD, nine UPDRS, nine
cognitive and 17 Tappy configurations. Every result is reported equally in
`runs/comparison.json`, including real-training fit, training out-of-fold (OOF),
validation and test metrics. No new training or test prediction was needed to
assemble this comparison from the saved artifacts.

SMOTE classification arms actually fitted these populations:

- Motor PD: **59 real + 61 synthetic = 120** training feature vectors;
  nine real validation people and 17 real test people.
- Cognitive AD/MCI: **21 real + 23 synthetic = 44** training feature vectors;
  three real validation people and six real test people.
- Tappy: **416 real + 788 synthetic = 1,204** training feature vectors;
  60 real validation people and 120 real test people.
- UPDRS regression: **59 real**, no synthetic severity targets;
  nine real validation people and 17 real test people.

All four families received matched unaugmented and SMOTE classification arms.
Motor and Tappy also received matched raw/pretrained feature arms. Synthetic
vectors are same-class interpolations of training summaries, not independent
patients or generated raw keystroke sequences. Training-fit metrics are scored
on real training people only, including in augmented arms. See every model's
train/test difference before interpreting its apparent fit.

The original run also recorded a training-OOF winner for each task and used it
in the initial summary. That never restricted which candidates were tested.
The current code removes automatic selection, and the comparison does not
filter or promote winners. Historical `registration.json`, `freeze.json`,
`results.json` and the pre-training `runs/README.md` remain unchanged. The exact
original source files were verified against their registered hashes and saved
in `runs/source.zip` before this reporting change.

Used all 85 neuroQWERTY people, all 30 writing participants/59 tasks, 2,571
exactly joined word-pause rows (77 quarantined), and 596 labeled Tappy people
with 39,220,502 retained observations across original/expanded releases. Aalto
pretraining used 168,592 people and 123,097,240 retained events. The secondary
folder contained 28,207 byte-identical exports, one unreadable duplicate,
and macOS sidecars; it added no new eligible people.

The original selected-model evidence criterion was not met for either primary
clinical endpoint. It is not applied to the full set of candidates.
Two verified-recipient inquiries are prepared in `runs/requests.json`;
sending is pending the sender-account choice. No requests have been sent.
Frozen artifacts and thresholds are unchanged after evaluation.

## Expanded motor experiment

The follow-up retains TabNet + shared frozen features + SMOTE and benchmarks
all four families equally. It lives in `runs/expanded/`; original results remain
unchanged. This is a prototype-development follow-up after observing the
original test results, not a fresh blind test.

The expanded training populations are:

- Physical-keyboard neuroQWERTY: 85 people, split 59/9/17; 812 real training
  rows from pooled summaries and recording windows. Classification arms fit
  812, 1,740, 6,960, 27,840 or 111,360 rows; the largest adds 110,548 synthetic rows.
- Physical-keyboard CoNLL: 232 people, split 161/24/47; 2,538 real training
  rows from pooled summaries and responses. Classification arms fit 2,538,
  5,724, 22,896, 91,584 or 366,336 rows; the largest adds 363,798 synthetic rows.
- Separate touchscreen cohort: 33 people, split 22/4/7; 249 real training
  rows. Classification arms fit 249, 536, 2,144, 8,576 or 34,304 rows.
- The unlabeled physical-keyboard encoder fits 169,705 rows from 169,172
  source participant identifiers for 100 epochs. Source counts are 168,592
  Aalto, 164 IKDD and 416 Tappy training participants; anonymous cross-source
  identities cannot be resolved.

The largest neuroQWERTY neural final fit has 21,800 optimizer updates; the
largest CoNLL final fit has 71,600. Each also has three separate training-fold
fits. Synthetic rows increase training exposure, not the participant counts.

All **117 configurations completed training and both held-out evaluations**:
41 neuroQWERTY PD, nine neuroQWERTY UPDRS, 41 CoNLL PD, 21 touchscreen PD
and five touchscreen UPDRS. This includes 112 learned configurations and five
dummy references. `runs/expanded/comparison.json` contains every configuration's
training-fit, training-OOF, validation and test scores; no model was filtered out.

For the fixed maximum-augmentation, pretrained physical-keyboard arms, test
accuracy / AUROC were:

- XGBoost: neuroQWERTY 41.2% / .583; CoNLL 63.8% / .683.
- Random forest: neuroQWERTY 47.1% / .625; CoNLL 66.0% / .645.
- TabNet: neuroQWERTY 64.7% / .597; CoNLL 55.3% / .654.
- NODE: neuroQWERTY 58.8% / .667; CoNLL 57.4% / .639.

The expanded pretrained TabNet's neuroQWERTY test accuracies were 52.9%, 58.8%,
47.1%, 52.9% and 64.7% for none/2/8/32/128 augmentation targets. Its maximum
augmentation arm fitted all 59 real training people correctly but scored 11/17
on test. The original pretrained SMOTE TabNet remains 12/17 (70.6%), AUROC .625;
more training exposure did not improve that reference. On CoNLL, the analogous
TabNet sequence was 61.7%, 63.8%, 57.4%, 48.9% and 55.3% on 47 test people.

A reporting-only correction prevents tiny floating-point differences from
ranking identical dummy predictions when averaging different numbers of views.
All classification dummies now have AUROC .5. Training dummy metrics use exact
participant priors in the existing folds. `reference_correction.json` records
the correction and source hashes; `reference_correction.zip` retains original
reports/predictions. Every learned-model metric and prediction, all 117 frozen
checkpoints, and the original baseline remain unchanged. The registered source
in `source.zip` records the exact implementation used for training.

Before feature extraction, the following primary results motivated the changes:
[Giancardo et al. 2016](https://doi.org/10.1038/srep34468) used non-overlapping
90-second windows with at least 30 holds and participant-level aggregation,
reporting AUROC .81; [Arroyo-Gallego et al. 2018](https://doi.org/10.2196/jmir.9462)
reported at-home AUROC .76 using the window approach. The existing seven hold
statistics are retained: this is an adaptation, not their exact feature pipeline.
Every valid hold contributes to its person's pooled summary; qualifying windows
are added as real recording-derived training rows. Windows never cross recording
boundaries. All sessions, windows and synthetic descendants stay with the same
participant split and training fold. Prediction is the unweighted mean across
that person's pooled-summary and window rows.

All 85 desktop motor participants and their original 59/9/17 assignments are
retained. Additional clinician-assessed touchscreen data from 33 people is
kept strictly separate: it never enters physical-keyboard training or shared
pretraining. It receives its own seed-matched participant split and raw-feature
models only. No cross-device motor transfer is assumed. Its timing-distribution
results are exploratory within the touchscreen cohort. PD and observed UPDRS-III
remain separate tasks with shared features/splits inside each source cohort.

The [i-PROGNOSIS release](https://doi.org/10.5281/zenodo.2571623) supplies observed
PD/control and UPDRS-III scores. Its [original study](https://doi.org/10.1038/s41598-018-25999-0)
reported AUROC .92 using timing statistics and session prediction aggregation;
we adapt only the hold-time family and aggregate all released sessions plus the
pooled-person vector. Raw press/release differences are converted from ms to s.
This CC BY 4.0 dataset's publisher MD5 is checked before extraction.

The authors' [CoNLL online English release](https://typingresearch.com/conll2020/)
is also included as a separate PD classification task. The released CSV has 232
identifiers (its README describes 230); all eligible identifiers are audited and
used, with a new participant split at the same seed. This is online-reported
prior diagnosis, not independently verified clinical ground truth. Its authors
already interpolated some missing events, without per-event provenance flags;
that limitation is preserved. Their [study](https://aclanthology.org/2020.conll-1.47/)
reported positive typing classification results. We use measured/released hold
summaries for each response and each person, not their CNN-LSTM architecture or
textual content. The publisher SHA-256 is checked before extraction.
PD compares no augmentation and SMOTE targets of 2, 8, 32 and 128 times the
largest original window-row class count, per class. SMOTE is applied to scaled
real training rows only. Each synthetic vector records two training parents
and its interpolation weight; the first parent defines its weighting group.
Weights sum equally per source participant (using first-parent ownership for
synthetic rows) and are supplied to every model. This controls domination by
participants with more recordings. Imputation/scaling still fit training views
only. Larger SMOTE ratios are experimental settings, not literature-proven
clinical improvements; [Chawla et al.](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/chawla2002.html)
support the interpolation method and comparison against unaugmented training.
UPDRS uses all real views without invented severity labels.

Every physical-keyboard model receives raw and pretrained feature arms.
Touchscreen models receive raw features only. Pretraining increases
from 10 to 100 fixed epochs and uses all cached Aalto participants, all 533
released [IKDD recordings](https://github.com/MachineLearningVisionRG/IKDD),
IKDD pooled participant summaries, and only the 416 already-assigned Tappy
training participants. Clinical labels never enter this encoder. IKDD's released
hold durations are in milliseconds and publisher-truncated at 500 ms; retain
this source difference in interpretation. Dataset citation:
[Tsimperidis et al. 2024](https://doi.org/10.3390/info15090511).
[TabNet](https://arxiv.org/abs/1908.07442) reports positive unlabeled-pretraining
results; this cross-source application remains an experimental adaptation.
The frozen 8-dimensional representation is identical for every downstream model.
[How-We-Type](https://zenodo.org/records/4047775) was downloaded and checked but
excluded because its released typing schema contains no hold/release events.
Its noncommercial research terms and original schema are retained with the data.

The downstream neural models train for 100 epochs in deterministic shuffled
minibatches of up to 512 rows, visiting every training row each epoch. All other
architecture/optimizer/tree settings remain those of the baseline. Three participant-level
training folds provide diagnostics. All 117 configurations (100 PD, 12 UPDRS,
five dummies) are frozen before any evaluation; none is selected or discarded
using training, validation or test performance. Threshold stays at .5.
Dataset/source hashes, training views, parent lineage and actual optimizer
updates are saved. Four CPU workers train independent configurations, each with one computational
thread and the same seed. This expands the motor prototype specifically; prior
writing and standalone Tappy benchmarks remain in the original run.

```bash
python3 -m prototype_net.exp4.model --phase expand --output prototype_net/exp4/runs/expanded
python3 -m prototype_net.exp4.model --phase train --output prototype_net/exp4/runs/expanded
python3 -m prototype_net.exp4.model --phase evaluate --output prototype_net/exp4/runs/expanded
python3 -m prototype_net.exp4.model --phase report --output prototype_net/exp4/runs/expanded
```

## Distribution augmentation with the existing TabNet encoder

`runs/distribution/` reuses the exact frozen encoder and 59/9/17 clinical
participant split from `runs/expanded/`. TypeNet is postponed. The seven real
hold features, eight frozen encoder features, model architectures, 100 neural
epochs, seed, threshold and training-fold protocol stay fixed. All four families
receive raw and pretrained arms, with no generation, 700 synthetic profiles,
and 7,000 synthetic profiles: 24 learned configurations plus one dummy.
Synthetic pools are balanced by label, with source participants sampled as
evenly as possible inside each label group.

The method was fixed before fitting after reviewing
[Lunardon, Menardi and Torelli (2014), ROSE](https://journal.r-project.org/articles/RJ-2014-008/).
Their class-conditional smoothed bootstrap samples Gaussian neighborhoods of
observations rather than interpolating pairs; their illustrated classification
tree achieved AUROC .989 versus .798 with ordinary oversampling. That result
motivates this experiment, not an expected clinical effect size.

Our adaptation selects pooled participant profiles equally within each label,
then adds zero-mean correlated Gaussian deviations. Kernel covariance combines
between-participant covariance of pooled summaries and the equally weighted
mean of within-participant window covariance. Its bandwidth is
`0.25 * (4 / (9 * class_participants)) ** (1 / 11)`, fixed before fitting.
The .25 multiplier, hierarchical covariance and clinical application are
experimental adaptations. Reject invalid profiles: nonfinite values, negative
duration/spread, misordered quantiles or IQR exceeding the q90-minus-q10 range.
These are synthetic timing-feature profiles, not generated raw keystrokes.

Every generator and preprocessing fit uses only the relevant training people,
including inside the three training folds. The first row per person is their
pooled profile; subsequent rows are their recording windows. Generated profiles
inherit their source label. Parent IDs, source rows, seed, covariance and hashes
are stored. Real training rows remain present; the largest fit has 812 real
rows plus 7,000 synthetic rows. Each source person receives equal total training
weight, including their synthetic descendants. The encoder derives its features
from each generated raw profile and remains frozen.

After every model is frozen, evaluation reports both original real participants
and 1,000 synthetic validation / 2,000 synthetic test profiles. The larger
synthetic pool thus has a 7,000/1,000/2,000 allocation. Held-out synthetic profiles
are centered on held-out people's pooled observations using the training-only
kernel for their observed label; no held-out observation fits that kernel or
enters training. Source people and all descendants remain in their original
role. Every model gets the same synthetic evaluation profiles. This is a
label-conditioned perturbation benchmark with 9/17 source people, not a new
independent clinical population. Synthetic row scores and scores averaged per
source person are both reported; confidence intervals resample source people.
No synthetic or real validation/test result adjusts a model or threshold.

`distribution.json` records class means, between-person spread, within-person
spread, generated means/spread and rejection counts. The generated training
profiles are saved once, shared identically across model families, and checked
against each fit's synthetic-data hash. Existing runs remain unchanged.

```bash
python3 -m prototype_net.exp4.model --phase distribution --baseline prototype_net/exp4/runs/expanded --output prototype_net/exp4/runs/distribution
python3 -m prototype_net.exp4.model --phase train --output prototype_net/exp4/runs/distribution
python3 -m prototype_net.exp4.model --phase evaluate --output prototype_net/exp4/runs/distribution
python3 -m prototype_net.exp4.model --phase report --output prototype_net/exp4/runs/distribution
```

## Run

From the repository root, with Python, NumPy, pandas, SciPy, scikit-learn, PyTorch,
XGBoost, joblib, threadpoolctl and openpyxl installed:

```bash
python3 -m prototype_net.exp4.model --phase fetch
python3 -m prototype_net.exp4.model --phase check
python3 -m prototype_net.exp4.model --phase prepare
python3 -m prototype_net.exp4.model --phase prepare-aalto
python3 -m prototype_net.exp4.model --phase train
python3 -m prototype_net.exp4.model --phase evaluate
python3 -m prototype_net.exp4.model --phase report
```

`fetch` downloads the expanded Tappy v3 release, verifies publisher SHA-256
checksums, joins its 13 archive parts, verifies ZIP CRCs and removes redundant
parts. Data goes under `data/clinical/tappy/expanded`. `prepare` also works with
only local data. Run it after downloading; a frozen cohort cannot be expanded.
`report` reads stored scores and verified checkpoints without fitting or
reopening held-out prediction. It writes `comparison.json` once.
`--output` changes the run directory; `--epochs` defaults to 100 and must be
decided before training. Do not change seeds or create fresh runs to hunt for
better held-out results. This code fixes the seed at **9172026**.

## Data and endpoints

- **motor_pd:** neuroQWERTY clinician-assessed PD versus control; all sessions
  pooled into one hold-time summary per participant.
- **motor_updrs:** the same people, features and splits; observed UPDRS-III
  regression. Diagnosis, tapping scores, published nQi and UPDRS never enter X.
- **cognitive_ci:** AD/MCI versus control; all released task summaries averaged
  per person, plus mean between-word pause from exact-ID word joins. An
  unresolved word-file ID mismatch is quarantined; no fuzzy identity repair.
  Binary pooling is supported by the study design; no fabricated severity tiers.
- **tappy_self_report:** original and expanded logs, unioned by stable user ID
  and exact record, with conflicting diagnosis reports excluded. This is a
  separate, weaker-label endpoint. Hold/latency/flight distributions use all
  eligible observations. Negative flight intervals are valid; missing events
  are never bridged into invented key sequences.

Seven timing descriptors: mean, population SD, 10th percentile, median, 90th
percentile, IQR and moment skewness. Writing uses eight released timing/speed
variables plus mean between-word pause. These compact adaptations are motivated
by the cited findings below, not claimed reproductions of published pipelines.
IDs, dates, study membership, diagnoses, medication and clinical scores are
excluded from predictors. Invalid/nonfinite timings and exact duplicates are
audited. No participant cap or arbitrary window minimum is imposed.

`prepare-aalto` ignores macOS resource-fork sidecars and extracts hold-time summaries from every available Aalto person
under `data/Keystrokes/files`, with only missing identities supplemented from
`data/keystrokes_f/files` (also a fallback for an unreadable primary header).
It deduplicates keystroke IDs within each identity.
Copies do not increase sample size. These people never receive clinical labels.
The optional **pretrained** arm uses a shared TabNet masked-reconstruction
encoder fitted solely on these external, unlabeled hold summaries, appending
eight frozen latent features to the observed clinical features. All four models
receive exactly the same features and clinical splits, with raw-feature arms as
matched controls. Motor and Tappy use this arm; writing lacks measured holds.

TabNet's published positive unsupervised-pretraining results motivate this arm
*before extraction*. The compact reconstruction encoder and shared embedding
comparison are adaptations; transfer from Aalto transcription to clinical
typing is an unproven hypothesis. Ten fixed epochs, 20% independent feature
masking, width 8/8, three steps, training-only standardization and masked MSE;
neural details otherwise match below. No clinical validation/test data enters
pretraining. Cross-source real-world identity overlap cannot be verified from
anonymous identifiers. Prior TypeNet weights are never loaded.
The external unlabeled pool is training-only auxiliary data; the 70/10/20
evaluation protocol applies to the clinically labeled participants.

## Common protocol

1. Assign people once using stratified 70/10/20 train/validation/test splitting.
   Integer rounding uses `ceil(.20*N)` test and `ceil(.125*remaining)` validation.
   Every session and synthetic descendant stays with its source participant.
   Actual counts, IDs, feature schemas and input hashes are recorded in
   `manifest.json`. Motor diagnosis and UPDRS share exactly the same people.
2. Register code hashes, package versions, all candidates and settings before
   fitting. Every candidate uses the same three training-only folds. Imputation,
   scaling, neighbor selection and augmentation are fitted inside each fold.
   Fit once on all training people after computing out-of-fold predictions.
3. Compare XGBoost, random forest, TabNet and NODE with **none** and **smote**
   augmentation for classification. UPDRS regression is real-only. An empirical
   training-prior/mean dummy establishes a reference. SMOTE creates convex
   interpolations between same-class training participants, using up to five
   neighbors and reaching twice the original majority count in each class.
   For pretrained arms, interpolate original summaries first, then derive the
   frozen embeddings, preserving the link between original and learned features.
   Store both parent IDs and interpolation weights. These are synthetic feature
   vectors, not new patients, raw typing sequences or clinical ground truth.
4. Hyperparameters are fixed: XGBoost 150 trees/depth 2/rate .03/L2 10;
   RF 300 trees/depth 4/minimum leaf 3; TabNet widths 8/8, three steps;
   NODE two layers of 16 trees/depth 3. Neural models use CPU, deterministic
   operations, AdamW .003/weight decay .0001, fixed 100 full-batch epochs, gradient
   clipping 5. TabNet entropy penalty .001. No early stopping or held-out eval
   set; no threshold tuning or probability calibration. Binary threshold = .5.
5. Freeze every candidate, then evaluate **all** candidates on both
   untouched-in-this-run held-out sets. Every candidate is reported equally;
   OOF scores are diagnostics and never choose a winner. Neither validation
   nor test chooses features, epochs, thresholds or models. Training can resume
   before freeze. Evaluation is single-use and refuses source/cache/model
   changes; an interrupted opening remains recorded.
6. Report participant-level AUROC, average precision, balanced accuracy,
   sensitivity/specificity, log loss and confusion matrices; MAE/RMSE/R² for
   motor score. Save predictions and 2,000 participant bootstrap replicates'
   percentile intervals (stratified by class for AUROC). Samples are people,
   never keystrokes, tasks, sessions or synthetic descendants.

Current reports are descriptive comparisons of every fixed candidate, with no
automatic winner or credibility gate. Per-candidate intervals are not adjusted
for searching across candidates; choosing the highest observed test score does
not establish superiority. The original run's selected-model signal rule is
preserved only in its immutable protocol snapshot and results. It must not be
applied across all configurations as if that comparison were preregistered.

A six-person cognitive test has only 20 possible balanced label assignments:
even perfect ranking gives a one-sided exact permutation p=.05. Its bootstrap
AUROC interval can be [1,1] without implying certain population performance.
Small cohorts, prior repository-wide exposure, confounding and domain shift
require a genuinely new external cohort. A new split cannot erase prior use.
UPDRS regression across cases and controls does not by itself demonstrate
within-patient progression tracking or severity discrimination among PD cases.
Seek independent clinician-labeled recordings with repeated assessments and
participant IDs; synthetic counts describe training augmentation, not additional
independent evidence.

## Literature (reviewed before implementation, 2026-09-17)

- [Giancardo et al., 2016](https://doi.org/10.1038/srep34468): hold-time variability
  and distribution features with UPDRS-based regression; reported combined
  AUROC .81. Supports the motor timing family and observed motor-score target.
  [neuroQWERTY data/schema](https://physionet.org/content/nqmitcsxpd/1.0.0/)
  (ODC Attribution v1.0; cite the study and PhysioNet).
- [Adams, 2017](https://doi.org/10.1371/journal.pone.0188226): timing features and
  classifier ensembles, reported AUROC .98 in its study. Supports Tappy timing
  summaries, not clinician-confirmed label quality.
- [Cognitive Writing Process Characteristics in Alzheimer's Disease, 2022](https://doi.org/10.3389/fpsyg.2022.872280):
  significant cognitive-group differences in speed, pauses and bursts after
  controlling for typing speed. Supports the released writing variables and
  between-word pauses; association is not validated diagnostic accuracy.
  [Released summaries](https://doi.org/10.5281/zenodo.5942517), CC BY 4.0.
- [Chawla et al., 2002, SMOTE](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/chawla2002.html):
  improved ROC performance on tabular benchmarks compared with undersampling
  and prior adjustments. Applying same-class interpolation to both clinical
  classes is an explicitly experimental extension, not proven clinical benefit.
  It is always compared against unaugmented training.
- [XGBoost](https://arxiv.org/abs/1603.02754),
  [TabNet](https://arxiv.org/abs/1908.07442), and
  [NODE](https://arxiv.org/abs/1909.06312) report positive tabular benchmarks.
  Local neural implementations are compact adaptations, not official paper
  reproductions. TabNet is a neural tabular comparator rather than a tree model.

## Additional data

- [Expanded Tappy v3](https://data.mendeley.com/datasets/z39mhdsynx/3),
  DOI 10.17632/z39mhdsynx.3, CC BY 4.0: over 500 subjects claimed by the provider;
  exact usable union and overlap must come from the adapter audit. Same-source
  expansion is not an external validation cohort. Still self-reported PD.
- [Longitudinal dementia typing corpus](https://doi.org/10.1007/s10579-023-09718-4):
  closer cognitive/longitudinal match, available by request rather than public
  download. Request raw events, timed clinical assessments and repeated visits.
- [Park MCI smartphone study](https://doi.org/10.2196/59247): direct cognitive
  assessment, request-access data; substantial smartphone/desktop domain shift.
  [Correction](https://doi.org/10.2196/86291) identifies the custom keyboard app.
- [2025 pressure-keyboard Parkinson study](https://doi.org/10.1126/sciadv.adt6631):
  specialized hardware/pressure signals and very few patients; not a compatible
  timing-only source. Do not merge its reported results into this benchmark.

Data requests should describe this as exploratory research, request deidentified
records under the provider's access terms, and never claim institutional
affiliation or ethics approval that has not been provided.
