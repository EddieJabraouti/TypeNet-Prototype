# Exp4: clinical keystroke benchmarks

One modular `model.py`; all generated files live in `runs/`. The original runs
reuse the TabNet and NODE architecture layers in `prototype_net/failed/exp3`.
The frozen TypeNet follow-up additionally reuses `paper_typenet/nn.py` and the
existing physical-keyboard encoder checkpoint; its protocol is documented below.

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

The primary combined dataset is `runs/distribution/combined/dataset.npz`:
**11,177 real and synthetic rows together**, with a saved split column:
7,812 training (812 real + 7,000 generated), 1,134 validation (134 + 1,000),
and 2,231 test (231 + 2,000). Source participants and descendants stay in one
partition. Origin and source IDs are retained as audit metadata, never predictors.
Maximum-generation models already trained on this exact combined training data.
The unaugmented and 700-profile controls retain their registered training sets.

`combined/results.json` reports every candidate on the same combined validation
and test rows. With 7,000 generated training profiles and frozen encoder features,
combined-test accuracy / AUROC are XGBoost 51.6% / .612, random forest
54.8% / .629, TabNet 47.7% / .491 and NODE 49.0% / .540. TabNet with 700
generated training profiles scores 52.4% / .623 on the same combined test set.
Row metrics pool real and synthetic observations; source-participant metrics
and intervals retain the original grouping. Earlier separate-origin reports
remain diagnostic records. Combining reports does not refit or tune any model.

The original 70.6% TabNet checkpoint was also evaluated unchanged on this same
combined dataset. It scores 74.4% accuracy / .650 AUROC on the 1,134 validation
rows and 57.5% / .593 on the 2,231 test rows. Its original 12/17 accuracy and
.625 AUROC were reproduced first. No retraining, encoder substitution or
threshold adjustment occurred. Results and verified checkpoint/input hashes
are in `combined/original_tabnet.json`, with saved predictions alongside it.

Five later stratified reshuffles of the 26 held-out source people, keeping
9 validation / 17 test people and all descendants grouped, yielded average
combined validation accuracy 59.1% / AUROC .615 and complementary test accuracy
65.4% / AUROC .654. Validation accuracies were 51.8%, 55.8%, 66.5%, 41.0% and
80.2%. This uses unchanged cached predictions, not retraining or five independent
datasets; original split files remain intact. Seeds, participant assignments and
every metric are saved in `combined/original_tabnet_resampling.json`.

Future distribution `report` runs also create the combined dataset and report.
For an existing completed report, run `--phase combined --output
prototype_net/exp4/runs/distribution` once. The combined folder stores the dataset,
predictions, all results and the reporting source snapshot with input hashes.

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

The run completed all 25 configurations. The largest pretrained arms scored
the following real-test accuracy / AUROC, then synthetic-test accuracy / AUROC:

- XGBoost: 47.1% / .597; 50.9% / .608.
- Random forest: 47.1% / .653; 54.8% / .623.
- TabNet: 52.9% / .583; 46.8% / .481.
- NODE: 52.9% / .597; 48.1% / .532.

Pretrained TabNet with 700 generated profiles scored 52.9% / .750 on real test
and 51.7% / .621 on synthetic test. With 7,000, its real-training accuracy was
91.5%, training-OOF accuracy 72.9%, real-validation accuracy 77.8% and
synthetic-validation accuracy 75.3%. All raw-feature controls and other settings
are retained in `runs/distribution/comparison.json`; no winner was selected.
All eight learned unaugmented controls reproduce the expanded run's predictions
exactly. The original 70.6% TabNet belongs to the earlier, different run.

Distribution checks found control/PD average participant median holds of
97.95/131.81 ms and IQRs of 33.64/57.88 ms. Generated median and IQR averages
stay within .5% of those values. Long-tail mean holds increase by 18.3%/12.9%
after Gaussian smoothing and admissibility rejection, so fidelity is strongest
for central quantiles and imperfect for tails. These shifts are reported without
retuning after evaluation. Matrix multiplication emitted floating-point status
warnings on this runtime; operands and outputs were finite and agreed with
direct summation within 1.8e-15. All generated profiles and predictions passed
finite-value, ancestry and metric checks. `audit.log` records verification.

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

## Frozen TypeNet repeated CV (registered before fitting)

This follow-up replaces the shared TabNet representation with the repository's
existing `typenet_68k_m50_g10_64x512_best_weights.pt`, the default of the earlier
TypeNet prototype. Its 200,458 parameters and running statistics stay frozen.
No checkpoint is chosen using clinical scores. Each classifier receives the
128 TypeNet coordinates plus the existing seven hold-time summary statistics.

Sources consulted before implementing the adapter and generator:

- Acien et al., [TypeNet](https://arxiv.org/abs/2101.05570): positive biometric
  results (2.2% physical-keyboard EER); 50 strokes, HL/IL/PL/RL and keycode/255.
  This supports the representation, not a claim about clinical accuracy.
- Lan and Yeo, [PLOS ONE 2019](https://doi.org/10.1371/journal.pone.0219114):
  printable-key/space filtering and hold-time variability differentiated early
  PD (SD-index AUC .752 in the full cohort). Exclude modifiers, backspace and
  other non-printing controls. Keep valid long holds; no clinical-score-driven
  cutoff. Alphanumeric/space keycodes have direct mappings; ambiguous symbols
  use the frozen encoder's input key-channel mean, with counts recorded.
- Bandara et al., [time-series augmentation](https://arxiv.org/abs/2008.02663):
  moving-block bootstrap and pooling augmented with original series improved
  forecasting. Applying contiguous block resampling to clinical typing is an
  adaptation evaluated here, not a previously established clinical result.

The new generator samples two contiguous blocks from the same participant's
recordings to create each 50-stroke sequence, with a successor event for timing
calculations. Holds, press intervals and keys are sampled jointly. Inter-key
and release intervals are recomputed, preserving timing identities and allowing
real key rollover. No Gaussian summary noise, class covariance, cross-person
mixing or label input is used. Labels are inherited after generation. Every
synthetic stroke retains its source-event index. Marginal timing quantiles,
means, correlations, lag-one dependence and sequence uniqueness are audited
using training data. Raw recording boundaries are never crossed within blocks.

Use the original 59/9/17 participant split. Inside the 59 training people, run
five seeded repeats of stratified five-fold CV (seeds 9172026 through 9172030).
All rows and descendants of a participant remain together. For each fold and
family, compare zero versus 128 synthetic sequences per training participant.
Use the same combined real-plus-128-synthetic validation rows for both arms.
Choose the highest CV combined-row accuracy, then AUROC, then less augmentation.
All classifier settings remain as in the preceding experiment (100 neural
epochs, fixed .5 threshold). Preprocessing is fitted only on that fold's fit rows.
There is no early stopping or external validation/test-based selection.

After all 200 fits and 100 family selections are frozen, evaluate each selected
fold model on the same external validation and test sets. Also record the best
CV-selected family for each fold, with fixed family ordering to break exact ties.
Report mean and sample SD over 25 fold models per family, plus the selected-family
pipeline. SD describes sensitivity to training folds, not 25 independent test
cohorts. Both combined-row and real-source-participant scores are retained.
Real and synthetic observations share one dataset with split, source and origin
metadata; evaluation views are generated independently within each held-out
source participant, without feeding labels into generation. The fixed test cohort
has been used in earlier experiments. It is excluded from this run's selection.


### Completed TypeNet results

All 200 fits completed: 25 folds × four classifier families × two augmentation
arms. The 100 family selections and 25 overall selections were frozen before
external scoring. TabNet here is a classifier head; every candidate uses frozen
TypeNet features. Neural heads trained for 100 epochs without early stopping.

One combined dataset contains 14,235 observations:

- Training pool: 2,282 real + 7,552 synthetic = 9,834 rows, from 59 participants.
- External validation: 306 real + 1,152 synthetic = 1,458 rows, from 9 participants.
- Test: 767 real + 2,176 synthetic = 2,943 rows, from 17 participants.

Each CV fit uses 47–48 training participants; augmented fits use 7,769–7,994
combined rows. Real-only fits remain a control. Augmentation was selected in
46/100 family-by-fold comparisons. The chosen model for every fold was evaluated,
including each family's winner. All 200 CV candidate scores remain available.

Test accuracy, mean ± sample SD over 25 fold models; combined rows first,
real-participant aggregation second:

- xgboost: 56.9% ± 4.8%; real participants 59.3% ± 8.5%.
- random_forest: 56.6% ± 5.3%; real participants 58.1% ± 7.8%.
- tabnet: 58.7% ± 3.8%; real participants 61.4% ± 6.8%.
- node: 64.8% ± 4.6%; real participants 74.1% ± 7.4%.
- cv_selected: 59.5% ± 4.7%; real participants 64.2% ± 9.0%.

The `cv_selected` row evaluates the procedure that chooses the winning family
within each fold. NODE's highest mean test score is a reported comparison,
not a test-based replacement for that preselected procedure. The fixed test
cohort is shared by all 25 models; the SD is across models, not cohorts.

Training-only fidelity checks found 100% unique generated sequences, no exact
real-row copies, valid timing identities and complete same-participant lineage.
Label-wise central hold/press quantiles shifted by at most 1.6%; hold means
shifted -7.6% for controls and -2.6% for PD. A source-disjoint five-fold XGBoost
real-versus-synthetic discriminator, using equal source/origin weights and full
50-stroke sequences, scored AUROC 0.505 ± 0.018 (chance is .5). This diagnostic
supports fidelity for these features, not a guarantee of every temporal property.
Mean log-hold lag-one correlation was .0155 real versus .0302 synthetic.

`results.json` contains all CV candidates, fold-level external results, aggregates,
AUROC and other metrics. `dataset.npz` holds the combined observations, splits,
origins, raw sequences and source-event lineage. Checkpoints are bundled by fold.
`audit.json`, `fidelity.json` and `verification.json` record the checks; audit
scripts and runtime versions are archived in `audit_source.zip`. Verification
recomputed accuracy and SD from saved predictions, checked fold-only scaler fit
statistics for all 200 models, and verified unchanged encoder and baseline hashes.

Run into a new output directory (existing registered runs are protected):

```sh
python3 -m prototype_net.exp4.model --phase typenet-prepare --output prototype_net/exp4/runs/typenet
python3 -m prototype_net.exp4.model --phase typenet-train --output prototype_net/exp4/runs/typenet
python3 -m prototype_net.exp4.model --phase typenet-evaluate --output prototype_net/exp4/runs/typenet
```

## Frozen NODE external validation

The external follow-up evaluates all 25 already-selected TypeNet → NODE models
without fitting, recalibration, threshold changes or model selection. Primary
cohort: all eligible participants in the released Online English CSV, external
to the neuroQWERTY-trained models. The cohort has been examined in earlier,
separate exp4 experiments, so this is model transfer, not a newly blinded dataset.
The publication describes self-reported early-stage PD; these are not independently
verified clinical diagnoses. The release has 232 labeled IDs, while the paper
reports 230 recruited and 229 analyzed. Retain all consistent IDs and record
eligibility rather than inventing exclusion criteria to reproduce that count.

The adapter uses the same 50-stroke TypeNet channels and seven hold statistics,
fixed printable-key filtering, direct alphanumeric/space mapping and neutral
symbol keycodes. Response boundaries are respected. No features are fitted on
external data. All retained real sequences are scored, with equal-participant
metrics after averaging each participant's sequence probabilities. Report mean
and sample SD across the 25 fixed models, plus a prespecified equal-weight model
ensemble. Report stratified participant-bootstrap confidence intervals for that
ensemble. The classification threshold remains .5. No synthetic external rows
are needed to count these additional real participants.

Sources read before implementation: [TypeNet](https://arxiv.org/abs/2101.05570),
[printable-key clinical analysis](https://doi.org/10.1371/journal.pone.0219114),
and [Dhir et al., CoNLL 2020](https://aclanthology.org/2020.conll-1.47/), whose
online typing model achieved participant AUROC .84/.75 in off/on medication
groups. Cross-cohort generalization is the question being tested here.

### External result

All 25 unchanged NODE heads were evaluated on 232 Online English participants
(101 PD, 131 controls), with 9,561 real sequences and no synthetic observations.
Participant accuracy was 60.4% ± 2.3% and AUROC .669 ± .024 across heads.
The fixed equal-weight ensemble achieved 62.5% accuracy and AUROC .679;
stratified participant-bootstrap 95% intervals were 56.5–68.5% and .613–.745.
This is transfer to participants absent from these models' clinical training;
the Online English cohort had already been explored with other project models.
Artifacts and verification are in `runs/external`.

## Real-data experiment

Combine all real neuroQWERTY and Online English sequences in one dataset;
exclude every synthetic observation. Use the verified frozen TypeNet embeddings
and the same seven hold statistics (135 features). The source studies contain
85 and 232 participants respectively. Online PD labels are self-reported;
neuroQWERTY labels are clinical. The source field is metadata, never an input.

Make a fresh, diagnosis-stratified 70/10/20 participant split with seed 9172026.
All observations belonging to one person stay in one split. Fit scalers only
on training rows and give each participant equal total training weight.
Use five repeats of five-fold CV within the training split, with identical folds
for NODE, XGBoost, Random Forest and TabNet: 25 fits per family, 100 total.
The four configurations are fixed to their existing settings; neural heads train
for 100 epochs. The encoder stays frozen. No augmentation, hyperparameter search,
early stopping or threshold adjustment. Freeze checkpoints before evaluating
validation and test participants; evaluate all four families at threshold .5.

Report participant accuracy and AUROC as mean and sample SD across 25 fold models.
Each participant prediction averages their real sequence probabilities.
Also save equal-weight ensemble results and source-specific breakdowns.
The SD measures variation across models on the same test participants.
Verify zero synthetic rows, disjoint participant splits, fold-only preprocessing,
checkpoint integrity and reproducibility of metrics from saved predictions.

The feature methods and positive literature precedents remain those cited above:
[TypeNet](https://arxiv.org/abs/2101.05570) for physical-keyboard sequence
representations, [NODE](https://arxiv.org/abs/1909.06312) for tabular prediction,
and the clinical typing studies. No new feature engineering is introduced.

No emails will be sent without the user's approval. Outreach is deferred.

```sh
python3 -m prototype_net.exp4.model --phase real-prepare --baseline prototype_net/exp4/runs --output prototype_net/exp4/runs/real
python3 -m prototype_net.exp4.model --phase typenet-train --output prototype_net/exp4/runs/real
python3 -m prototype_net.exp4.model --phase real-evaluate --output prototype_net/exp4/runs/real
```

### Completed real-data results
All 100 fits completed, using only 12,916 real sequences from 317 participants.
- Training: 221 participants, 8,979 real sequences.
- Validation: 32 participants, 1,366 real sequences.
- Test: 64 participants, 2,571 real sequences.

Each CV fit trained on 176–177 people within the 221-person training split.
TypeNet remained frozen; NODE and TabNet heads trained for 100 epochs per fit.
All test results below are real-participant accuracy, mean ± sample SD across
25 fold models per family, followed by mean AUROC. Validation uses 32 people.

- xgboost: test 72.6% ± 1.8%, AUROC 0.776; validation 61.9% ± 3.3%, AUROC 0.736.
- random_forest: test 72.4% ± 2.0%, AUROC 0.777; validation 60.8% ± 3.4%, AUROC 0.733.
- tabnet: test 69.8% ± 2.5%, AUROC 0.739; validation 57.9% ± 4.5%, AUROC 0.687.
- node: test 69.1% ± 3.0%, AUROC 0.748; validation 61.4% ± 4.9%, AUROC 0.709.

The prespecified equal-weight NODE ensemble scored 73.4% accuracy (47/64)
and AUROC .757 on test; its validation accuracy was 59.4% (19/32).
This ensemble is a separate summary from the mean accuracy across fold models.

Verification passed: fresh split reproducibility, participant-disjoint CV,
exact real-only feature provenance, all 100 fold-specific scaler/imputer fits,
unchanged encoder/checkpoint hashes, and accuracy/AUROC/SD reproduced from saved
predictions. All results, source-specific breakdowns and artifacts are in
`runs/real`; the independent verification script is archived in `audit_source.zip`.

## TypeNet embeddings only

`runs/embeddings` repeats the real-data experiment using only the 128 frozen
TypeNet coordinates. Remove the seven hold-time summaries from each sequence;
retain every real observation, participant label, participant split and CV fold
from `runs/real`. No personal baseline features or synthetic rows are used.
The representation is the existing [TypeNet](https://arxiv.org/abs/2101.05570)
embedding; no new extraction or generation method is introduced.

Train all four classifiers afresh with the unchanged training and evaluation
functions: five repeats of five-fold CV, 25 fits per family, 100 fits total.
Retain fold-only standardization, participant weighting, seed 9172026, fixed
hyperparameters, 100 neural epochs and threshold .5. Input width changes from
135 to 128; all other settings remain the same. Freeze all checkpoints before
evaluating validation and test. Report participant accuracy mean and sample SD
and mean AUROC, with equal-weight ensemble metrics separately.

Preparation verifies the original source archive and unchanged training functions.
Audit exact feature-column identity, preserved observations and folds, train-only
preprocessing, all checkpoint dimensions and independently reproduced metrics.

```sh
python3 -m prototype_net.exp4.model --phase embeddings-prepare --baseline prototype_net/exp4/runs/real --output prototype_net/exp4/runs/embeddings
python3 -m prototype_net.exp4.model --phase typenet-train --output prototype_net/exp4/runs/embeddings
python3 -m prototype_net.exp4.model --phase real-evaluate --output prototype_net/exp4/runs/embeddings
```

### Completed embeddings-only results

All 100 fits completed on the same 12,916 real sequences from 317 participants:
221 training, 32 validation and 64 test. Each classifier used 128 inputs.
Results below are participant accuracy mean ± sample SD across 25 models,
followed by mean AUROC. Parentheses give the original 135-feature test accuracy.

- XGBoost: test 71.1% ± 2.7%, AUROC .731 (72.6%); validation 59.0% ± 2.6%, AUROC .697.
- Random Forest: test 70.1% ± 2.4%, AUROC .718 (72.4%); validation 58.1% ± 2.6%, AUROC .686.
- TabNet: test 69.7% ± 3.0%, AUROC .740 (69.8%); validation 57.1% ± 3.2%, AUROC .643.
- NODE: test 67.1% ± 2.9%, AUROC .743 (69.1%); validation 59.8% ± 4.1%, AUROC .675.

Separate equal-weight ensemble test results: XGBoost 73.4% (47/64), AUROC .737;
Random Forest 75.0% (48/64), AUROC .724; TabNet 75.0% (48/64), AUROC .754;
NODE 70.3% (45/64), AUROC .750. These ensemble accuracies are not the mean
accuracy across individual fits. No model or threshold was changed after scoring.

Removing the summaries reduced mean test accuracy by 1.44 percentage points
for XGBoost, 2.25 for Random Forest, 0.13 for TabNet and 2.00 for NODE.
Mean test AUROC fell except for TabNet, which was essentially unchanged.
This is a feature comparison with 25 fits per family, not a comparison of fit counts.

Verification passed: exact original embedding columns and metadata, identical
participant splits and folds, unchanged training/evaluation functions and settings,
100 fold-only scaler/imputer fits, 128 inputs on every checkpoint, every saved
participant prediction reproduced, and all accuracy/AUROC/SD/ensemble metrics
independently checked. Artifacts are in `runs/embeddings`; verification code is
archived in its `audit_source.zip`.

## Personal baseline monitor

`PersonalMonitor` implements fixed-baseline, repeated-batch inference using the
real-only XGBoost ensemble and frozen TypeNet. Each accepted batch contains
50 complete, non-overlapping sequences of 50 printable keystrokes. TypeNet
produces a 50 × 128 matrix; seven timing statistics per sequence extend it to
50 × 135 for the classifier. Session boundaries are respected. The first batch
becomes the baseline. Each later batch is scored against that unchanged baseline.
No online fitting or synthetic generation occurs.

Literature read before implementation:

- [TypeNet's author implementation](https://github.com/BiDAlab/TypeNet) compares
  query embeddings with a gallery using mean pairwise Euclidean distance; its
  published authentication experiments showed effective typing representations.
- [Gretton et al., A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html)
  reports strong distribution-comparison results on attribute matching and graph
  problems. We use its descriptive squared MMD statistic with an RBF kernel.
- [Holmes et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10585965/) found
  group differences in longitudinal change from baseline using a separately
  trained motor-impairment score. This supports investigating personal change,
  not interpreting TypeNet distance as clinically validated deterioration.

Outputs include the mean classifier score, its change from the baseline score,
mean cross-batch embedding distance, distance relative to the baseline's mean
within-batch distance, and biased squared MMD. The RBF denominator is the median
nonzero squared distance among baseline embeddings, fixed for all follow-ups.
If the baseline is degenerate, the distance ratio is null and the denominator
uses a numerical floor. MMD is descriptive: no significance test, alert threshold
or medical-change classification is fitted. It compares sets, without assuming
that row positions represent matched text. The classifier score is not calibrated
as an individual disease probability. Baseline-relative metrics provide
personalization while classifier weights remain frozen.

Each event requires `session_id`, `event_id`, `keydown_ms`, `keyup_ms`, and either
`key` (a printable character; space accepted) or a pre-normalized `keycode`.
Standard keycodes are in [0,255]; internal replay also accepts the reserved
neutral symbol code derived from the frozen encoder. OS-specific scan codes
must be mapped before using `keycode`. Timestamps
share one clock within each session, and sessions are supplied in collection order.
A batch payload contains `participant_id`, unique `batch_id`, monotonically
increasing `timestamp_ms`, and the `events` array. IDs preserve event provenance;
labels are never passed to the monitor. Incomplete windows at session ends are
not counted. The canonical timing adapter can use the next event in the same
supplied session to complete a window's final interval. It never reads future
batches. Fewer than 50 full windows returns `insufficient_data` without changing
state; more than 50 requires the caller to divide the input into batches.

`observe(payload)` establishes the baseline on the first accepted batch and
scores subsequent ones. `save(path)`/`restore(path)` persist the baseline,
processed batch/event IDs and model fingerprint. Duplicate events, repeated
batches, wrong participants, out-of-order batches and mismatched model state
are rejected. Baseline and follow-up contain no shared source events.

```sh
python3 -m prototype_net.exp4.model --phase simulate --baseline prototype_net/exp4/runs/real --output prototype_net/exp4/runs/monitor
python3 -m prototype_net.exp4.model --phase monitor --baseline prototype_net/exp4/runs/real --input baseline.json --state person.npz --output baseline_result.json
python3 -m prototype_net.exp4.model --phase monitor --baseline prototype_net/exp4/runs/real --input followup.json --state person.npz --output followup_result.json
```

Simulation protocol: use every held-out neuroQWERTY participant with at least two
recorded visits containing 2,500 valid printable keys each. Take the first 2,500
keys of each eligible visit in chronological order. Recompute TypeNet embeddings
from these real events; do not reuse fitted participant baselines or cached
classifier predictions. Restart the monitor and restore its state before each
follow-up. Record the actual visit dates inferred from source filenames. These
are sampled typing visits, not full days of passive observation. No new clinical
accuracy or change-detection accuracy is inferred from this functional replay.
Save replay inputs, embedding/feature matrices, persistent state and results.

### Completed monitor simulation

The exact 50-sequence baseline plus 50-sequence follow-up replay completed for
held-out participant `nq:95`, a control from the validation split. Their recorded
visits were September 11 and September 25, 2014, approximately 14 days apart.
Each batch contains 2,500 real keystrokes, producing a 50 × 128 embedding matrix
and a 50 × 135 classifier feature matrix. This was the only held-out participant
with two qualifying visits; no observations were repeated or generated to fill
batches. Other participants with shorter recordings were not silently padded
into full 50-keystroke sequences.

- Baseline classifier score: .370206; control-class prediction.
- Follow-up classifier score: .406874; control-class prediction.
- Classifier-score change: +.036668.
- Mean cross-batch embedding distance: 1.278269.
- Mean baseline within-batch distance: 1.164350; ratio 1.097840.
- Squared MMD: .128308, using the fixed baseline bandwidth.

The baseline was restored from disk before follow-up and remained byte-identical.
`runs/monitor/replay.jsonl` contains both real input batches. `matrices.npz`
contains explicit embedding and feature matrices; `nq_95.npz` contains persistent
monitor state. `results.json` records outputs and source dates. Verification
passed for independent distance/MMD calculations, API/CLI parity, frozen weights,
real event separation, restart behavior, incomplete batches and invalid/repeated
inputs. A separate state-machine check processed four disjoint real 10-window
excerpts to exercise repeated updates; it does not add participants or observations
to the primary 50-window simulation. Its script is in `audit_source.zip`.

### Control baseline with PD-labeled follow-up

`simulate-impaired` runs the same fixed XGBoost monitor with `nq:95` as the
control baseline. It first replays that participant's control follow-up, then
replays every PD-labeled participant with at least 50 complete windows across
their recordings. Training participants are included as requested. Select the
first 50 windows in source-recording order, preserving session boundaries and
using only real keystrokes. Record all eligible participants without selection
based on predictions. Each batch contains 2,500 real keys and produces a
50 × 128 embedding matrix. No new features, fitting or thresholds are introduced;
the previously cited TypeNet and MMD methods remain unchanged.

The PD batches come from other people: this is an explicitly cross-person demo,
not a claim that the control participant developed impairment. The payload's
participant ID identifies the demo stream; each result records the actual source
participant, diagnosis label, training split and source recordings separately.
Payload timestamps specify simulated arrival order, while actual recording
timestamps are retained as source metadata. The reference baseline is saved and
restored between every batch and never updated. Report every score and prediction,
including PD cases classified as control. PD diagnosis is a source label, not a
measurement of that batch's momentary severity. No new accuracy claim is made.

```sh
python3 -m prototype_net.exp4.model --phase simulate-impaired --baseline prototype_net/exp4/runs/real --output prototype_net/exp4/runs/impaired
```

The impaired replay completed and passed verification. Six of eight PD-labeled
batches were classified PD; `nq:84` and `nq:86` were classified control. The
control follow-up remained control (.406874). All ten batches (one baseline,
one control follow-up, eight PD follow-ups) used 25,000 unique real events.
The original baseline remained unchanged through every saved/restored update.
Scores, source IDs and training splits are in `runs/impaired/results.json`.

### Batch-size coverage

The user clarified that 2,500 keys per batch was an example, not a required
minimum. With complete 50-key windows and equal-size disjoint baseline/follow-up
segments, eligibility across both datasets is: 50 keys per batch, 317 people;
100, 316; 200, 312; 300, 311; 400, 304; 500, 297; 600, 283; 1,000, 35;
2,500, 2. The broad counts permit earlier/later segments from the same visit.
For actual separate-visit neuroQWERTY pairs, 400 keys per visit retains all
31 paired participants (18 PD, 13 controls); 500 retains 30; 1,000 retains 25;
2,500 retains 2. These are data-availability counts, not outcome metrics.
A single 50-key sequence cannot estimate within-person embedding variability;
the existing monitor requires at least two embeddings. TypeNet stays at 50 keys
per sequence; the number of sequences per baseline/follow-up can be reduced.

## Baseline-aware 300/400-key experiments

Compare fixed population-input and baseline-aware XGBoost/NODE configurations
using the same real follow-up rows, participant splits, five repeats of five-fold
CV, seeds, thresholds and optimizer settings. Retrain both arms so differences
in training data do not confound the input-feature comparison. Each scenario
has 100 fits: 25 folds × two families × two feature arms; 200 total. Neural heads
train for 100 epochs. TypeNet remains frozen. No new synthetic observations.

The 300-key scenario uses six complete 50-key sequences as each participant's
baseline and all later complete six-sequence batches. It includes earlier/later
segments within a recording, not only separate visits. Recordings are ordered by
filename timestamps for neuroQWERTY and acquisition response ID for Online English;
keys are time-sorted within each recording. Online clocks sometimes reset or
responses overlap in their timestamp ranges, so response order is used between
recordings and no absolute calendar-day interpretation is assigned.

The 400-key scenario uses eight sequences from the first neuroQWERTY visit as
the baseline and all complete eight-sequence batches from later visits. All
baseline and follow-up windows belong to the same participant. No raw source
keys are shared across baseline or follow-up batches. Sequence timing is recomputed
within each supplied batch; terminal intervals never read into the next batch.
A remainder smaller than a complete batch is excluded, without resampling.

Population inputs contain the original 135 features. Personal inputs contain
those features, the earlier baseline's per-feature mean and standard deviation,
and (current − baseline mean)/baseline standard deviation: 540 total columns.
Standard deviations at or below 1e-8 use a denominator of one, a fixed numerical
constant guard. Only baseline observations determine these individual statistics.
Population preprocessing is fitted on each CV training fold only. All people have
equal total training weight regardless of their number of follow-up batches.
Baseline observations are reference inputs; they are not supervised training rows.
The reference is fixed for subsequent batches and its diagnosis is never supplied
as a personalization input. This is baseline-aware PD classification, not a
trained detector of within-person clinical progression.

Literature read before feature implementation: Tervonen et al.'s
[2026 comparison](https://link.springer.com/article/10.1007/s11257-026-09444-w)
evaluates baseline mean/SD context columns, baseline normalization and related
methods on 170 participants. It reports improvements from personalization in
many feature-based tasks, mixed results for simple baseline calibration, and
positive prior baseline-calibration findings in its 2023/2024 studies. Our
combination of raw features, baseline moments and normalized deviations is an
experimental adaptation of those components, not a claimed replication of their
performance. Earlier TypeNet and NODE sources remain applicable.

Keep the existing 317-person experiment's participant assignments, restricted
to eligible people. Expected counts: 300 keys, 218/31/62 training/validation/test;
400 keys, 18/5/8. The methods within each scenario share exactly the same folds.
Both scenarios are registered before training. Freeze every checkpoint before
validation/test scoring. Report participant accuracy and AUROC mean/sample SD
over 25 fold models, with fixed probability-mean ensembles as secondary results.
The 300/400 scenarios have different populations, so their scores are not a
controlled comparison of batch size alone.

```sh
python3 -m prototype_net.exp4.model --phase personal-prepare --baseline prototype_net/exp4/runs/real --output prototype_net/exp4/runs/personal
python3 -m prototype_net.exp4.model --phase personal-train --output prototype_net/exp4/runs/personal
python3 -m prototype_net.exp4.model --phase personal-evaluate --output prototype_net/exp4/runs/personal
```

### Completed baseline-aware results

All 200 fits completed and verification passed. Accuracy below is mean ± sample
SD across 25 fold models, evaluated on the same held-out people within each
scenario. Both arms were retrained on identical follow-up observations.

300 keys per batch: 218 training, 31 validation, 62 test participants.

- validation, xgboost: population 69.4% ± 3.4% (AUROC 0.768); baseline-aware 64.1% ± 5.9% (AUROC 0.651).
- validation, node: population 64.9% ± 3.6% (AUROC 0.739); baseline-aware 66.3% ± 4.1% (AUROC 0.670).
- test, xgboost: population 70.6% ± 2.1% (AUROC 0.758); baseline-aware 67.7% ± 2.6% (AUROC 0.736).
- test, node: population 66.4% ± 3.2% (AUROC 0.726); baseline-aware 69.5% ± 4.1% (AUROC 0.748).

400 keys per batch: 18 training, 5 validation, 8 test participants.

- validation, xgboost: population 66.4% ± 9.5% (AUROC 0.990); baseline-aware 62.4% ± 15.6% (AUROC 0.730).
- validation, node: population 70.4% ± 10.2% (AUROC 1.000); baseline-aware 80.8% ± 13.5% (AUROC 0.980).
- test, xgboost: population 49.5% ± 7.6% (AUROC 0.693); baseline-aware 39.5% ± 27.9% (AUROC 0.357).
- test, node: population 69.0% ± 19.1% (AUROC 0.765); baseline-aware 49.0% ± 14.8% (AUROC 0.620).

In the 300-key experiment, baseline-aware NODE improved mean test accuracy
by 3.10 percentage points and mean AUROC by .022 versus the matched population
arm. Its fixed equal-weight ensemble scored 75.8% (47/62) and AUROC .782,
compared with 67.7% (42/62) and .738 for the population NODE ensemble. The
ensemble is a separate result from the 25-model mean. Baseline-aware XGBoost
performed worse on mean test accuracy and AUROC. Both baseline-aware classifiers
performed worse on the 400-key separate-visit test cohort of eight participants.

Verification independently reproduced all reported accuracy, AUROC, SD and
ensemble results, checked all 200 fold-only preprocessing states, verified
source-event separation and chronological baseline construction, and confirmed
unchanged checkpoints. Full results and verification are in `runs/personal`.

Additional primary precedent read before preparation: [Tervonen et al., 2024](https://eudl.eu/doi/10.1007/978-3-031-59717-6_3)
reported better cognitive-load classification with short personal-baseline
calibration than without personalization. That result supports testing the
normalization component; it does not establish performance on these keystrokes.

## Full baseline embedding concatenation

`runs/baseline_embeddings` tests XGBoost, Random Forest, TabNet and NODE with
and without the full baseline embedding matrix. Retain the exact 300/400-key
observations, chronology, participant assignments and CV folds from `runs/personal`.
Discard the seven hold-time summaries. Each population input is the current
128-coordinate TypeNet embedding. Each personal input appends every baseline
embedding in chronological row-major order: six baseline embeddings at 300 keys
(896 total input columns), eight at 400 keys (1,152 columns). No baseline mean,
standard deviation or deviation features are calculated. The baseline stays
fixed across all later sequences. No supervised training row is a baseline row.

The full follow-up matrix is still processed sequence by sequence, keeping the
original training unit and prediction aggregation. Every post-baseline sequence
receives the same full baseline context for that person. Predictions are averaged
per participant for evaluation; there is no averaging of baseline input vectors.
Both arms use exactly the same follow-up observations and participant weights.

[TypeNet](https://arxiv.org/abs/2101.05570) supplies the existing pretrained
sequence representation. A primary positive precedent read before preparation,
[HONeYBEE, 2025](https://www.nature.com/articles/s41746-025-02003-4), concatenates
foundation-model embeddings and evaluates Random Forest classifiers; concatenation
achieved the highest classification accuracy among its compared fusion methods.
That study supports testing embedding concatenation, not an expectation of the
same gains for longitudinal keystrokes. The requested baseline-matrix arrangement
is an experiment rather than a replication of that oncology study.

Train both arms afresh for all four families, with five repeats of five-fold CV
and 25 fits per configuration: 400 fits across the two scenarios. Retain the
existing seed, model hyperparameters, train-fold-only standardization, 100 neural
epochs and fixed .5 threshold. TypeNet remains frozen. No synthetic observations,
hyperparameter selection, early stopping or validation/test-based adjustments.
Freeze all checkpoints before validation/test scoring. Primary metrics are
participant accuracy mean/sample SD and mean AUROC; fixed equal-weight ensembles
are secondary. The 300-key cohort retains 218/31/62 training/validation/test people;
the separate-visit 400-key cohort retains 18/5/8.

```sh
python3 -m prototype_net.exp4.model --phase personal-embeddings-prepare --baseline prototype_net/exp4/runs/personal --output prototype_net/exp4/runs/baseline_embeddings
python3 -m prototype_net.exp4.model --phase personal-train --output prototype_net/exp4/runs/baseline_embeddings
python3 -m prototype_net.exp4.model --phase personal-evaluate --output prototype_net/exp4/runs/baseline_embeddings
```

### Completed full-matrix results

All 400 fits completed. The figures below compare embeddings without baseline
against full baseline-matrix concatenation, using identical follow-up observations.
Accuracy is participant mean ± sample SD across 25 models; AUROC is the mean.

300 keys: 218 training, 31 validation and 62 test participants.

- validation, XGBoost: without baseline 64.3% ± 2.4% (AUROC 0.710); full baseline 55.2% ± 5.4% (AUROC 0.605).
- validation, Random Forest: without baseline 63.1% ± 3.2% (AUROC 0.693); full baseline 54.7% ± 7.6% (AUROC 0.543).
- validation, TabNet: without baseline 60.5% ± 3.3% (AUROC 0.662); full baseline 52.6% ± 6.4% (AUROC 0.560).
- validation, NODE: without baseline 59.6% ± 3.4% (AUROC 0.665); full baseline 53.7% ± 7.0% (AUROC 0.558).

- test, XGBoost: without baseline 69.7% ± 2.5% (AUROC 0.717); full baseline 63.5% ± 4.9% (AUROC 0.676).
- test, Random Forest: without baseline 68.6% ± 2.5% (AUROC 0.697); full baseline 61.2% ± 4.5% (AUROC 0.647).
- test, TabNet: without baseline 69.0% ± 3.1% (AUROC 0.714); full baseline 61.9% ± 5.8% (AUROC 0.628).
- test, NODE: without baseline 64.7% ± 2.1% (AUROC 0.710); full baseline 61.2% ± 4.2% (AUROC 0.640).

Separate equal-weight ensemble test results, without baseline → full baseline:
- XGBoost: 71.0% → 67.7%; AUROC 0.718 → 0.686.
- Random Forest: 71.0% → 64.5%; AUROC 0.692 → 0.694.
- TabNet: 71.0% → 64.5%; AUROC 0.725 → 0.680.
- NODE: 66.1% → 58.1%; AUROC 0.717 → 0.664.

400 keys: 18 training, 5 validation and 8 test participants.

- validation, XGBoost: without baseline 75.2% ± 8.7% (AUROC 0.900); full baseline 68.8% ± 20.1% (AUROC 0.830).
- validation, Random Forest: without baseline 70.4% ± 11.7% (AUROC 0.930); full baseline 65.6% ± 23.5% (AUROC 0.800).
- validation, TabNet: without baseline 79.2% ± 4.0% (AUROC 0.970); full baseline 65.6% ± 18.7% (AUROC 0.760).
- validation, NODE: without baseline 76.0% ± 8.2% (AUROC 1.000); full baseline 72.0% ± 14.1% (AUROC 0.820).

- test, XGBoost: without baseline 48.5% ± 13.7% (AUROC 0.550); full baseline 50.5% ± 22.7% (AUROC 0.497).
- test, Random Forest: without baseline 43.0% ± 13.1% (AUROC 0.432); full baseline 49.5% ± 17.9% (AUROC 0.491).
- test, TabNet: without baseline 52.0% ± 18.6% (AUROC 0.613); full baseline 42.5% ± 20.1% (AUROC 0.385).
- test, NODE: without baseline 64.5% ± 16.0% (AUROC 0.733); full baseline 54.5% ± 7.1% (AUROC 0.782).

Separate equal-weight ensemble test results, without baseline → full baseline:
- XGBoost: 50.0% → 37.5%; AUROC 0.562 → 0.438.
- Random Forest: 50.0% → 50.0%; AUROC 0.375 → 0.312.
- TabNet: 37.5% → 37.5%; AUROC 0.688 → 0.188.
- NODE: 62.5% → 50.0%; AUROC 0.875 → 0.938.

Full-matrix concatenation reduced mean test accuracy and AUROC for every
classifier at 300 keys. At 400 keys, accuracy increased for XGBoost and Random
Forest but decreased for TabNet and NODE; AUROC moved in different directions.
The 400-key cohort has eight test participants and is a different cohort from
the 300-key experiment. These results concern direct matrix concatenation,
not the earlier baseline mean/SD/deviation representation.

Verification passed for all 400 fits: original real observations and chronology,
exact ordered baseline-vector concatenation, no timing-summary input columns,
identical participant assignments and folds, fold-only preprocessing, all saved
participant predictions reproduced from checkpoints, and accuracy/AUROC/SD and
ensemble metrics independently recalculated. Source, checkpoints, predictions,
results and verification are archived in `runs/baseline_embeddings`.

## Learned baseline correction with NODE

`runs/residual` tests only the 300-key scenario. Keep the exact follow-up rows,
baseline matrices, participant splits and repeated CV folds from
`runs/baseline_embeddings/300`. All inputs are frozen TypeNet coordinates;
there are no appended hold-time statistics or synthetic observations.
Train three prespecified configurations: population NODE, gated residual NODE,
and NODE with late concatenation. Each receives 25 fits, 75 total.

For each current embedding x and its owner's earlier 6 × 128 baseline B, a
single attention head projects queries and keys to 16 coordinates, applies
softmax(q(x)k(B)^T / 4), and uses the original baseline vectors as values.
A 256 → 32 → 128 GELU network maps [x, context − x] to a learned correction z.
The residual representation is x + tanh(g) * z, with a learned 128-coordinate
gate initialized at zero. Late concatenation uses [x, z]. NODE receives 128
or 256 features respectively. The same baseline stays fixed for later batches;
attention weights can change with the current sequence. No positional encoding
is used, so permuting the baseline rows does not change the representation.
These weights aggregate baseline information inside the trainable network;
the input remains the full baseline matrix rather than fixed mean/SD features.

The NODE head retains its original two layers, 16 trees per layer and depth 3.
Both attention projections and the correction network train jointly with NODE.
Adapter initialization uses seed 9172027 without consuming the NODE head's RNG
stream. The residual model starts with exactly the same head initialization and
predictions as population NODE. Fit standardization on the fold's training
follow-up embeddings only, then apply that same transform to every baseline
coordinate. Neither a baseline label nor another participant's baseline is used.

Keep seed 9172026, 100 epochs, AdamW learning rate .003, weight decay .0001,
gradient clipping 5, batches of at most 512 and equal total participant weights.
Use five repeats of five-fold participant CV within 218 training people. Freeze
all 75 checkpoints before scoring the 31 validation and 62 test participants at
threshold .5. No early stopping, tuning or selection on validation/test.
Report every configuration's participant accuracy mean/sample SD and mean AUROC;
fixed equal-weight ensemble metrics are secondary. There is no 400-key run.

Primary positive precedents read before implementation:
- [Gorishniy et al., 2021](https://arxiv.org/abs/2106.11959): residual networks were strong tabular baselines.
- [Lee et al., 2019](https://proceedings.mlr.press/v97/lee19d.html): attention-based set models achieved strong results on set-structured tasks.
- [Bachlechner et al., 2021](https://proceedings.mlr.press/v161/bachlechner21a.html): zero-initialized residual gates improved optimization in their experiments.

Our small attention adapter, coordinate-wise bounded gate and NODE combination
adapt these components; they are not a replication of those papers or an
established clinical-change model. Population NODE is retrained unchanged to
isolate the effect of the learned baseline correction on matched observations.

```sh
python3 -m prototype_net.exp4.model --phase residual-prepare --baseline prototype_net/exp4/runs/baseline_embeddings --output prototype_net/exp4/runs/residual
python3 -m prototype_net.exp4.model --phase residual-train --output prototype_net/exp4/runs/residual
python3 -m prototype_net.exp4.model --phase residual-evaluate --output prototype_net/exp4/runs/residual
```

### Completed learned-correction results

All 75 fits completed on the 300-key cohort: 218 training, 31 validation and
62 test participants. TypeNet remained frozen; no timing summaries or synthetic
rows were added. Accuracy is mean ± sample SD across 25 fold models.

- validation, No-baseline NODE: 59.6% ± 3.4%, mean AUROC 0.665.
- validation, Gated residual NODE: 54.6% ± 5.8%, mean AUROC 0.556.
- validation, Late-concatenation NODE: 58.7% ± 6.2%, mean AUROC 0.647.

- test, No-baseline NODE: 64.7% ± 2.1%, mean AUROC 0.710.
- test, Gated residual NODE: 62.6% ± 3.9%, mean AUROC 0.666.
- test, Late-concatenation NODE: 66.0% ± 4.4%, mean AUROC 0.702.

Separate fixed equal-weight ensemble test results:
- No-baseline NODE: 66.1% (41/62), AUROC 0.717.
- Gated residual NODE: 59.7% (37/62), AUROC 0.741.
- Late-concatenation NODE: 64.5% (40/62), AUROC 0.765.

Late concatenation increased mean test accuracy by 1.29 percentage points,
but reduced mean test AUROC by .0079 versus population NODE. Residual addition
reduced mean test accuracy by 2.06 points and AUROC by .0432. Ensemble AUROCs
increased for both adapters, while ensemble accuracies decreased at the fixed
.5 threshold. Neither adapter improved every reported test metric. No threshold
or training setting was changed after observing these results.

Verification passed: unchanged prior functions; original real follow-up rows and
earlier same-person baseline matrices; participant-disjoint CV; all 75 training-only
preprocessing states; all 25 fresh control checkpoints exactly matching the previous
NODE weights; adapter parameter updates and nonzero residual gates; baseline-order
invariance; every saved prediction reproduced; unchanged encoder and checkpoint
hashes; all accuracy/AUROC/SD and ensemble metrics recalculated. Source archives,
checkpoints, predictions, results and verification are in `runs/residual`.

## Fresh adapters for XGBoost, Random Forest and TabNet

`runs/adapters` completes the same 300-key comparison for the three other model
families: population input, gated residual input, and late concatenation. Keep
the exact data, personal baseline matrices, folds and settings from `runs/residual`.
There are 225 fresh final-classifier fits, 25 for each family/configuration.
No previously trained classifier or adapter checkpoint initializes any fit.

Each XGBoost and Random Forest pipeline independently trains its own adapter from
scratch in its own CV training fold, using the same temporary differentiable NODE
head and objective as the earlier adapter experiment. It then freezes its own
adapter, discards that head's predictions, and fits its tree classifier on the
learned 128- or 256-coordinate representations. This adds 100 fresh adapter
training runs: two tree families × two adapter modes × 25 folds. No adapter object
or checkpoint is shared between tree families. Identical data, initialization and
training objectives can yield identical weights under the common deterministic
seed; independent training does not imply different numerical weights.

TabNet trains a fresh adapter jointly with its own head. Preserve its original
architecture and .001 sparse-attention penalty. The residual TabNet model starts
with exactly the no-baseline TabNet head state and predictions. All attention and
correction parameters remain trainable throughout its 100 epochs.

All three no-baseline controls are freshly trained on the same follow-up rows.
Keep 100 neural epochs, the existing AdamW settings, participant weights, model
hyperparameters, seed 9172026 and fixed .5 threshold. Each adapter's standardizer
uses only its fold's training follow-up embeddings; the same transform applies
to that person's baseline. Tree-stage standardizers fit only the transformed
training rows. No timing summaries, synthetic rows, tuning or early stopping.
Freeze all models before evaluating 31 validation and 62 test participants.
Literature precedents and experimental qualifications are those in the preceding
learned-correction section; no new feature-extraction method is introduced.

```sh
python3 -m prototype_net.exp4.model --phase adapters-prepare --baseline prototype_net/exp4/runs/residual --output prototype_net/exp4/runs/adapters
python3 -m prototype_net.exp4.model --phase adapters-train --output prototype_net/exp4/runs/adapters
python3 -m prototype_net.exp4.model --phase adapters-evaluate --output prototype_net/exp4/runs/adapters
```

### Completed three-family adapter results

All 225 classifier fits and 100 fresh tree-adapter training runs completed.
Each tree family trained its own adapters separately; TabNet trained its own
adapters jointly. Only the frozen TypeNet representation and dataset were reused.
Results use 300 keys, 218 training, 31 validation and 62 test participants.
Accuracy is mean ± sample SD across 25 fits; AUROC is the mean.

XGBoost

- validation, no baseline: 64.3% ± 2.4%, AUROC 0.710.
- validation, gated residual: 53.9% ± 5.0%, AUROC 0.559.
- validation, late concatenation: 59.5% ± 5.8%, AUROC 0.647.
- test, no baseline: 69.7% ± 2.5%, AUROC 0.717.
- test, gated residual: 61.9% ± 4.1%, AUROC 0.661.
- test, late concatenation: 65.9% ± 4.3%, AUROC 0.706.

Random Forest

- validation, no baseline: 63.1% ± 3.2%, AUROC 0.693.
- validation, gated residual: 54.1% ± 4.5%, AUROC 0.556.
- validation, late concatenation: 59.6% ± 6.3%, AUROC 0.638.
- test, no baseline: 68.6% ± 2.5%, AUROC 0.697.
- test, gated residual: 62.0% ± 4.1%, AUROC 0.661.
- test, late concatenation: 65.9% ± 4.8%, AUROC 0.704.

TabNet

- validation, no baseline: 60.5% ± 3.3%, AUROC 0.662.
- validation, gated residual: 57.8% ± 6.3%, AUROC 0.621.
- validation, late concatenation: 60.8% ± 6.2%, AUROC 0.641.
- test, no baseline: 69.0% ± 3.1%, AUROC 0.714.
- test, gated residual: 63.9% ± 5.0%, AUROC 0.692.
- test, late concatenation: 64.9% ± 3.3%, AUROC 0.692.

The no-baseline control had the highest mean test accuracy for every family.
Random Forest late concatenation slightly improved mean AUROC while reducing
accuracy. Other adapter configurations reduced both mean accuracy and AUROC
relative to their corresponding controls. All configurations retained threshold .5.

Separate fixed equal-weight ensemble test results:
- XGBoost, no baseline: 71.0% accuracy, AUROC 0.718.
- XGBoost, gated residual: 59.7% accuracy, AUROC 0.738.
- XGBoost, late concatenation: 64.5% accuracy, AUROC 0.766.
- Random Forest, no baseline: 71.0% accuracy, AUROC 0.692.
- Random Forest, gated residual: 59.7% accuracy, AUROC 0.735.
- Random Forest, late concatenation: 64.5% accuracy, AUROC 0.761.
- TabNet, no baseline: 71.0% accuracy, AUROC 0.725.
- TabNet, gated residual: 67.7% accuracy, AUROC 0.754.
- TabNet, late concatenation: 69.4% accuracy, AUROC 0.735.

Verification passed: matched original real data and folds, independent fresh
adapter training with checkpoint loading blocked in the mechanism test, no shared
adapter objects between tree families, all 325 fold-only preprocessing states,
fresh controls reproducing the prior predictions, correct neural epochs and
TabNet penalty, and all 225 saved predictions independently reproduced. Metrics
and checkpoint/encoder/dataset hashes were verified. Artifacts and audit source
are archived in `runs/adapters`.
