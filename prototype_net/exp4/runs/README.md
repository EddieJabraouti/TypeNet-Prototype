# Exp4: clinical keystroke benchmarks

One modular `model.py`; all generated files live in `runs/`. Reuses only the
TabNet and NODE architecture layers in `prototype_net/failed/exp3`. No prior
checkpoints, synthetic disease labels or clinical generators are reused.

## Run

From the repository root, with Python, NumPy, pandas, SciPy, scikit-learn, PyTorch,
XGBoost, joblib and threadpoolctl installed:

```bash
python3 -m prototype_net.exp4.model --phase fetch
python3 -m prototype_net.exp4.model --phase check
python3 -m prototype_net.exp4.model --phase prepare
python3 -m prototype_net.exp4.model --phase prepare-aalto
python3 -m prototype_net.exp4.model --phase train
python3 -m prototype_net.exp4.model --phase evaluate
```

`fetch` downloads the expanded Tappy v3 release, verifies publisher SHA-256
checksums, joins its 13 archive parts, verifies ZIP CRCs and removes redundant
parts. Data goes under `data/clinical/tappy/expanded`. `prepare` also works with
only local data. Run it after downloading; a frozen cohort cannot be expanded.
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
5. Select one candidate per endpoint using training-only OOF AUROC (minimum MAE
   for regression), lexical tie break. OOF scores used for selection are
   development estimates, not unbiased final estimates. Freeze every candidate,
   then evaluate **all** candidates on both untouched-in-this-run held-out sets.
   Neither validation nor test chooses features, epochs, thresholds or winners.
   Training can resume before freeze. Evaluation is single-use and refuses
   source/cache/model changes; an interrupted opening remains recorded.
6. Report participant-level AUROC, average precision, balanced accuracy,
   sensitivity/specificity, log loss and confusion matrices; MAE/RMSE/R² for
   motor score. Save predictions and 2,000 participant bootstrap replicates'
   percentile intervals (stratified by class for AUROC). Samples are people,
   never keystrokes, tasks, sessions or synthetic descendants.

The prespecified exploratory signal rule for each of the two primary clinical
classifiers is test AUROC's **97.5% interval lower bound > .5**, with a one-sided
label-permutation **p ≤ .025** (exact enumeration up to 20,000 allocations;
otherwise 10,000 seeded permutations with a plus-one correction). The two
primary endpoints share a Bonferroni .05 error budget; this does not cover
arbitrary later analyses. A six-person cognitive test cannot clear this exact
permutation criterion even with perfect ranking. Tappy and regression remain secondary.
Even passing this rule does not establish clinical utility or validation.
Small cohorts, prior repository-wide exposure, confounding and domain shift
require a genuinely new external cohort. A new split cannot erase prior use.
UPDRS regression across cases and controls does not by itself demonstrate
within-patient progression tracking or severity discrimination among PD cases.

If either primary endpoint fails the rule, `results.json` flags `request_data`.
Seek independent clinician-labeled recordings with repeated assessments and
participant IDs rather than increasing synthetic sample counts.

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
