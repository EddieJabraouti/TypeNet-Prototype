# Accumulated-window results

This is a matched 5-versus-10-window experiment. Neither budget reached the requested 70–80% validation/test accuracy. It does not evaluate hundreds of windows or clinical severity.

## Four-class accuracy and calibration

### Validation

- **5 baseline + 5 query windows:** accuracy 58.38% (95% participant-bootstrap CI 57.03–59.75%), ECE 2.31 percentage points, pooled detection EER 20.67%.
  Log loss 0.8980; multiclass Brier 0.5197; mean confidence 59.80%.
  Class recall: normal 71.2%, mild 46.9%, moderate 44.8%, severe 70.6%.
- **10 baseline + 10 query windows:** accuracy 61.75% (95% participant-bootstrap CI 60.22–63.22%), ECE 3.41 percentage points, pooled detection EER 16.40%.
  Log loss 0.8162; multiclass Brier 0.4774; mean confidence 64.72%.
  Class recall: normal 74.6%, mild 48.5%, moderate 50.5%, severe 73.4%.

Paired 10-minus-5 improvement: **+3.38 percentage points**, 95% participant-bootstrap CI +1.75 to +5.03.

### Test

- **5 baseline + 5 query windows:** accuracy 59.54% (95% participant-bootstrap CI 58.45–60.46%), ECE 1.16 percentage points, pooled detection EER 19.20%.
  Log loss 0.8847; multiclass Brier 0.5130; mean confidence 59.68%.
  Class recall: normal 71.5%, mild 48.4%, moderate 46.4%, severe 71.8%.
- **10 baseline + 10 query windows:** accuracy 62.80% (95% participant-bootstrap CI 61.82–63.78%), ECE 2.25 percentage points, pooled detection EER 16.18%.
  Log loss 0.8055; multiclass Brier 0.4754; mean confidence 64.87%.
  Class recall: normal 75.9%, mild 52.5%, moderate 50.0%, severe 72.8%.

Paired 10-minus-5 improvement: **+3.26 percentage points**, 95% participant-bootstrap CI +2.21 to +4.28.

## Training and selection

Both models are new gradient-boosted timing-summary trees. Neither uses TypeNet or prior exp3 weights.

- 5 windows per side: fitted-training accuracy 83.67%; selection accuracy 59.25%. Training accuracy is in-sample, not evidence of generalization.
- 10 windows per side: fitted-training accuracy 96.53%; selection accuracy 64.25%. Training accuracy is in-sample, not evidence of generalization.

The large training-to-selection gaps indicate substantial overfitting. The measured benefit of more windows does not establish a performance ceiling or imply that this model family is optimal.

Cohorts: train 12,000, selection 800, calibration 600, validation 800, test 2,000.

Three configurations per budget were selected using only selection identities. Separate scalar temperatures used calibration identities. Models, temperatures, participant lists, preprocessing and generator hashes were frozen before a single final evaluation. Four labels of a person remain together for splitting and bootstrap uncertainty.

## Data limitations

The complete Aalto census found 168,593 people, each with 15 sentences. The second Aalto folder adds no participants or histories. Recovering discarded keystrokes yields enough distinct variable-length windows for this comparison, but only three people have 40 total windows and nobody has 100. The planned 20/50/100/200 windows per side cannot be evaluated at cohort scale.

Participants in this experiment must have at least ten windows in each chronological sentence half. Both budgets use the same eligible people, nested baseline/query observations, and shared synthetic query profiles. This conditions the results on people with sufficient recording length; it is not directly comparable to earlier exp3 cohorts or models. Windows within one sentence may be correlated. No sentence crosses the baseline/query boundary.

All final evaluation identities are excluded from the previous exp3 selection/calibration/validation/test assignments. The historical v4 cohort has nevertheless contributed earlier aggregate results: this is not untouched external validation.

The generator is unchanged, severity is stable throughout each query period, and calibration assumes balanced synthetic classes. These recordings cannot test long-term baseline drift, changing severity, real-world prevalence, or cognitive-health validity. More distinct chronological recordings per person are required for the intended deployment-scale experiment.

## Verification

17 control tests passed. A training-only smoke run exercised fitting, calibration, freezing and sealing. Feature generation and inference matched exactly in eight smoke cases (all four classes at both budgets). Existing exp3 frozen sources/results were not modified.

See `README.md` for methods and inference, and `runs/v1/` for protocol, complete selection history, audits, models, calibrated predictions and metrics.
