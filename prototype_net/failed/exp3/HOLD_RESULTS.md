# Hold-rate experiment on the full Aalto freeze

Same people and synthetic draws as the full-Aalto parent freeze. Hold features use observed hold times versus each person's clean baseline, with the same 42-d recipe as the press-to-press pause vector. Selection is among hold, augmented, and augmented+hold by 4-way accuracy, then log loss. Chance accuracy is 25%.

Cohort: train 55,732; selection 1,306; calibration 1,034; validation 1,202; test 2,738.

Selection candidates:
- summary (618 features): selection 64.34%, log loss 0.7894; train 66.41% (4-way); parent 64.34%
- hold (42 features): selection 54.23%, log loss 0.9870; train 55.68% (4-way)
- augmented (1443 features): selection 66.21%, log loss 0.7527; train 68.64% (4-way); parent 66.21%
- augmented_hold (1485 features): selection 65.98%, log loss 0.7458; train 68.98% (4-way)

Selected: **augmented**. Train accuracy 68.64%; selection accuracy 66.21% (4-way, chance 0.25).

## Parent freeze (augmented, no raw-series concat)

- Validation: 64.70%; EER 14.81%. Recall normal 78.5%, mild 55.7%, moderate 50.7%, severe 74.0%.
- Test: 65.75%; EER 13.93%. Recall normal 79.8%, mild 57.6%, moderate 51.5%, severe 74.1%.

## Four-class accuracy versus the 618-feature summary

- **Validation:** 63.14% → **64.70%**; change +1.56 pp (95% participant-bootstrap CI +0.71, +2.45).
  EER 15.86% → 14.81%; ECE 1.85 → 1.49 pp; log loss 0.7982 → 0.7677.
  Recall: normal 78.5%, mild 55.7%, moderate 50.7%, severe 74.0%.
- **Test:** 64.11% → **65.75%**; change +1.64 pp (95% participant-bootstrap CI +1.09, +2.15).
  EER 15.75% → 13.93%; ECE 1.06 → 1.09 pp; log loss 0.7817 → 0.7443.
  Recall: normal 79.8%, mild 57.6%, moderate 51.5%, severe 74.1%.

## Pause concat on the same parent

Selected pause representation: augmented_pause. Selection 67.40%; train 69.06% (4-way).
- Validation: 64.70% → 65.45%; change +0.75 pp (95% CI +0.06, +1.50). Recall normal 78.4%, mild 55.4%, moderate 51.5%, severe 76.5%.
- Test: 65.75% → 66.66%; change +0.91 pp (95% CI +0.43, +1.40). Recall normal 80.1%, mild 58.7%, moderate 53.1%, severe 74.8%.

Hold descriptors are observed hold-time rates, excess mass, and quantile ratios relative to the enrolled person. They are not minimum-entropy deconvolution. These synthetic classes are not clinically validated severity labels.
