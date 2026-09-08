# Clinical keystroke data and synthetic-impairment review

Date reviewed: 2026-09-08

## Best candidate datasets

| Priority | Dataset | Population and measurements | Access | Appropriate use | Main limitation |
|---|---|---|---|---|---|
| 1 | [Longitudinal multimodal dementia corpus](https://doi.org/10.1007/s10579-023-09718-4) | 22 participants; 17 provided 271 longitudinal typed sessions; dementia/MCI and age-matched controls; repeated cognitive assessments and raw keystrokes | NDA request to Dimitris Gkoumas (`d.gkoumas@qmul.ac.uk`) or Maria Liakata (`m.liakata@qmul.ac.uk`) | Closest match to personalized longitudinal change detection | Small cohort; participants are not observed before disease onset |
| 2 | [Park MCI smartphone study](https://doi.org/10.2196/59247) | 111 clinically assessed participants; 2,740 usable naturalistic smartphone sessions; hold and flight times; MoCA-K and other tests | Request from Jin-Hyuck Park (`roophy@naver.com`) | Direct MCI labels and cognitive-score association | Smartphone-to-desktop domain shift; cross-sectional; a [2025 correction](https://doi.org/10.2196/86291) states that the app was custom-built, not Neurokeys as originally reported |
| 3 | [TypeOfMood MCI study](https://doi.org/10.3389/fdgth.2020.567158) | 11 MCI and 12 matched controls; 3,139 usable naturalistic sessions with more than 40 keypresses | Request from Leontios J. Hadjileontiadis (`leontios@auth.gr`) | Raw timing sequences plus typing metadata | Very few independent participants; smartphone domain |
| 4 | [AD/MCI writing-process data](https://doi.org/10.5281/zenodo.5942517) | 15 MCI/mild-AD and 15 age-/gender-matched controls; two typed picture descriptions | Open, CC BY 4.0; downloaded under `data/clinical/ad_mci_writing` | Calibrate cognitive pauses, fluency, and burst structure | Public files are participant- and word-level summaries, not complete raw key events |
| 5 | [neuroQWERTY MIT-CSXPD](https://doi.org/10.13026/C2859Q) | 42 clinically assessed Parkinson cases and 43 controls; raw desktop key events and UPDRS-III | Open, ODC Attribution; downloaded under `data/clinical/neuroqwerty` | Calibrate and externally test a motor-impairment component | Parkinson motor impairment is not cognitive decline |
| 6 | [Tappy](https://archive.physionet.org/pn6/tappy/) | Naturalistic desktop typing from more than 200 participants over months; raw hold, latency, flight, and hand-transition data | Open PhysioNet archive; downloaded under `data/clinical/tappy` | Estimate natural timing heterogeneity and hand asymmetry | Diagnosis and severity are self-reported; malformed and missing records exist |

The [RADAR-AD](https://doi.org/10.1186/s13195-021-00825-4) and ABOARD/Neurokeys
programs are additional collaboration targets, but their raw typing data are
not publicly downloadable.


## Local empirical audit

All comparisons below use the participant, not the keystroke or session, as the
independent unit.

### Direct AD/MCI writing data

Participant-level means from the open Zenodo dataset:

| Feature | Healthy control | Cognitive impairment | CI/HC ratio |
|---|---:|---:|---:|
| Characters/minute | 90.27 | 48.96 | 0.54 |
| Proportion of time paused (>2 s) | 0.348 | 0.565 | 1.62 |
| Production bursts/minute | 4.31 | 5.99 | 1.39 |
| Production-burst duration (s) | 9.78 | 4.75 | 0.49 |
| Mean within-word pause (ms) | 362.77 | 622.00 | 1.71 |
| Personal median character-to-character latency (ms) | 173.20 | 200.37 | 1.16 |

The dominant signal is sparse cognitive pausing and shorter production bursts,
not uniform slowing of every keystroke. The source study adjusted pause
analyses for each participant's personal typing speed.

### Raw Parkinson typing data

Participant-level estimates calculated from the downloaded raw files:

| Dataset | Feature | Control | Parkinson | PD/control ratio |
|---|---|---:|---:|---:|
| neuroQWERTY | Mean hold time (s) | 0.107 | 0.137 | 1.28 |
| neuroQWERTY | Hold-time SD (s) | 0.045 | 0.064 | 1.44 |
| neuroQWERTY | Mean inter-key latency (s) | 0.567 | 0.550 | 0.97 |
| Tappy | Mean hold time (ms) | 112.51 | 120.72 | 1.07 |
| Tappy | Hold-time SD (ms) | 38.90 | 45.77 | 1.18 |
| Tappy | Mean press-to-press latency (ms) | 321.01 | 326.19 | 1.02 |

These datasets disagree in effect magnitude but agree that increased hold-time
variability is more defensible than adding a large delay to every inter-key
transition. They also illustrate the substantial dataset and device
heterogeneity reported in the systematic review.


### 1. Separate constructs

- **Motor component:** hold-time variability, occasional release delay,
  hand-transition asymmetry, and temporally correlated motor states.
- **Cognitive component:** sparse long pauses at word boundaries, delayed
  post-correction restart, shorter production bursts, and session-level
  disfluency.


## Evidence quality and interpretation

The 2022 [systematic review and meta-analysis](https://doi.org/10.1038/s41598-022-11865-7)
covered 41 studies. It reported pooled MCI/early-AD sensitivity of 0.85 and
specificity of 0.82, but with substantial heterogeneity and only 254 MCI
participants across the literature. In-clinic performance was generally higher
than in-the-wild performance. Self-reported labels, small samples, device
heterogeneity, and insufficient control of age, medication, mood, and typing
experience remain important risks of bias.

## Implemented evidence calibration

`prototype_net/nn.py` now uses:

- a severe hold-time location ratio of 1.20, bounded by the 1.07 Tappy and
  1.28 neuroQWERTY participant-level estimates;
- small press-to-press location shifts;
- bidirectional, temporally correlated motor variability;
- sparse heavy-tailed pauses weighted toward word boundaries and correction
  restarts;
- heterogeneous motor/cognitive participant profiles and session effects; and
- 50% speed-matched profiles to prevent a global-speed shortcut.

A 100-user pretrained selection preflight produced overall AUROC 0.657 and EER
0.385 (mild AUROC 0.541; moderate 0.641; severe 0.789). This is substantially
harder than the former matched additive generator and is not near-perfect
before fine-tuning.

The v3 protocol preserves the 68,000 training identities, quarantines 2,000
legacy and 7,000 historically exposed identities, and assigns fresh,
identity-disjoint groups of 1,000 participants each to model selection,
threshold calibration, and final testing. It remains a synthetic stress test,
not clinical validation.

