# C0 baseline report

C0 is the paper's cross-attention model (part 1, code unchanged) retrained on
this study's folds and inputs. Every graph approach is compared against it.
Numbers below were produced on Perun on 2026-10-27/28; the raw tables are in
`results/C0/` there (`runs.csv`, `summary.csv`, `quad_vs_trio.csv`,
`ensemble/`, `submissions/`).

## Protocol

| | Paper (part 1) | C0 here |
| --- | --- | --- |
| Folds | stratified 5-fold, per task; 25 % of validation videos share an author with training | author- and duplicate-grouped 5-fold (`outputs/folds.csv`), leakage 0 % |
| Checkpoint selection | best ICM-Soft-Norm on the fold that is reported | lowest validation loss on an inner fold of the training part; the outer fold is predicted once (nested) |
| Early stopping | patience 4–6 | patience 10 (EMA starts late; data_note 13) |
| Descriptions | three VLM sources, uneven train/test | regenerated for all 3198 videos, one model / prompt / sampler |
| Transcript | organisers' `text` | organisers' text, cleaned (VLM prompt: our ASR where reliable) |
| VideoMAE | attention biases not loaded | biases loaded (`videomae_fix.py`) |
| Configs | runs 164307 / 184649 / 202433 | the same configs (`configs/c0/`), max VLM tokens 512 |
| Seeds | 1 | 5 (42–46) per modality set and task: 30 runs |
| Evaluation | PyEvALL, 3.2 / 3.3 without the EXIST hierarchy | `evaluate.py`: EXIST hierarchy, 2508 val-eligible videos, author bootstrap |

## C0 out-of-fold results (5 seeds)

ICM-Soft-Norm as mean ± sd over seeds, then the seed ensemble; the other
columns are for the ensemble (AUC and F1: seed mean).

| Task | Model | ICM-Soft-Norm | ens | ICM-Norm (hard) | AUC | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 3.1 | quad | 0.4879 ± 0.0060 | 0.4884 | 0.6024 | 0.8167 | 0.7320 |
| 3.1 | trio | 0.4812 ± 0.0105 | 0.4818 | 0.6076 | 0.8119 | 0.7294 |
| 3.2 | quad | 0.3850 ± 0.0047 | 0.3851 | 0.5602 | 0.8298 | 0.6501 |
| 3.2 | trio | 0.3832 ± 0.0042 | 0.3836 | 0.5613 | 0.8271 | 0.6497 |
| 3.3 | quad | 0.0690 ± 0.0019 | 0.0710 | 0.3610 | 0.7957 | 0.3448 |
| 3.3 | trio | 0.0658 ± 0.0030 | 0.0690 | 0.3677 | 0.7973 | 0.3545 |

**Seed spread = tolerance for later comparisons:** sd of ICM-Soft-Norm is
0.006–0.010 (3.1), ~0.004 (3.2), 0.002–0.003 (3.3). Seed ensembling adds
almost nothing out of fold (3.1: 0.4879 → 0.4884).

### Subsets (quad, seed ensemble, ICM-Soft-Norm / ICM-Norm)

| Task | contested | speech | no speech |
| --- | --- | --- | --- |
| 3.1 | 0.0000 / 0.2467 | 0.4864 / 0.6022 | 0.4767 / 0.5831 |
| 3.2 | 0.2838 / 0.2957 | 0.3852 / 0.5552 | 0.3735 / 0.5593 |
| 3.3 | 0.1161 / 0.1933 | 0.0716 / 0.3619 | 0.0581 / 0.3386 |

- **3.1 contested = 0 is calibration, not a bug** (data_note 13): shrinking
  the probabilities towards 0.5 raises it to 0.33 at unchanged AUC, and the
  paper's own OOF also scores 0 there (below). Graph models are compared on
  contested videos against confidence-matched controls.
- **No-speech videos** score a little lower at the same AUC, so C0 does not
  depend heavily on speech; the test set has more of them (30 % vs 19 %).

## Quad vs trio (does physiology help?)

Paired author bootstrap of the seed ensembles, 2000 resamples, 95 % CI.
Bold = interval excludes 0.

| Task | Subset | ICM-Soft-Norm | ICM-Norm | AUC |
| --- | --- | --- | --- | --- |
| 3.1 | all | **+0.0066** [0.0038, 0.0094] | −0.0051 [−0.016, 0.006] | +0.0036 [−0.0003, 0.0077] |
| 3.1 | contested | 0 [−0.029, 0] | **−0.0347** [−0.069, −0.002] | −0.0070 [−0.017, 0.004] |
| 3.2 | all | +0.0015 [−0.001, 0.004] | −0.0011 [−0.012, 0.009] | +0.0020 [−0.0004, 0.0044] |
| 3.2 | contested | **+0.0113** [0.004, 0.019] | +0.0065 [−0.016, 0.031] | **+0.0092** [0.004, 0.014] |
| 3.3 | all | +0.0021 [−0.001, 0.007] | −0.0067 [−0.017, 0.004] | −0.0002 [−0.004, 0.004] |
| 3.3 | contested | −0.0006 [−0.003, 0.002] | +0.0008 [−0.016, 0.018] | −0.0014 [−0.007, 0.005] |

Physiology moves C0 by at most ~0.01, in both directions; with 18
comparisons one or two "significant" cells are expected by chance. The
paper's test-set gains (one run, no interval) are not reproduced:

| | Paper, test | C0, CV |
| --- | --- | --- |
| 3.1 soft | +0.044 | +0.007 |
| 3.1 hard | +0.010 | −0.005 (tie) |
| 3.2 soft | +0.009 | +0.002 (tie) |
| 3.3 hard | +0.033 | −0.007 (tie) |

→ The reference for Q4 is quad (as in the paper); trio is equivalent.
Whether physiology enters the graphs is left to the audit (Step 28).

## Text only vs video (does VideoMAE add anything?)

`text` = the same model and configs without video and physiology (the
paper's vlm + trn; Step 28's text-only control), 5 seeds per task. Seed
ensembles, ICM-Soft-Norm / ICM-Norm / AUC:

| Task | text | trio | quad |
| --- | --- | --- | --- |
| 3.1 | 0.4836 / 0.6043 / 0.8269 | 0.4818 / 0.6076 / 0.8216 | 0.4884 / 0.6024 / 0.8252 |
| 3.2 | 0.3688 / 0.5525 / 0.8323 | 0.3836 / 0.5613 / 0.8327 | 0.3851 / 0.5602 / 0.8347 |
| 3.3 | 0.0583 / 0.3537 / 0.8033 | 0.0690 / 0.3677 / 0.8055 | 0.0710 / 0.3610 / 0.8053 |

Paired author bootstrap, trio − text (the video), 95 % CI; bold = excludes 0:

| Task | all: ICM-Soft-Norm | all: ICM-Norm | all: AUC | contested: ICM-Soft-Norm |
| --- | --- | --- | --- | --- |
| 3.1 | −0.002 [−0.006, 0.002] | +0.003 [−0.010, 0.017] | −0.005 [−0.012, 0.001] | 0 [−0.018, 0.017] |
| 3.2 | **+0.015** [0.009, 0.020] | +0.009 [−0.007, 0.025] | 0.000 [−0.006, 0.007] | **−0.034** [−0.050, −0.018] |
| 3.3 | **+0.011** [0.003, 0.015] | **+0.014** [0.001, 0.029] | +0.002 [−0.003, 0.008] | **+0.005** [0.002, 0.008] |

quad − text is the same picture (3.1 +0.005 [0.001, 0.009] soft, AUC ties
on every task). **VideoMAE adds no discrimination:** AUC ties on all three
tasks, all videos and contested. The soft-metric changes on 3.2 go up on
all videos and down on contested, which is the signature of a change in
confidence (data_note 13). Only 3.3 gains a little on every subset. The
visual content mostly reaches the model already through the text: the VLM
description is written from the frames. This matches data_note 16 (frames
inside a video rarely differ) and says where graphs can add information:
not in clip nodes, but in links between videos (data_note 17).

## C0 against the paper on the same validation videos

The paper's saved out-of-fold predictions (the runs behind its submissions)
re-scored with `evaluate.py` on the same 2508 videos (`convert_part1_oof.py`;
3.3 hard threshold 0.4 for both).

| Task | Metric | Paper OOF | C0 OOF (quad ens) | C0 − paper |
| --- | --- | --- | --- | --- |
| 3.1 | ICM-Soft-Norm | 0.5451 | 0.4884 | −0.057 |
| | ICM-Norm | 0.6359 | 0.6024 | −0.034 |
| | AUC | 0.8259 | 0.8252* | ≈ 0 |
| | Macro-F1 | 0.7574 | 0.7320 | −0.025 |
| | contested ICM-Soft-Norm | 0.0000 | 0.0000 | 0 |
| 3.2 | ICM-Soft-Norm | 0.4371 | 0.3851 | −0.052 |
| | ICM-Norm | 0.5761 | 0.5602 | −0.016 |
| | AUC | 0.8453 | 0.8347* | −0.011 |
| | Macro-F1 | 0.6680 | 0.6501 | −0.018 |
| 3.3 | ICM-Soft-Norm | 0.0877 | 0.0710 | −0.017 |
| | ICM-Norm | 0.3555 | 0.3610 | +0.006 |
| | AUC | 0.7855 | 0.8053* | +0.020 |
| | Macro-F1 | 0.4099 | 0.3448 | −0.065 |

\* ensemble AUC from the bootstrap table; the seed mean is slightly lower.

**Reading:** C0 ranks videos as well as the paper's model (AUC equal on 3.1,
−0.01 on 3.2, +0.02 on 3.3). The gap is concentrated in ICM-Soft-Norm, the
metric the paper selected its epochs by, on the very fold it reported, on
folds with author leakage. A rough decomposition (ICM-Soft-Norm, ensembles):

| Step | 3.1 | 3.2 |
| --- | --- | --- |
| Paper OOF: ICM selection on the reported fold, leaky folds | 0.545 | 0.437 |
| C0, ICM selection on an inner fold, grouped folds (`runs/C0_select_icm`) | 0.508 | 0.420 |
| C0, validation-loss selection (final) | 0.488 | 0.385 |

The last step trades ICM-Soft-Norm for calibration: ICM selection broke 3.3
(epoch 1 kept) and inflated quad − trio to +0.054 (data_note 13). The paper
itself reports that validation-loss selection generalised better to the
hidden test set. Temperature scaling (Step 27), applied to every model
alike, is where the soft-metric gap can be recovered.

## Paper test scores (reference) and C0 test scores

Paper, official EXIST 2026 test (ICM-Soft-Norm / ICM-Norm / Macro-F1):

| Task | quad | trio | best of the paper's 8 configurations |
| --- | --- | --- | --- |
| 3.1 | 0.5457 / 0.6563 / 0.7695 | 0.5013 / 0.6464 / 0.7631 | quad |
| 3.2 | 0.3845 / 0.5450 / 0.6364 | 0.3759 / 0.5406 / 0.6326 | vlm+trn 0.3951 (soft), vlm+video 0.5605 (hard) |
| 3.3 | 0.1297 / 0.4549 / 0.4193 | 0.1297 / 0.4215 / 0.4106 | quad |

Paper OOF vs paper test: 3.1 0.545 vs 0.546 (equal); 3.2 0.437 vs 0.385
(OOF optimistic); 3.3 hard 0.356 vs 0.455 (test higher). So C0's OOF does
not predict its test score; only the test run can.

**C0 test: pending** (EXIST site down on 2026-10-28). Submissions are ready
in `results/C0/submissions/exist2026_AegisAthena/` (run 1 = quad, run 2 =
trio; each the mean of 25 models: 5 seeds × 5 folds). Record the site's
scores in `results/C0/test_scores.csv`.

## Open points

- C0 test scores (above).
- 3.3 hard labels use the paper's fixed threshold 0.4; per-category
  thresholds fitted on the other folds' OOF (Step 27) are not applied yet,
  and will be applied to every model alike.
- Text-only control: OOF done (above); its test submissions are run 3.
