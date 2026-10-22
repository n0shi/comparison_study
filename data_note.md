# Data note (Step 1–3, stage 0)

Generated from `outputs/videos_stats.json`, `outputs/near_duplicates_stats.json`
and `outputs/folds_stats.json` (2026-10-21). Re-run `s01`–`s03` to refresh.

## Inventory

| | train | test |
|---|---|---|
| videos (JSON, H5, mp4 all present) | 2524 (1524 ES / 1000 EN) | 674 (304 ES / 370 EN) |
| labels | official gold, 3 tasks | none (blind; scored on the EXIST site) |
| mp4 with an audio stream | 2524 | 674 |

- Duration: median 17.2 s, 5–95 % 6.4–59 s, max 135 s. Audio 44.1 kHz in every file. Mostly 30 fps (also 25/24/50/60, a few ~17).
- Annotators: 2 per video for 1780 videos, 3 for 744. Only gender is known about them.
- Physiology: 2 subjects per video, ET/HR/EEG keys present for all 3198 videos. Subject ids (`ES3_2`…) are a different id space from the annotators (`Annotator_4`…). Physiology is kept out of every approach until the Step 28 audit.

## Labels (official gold, consumed as-is)

| task | unanimous | contested | with hard gold |
|---|---|---|---|
| 3.1 | 1946 | 578 | 2508 (16 ties 0.5/0.5) |
| 3.2 | 1836 | 688 | 2466 |
| 3.3 | 1446 | 1078 | 2279 |

- P(YES) takes 5 values: 0 (1104), 1/3 (202), 1/2 (16), 2/3 (360), 1 (842).
- Hard 3.2: NO 1306 / DIRECT 736 / JUDGEMENTAL 424.
- Hard 3.3 positives: STEREOTYPING 582, IDEOLOGICAL 229, OBJECTIFICATION 131, **SEXUAL-VIOLENCE 57, MISOGYNY 37**. The two rare classes have about 11 and 7 hard positives per validation fold.
- 3.3 unanimity uses the raw annotator label sets (flag only, never a target).
- The 3.3 soft gold rows do **not** sum to 1 (mean 1.11; see `project_sexism/models/task3_3_metric_audit`). DATA.md says otherwise and is wrong on this point.

## Problems found

1. **Author leakage in the part-1 folds.** The author comes from the TikTok URL. 2036 authors in train, 710 train videos come from authors with more than one video (up to 23). In the part-1 folds, **25 % of validation videos have their author in the training part**. Predicting a video's label as the author's majority label gets 75 % accuracy on those 633 videos. The test set shares an author with train for only 1 of 674 videos, so part-1 CV is optimistic relative to test. Between-video approaches (B1–B5, C4) would turn this leak into gains that cannot appear on test.
   → `grouped` folds (StratifiedGroupKFold by author, seed 42): leakage 0 %, same balance by language, label and contested share.
2. **The `text` field is not a clean ASR transcript.** It mixes speech, on-screen text and emoji names (`face_with_tears_of_joy`…). 131 train / 30 test texts contain **leaked LLM cleaning prompts** ("clean the previously detected text by removing repetitions…", "cleaned text"). 339 train texts contain the author's handle, a watermark. 217 texts have degenerate repetition (ASR loops on music). Words per second: median 3.6, but 807 videos are above 5 w/s, which is more than speech, so this is overlay text. 6 videos have almost no text.
   → The text cannot be force-aligned to audio as a whole (see "Text–audio synchronisation" below).
3. **Part 1 scored 3.2 / 3.3 without the EXIST hierarchy.** PyEvALL takes a `hierarchy` parameter, and part 1 never passed it. Under the hierarchy, the same predictions score differently. For example, the constant (0.5, 0.3, 0.2) prediction on 3.2 gets ICM-Soft-Norm 0.066 flat and 0.123 hierarchical. Flat multi-label hard ICM is **not implemented** in PyEvALL, which explains the NaN 3.3 hard ICM in part 1. Our evaluation uses the hierarchy (YES → subclasses). **Confirmed** by the EXIST 2026 Guidelines (V0.5, "Evaluation" section), which give exactly these hierarchies for 3.2 and 3.3 and no hierarchy for 3.1. The official F1 for 3.1 is positive-class F1 (`f1_yes`); for 3.2 and 3.3 it is macro F1. Part-1 CV numbers for 3.2 / 3.3 must be re-scored before any comparison.
4. **Part-1 folds differ per task.** The stratification key depends on the task, so only about 21 % of videos share a fold between 3.1 and 3.2/3.3. Part-1 OOF predictions are still valid as OOF, but they are not paired by fold across tasks.
5. **Near-duplicates are negligible.** pHash on 4 clips gives 4 pairs at Hamming distance ≤ 6 with ≥ 3 clips matching (2 of them same author), and no train–test pairs. Looser thresholds mostly catch dark or template frames whose labels disagree, so they were rejected.
6. **The VLM descriptions come from two models**: qwen3-vl 30b (1794 train / 556 test) and 235b-cloud (730 / 118, 90 % ES). YES rate is 49 % for 30b and 44 % for 235b. Weak, and confounded with language. W3 and B2 will use one model (30B) only.
7. **H5 clips cover little of long videos.** There are 4 clips × 16 frames, stride 4, which is 64 source frames (~2 s at 30 fps). The first clip starts at frame 0, the last ends at the final frame, and the gaps between them are equal (checked on 6 videos against the mp4). On a 55 s video the clips cover ~8 s. For low-fps or very short videos the clips overlap.

8. **VideoMAE attention biases were never loaded in part 1.** `MCG-NJU/videomae-base` stores the attention biases as `q_bias` / `v_bias`. Under transformers 5.6.2, installed in `exist_vt` on 2026-04-27 (before the paper's runs on 2026-05-09/10), they load as `UNEXPECTED`, and `query/key/value.bias` of all 12 encoder layers are reported `MISSING` and initialised afresh. The same warning is in part-1 logs written the same day by the same environment (`vlm_quad/logs/exist_kfold_3164{5,6,7}.err`, inference jobs of 2026-05-09). So the paper's video stream ran with re-initialised attention biases, and C0 reproduces that faithfully. The graph features (Step 6) load VideoMAE with the biases mapped in, and check that no `MISSING` keys remain.

## Folds

`outputs/folds.csv`: `outer_fold` (scheme passed to `s03`, currently `grouped`), `inner_for_outer_k` (4 grouped inner folds nested in each outer training part, -1 for the outer fold's own videos), `val_eligible` (False for the 16 ties), `part1_fold_{3_1,3_2,3_3}`, `grouped_fold`.

Part-1 task 3.1 folds are reproduced exactly: all 5 SHA256 values from DATA.md match.

Under `grouped`, a tie takes its author's fold so that this fold's model never trains on it. It never validates.

**Consequence:** if `grouped` is kept, C0 must be retrained (Step 4, 75 runs), because part-1 OOF predictions come from leaky folds. Part-1 OOF can still be re-scored as a reference.

## Evaluation (`src/evaluate.py`, `src/icm_fast.py`)

- ICM, ICM-Norm, ICM-Soft and ICM-Soft-Norm come from a vectorised port of PyEvALL that is **identical to PyEvALL to 1e-6**. It is checked by `evaluate.py selftest` on gold, prior, noisy and random predictions for all 3 tasks, on the contested subset, and with bootstrap weights against physically duplicated rows. PyEvALL itself takes ~10–20 s per call; the port allows a 10k-resample bootstrap in ~4 min per subset.
- Also reported: CE, Brier, class-wise ECE against the soft targets, macro AUC on hard labels, macro F1 and mean entropy. Each is given overall, for unanimous and contested videos, and for EN and ES.
- `compare` runs a paired **cluster** bootstrap: it draws authors, each with all of their videos, within each language. `--resample video` gives the plain per-video version.
- The evaluated population is the 2508 validatable videos. The 16 task-3.1 ties are train-only in every fold and are excluded for all tasks, so all three tasks score the same videos.
- `controls` builds confidence-matched graph-free controls from a base model (blend toward the fold's training prior, and temperature), each matched to the candidate's mean entropy. A graph gain counts only if it beats these controls and moves AUC.
- Reference points (3.1): gold 1.0; class prior 0.247; uniform random 0.246. For 3.3 the class prior is truncated to **0.0** ICM-Soft-Norm, so small 3.3 numbers are expected.

## Text–audio synchronisation (design for Step 5)

The organisers' text cannot be aligned as one block, because part of it is never spoken. Plan:

1. Extract audio as 16 kHz mono WAV from the mp4. Run VAD (Silero) to get a speech ratio per video and flag music-only videos.
2. Run our own ASR with word timestamps: faster-whisper large-v3, language from `lang`, then WhisperX wav2vec2 alignment for word times. Keep word probabilities.
3. Clean the organisers' text: drop the leaked prompt boilerplate and the `text` / `cleaned text` markers, collapse n-gram loops, and pull emoji names into a separate token list.
4. Split the cleaned text into sentences (6 texts lack punctuation; long sentences are split at ASR pauses > 0.7 s).
5. Anchor each sentence by fuzzy word-sequence matching against the ASR word stream (local alignment on normalised tokens, match ratio ≥ 0.6). A matched sentence gets the time span of its matched words. **An unmatched sentence is on-screen text**: it becomes an untimed `overlay` node linked to the hub, not to a moment.
6. Moments: 4 windows whose boundaries are the midpoints between consecutive H5 clip centres, so each window holds exactly one clip. A sentence goes to the window that contains most of its span.
7. Checks: per-video share of text matched, speech ratio, WER between our ASR and the organisers' text where both exist, and a hand check of 50 videos (Step 9).

The audio nodes for Step 23 (Whisper encoder per window) come from the same WAV.
