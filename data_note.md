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
6. **The VLM descriptions come from three sources, split unevenly between train and test.** qwen3-vl:30b with prompt v2 (1794 train / 0 test), qwen3-vl:30b with prompt v2_5sections_fix1 (0 / 556), and qwen3-vl:235b-cloud with fix1 (730 / 118, 90 % ES). DATA.md's claim of a uniform prompt version is wrong. A character n-gram classifier tells 235b from 30b text with AUC 0.998 (ES) and 0.987 (EN); 235b fields are about 25 % longer. Within ES, the YES rate is 44.7 % for 235b and 52.3 % for 30b. The frame sampler also differed by source: 18–24 % of 30b videos exceed the adaptive frame cap, while 0 % of 235b videos do (median 10 frames vs 7). Re-running the current sampler reproduces the stored timestamps for only 7 of 16 sampled videos, so the old frames cannot be recovered. 83 % of the test descriptions (30b + fix1) come from a source with no training example. → All 3198 descriptions are regenerated with one sampler, one model and one prompt: `src/vlm_frames.py` (part 1's sampler, verbatim, in `src/vlm/`) and `src/vlm_describe.py` (Qwen3-VL-30B-A3B-Instruct through vLLM, part 1's fix1 prompt and generation settings). C0 and every graph use the regenerated descriptions.
7. **H5 clips cover little of long videos.** There are 4 clips × 16 frames, stride 4, which is 64 source frames (~2 s at 30 fps). The first clip starts at frame 0, the last ends at the final frame, and the gaps between them are equal (checked on 6 videos against the mp4). On a 55 s video the clips cover ~8 s. For low-fps or very short videos the clips overlap.

8. **VideoMAE attention biases were never loaded in part 1.** `MCG-NJU/videomae-base` stores the attention biases as `q_bias` / `v_bias`. Under transformers 5.6.2, installed in `exist_vt` on 2026-04-27 (before the paper's runs on 2026-05-09/10), they load as `UNEXPECTED`, and `query/key/value.bias` of all 12 encoder layers are reported `MISSING` and initialised afresh. The same warning is in part-1 logs written the same day by the same environment (`vlm_quad/logs/exist_kfold_3164{5,6,7}.err`, inference jobs of 2026-05-09). So the paper's video stream ran with re-initialised attention biases, and C0 reproduces that faithfully. The graph features (Step 6) load VideoMAE with the biases mapped in, and check that no `MISSING` keys remain.
9. **Part 1 selects its checkpoint on the fold it reports.** Each fold's best epoch (by ICM-Soft-Norm) and early stopping are decided on that fold's validation set, and the OOF predictions come from the same set, so part-1 CV scores are optimistic. C0 here uses a nested protocol: select on an inner fold of the training part, then predict the outer fold once (`src/c0_part1_folds.py`).
10. **Part 1's saved checkpoints are not the selected model.** Validation runs with EMA weights (`apply_shadow`), but `best_model.pt` is written after `restore`, i.e. with the raw weights. The paper's test predictions, made from `best_model.pt`, therefore came from a different model from the one selected. C0 predicts the outer fold and the test set inside the evaluation step, with the EMA weights that were scored.

11. **Much of the organisers' `text` is not a transcript but an LLM's account of the video.** In the 40-video review, the organisers' sentences that do not occur in the speech were judged to be mostly an LLM describing or interpreting the video, rather than on-screen text. Some are explicit judgements, e.g. 120393: "ser machista en 2021 es una actitud desactualizada y perjudicial…". On some videos the speech is almost absent from the text: in 220037, 92 % of the video is speech (216 ASR words), but only 6 % of them are found in the organisers' text by sentence alignment. The field is official input, also given at test time, so this is not leakage in the strict sense. It is, however, interpretation rather than observation, and the Step 32 deletion test of the organisers'-text node measures how much the models rely on it.

## Regenerated VLM descriptions (2026-10-24)

All 3198 videos were redone with one model, prompt and sampler (problem 6): Qwen3-VL-30B-A3B-Instruct in bf16 through vLLM 0.11, using the paper's `v2_5sections_fix1` prompt and frame sampler. The transcript slot gets our ASR where reliable (1791 train / 394 test), otherwise the organisers' cleaned text. All descriptions are status ok, none retried; 14 of ~26 k frames are blank (fades). Output: `training_vlm_v3.json` / `test_vlm_v3.json` (config keys `train_json_v3` / `test_json_v3`).

- **Length:** the new descriptions are longer. The four fields the classifier reads (TARGET, MESSAGE, SITUATION, PEOPLE) have a median of 348 XLM-R tokens (old: 252). 23 % exceed part 1's 384-token limit, 0.4 % exceed 512, so C0 reads up to 512 tokens. The graphs encode each field separately and are unaffected. 0.35 sections per video exceed the prompt's sentence limits (old: 0.01).
- **Signal** (`src/vlm_check.py`: TF-IDF + logistic regression on the four fields, OOF on our folds, author bootstrap, 1000 resamples; paper → new):

| task | ICM-Soft-Norm | AUC | CE |
|---|---|---|---|
| 3.1 | 0.398 → 0.408 (+0.010 [+0.002, +0.017]) | 0.797 → 0.802 (n.s.) | 0.565 → 0.558 (n.s.) |
| 3.2 | 0.264 → 0.279 (+0.015 [+0.009, +0.021]) | 0.814 → 0.819 (n.s.) | 0.787 → 0.771 (better) |
| 3.3 | 0.051 → 0.058 (+0.006 [+0.003, +0.010]) | 0.776 → 0.771 (n.s.) | 0.325 → 0.322 (better) |

  No label information was lost; the soft metric improves on all three tasks.
- **Shift** (train vs test separability, 5-fold AUC). The reference is the organisers' text, which one pipeline produced for both splits, so it reflects content alone: 0.624. Paper descriptions: 0.640. New descriptions: 0.667. The increase is not driven by the transcript source: it holds within reliable-ASR videos (0.617 → 0.646) and within the rest (0.669 → 0.686). The separating words are topical (test: fashion, dating, relationship, text overlay, voiceover; train: sexual, family, pareja, estereotipos), and no source signature remains since there is one source. **The test set differs in topic from train**, and the more detailed descriptions show it, so test scores may sit below CV.
- Examples: in 220037 the organisers' text is garbled OCR. The old description ("a fake method … absurd criteria like 1j7302") followed it; the new one, made from our ASR, gives the actual argument about male validation. Occasional over-interpretation remains: in 120127 the new TARGET calls a sincere feminist speech possibly a parody.

## Speech recognition (decided 2026-10-23)

**Bake-off** (`src/asr_bakeoff.py`, 30 videos, 15 EN + 15 ES, strata music / dense / random; results in `outputs/asr/bakeoff_*.csv`). Whisper large-v3 + WhisperX was compared with Parakeet-TDT-0.6B-v3:
- On clear speech the two agree on 90–100 % of words.
- Whisper with a forced language translated foreign speech and lyrics ("sir, are you there?" → "Señor, ¿estás ahí?") and produced classic silence phrases ("Thanks for watching!", "¡Suscríbete!").
- Parakeet stayed quiet on silence, but misidentified the language of ordinary Spanish speech (120986) and output an English pseudo-translation, a failure that cannot be filtered out.

→ **Whisper**, with language ID on the VAD speech (record language when the probability is below 0.7) and filters that drop segments outside VAD speech, with high no-speech probability, looping, or a known silence phrase where VAD finds little speech. Re-run on the same videos: the translations and silence phrases are gone. One Portuguese song remains translated (language probability 0.59).

**Review** (`src/asr_review.py`, 40 videos, listened to by the author). Decisions:
- the organisers' text is kept whole as one untimed document (problem 11), not split into speech and overlay;
- our ASR is trusted when it has ≥ 5 timed words and ≥ 20 % speech (29 of 40 reviewed videos); otherwise there is no timed speech, and the organisers' text is the only text;
- every video gets 4 moments, one H5 clip each, with boundaries moved to the nearest pause within 1 s (`src/s05_moments.py`);
- the VLM transcript slot gets our ASR when reliable, otherwise the organisers' cleaned text (as in the paper); the OCR slot stays empty.

**Full run** (all 3198 videos). One mp4 has a truncated audio track (220089, 1 broken AAC packet of 182). It is decoded around the broken packet, and the count is recorded as `audio_bad_packets`; no other video has broken audio.
- Reliable ASR: train 1791 / 2524 (71 %), test 394 / 674 (58 %).
- Reliable videos have speech in 3.52 of 4 windows on average; all 4 windows are filled in 69 %. 70 % of window boundaries moved to a pause.
- The organisers' text contains a median 95 % of the recognised speech, and less than 30 % in only 4 % of reliable videos.
- Whisper switched away from the record language in 10 % of ES train videos, mostly to English (songs, quoted speech).
- Outer folds hold 69–73 % reliable videos each.

**The train/test gap is in the videos, not in the ASR.** It is entirely in English: reliable ASR is 72 % (train) vs 52 % (test) in EN, and 70 % vs 66 % in ES. The main cause is videos with no speech at all (< 5 % VAD): 30 % of test vs 19 % of train. The VAD speech share of EN videos is 0.37 (test median) vs 0.65 (train). The organisers' text agrees independently: the EN test text is shorter (46 vs 64 words), and only 33 words for no-speech test videos. The language mix explains little (58.5 % → 60.5 % at the train mix). In train, videos with speech are more often sexist (YES 50 % vs 41 %). **→ Test results are reported separately for videos with and without reliable speech.**

12. **The paper's text and the configs that actually ran differ.** The paper states 2 fusion layers, the bottom 8 VideoMAE layers frozen, and learning rates 1e-5 / 2e-5 / 5e-5 / 1e-4 (text / video / physio / head), with checkpoints selected by validation loss. The runs behind the submissions (`configs/c0/`) used 1 fusion layer for 3.1 and 3.3 (2 for 3.2) and 6 frozen VideoMAE layers. Their learning rates were 5e-6 / 5e-6 / 5e-5 / 5e-5 (3.1) and 8e-6 / 8e-6 / 1e-5 / 1e-5–1.5e-5 (3.2, 3.3). They selected checkpoints by ICM-Soft-Norm, with 18 frozen XLM-R layers (16 for 3.3), mixup only in 3.2, and a 3.3 hard threshold of 0.4. C0 uses what ran, not what the paper says.
13. **Selecting by ICM-Soft-Norm breaks on 3.3, and part 1's patience stops training too early.** The inner ICM-Soft-Norm on 3.3 is 0 on every epoch (truncation), so the first, still-frozen epoch was kept: OOF AUC 0.57–0.72, against ~0.81 in part 1. Part 1's EMA is rebuilt when the backbones unfreeze, applied after 500 updates, and lags ~1000 updates. On 1520 training videos (~190 updates per epoch) the validation loss stays flat for ~6 epochs, and a patience of 4–6 stops training first. C0 therefore selects by inner validation loss with patience 10 for all tasks. The earlier runs are kept in `runs/C0_select_icm`. On 3.1 that ICM-selected run also preferred late, overconfident epochs: ICM-Soft-Norm 0 on contested videos, and quad beat trio by +0.054 ICM-Soft-Norm at equal AUC.
    With selection by inner loss, the 3.1 run (quad, seed 42) still scores ICM-Soft-Norm 0 on contested videos, while its contested AUC is 0.605. This is calibration, not a bug. Shrinking its probabilities towards 0.5 (q = 0.5 + λ(p − 0.5)) leaves AUC unchanged. On contested videos it raises ICM-Soft-Norm from 0 (λ = 1) to 0.167 (λ = 0.75) and 0.327 (λ = 0.5). On all videos it lowers it from 0.486 to 0.425 and 0.369. 38 % of contested videos get a probability above 0.83, where the gold never exceeds 5/6. So a model that is merely less confident gains on contested videos. Graph models are therefore compared there against the confidence-matched controls of `evaluate.py controls`, not against raw C0.
14. **The paper's numbers are test scores of a five-model ensemble; C0's CV scores are OOF of one model.** On the test set each video is the mean of the five fold models. Out of fold it is predicted by one model only, so OOF ICM-Soft-Norm is lower wherever averaging would soften the probabilities. Quad, seed 42, OOF scores and the paper's test scores (ICM-Soft-Norm / ICM-Norm / F1):

    | task | OOF | paper test |
    |---|---|---|
    | 3.1 | 0.486 / 0.609 / 0.736 | 0.546 / 0.656 / 0.770 |
    | 3.2 | 0.382 / 0.561 / 0.653 | 0.385 / 0.545 / 0.636 |
    | 3.3 | 0.072 / 0.375 / 0.364 | 0.130 / 0.455 / 0.419 |

    C0 is compared with the paper only on the test set, through the same fold ensemble (`test_mean.csv`, see `results/C0/test_scores.csv`).

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

## Text–audio synchronisation (original design, superseded by "Speech recognition" above)

The organisers' text cannot be aligned as one block, because part of it is never spoken. Plan:

1. Extract audio as 16 kHz mono WAV from the mp4. Run VAD (Silero) to get a speech ratio per video and flag music-only videos.
2. Run our own ASR with word timestamps: faster-whisper large-v3, language from `lang`, then WhisperX wav2vec2 alignment for word times. Keep word probabilities.
3. Clean the organisers' text: drop the leaked prompt boilerplate and the `text` / `cleaned text` markers, collapse n-gram loops, and pull emoji names into a separate token list.
4. Split the cleaned text into sentences (6 texts lack punctuation; long sentences are split at ASR pauses > 0.7 s).
5. Anchor each sentence by fuzzy word-sequence matching against the ASR word stream (local alignment on normalised tokens, match ratio ≥ 0.6). A matched sentence gets the time span of its matched words. **An unmatched sentence is on-screen text**: it becomes an untimed `overlay` node linked to the hub, not to a moment.
6. Moments: 4 windows whose boundaries are the midpoints between consecutive H5 clip centres, so each window holds exactly one clip. A sentence goes to the window that contains most of its span.
7. Checks: per-video share of text matched, speech ratio, WER between our ASR and the organisers' text where both exist, and a hand check of 50 videos (Step 9).

The audio nodes for Step 23 (Whisper encoder per window) come from the same WAV.
