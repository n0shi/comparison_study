# Choices made once for every approach

Each choice below holds for W1–W6, B1–B5 and the controls, so that only the
graph changes between approaches. Open choices are marked **open** with the
evidence that will settle them.

## Fixed

| Choice | Value | Why |
|---|---|---|
| Folds | author-grouped, 5 outer, inner fold 1 for epoch selection (`folds.csv`) | C0's protocol; no author on both sides |
| Evaluation population | the 2508 `val_eligible` training videos, all three tasks | the 16 task-3.1 ties are never predicted out of fold |
| Network | HeteroConv(GATv2 per edge type), own-path Linear, residual, LayerNorm; hub + attention pool per node type + presence flags; 3-layer MLP head | `train.py`; C1 is the same network with every message empty |
| Losses | part 1's TaskLoss, class weights, Dirichlet / Beta, label smoothing from `configs/c0/task*.json` | identical to C0 |
| Tuning budget | 20 fixed trials per approach × task × outer fold (trial 0 = defaults), scored on inner fold 1; then 5 seeds (42–46) | Errica et al.; same candidates for every approach |
| W2 grounding rule | link a field to every moment within 0.25 × real-match gap of its best; none if the best is below the 95th percentile of a random video's best; thresholds from training features only | top-2 of 4 is decided by noise (clips have cosine 0.91 to each other; data_note 16) |
| W3 prompt | v4: every moment scored 0/1/2, edge = 2, no budget | v2/v3 budget caused a position bias |
| C4 | BGE description neighbours, k and prior weight m chosen on inner folds by soft CE | `c4_knn_vote.py` |
| **Text node encoder** | multilingual-e5-large, mean over tokens, "query: " prefix (setup v2) | linear probes on outer folds: e5 best on all three tasks (all text AUC 0.795 / 0.815 / 0.812 vs XLM-R 0.787 / 0.795 / 0.767); C1 tuning, inner loss e5 / BGE / XLM-R: 0.540 / 0.563 / 0.569 (3.1), 0.797 / 0.856 / 0.873 (3.2), 0.251 / 0.260 / 0.265 (3.3), larger than any gap between graph approaches |
| Node-type dropout | 0.15 per video; the dropped type's nodes are zeroed **and their edges removed** (setup v2) | in v1 the zeroed nodes still sent GATv2's bias |
| `feat_dropout` grid | {0, 0.2, 0.4, 0.6} (setup v2) | in v1, 0.4, the grid's edge, won 66 of 90 tuning runs |
| **Audio nodes** (Step 23) | none, for every approach | W2 with audio − W2, seed ensembles, author bootstrap: ICM-Soft-Norm −0.007\* / −0.017\* / −0.027\*, AUC −0.005 / −0.011\* / −0.019\* (3.1 / 3.2 / 3.3; on 3.1 W2 had 2 seeds then); inner loss also worse on all three tasks (0.553 vs 0.543, 0.820 vs 0.812, 0.257 vs 0.253) |

## Decision rules (written 2026-11-04, before the bootstrap of W3 and of the 5-seed 3.1 runs was read)

These rules settle every choice between approaches, so that no choice is fitted
to the numbers after they are seen.

- **Primary metric:** ICM-Soft-Norm of the seed ensemble on all 2508 videos,
  tasks 3.1 and 3.2. Task 3.3 is exploratory (its rarest categories have 7–11
  hard positives per fold), as in Step 30. Hard ICM-Norm, AUC, CE and the
  contested subset are reported, but do not decide.
- **Significance:** the paired author bootstrap of `graph_report.py`; a
  difference counts when its 95 % interval excludes 0.
- **An inside-video approach beats C1** when it is significantly better on at
  least one of 3.1 and 3.2 and not significantly worse on the other.
- **The base for the between-video approaches** is the cheapest approach that
  beats C1 (cost from `outputs/costs.jsonl` plus build cost: VLM hours count).
  If none beats C1, the base is C1. One base for all three tasks.
- **A between-video approach beats the base** by the same rule, against the
  base alone and against the late blend of the base with C4 (the base's OOF
  mixed with the kNN vote, the weight per outer fold chosen on the other four
  folds' OOF). An approach that does not beat the blend is reported as "the
  network adds nothing over averaging the neighbours' labels".
- **Ties go to the cheaper approach.** Order of cost: C1 < C3 ≈ W2 < W1/W5 <
  W4/W6 (learned or second graph) < W3 (VLM hours); C4 < B1 < B4 < B3 < B2.

### Applied to Step 24 so far (2026-11-04)

ICM-Soft-Norm, approach − C1, seed ensembles, all videos (3.1 / 3.2):

| Approach | 3.1 | 3.2 | Beats C1 |
|---|---|---|---|
| C2 | −0.022\* | −0.021\* | no |
| C3 | −0.002 [−0.005, 0.002] | +0.015\* [0.011, 0.020] | **yes** |
| W2 | −0.006\* | −0.007\* | no (worse on both) |
| W3 | −0.006\* | +0.009\* | no (worse on 3.1) |

**Base for B1–B5: C3.** It is the only approach that beats C1, and in the cost
order above only C1 is cheaper, so W1 and W4–W6 cannot replace it under these
rules; they are still run for Q1.

### Late checks on the base (2026-11-04, `late_checks.py`, 2000 resamples)

ICM-Soft-Norm, B − A, all videos (3.1 / 3.2 / 3.3):

| Check | 3.1 | 3.2 | 3.3 |
|---|---|---|---|
| C3 − C0 at C3's entropy (prior blend) | +0.020\* | +0.002 | −0.002 |
| C3 − C0 at C3's entropy (temperature) | +0.018\* | −0.000 | −0.001 |
| C3+C4 − C3 (late blend) | −0.008\* | −0.034\* | +0.009 |
| C3+C4 − C4 | +0.056\* | +0.075\* | +0.003 |
| C0+C4 − C0 (Step 28 control) | −0.028\* | −0.035\* | −0.007 |
| C3 T − C0 T (temperature per fold by soft CE) | +0.030\* | −0.005 | −0.006 |

- **C0's lead in ICM-Soft-Norm is confidence, not information.** Made as
  uncertain as C3, C0 is significantly worse on 3.1 and equal on 3.2 and 3.3;
  on the contested subset C3 is ahead on 3.1 and 3.2 (+0.11\*, +0.03\*).
- **The late blend lowers ICM-Soft-Norm on all videos** for 3.1 and 3.2 (it
  raises it on the contested subset), although its weight minimises soft CE:
  soft CE and ICM-Soft-Norm disagree. Chosen weights of C4: 0.10–0.15 for C3,
  0.35–0.40 for C0 on 3.1. The rule stays as written (a B must beat the base
  and the blend); on 3.1 and 3.2 beating the base now implies beating the blend.
- **Temperature fitted by soft CE** lowers ICM-Soft-Norm on 3.2 and 3.3 for both
  models (on 3.3 to ~0.01–0.02). Calibration for Step 27 must therefore be
  judged on ICM-Soft-Norm as well as CE, not on CE alone.

### Rules for W4 and B1, written before either is run (2026-11-04)

- **W4's k grid: {2, 3, 5, 8}**, 5 of the 20 trials each (trial j gets
  k = [3, 2, 5, 8][j mod 4]; the network settings of trial j are those of every
  other approach). A video has 9–13 non-hub nodes (median 12), so the grid runs
  from ~20 % of a node's possible partners to nearly all of them (C3). Which k
  the tuning picks is itself the answer to "does pruning help": if 8 wins in
  most folds, the learned scorer only reweights, as C3's attention already does.
- **W4-frozen** (exploratory, outside the main table): fold k's trained W4
  scorer fixes the edges of fold k's videos, and C3's network is trained on
  them with the same 20 trials. Run whatever W4 scores. It tells whether the
  learned edges are worth having in themselves (W4-frozen ≥ C3) or only while
  trained with the task (W4-frozen ≈ W2 or below).
- **B1's k grid: {5, 10, 20}**, not Step 16's {3, 5, 10}. C4 picked k = 20 in
  13 of 15 folds (10 in the others), so {3, 5, 10} would stop below the
  plain vote's optimum.
- **B1's label masking rate is tuned** (added 2026-11-04, before any B1 run):
  hide ∈ {0.5, 0.25, 0.75, 0} cycled with nbr_k ∈ {10, 5, 20} over the same 20
  trials (periods 4 and 3: all 12 pairs occur). hide = 0 (never hidden) tells
  whether UniMP masking is needed at all. B1N hides every label, in training
  and when predicting, and tunes nbr_k only.
- **B1's neighbour pool**, settled by C4 run with each pool
  (`c4_knn_vote.py --predict-pool inner1`): if predicting from inner split k.1
  instead of outer split k changes C4's ICM-Soft-Norm by less than 0.005 on
  3.1 and 3.2 (and not significantly), B1 uses inner split k.1 for training,
  epoch selection and prediction alike; otherwise it trains and selects on
  inner split k.1 and predicts from outer split k, as C4 does.
  **Result (2026-11-04):** C4 from inner split k.1 − C4 from outer split k,
  ICM-Soft-Norm all videos: −0.0073 (3.1), −0.0077 (3.2), −0.0143 (3.3);
  AUC −0.003 / −0.004 / −0.007. Above 0.005 on both tasks, so **B1 trains and
  selects epochs on inner split k.1 and predicts the outer fold and the test
  set from outer split k.** The pool at prediction is ~1/3 larger than in
  training; the neighbour count k is fixed, so only the neighbours get closer.

Setup v1 (XLM-R, soft node-type dropout, `feat_dropout` up to 0.4) is kept in
`runs/graphs` / `results/graphs` as the record of the encoder choice; every
comparison between approaches uses setup v2 (`runs/graphs_v2`).

### W4 and B1 by these rules (2026-11-05, 5 seeds, 2000 resamples)

ICM-Soft-Norm, seed ensembles, all videos (3.1 / 3.2; 3.3 exploratory):

| Comparison | 3.1 | 3.2 | 3.3 | Verdict |
|---|---|---|---|---|
| W4 − C1 | −0.009\* | +0.005\* | +0.012\* | does not beat C1 (worse on 3.1) |
| W4 − C3 | −0.007\* | −0.011\* | +0.003 | |
| B1 − C3 (base) | −0.001 | −0.014\* | +0.011\* | does not beat the base (worse on 3.2) |
| B1N − C3 | +0.002 | −0.017\* | +0.009 | does not beat the base |
| B1 − B1N | −0.003 | +0.004 | +0.002 | the neighbours' labels add nothing |

- No AUC difference against C3 is significant on 3.1 or 3.2. Neither B
  beats the base, so the blend comparison is not needed.
- W4's k varies across 3.1's folds (8, 3, 2, 8, 5), is 5 on 3.2 and 3 on 3.3;
  its scorer's scores are narrow (field–clip p5–p95 −0.10 to 0.14), and one
  seed is an outlier (3.1 seed 44: 0.456, the others 0.467–0.480).
- B1 and B1N pick nbr_k = 5, the grid's lowest value, in 21 of 30 folds:
  tuning moves towards fewer neighbours. The grid is not widened after the fact.
- **Q1 so far: C3.** W1, W5, W6 not run.

### Robustness check of the message function (written 2026-11-05, before any run)

Every approach is within ±0.015 of C1, so "edges do not help" could be
GATv2's doing. C1, C3 and W2 are run again with GATv2 replaced by a plain mean
of Linear(x_j) over a node's in-edges (`--conv mean`, GraphSAGE-mean without
its root term, which the own path already is) and by the same mean with each
message times the edge attribute (`--conv wmean`, cosine or 1, so C3 keeps its
similarity weights). No bias, so a relation without edges sends 0 and C1 stays
the network with every message empty. Same 20 trials, 5 seeds, metrics and
bootstrap; runs in `runs/graphs_v2_<conv>`.

- **Robust** if, under both, neither C3 nor W2 beats C1 by the rule above
  except where it does under GATv2 (C3 on 3.2).
- **Depends on the message function** if C3 or W2 beats C1 under mean or wmean
  where it does not under GATv2, or loses where it wins; reported as such.
- The base for B2–B5 stays C3 under GATv2 whatever this check shows; the
  check qualifies the conclusion, it does not re-choose the network.

### Encoder check: cheaper and lower-dimensional inputs (written 2026-11-05, before any run)

The tuning pushes `feat_dropout` to the grid's edge (0.6) and every graph is
within ±0.015 of C1. Is the network overfitting 768–1024-dimensional frozen
inputs on ~1600 training videos per fold, and would edges matter more with
smaller inputs? Exploratory: the main table stays on setup v2 (e5, VideoMAE).

- **Step 1, linear probes** (`text_probe.py`, the same nested protocol), all
  text sources: e5 (1024), e5 after one label-free PCA to 16 / 64 / 256,
  multilingual-e5-small (384), Model2Vec potion-multilingual-128M (256, static),
  TF-IDF of the same texts (char 2–5-grams, words 1–2) with SVD to 256; video:
  VideoMAE, VideoMAE PCA-64, SigLIP 2, and text + each video source. Every
  reduction (PCA, TF-IDF vocabulary, SVD) is label-free and fitted on all
  videos, train and test.
- **Step 2, graphs**, only if step 1 finds an input whose probe ICM-Soft-Norm on
  3.1 and 3.2 is within 0.01 of e5's (or better) with at most a quarter of the
  dimensions: C1 and C3 on it (s06 `pca64` files or the cheap encoder), same 20
  trials and 5 seeds. Read: C3 − C1 under the smaller input against C3 − C1
  under e5, and each against its setup-v2 run (`graph_report.py` compares an
  approach with its `tuned+e5` run).
- Step 1 alone settles nothing in the main table; step 2 qualifies the
  conclusion "edges do not help" as the message-function check does.

### Message-function check: result (2026-11-07) and setup v3 (written before any v3 run)

ICM-Soft-Norm, seed ensembles, all videos (3.1 / 3.2 / 3.3); point estimates
(the bootstrap of these pairs is in `results/conv_check_compare.csv`):

| | GATv2 | wmean | mean |
|---|---|---|---|
| C3 − C1 | −0.002 / +0.015 / +0.010 | +0.004 / −0.008 / −0.004 | +0.012 / +0.003 / 0.000 |
| W2 − C1 | −0.006 / −0.007 / +0.014 | +0.001 / −0.001 / −0.009 | +0.011 / −0.011 / −0.010 |
| **C1 − C1 (GATv2)** | 0 | −0.009 / +0.014 / +0.008 | −0.012 / +0.009 / +0.007 |

- **C1 has no edges, so its three runs are the same network** (wrong for
  GATv2 vs mean / wmean: the hub's member edges use the message function; see
  "Setup v3 noise floor" below). They differ
  only in initialisation (the unused message modules draw from the RNG) and
  so in which trial wins the tuning. Their gaps, up to 0.012 (3.1) and 0.014
  (3.2), are as large as any effect of a graph, and the author bootstrap
  marks them significant: it resamples videos, not training runs.
- The sign of every edge effect changes with the message function. C3's
  win on 3.2 (+0.015, the reason it became the base) is not reproduced.
- So setup v2's decisions between approaches are within training noise.

**Setup v3** (added 2026-11-07, before any v3 run):

- Each tuning trial is scored by its mean inner-fold-1 loss over 3 seeds
  (42, 43, 44; `train.py --tune-seeds 3`), so the winning trial depends less
  on one initialisation. Everything else as in v2: 20 trials, 5 seeds (42–46),
  seed ensembles, author bootstrap. Runs in `runs/graphs_v3`; v2 stays as the
  record.
- Re-run in v3: C1, C3, W2, W3, C2, W4, B1, B1N; and C1 with `--conv mean`
  and `--conv wmean` (`runs/graphs_v3_<conv>`), as the noise floor.
- **Noise floor** per task and metric: the largest |C1 replica − C1| over the
  two replicas in v3.
- **A difference counts** when its 95 % interval excludes 0 **and** it is
  larger than the noise floor of its task. The decision rules above (beats C1,
  the base, ties to the cheaper) are applied again on v3 with this test; the
  v2 tables are not used for decisions any more.
- If the v3 floor is still above 0.01 on 3.1 or 3.2, that is reported as the
  resolution limit of the comparison, not fixed by more runs.

### Setup v3 noise floor: result and a correction (2026-11-10, before any v3 approach comparison was read)

ICM-Soft-Norm, seed ensembles, all videos (3.1 / 3.2 / 3.3), 1000 resamples
(`results/conv_check_compare_v3.csv`; AUC differences all below 0.003):

| | v2 | v3 |
|---|---|---|
| C1 mean − C1 GATv2 | −0.012 / +0.009 / +0.007 | −0.012 / +0.004 / 0.000 |
| C1 wmean − C1 GATv2 | −0.009 / +0.014 / +0.008 | −0.009 / +0.004 / +0.005 |
| C1 mean − C1 wmean | −0.003 / −0.005 / −0.001 | −0.003 / 0.000 / −0.005 |

- **Correction to the section above:** C1 has no edges between content
  nodes, but it keeps the member edges to the hub, and those go through the
  message function. So C1 under GATv2 and C1 under mean / wmean are not the
  same network: the hub attends over its members in one and averages them in
  the other. Only C1 mean and C1 wmean are the same function (member edges
  carry attribute 1, so the weighting does nothing; same initialisation), so
  their gap is the only pure run-to-run noise: at most 0.005 in v2 and in v3.
- GATv2's C1 is ahead on 3.1 by the same 0.009–0.012 in both setups, so that
  gap is most likely a real effect of the hub's aggregation, not noise.
- 3-seed tuning halved the GATv2-vs-mean gaps on 3.2 (0.014 → 0.004) and left
  3.1 unchanged.
- **Floor applied (the rule as written):** 0.012 / 0.004 / 0.005. 3.1's floor
  is above 0.01, so 0.012 is reported as the resolution limit on 3.1. The pure
  replica gap (≤ 0.005 on every task) is reported next to every decision as a
  sensitivity check; it does not replace the rule, since it was found after
  the rule was written and comes from one pair.

### Sharpness control (written 2026-11-11, before any v3 comparison between approaches was read)

Found on setup v2's seed ensembles (exploratory, one temperature for all
videos, not cross-fitted):

- **ICM-Soft-Norm rewards sharper predictions.** Dividing the logits by a
  temperature T < 1 changes neither the model nor the ranking, yet C1 goes
  from 0.480 (T = 1) to 0.557 (T = 0.5) and 0.596 (T = 0.1) on 3.1, and from
  0.357 to 0.442 and 0.475 on 3.2. Soft cross-entropy, a proper score, is
  best at T = 1 (0.520 on 3.1) and worse at every sharper T. So the models
  are calibrated, and the metric prefers them over-confident: +0.10, against
  graph effects of at most 0.015.
- **At equal sharpness the approaches come together.** With every model
  tempered to one mean entropy, the nine runs (C1 / C3 / W2 × GATv2 / wmean /
  mean) span 0.007 on 3.1 and on 3.2, against 0.012 and 0.023 raw. C3's raw
  +0.015 over C1 on 3.2 (why it became the base) becomes −0.0004: C3 was only
  sharper (mean entropy 0.737 against 0.780).
- The linear probe on e5 (0.521 raw on 3.1) is above every graph network raw
  and below them at equal sharpness, for the same reason.

**Rule, for every comparison from here on (v3 and later):**

- Each difference in ICM-Soft-Norm is read twice: **raw**, as before, and
  **sharpness-controlled**: every approach's seed ensemble is tempered with a
  temperature chosen by cross-fitting (for outer fold k, the grid value that
  maximises ICM-Soft-Norm on the out-of-fold predictions of the other four
  folds; grid 0.02–1.5; `sharp_check.py`). Hard decisions are not changed.
- **A difference counts** when it passes the v3 test (95 % interval excludes 0
  and larger than the task's noise floor) **in both readings**.
- Soft cross-entropy, Brier and AUC are reported next to every comparison as
  scores that sharpening cannot raise; they describe, they do not decide.
- ICM-Soft-Norm stays the primary metric (it is EXIST's), and its raw value is
  what is reported against other systems.
- The noise floor of the sharpness-controlled reading is computed the same
  way (largest |C1 replica − C1| in v3, tempered; `--replicas`).
- Checked on v2 before the rule was applied (`results/graphs_v2/sharp_*.csv`,
  200 resamples): the chosen temperatures are 0.05–0.2, so the controlled
  reading is close to scoring hard decisions and its intervals are wider.
  Raw ICM-Soft-Norm and cross-entropy can disagree in sign: on 3.1 C0 quad is
  +0.009 over C1 raw, and worse in cross-entropy (+0.026) and Brier.

### W1, W5, W6 settings (written 2026-11-07, before any run)

Same nodes' features, network, losses, 20 trials and 5 seeds as every
approach (setup v3 if the noise check keeps it, else v2); only the graph and
what it needs change (`graphs_within.py`, `train.py`).

- **W1, modalities as nodes:** fields, speech, clips each become one node, the
  mean of their vectors (the fields keep the mean visibility feature; a video
  without speech has no speech node), plus the organisers' text and the hub.
  Every pair of modalities is linked by its own relation (new: `with_org` for
  the three pairs with the organisers' text); grounds / paraphrases carry the
  mean cosine of the pairs W2 scores, the rest 1.
- **W5, hypergraph, as a star expansion:** a hyperedge is a node (type `hedge`,
  a learned vector per position, pooled in the readout like any node type).
  Hyperedges 0–3: moment m's clip, its speech node and the fields W2 grounds
  to that clip (attribute: their SigLIP cosine); hyperedge 4: the four fields.
  No pairwise relations besides member edges to the hub, so two members reach
  each other in two layers (through their hyperedge); the layer count is tuned
  as for everyone (1–3).
- **W6, dual-stream:** W2's graph unchanged, plus a moment graph on the clip
  nodes: `mnext` (consecutive moments) and `msim` (pairs whose VideoMAE clips
  are more similar than the video's median pair; attribute: that cosine).
  After the main layers each moment's content vector is its clip plus its
  speech; one GATv2 layer (with self-loops) over the moment graph and a linear
  score give one weight per moment (softmax within the video); the weighted
  sum of the content vectors is added to the readout. One layer, not tuned.
- **Fix found while testing (2026-11-07):** attention pooling passed the number
  of nodes, not of videos, as softmax's group count. Values are the same (it
  only sizes a buffer), so earlier runs stand, but a batch with fewer nodes of
  a type than videos crashed; W1 (at most one speech node per video) needs it.
- Checked on synthetic graphs: C1, C2, C3, W2, W3, W4 give graphs identical
  to the previous code, and C1, C3, W2, W4 identical initial weights and
  outputs; W1 / W5 / W6 train with 1 and 3 layers, W6's weights sum to 1.

### Setup v3 read by the sharpness rule (2026-11-12, `sharp_check.py`, 1000 resamples)

All 11 approaches, 5 seeds, all videos; `results/graphs_v3/sharp_compare.csv`.

**Noise floors.** Raw: 0.012 / 0.004 / 0.005 (3.1 / 3.2 / 3.3, as written
above). Sharp, from the three C1 replicas: 0.001 / 0.004 / 0.003. The raw gap
of 0.012 between C1's replicas on 3.1 is 0.0001 at matched sharpness, with
equal cross-entropy, Brier and AUC.

**A limit of these floors, found now.** C1's only edges are member edges with
attribute 1, so `mean` and `wmean` are the same network with the same seeds:
on 3.2 their predictions are identical (difference 0.0000). That pair measures
run-to-run nondeterminism only, and GATv2 against `mean` is another network.
Neither is a retraining of one network with other seeds, so the floors are
lower bounds. A replica of C1 with seeds 47–51 would measure it.

**What counts** (interval excludes 0 and above the floor, in both readings):

| task | difference | raw | sharp | cross-entropy | AUC |
|---|---|---|---|---|---|
| 3.1 | none | | | | |
| 3.2 | W4 − W2 | +0.0118\* | +0.0155\* | −0.0138\* | +0.0053\* |
| 3.3 | B1 − C1 | +0.0236\* | +0.0198\* | +0.0008 | +0.0033 |
| 3.3 | B1 − C3 | +0.0252\* | +0.0306\* | −0.0026\* | +0.0035 |
| 3.3 | B1 − B1N | +0.0095\* | +0.0127\* | +0.0006 | +0.0033 |
| 3.3 | B1N − C3 | +0.0156\* | +0.0179\* | −0.0031\* | +0.0002 |
| 3.3 | C2 − C1 | −0.0173\* | −0.0130\* | +0.0039\* | −0.0162\* |

- **No approach beats C1 (no edges) on 3.1 or 3.2.** Against C1, 13 of the 20
  raw differences on 3.1 and 3.2 have an interval excluding 0, all negative;
  at matched sharpness 2 do, and one of them (W4 on 3.2: −0.0055\* raw,
  +0.0141\* sharp) changes sign.
- **W1 on 3.1** is below C1 in both readings (−0.0055\*, −0.0114\*) but under
  the raw floor, so it does not count.
- **On 3.2 most edge sets are worse than no edges by the descriptive scores.**
  Cross-entropy against C1: W2 +0.013\*, W6 +0.019\*, C2 +0.018\*, W1 +0.010\*,
  W5 +0.010\*, B1 +0.014\*; AUC lower for W2, W5, W6, C2. C3 and W4 equal C1.
  W4 beats W2 by returning to C1's level, not by passing it.
- **C2 on 3.2**, which held in both readings in setup v2, does not here
  (−0.0234\* raw, −0.0096 sharp); its cross-entropy and AUC are still worse.
- **C0 quad against C1:** higher raw ICM-Soft-Norm on all three tasks
  (+0.010\*, +0.017\*, +0.017\*), not at matched sharpness; worse cross-entropy
  and Brier on 3.1 and 3.3, lower AUC on 3.1 and 3.2. C0 is the sharper model
  (entropy 0.48 against 0.53 on 3.1).
- **3.3 is exploratory**, and its chosen temperatures vary by fold (0.02–0.5).
- The temperature is at the grid's lower edge (0.02) in some folds (W3 on 3.1
  and C0 quad on 3.2: 3 of 5). The grid was not changed after reading.

### Seed replica of C1 (written 2026-11-12, before it is run)

- C1 is trained again with seeds 47–51 and setup v3's own tuning file, in
  `runs/graphs_v3_rep` (`cluster/rep_c1.sh`). Nothing else changes.
- The noise floor of each reading and task becomes the larger of the floor
  above and |C1 replica − C1| in that reading.
- The differences listed as counting above are read again against the new
  floors. One that falls under a floor is reported as not resolved.
- B1's differences on 3.3 use the same floors; a replica of B1 is run only if
  one of them is within twice the floor.

### Seed replica of C1: result (2026-11-14, 1000 resamples)

C1 with seeds 47–51 minus C1 with seeds 42–46, the same network and tuning:

| task | raw | sharp | cross-entropy | Brier | AUC |
|---|---|---|---|---|---|
| 3.1 | −0.0029\* | −0.0011 | +0.0016 | +0.0016 | −0.0021 |
| 3.2 | −0.0063\* | −0.0023 | +0.0074\* | +0.0050\* | −0.0039\* |
| 3.3 | −0.0047 | −0.0055 | +0.0009 | +0.0002 | +0.0025 |

- **The author bootstrap calls a pure seed change significant** on 3.2 in four
  of the five scores, and in raw ICM-Soft-Norm on 3.1. It resamples videos,
  not trainings.
- **Floors by the rule** (the larger of the old floor and this replica):
  raw 0.012 / 0.0063 / 0.005, sharp 0.001 / 0.004 / 0.0055. One replica is
  one draw of the noise, so a difference under twice the floor is fragile.
- **Every difference listed as counting still counts.** W4 − W2 on 3.2
  (+0.0118 raw, +0.0155 sharp) is under twice the raw floor. B1 − B1N on 3.3
  (+0.0095, +0.0127) is about twice the floors. B1 − C1, B1 − C3, B1N − C3 and
  C2 − C1 on 3.3 are 2.4 to 6 times the floors.
- **The descriptive line on 3.2 is weaker than written above.** Seed noise in
  cross-entropy there is 0.0074, so W1 and W5 (+0.010) are within 1.4 times
  it and W2, B1 (+0.013, +0.014) within 2 times; only W6 and C2 (+0.019,
  +0.018) are clearly above. C1's first five seeds are the better draw on 3.2.
- On 3.3 the replica's temperature is at the grid's edge in 2 folds, so the
  sharp reading there is the less stable one.

### k sweep of B1: result (2026-11-14, 1000 resamples, `k_sweep_report.py`)

B1 with `nbr_k` fixed, minus C1 (raw / sharp ICM-Soft-Norm, cross-entropy, AUC):

| nbr_k | 3.1 | 3.2 | 3.3 |
|---|---|---|---|
| 1 | +0.001 / −0.002 | −0.017\* / −0.013; CE +0.030\*, AUC −0.015\* | +0.005 / −0.007 |
| 3 | +0.004 / −0.001 | −0.007\* / +0.003; CE +0.016\*, AUC −0.007\* | +0.015\* / +0.008 |
| 5 | −0.002 / +0.001 | +0.006\* / −0.001; CE +0.006 | +0.023\* / +0.020\* |
| 10 | +0.002 / −0.005 | −0.001 / +0.002; CE +0.004 | +0.015 / +0.008; CE −0.003\*, AUC +0.007\* |
| 20 | −0.000 / −0.004 | −0.002 / +0.005; CE +0.006 | +0.005 / −0.011\*; AUC +0.007\* |

- **3.1: flat.** No k differs from C1 in any score.
- **3.2: too few neighbours hurt.** One neighbour per source is worse than no
  neighbours in three scores, by 4 times the seed noise in cross-entropy;
  from 5 on, B1 is at C1's level and never above it.
- **3.3: B1's gain is a narrow peak at `nbr_k` = 5**, the value tuning chose
  (26 of 30 folds), so "B1 tuned" and `nbr_k` = 5 are nearly the same runs.
  At 3 and 10 the gain is about a third in the sharp reading and inside
  1.5 floors; at 1 and 20 it is gone or negative. AUC and cross-entropy
  instead improve with k up to 10–20. B1's result on 3.3 stands by the rule
  but is not robust to k.

### Rules for B3, B4 and the k sweep (written 2026-11-12, before any of them is run)

Setup v3 throughout (3 seeds per tuning trial, 5 seeds, base C3, B1's
neighbour pools and label masking). B2 and B5 are not run.

- **B3, learned links.** Candidates: the top 50 of each of the 3 sources (all
  that `s07` stores). The scorer is a weighted cosine with 4 weight vectors
  between the video's vector and the neighbour's, each encoder's part
  L2-normalised; at the start it is the mean of the two plain cosines. Each
  video keeps its `keep_k` best; grid {5, 10, 20}. Masking rate as B1.
- **B4, theme hyperedges.** k-means on the BGE-M3 vectors of TARGET of the
  split's training videos; grid of clusters {16, 64, 256}. A cluster is one
  node: its centroid and the mean soft label of its members outside the
  video's own author / duplicate group. **B4F** has one such node for each of
  TARGET, MESSAGE, SITUATION, PEOPLE.
- **Compared with:** B3 against C3 and B1; B4 against C3; B4F against C3 and
  B4; all against C1. Read by the sharpness rule, with the floors of the seed
  replica.
- **k sweep (descriptive, outside the main table).** B1 with `nbr_k` fixed to
  1, 3, 5, 10, 20 in every fold (per source, so up to 3 `nbr_k` neighbours),
  the other settings as tuned, 5 seeds, in `runs/graphs_v3_k`. If B3 is tuned:
  `keep_k` fixed to 3, 5, 10, 20, 40. Reported as a curve of ICM-Soft-Norm
  (raw and sharp), cross-entropy and AUC against k. It does not choose k: the
  main table keeps the k tuned on inner folds.
- Checked on synthetic features before the first run: B1 gives the same
  predictions as before the change; a video's own label never reaches its
  theme node; B3 keeps exactly `keep_k` neighbours, the best by score, and
  the scorer receives a gradient.

## Open

| Choice | Candidates | Settled by |
|---|---|---|
| **Best inside-video approach** (Q1) | C3 so far; W1, W4–W6 not run | Step 24 with all W, by the rules above |

## Known limits (same for every approach)

- **C3 is not structure-free.** Its field–clip and field–speech edges carry the
  cosine as an edge feature, so attention can reweight them: C3 is "all pairs,
  similarity as weight", a soft W2.
- **Inner-fold losses are optimistic.** Tuning and epoch selection share inner
  fold 1; only outer folds are used for conclusions.
- **`feat_dropout` 0.6 is again the grid's edge.** It won 69 of 90 tuning
  runs in setup v2. The grid is not widened: all approaches share it, and
  widening it would mean re-tuning every approach.
- **Message passing has little room inside a video.** About 13 nodes, the hub
  links all of them (every node sees every other in 2 hops), the 4 clips of a
  video have cosine ~0.91 to each other, and 1 layer won 44 of 90 tuning runs.
- **ICM-Soft-Norm is clipped at 0.** On task 3.3 the graphs score below the
  baseline and show 0; read ICM-Norm, AUC and CE there as well.
