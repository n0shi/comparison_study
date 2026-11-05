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
