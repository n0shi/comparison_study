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

Setup v1 (XLM-R, soft node-type dropout, `feat_dropout` up to 0.4) is kept in
`runs/graphs` / `results/graphs` as the record of the encoder choice; every
comparison between approaches uses setup v2 (`runs/graphs_v2`).

## Open

| Choice | Candidates | Settled by |
|---|---|---|
| **Base for B1–B5** | C1, C3, W2, W3 (W1, W4–W6 when they finish) | the decision rules above, on the Step 24 bootstrap |

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
