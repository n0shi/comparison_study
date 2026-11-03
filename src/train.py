"""Step 22: one training script for every graph approach (first version: inside a video).

Only the graph changes between approaches (graphs_within.py builds W2, C1-C3
from the same frozen features); network, losses, protocol and tuning budget
are shared.

Network
  input   per node type: dropout on the frozen features, Linear + embedding of
          the node's position (which field / which moment) -> LayerNorm -> d;
          the hub is a learned vector
  layers  L x [GATv2 per relation (edge attribute: cosine or 1), mean over
          relations] + each node's own Linear, residual, LayerNorm. The own
          path keeps a node type alive when no edge reaches it, so C1 (no
          edges) is the same network with every message empty.
  readout hub vector, an attention-pooled vector per node type and a presence
          flag per node type (a video without speech has no speech nodes)
  head    3-layer MLP, as in the paper
  regularisation  dropout; during training each video drops one node type
          with probability TYPE_DROP (the plan's Step 22): its nodes are zeroed,
          their edges removed (so they send no message, not even GATv2's bias)
          and that type's presence flag cleared

Setup v2 (choices.md): e5 text nodes by default, the edge removal above and
feat_dropout up to 0.6; runs go to runs/graphs_v2. The XLM-R runs of setup v1
(soft node-type dropout, feat_dropout up to 0.4) stay in runs/graphs.

Losses and selection are part 1's own code, imported from --part1-root (never
copied): TaskLoss (soft CE + Dirichlet for 3.1 / 3.2, BCE + Beta with
pos_weight for 3.3, label smoothing), class weights from the training part,
and the weighted validation loss C0 selects by. Dirichlet alpha, smoothing
and class weighting come from configs/c0/task*.json, as for C0.

Protocol (C0's, nested): for outer fold k, train on the outer training part
minus inner fold 1, keep the epoch with the lowest loss on inner fold 1
(patience --patience), then predict outer fold k and the test set once.

Tuning (the plan's budget, Errica et al.): --tune 20 trains the same 20
hyperparameter sets (TRIALS: drawn once with a fixed seed, so every approach,
task and fold gets identical candidates) on each outer fold's training part
and scores each by its best inner-fold-1 loss; the outer fold is never
predicted. The winners per fold go to runs/graphs_v2/<approach>/tuning_<task>_<text>.json.
A normal run with --hyper tuned then uses each fold's winner; run it with
5 seeds.

Outputs in runs/graphs_v2/<approach>/<approach>_<task>_seed<seed>[_<tag>]/:
oof_outer.csv, test_fold{k}.csv, test_mean.csv (all 5 folds), manifest.json;
cost to outputs/costs.jsonl. Scores are printed next to C0 and C4 on the same
videos.

Usage:
    python src/train.py --config config.perun.yaml --part1-root ... --approach W2 --task 3_1 --tune 20
    python src/train.py ... --approach W2 --task 3_1 --seed 42 --hyper tuned
    python src/train.py ... --approach W2 --task 3_1 --folds 1      # defaults, one fold (first slice)
    python src/train.py ... --smoke                                 # 300 videos, 2 epochs
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as ev
import graphs_within as gw
from common import TASKS, load_config, require, setup_stdout

ES_INNER = 1
POS_TYPES = {"field": 4, "speech": gw.N_WIN, "clip": gw.N_WIN, "audio": gw.N_WIN}
POOLED = ["field", "speech", "clip", "org", "audio"]       # audio only with --audio (Step 23)
TYPE_DROP = 0.15
HEADS = 4
DEFAULT_HP = {"hidden": 256, "layers": 2, "dropout": 0.2, "feat_dropout": 0.0,
              "lr": 1e-3, "weight_decay": 0.01, "batch": 64}
SPACE = {"hidden": [128, 256], "layers": [1, 2, 3], "dropout": [0.1, 0.3, 0.5],
         "feat_dropout": [0.0, 0.2, 0.4, 0.6], "lr": [3e-4, 1e-3, 3e-3],
         "weight_decay": [0.0, 0.01, 0.1], "batch": [32, 64, 128]}
TUNE_SEED = 2026


def trials(n: int) -> list[dict]:
    """The first n hyperparameter sets: the defaults, then a fixed random draw from SPACE."""
    grid = [dict(zip(SPACE, v)) for v in itertools.product(*SPACE.values())]
    order = np.random.default_rng(TUNE_SEED).permutation(len(grid))
    out = [dict(DEFAULT_HP)]
    out += [grid[i] for i in order if grid[i] != DEFAULT_HP][: n - 1]
    return out[:n]


# ------------------------------------------------------------------ model
def make_model(in_dims: dict, n_out: int, hp: dict):
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATv2Conv, HeteroConv
    from torch_geometric.utils import scatter, softmax
    d, layers, dropout = hp["hidden"], hp["layers"], hp["dropout"]
    # only the node types present; without audio the modules and their init order are as before
    pos_types = {t: n for t, n in POS_TYPES.items() if t in in_dims}
    pooled = [t for t in POOLED if t in in_dims]
    audio = "audio" in in_dims

    class GraphModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_drop = nn.Dropout(hp["feat_dropout"])
            self.inp = nn.ModuleDict({t: nn.Linear(n, d) for t, n in in_dims.items() if t != "hub"})
            self.pos = nn.ModuleDict({t: nn.Embedding(n, d) for t, n in pos_types.items()})
            self.norm_in = nn.ModuleDict({t: nn.LayerNorm(d) for t in in_dims})
            self.hub = nn.Parameter(torch.randn(1, d) * 0.02)
            self.convs = nn.ModuleList([HeteroConv({
                et: GATv2Conv(d, d // HEADS, heads=HEADS, edge_dim=1, add_self_loops=False,
                              dropout=dropout) for et in gw.edge_types(audio)}, aggr="mean")
                for _ in range(layers)])
            self.self_lin = nn.ModuleList([nn.ModuleDict({t: nn.Linear(d, d) for t in in_dims})
                                           for _ in range(layers)])
            self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(d) for t in in_dims})
                                        for _ in range(layers)])
            self.gate = nn.ModuleDict({t: nn.Linear(d, 1) for t in pooled})
            self.drop = nn.Dropout(dropout)
            width = d * (1 + len(pooled)) + len(pooled)
            self.head = nn.Sequential(nn.Linear(width, d), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(d // 2, n_out))

        def forward(self, b):
            n_graphs = b.num_graphs
            # node-type dropout: per video, with prob TYPE_DROP, one pooled type is removed
            keep = {}
            if self.training and TYPE_DROP > 0:
                hit = torch.rand(n_graphs, device=b["hub"].x.device) < TYPE_DROP
                which = torch.randint(len(pooled), (n_graphs,), device=hit.device)
                for j, t in enumerate(pooled):
                    keep[t] = (~(hit & (which == j)))[b[t].batch].unsqueeze(-1).to(b[t].x.dtype)
            # a dropped node type also loses its edges, so it sends and receives nothing
            ei_d, ea_d = b.edge_index_dict, b.edge_attr_dict
            if keep:
                ei_d, ea_d = dict(ei_d), dict(ea_d)
                for et, ei in b.edge_index_dict.items():
                    src, _, dst = et
                    m = torch.ones(ei.size(1), dtype=torch.bool, device=ei.device)
                    if src in keep:
                        m &= keep[src][ei[0], 0] > 0
                    if dst in keep:
                        m &= keep[dst][ei[1], 0] > 0
                    ei_d[et], ea_d[et] = ei[:, m], ea_d[et][m]
            h = {}
            for t in in_dims:
                if t == "hub":
                    z = self.hub.expand(b["hub"].num_nodes, -1)
                else:
                    z = self.inp[t](self.feat_drop(b[t].x))
                    if t in pos_types:
                        z = z + self.pos[t](b[t].pos_id)
                h[t] = self.norm_in[t](z) * keep.get(t, 1)
            for conv, own, norm in zip(self.convs, self.self_lin, self.norms):
                msg = conv(h, ei_d, edge_attr_dict=ea_d)
                h = {t: norm[t](h[t] + self.drop(nn.functional.gelu(
                    own[t](h[t]) + (msg[t] if t in msg else 0)))) * keep.get(t, 1) for t in h}
            parts = [h["hub"]]                     # one hub per graph, in graph order
            flags = []
            for t in pooled:
                bt = b[t].batch
                k = keep[t].squeeze(-1) if t in keep else torch.ones_like(bt, dtype=h[t].dtype)
                if h[t].size(0):
                    g = self.gate[t](h[t]).squeeze(-1).masked_fill(k == 0, -1e4)
                    a = softmax(g, bt, num_nodes=h[t].size(0)) * k
                else:
                    a = h[t].new_zeros(0)
                parts.append(scatter(a.unsqueeze(-1) * h[t], bt, dim=0, dim_size=n_graphs, reduce="sum"))
                flags.append((scatter(k, bt, dim=0, dim_size=n_graphs, reduce="sum") > 0).to(h[t].dtype))
            return self.head(torch.cat(parts + [torch.stack(flags, 1)], 1))

    return GraphModel()


# ------------------------------------------------------------------ data
def to_hetero(g: dict, y: np.ndarray | None):
    import torch
    from torch_geometric.data import HeteroData
    data = HeteroData()
    for t, x in g["x"].items():
        data[t].x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        data[t].num_nodes = len(x)
        data[t].pos_id = torch.from_numpy(np.asarray(g["pos"][t], dtype=np.int64))
    for et, (ei, attr) in g["edges"].items():
        data[et].edge_index = torch.from_numpy(np.ascontiguousarray(ei, dtype=np.int64))
        data[et].edge_attr = torch.from_numpy(np.ascontiguousarray(attr, dtype=np.float32))
    if y is not None:
        data.y = torch.from_numpy(y.astype(np.float32))[None]
    return data


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parent).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ main
def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--part1-root", required=True, help="part-1 vlm_quad package (losses, labels)")
    ap.add_argument("--approach", required=True, choices=gw.APPROACHES)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--runs-dir", default="runs/graphs_v2", help="setup v1 (XLM-R) runs are in runs/graphs")
    ap.add_argument("--features", default=None, help="default: outputs/features")
    ap.add_argument("--tune", type=int, default=0, help="tune: score this many trials per fold, no outer fold")
    ap.add_argument("--hyper", choices=["default", "tuned"], default="default")
    ap.add_argument("--text-features", choices=gw.TEXT_FEATURES, default="e5",
                    help="encoder of the text nodes (fields, speech, organisers' text)")
    ap.add_argument("--audio", action="store_true", help="add Whisper audio-window nodes (Step 23)")
    ap.add_argument("--w3", default=None, help="W3 edges (default outputs/w3/edges.jsonl)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="300 training videos, 2 epochs, fold 1")
    a = ap.parse_args()

    import torch
    from torch_geometric.loader import DataLoader
    sys.path.insert(0, str(Path(a.part1_root).resolve()))
    from labels import TASK_META, get_class_weights, load_labels          # part 1, read-only
    from losses import TaskLoss, compute_per_sample_eval_losses           # part 1, read-only
    from graph_stats import cost_log

    cfg = load_config(a.config)
    task, classes = a.task, TASKS[a.task]["classes"]
    p1task = f"task{task}"
    require(TASK_META[p1task]["classes"] == classes, "part 1's class order equals ours",
            expected=classes, found=TASK_META[p1task]["classes"])
    multilabel = TASKS[task]["type"] == "multilabel"
    p1cfg = json.load(open(Path(__file__).resolve().parents[1] / "configs" / "c0" / f"task{task}.json"))
    loss_cfg = {"dirichlet_alpha": float(p1cfg["dirichlet_alpha"]),
                "label_smoothing": float(p1cfg.get("label_smoothing", 0.0)),
                "use_class_weights": bool(p1cfg.get("use_class_weights", True)),
                "threshold": float(p1cfg.get("multilabel_threshold", 0.5))}
    if a.smoke:
        a.folds, a.epochs = [1], 2
        a.tune = min(a.tune, 2)

    adir = Path(a.runs_dir) / a.approach
    text_tag = ("" if a.text_features == "xlmr" else f"_{a.text_features}") + ("_audio" if a.audio else "")
    tuning_path = adir / f"tuning_{task}{text_tag}.json"
    if a.tune:
        out = adir
        require(not tuning_path.exists() or a.smoke, f"{tuning_path} not there yet",
                expected="no tuning file", found="exists (delete it or keep it)")
    else:
        tag = "" if sorted(a.folds) == [1, 2, 3, 4, 5] else "_f" + "".join(map(str, a.folds))
        tag += ("" if a.hyper == "default" else "_tuned") + text_tag
        out = adir / (f"{a.approach}_{task}_seed{a.seed}{tag}" + ("_smoke" if a.smoke else ""))
        require(not (out / "manifest.json").exists(), f"{out} has no finished run",
                expected="a fresh run directory", found="manifest.json already there (delete it or keep it)")
    out.mkdir(parents=True, exist_ok=True)
    fold_hp = {k: dict(DEFAULT_HP) for k in a.folds}
    if a.hyper == "tuned" and not a.tune:
        require(tuning_path.is_file(), f"{tuning_path} exists (run --tune first)", expected="file", found="missing")
        tuned = json.load(open(tuning_path))
        require(all(str(k) in tuned["folds"] for k in a.folds), "tuning covers every fold run",
                expected=a.folds, found=sorted(tuned["folds"]))
        fold_hp = {k: tuned["folds"][str(k)]["best"] for k in a.folds}

    # ---- labels (part 1's loader: soft vectors, hard labels for class weights)
    lab = load_labels(cfg.gold_path(task, "soft"), cfg.gold_path(task, "hard"), p1task)
    lab["video_id"] = lab.video_id.astype(str)
    lab = lab.set_index("video_id")
    soft_cols = [f"soft_{c}" for c in classes]

    # ---- graphs
    t0 = time.time()
    ft, cal, ids = gw.load_inputs(cfg, Path(a.features) if a.features else None, a.text_features, a.audio)
    if a.approach == "W3":
        w3_path = Path(a.w3) if a.w3 else cfg.out("w3/edges.jsonl")
        gw.load_w3(ft, w3_path)
    folds = pd.read_csv(cfg.out("folds.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    vids = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str}).set_index("id_EXIST")
    train_ids = [i for i in ids if vids.loc[i, "split"] == "train"]
    test_ids = [i for i in ids if vids.loc[i, "split"] == "test"]
    missing = [i for i in train_ids if i not in lab.index]
    require(not missing, "every training video has a soft label", expected=0, found=missing[:5])
    graphs = {}
    for k in ids:
        y = lab.loc[k, soft_cols].to_numpy(np.float32) if k in lab.index else None
        graphs[k] = to_hetero(gw.build(ft, k, a.approach, cal), y)
    in_dims = {t: graphs[ids[0]][t].x.shape[1] for t in gw.node_types(a.audio)}
    print(f"{len(graphs)} graphs ({a.approach}) built in {time.time() - t0:.0f} s; node features {in_dims}",
          flush=True)

    dev = torch.device(a.device)
    require(dev.type != "cuda" or torch.cuda.is_available(), "a GPU is visible (or pass --device cpu)",
            expected="torch.cuda.is_available()", found=False)

    def predict(model, keys):
        model.eval()
        probs = []
        with torch.no_grad():
            for b in DataLoader([graphs[k] for k in keys], batch_size=256, shuffle=False):
                logits = model(b.to(dev)).float()
                probs.append((torch.sigmoid(logits) if multilabel else torch.softmax(logits, -1)).cpu().numpy())
        return pd.DataFrame(np.concatenate(probs), columns=[f"p_{c}" for c in classes]).assign(id_EXIST=keys)

    def val_loss(model, criterion, keys):
        model.eval()
        num = den = 0.0
        with torch.no_grad():
            for b in DataLoader([graphs[k] for k in keys], batch_size=256, shuffle=False):
                b = b.to(dev)
                _, lw, wsl = compute_per_sample_eval_losses(criterion, model(b).float(), b.y)
                num += float(wsl.sum())
                den += float(lw.sum())
        return num / max(den, 1e-12)

    def split(k):
        inner = folds[f"inner_for_outer_{k}"]
        rest = folds.outer_fold != k
        tr = [i for i in train_ids if rest[i] and inner[i] != ES_INNER]
        sel = [i for i in train_ids if rest[i] and inner[i] == ES_INNER and folds.val_eligible[i]]
        outer = [i for i in train_ids if folds.outer_fold[i] == k and folds.val_eligible[i]]
        require(not (set(tr) | set(sel)) & set(outer), f"fold {k}: outer videos unused in training",
                expected="disjoint", found="overlap")
        if a.smoke:
            tr = list(np.random.default_rng(0).choice(tr, size=min(300, len(tr)), replace=False))
        return tr, sel, outer

    def fit(k, hp, seed, tr, sel, verbose=True):
        """Train on tr, keep the epoch with the lowest loss on sel. Returns (model, best, epoch, log)."""
        torch.manual_seed(seed * 100 + k)
        np.random.seed(seed * 100 + k)
        cw = get_class_weights(lab.loc[tr].reset_index(), p1task) if loss_cfg["use_class_weights"] else None
        criterion = TaskLoss(p1task, cw, loss_cfg["dirichlet_alpha"],
                             label_smoothing=loss_cfg["label_smoothing"]).to(dev)
        model = make_model(in_dims, len(classes), hp).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        gen = torch.Generator().manual_seed(seed * 100 + k)
        loader = DataLoader([graphs[i] for i in tr], batch_size=hp["batch"], shuffle=True, generator=gen)
        best, best_epoch, best_state, log, waited = np.inf, None, None, [], 0
        for epoch in range(1, a.epochs + 1):
            model.train()
            criterion.train()
            tot = n = 0
            for b in loader:
                b = b.to(dev)
                loss = criterion(model(b).float(), b.y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss) * b.num_graphs
                n += b.num_graphs
            criterion.eval()
            vl = val_loss(model, criterion, sel)
            log.append((epoch, round(tot / n, 5), round(vl, 5)))
            if vl < best - 1e-4:
                best, best_epoch, waited = vl, epoch, 0
                best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            else:
                waited += 1
            if verbose and (epoch == 1 or epoch % 10 == 0):
                print(f"  fold {k} epoch {epoch}: train {tot / n:.4f}  inner val {vl:.4f}  "
                      f"(best {best:.4f} @ {best_epoch})", flush=True)
            if waited >= a.patience:
                break
        require(best_state is not None, f"fold {k} selected an epoch", expected="finite loss", found=log)
        model.load_state_dict(best_state)
        return model, best, best_epoch, log

    # ---- tuning: inner folds only
    if a.tune:
        cand = trials(a.tune)
        result = {"approach": a.approach, "task": task, "seed": a.seed, "git_commit": git_commit(),
                  "text_features": a.text_features, "audio": a.audio,
                  "space": SPACE, "type_drop": TYPE_DROP, "type_drop_edges": True, "n_trials": len(cand),
                  "folds": {}}
        with cost_log(cfg.out("costs.jsonl"), a.approach, f"{task}_tune", "tune",
                      gpus=int(dev.type == "cuda"), folds=a.folds, trials=len(cand), smoke=a.smoke):
            for k in a.folds:
                tr, sel, _ = split(k)
                rows = []
                for j, hp in enumerate(cand):
                    t1 = time.time()
                    _, best, ep, log = fit(k, hp, a.seed, tr, sel, verbose=False)
                    rows.append({"trial": j, **hp, "inner_loss": round(best, 5), "best_epoch": ep,
                                 "epochs": len(log), "seconds": round(time.time() - t1, 1)})
                    print(f"  fold {k} trial {j:2d}: inner loss {best:.4f} @ {ep:3d}  {hp}", flush=True)
                win = min(rows, key=lambda r: r["inner_loss"])
                result["folds"][str(k)] = {"best": {p: win[p] for p in SPACE}, "best_trial": win["trial"],
                                           "best_inner_loss": win["inner_loss"], "trials": rows}
                print(f"  fold {k}: trial {win['trial']} wins, inner loss {win['inner_loss']:.4f}", flush=True)
        if not a.smoke:
            with open(tuning_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(result, f, indent=1)
            print(f"wrote {tuning_path}")
        return

    # ---- training with outer predictions
    manifest = {"approach": a.approach, "task": task, "seed": a.seed, "git_commit": git_commit(),
                "protocol": "nested: lowest validation loss on inner fold 1 of the outer training part",
                "hyper_source": a.hyper, "text_features": a.text_features, "audio": a.audio, "type_drop": TYPE_DROP, "type_drop_edges": True, "epochs": a.epochs, "patience": a.patience,
                "loss": loss_cfg, "grounding_calibration": cal,
                "w3_edges": str(w3_path) if a.approach == "W3" else None, "folds": {}}
    oof, tests = [], {}
    with cost_log(cfg.out("costs.jsonl"), a.approach, f"{task}_seed{a.seed}", "train",
                  gpus=int(dev.type == "cuda"), folds=a.folds, hyper=a.hyper, smoke=a.smoke):
        for k in a.folds:
            tr, sel, outer = split(k)
            t_fold = time.time()
            model, best, best_epoch, log = fit(k, fold_hp[k], a.seed, tr, sel)
            oof.append(predict(model, outer).assign(fold=k, best_epoch=best_epoch))
            tests[k] = predict(model, test_ids)
            tests[k].to_csv(out / f"test_fold{k}.csv", index=False,
                            columns=["id_EXIST", *[f"p_{c}" for c in classes]])
            manifest["folds"][k] = {"hyper": fold_hp[k], "best_epoch": best_epoch, "inner_best_loss": best,
                                    "epochs_run": len(log), "n_train": len(tr), "n_select": len(sel),
                                    "n_outer": len(outer), "n_params": int(sum(p.numel() for p in model.parameters())),
                                    "seconds": round(time.time() - t_fold, 1), "log": log}
            print(f"  fold {k}: best epoch {best_epoch} of {len(log)}, inner loss {best:.4f}, "
                  f"{time.time() - t_fold:.0f} s", flush=True)

    o = pd.concat(oof)[["id_EXIST", *[f"p_{c}" for c in classes], "fold", "best_epoch"]]
    o.to_csv(out / "oof_outer.csv", index=False)
    if len(tests) == 5:
        cols = [f"p_{c}" for c in classes]
        mean = sum(t[cols].to_numpy() for t in tests.values()) / 5
        pd.DataFrame(mean, columns=cols).assign(id_EXIST=test_ids)[["id_EXIST", *cols]] \
            .to_csv(out / "test_mean.csv", index=False)
    with open(out / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    report(cfg, task, o, loss_cfg["threshold"], a.approach)


def report(cfg, task: str, oof: pd.DataFrame, threshold: float, name: str) -> None:
    """Scores on exactly the videos this run predicted, next to C0 and C4 on the same videos."""
    gold = ev.load_gold_frame(cfg, task)
    cols = [f"p_{c}" for c in TASKS[task]["classes"]]
    have = set(oof.id_EXIST)
    mask = np.array([i in have for i in gold.ids])
    prior = gold.soft[mask].mean(0)

    def full(df):
        p = np.tile(prior, (len(gold.ids), 1))
        d = df.set_index("id_EXIST")
        rows = [j for j, i in enumerate(gold.ids) if i in have]
        p[rows] = d.loc[[gold.ids[j] for j in rows], cols].to_numpy(float)
        return p

    preds = {name: full(oof)}
    for other, path in (("C0 quad", Path("results/C0/ensemble") / f"C0_quad_{task}_oof.csv"),
                        ("C4", Path("runs/C4") / f"C4_{task}" / "oof_outer.csv")):
        if path.is_file():
            preds[other] = full(pd.read_csv(path, dtype={"id_EXIST": str}))
    preds["prior"] = np.tile(prior, (len(gold.ids), 1))
    subs = ev.subsets(gold)
    rows = []
    for m, p in preds.items():
        hard = ev.derive_hard(p, task, threshold)
        for s in ("all", "contested"):
            sc = ev.Evaluator(gold, p, hard, mask & subs[s]).scores()
            rows.append({"model": m, "subset": s, "n": int((mask & subs[s]).sum()),
                         **{k: round(sc[k], 4) for k in ("icm_soft_norm", "icm_norm", "auc", "ce")}})
    print(f"\ntask {task}, out of fold, on the {int(mask.sum())} videos of the folds run:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
