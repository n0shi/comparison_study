"""Step 12: W3 edges, chosen by Qwen3-VL instead of by similarity.

For every video the VLM sees the same inputs W2's similarity edges are built
from: the middle frame of each of the 4 H5 clips (SigLIP 2's keyframe in W2),
the speech of the 4 moments (s05_moments.py) and the four description fields
(TARGET, MESSAGE, SITUATION, PEOPLE). It answers, as JSON, which moments'
frames and which moments' speech support each field. These edges replace
W2's `grounds` (field-clip) and `paraphrases` (field-speech); the other W2
relations stay.

One prompt for every video, no labels, greedy decoding. An answer that is not
valid JSON, breaks the schema or cites a moment without speech is asked again
(new seed, temperature 0.3), up to --max-retries times; what is still wrong
after that is cleaned (impossible citations dropped) and flagged.

Output: outputs/w3/edges.jsonl, one line per video, appended per batch, so a
killed job resumes. Build cost goes to outputs/costs.jsonl (graph_stats.py).
--review writes a self-contained HTML page for the hand check: frames,
speech, fields, the chosen links and the VLM's evidence, with a verdict per
field that can be downloaded as CSV.

Usage:
    python src/w3_vlm_edges.py --config config.perun.yaml --sample 20 --out outputs/w3/edges_smoke.jsonl
    python src/w3_vlm_edges.py --config config.perun.yaml --review outputs/w3/edges_smoke.jsonl
    python src/w3_vlm_edges.py --config config.perun.yaml            # every video; re-running resumes
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import load_config, load_moments, load_records, require, setup_stdout

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
FIELDS = ["TARGET", "MESSAGE", "SITUATION", "PEOPLE"]
N_WIN = 4
GREEDY = dict(temperature=0.0, max_tokens=600)
RETRY = dict(temperature=0.3, top_p=0.9, top_k=20, max_tokens=600)
BASE_SEED = 4321

SYSTEM = ("You link the parts of a short video. You do not judge or classify the video; "
          "you only say which parts of it support which statements about it.")

TASK_V1 = """Statements about the video:
{fields}

For each statement, list
- "frames": the moments whose frame shows something the statement describes (a person, object, action, setting or on-screen text it mentions);
- "speech": the moments whose speech says something the statement describes or relies on.

Rules:
- Use moment numbers 1 to 4.
- A list may be empty when no moment supports the statement.
- Never list a moment under "speech" if it has no speech.
- List a moment only if its frame or speech clearly supports the statement; do not guess.
- "evidence": at most 12 words naming what links them.

Answer with JSON only, in this form:
{{"TARGET": {{"frames": [], "speech": [], "evidence": ""}}, "MESSAGE": {{...}}, "SITUATION": {{...}}, "PEOPLE": {{...}}}}"""

# v1 linked almost every field to all 4 frames (smoke, 20 videos: 3.94 of 4),
# even where its own evidence said the frames do not show the field: a field
# describes the whole video, so every moment fits it loosely. v2 asks for the
# most specific moments, at most 2 per list (W2's budget for grounds and
# paraphrases, so W2 and W3 differ only in who picks the edges), and an empty
# list when the field is not visible or audible at all.
TASK_V2 = """Statements about the video:
{fields}

The four moments are parts of one video, so a statement about the whole video often fits all of them loosely. Pick only the moments that show or say it most specifically, compared with the other moments.

For each statement, give
- "frames": up to 2 moments whose frame shows most specifically what the statement describes (a person, object, action, setting or on-screen text it names), strongest first;
- "speech": up to 2 moments whose speech says most specifically what the statement describes or relies on, strongest first.

Rules:
- Use moment numbers 1 to 4, at most 2 per list.
- Leave a list empty when no frame shows the statement, or no speech says it, for example when the statement comes from text that is not visible or audible in these moments.
- Never list a moment under "speech" if it has no speech.
- "evidence": at most 12 words naming what links them.

Answer with JSON only, in this form:
{{"TARGET": {{"frames": [], "speech": [], "evidence": ""}}, "MESSAGE": {{...}}, "SITUATION": {{...}}, "PEOPLE": {{...}}}}"""

# v2 still started at moment 1 for 50 of 79 fields (moments 3 and 4: 6 and 5)
# and filled its budget even where its own evidence said nothing supports the
# field: the lists came before the evidence, so the evidence could not steer
# them. v3 writes the evidence first and scores every moment (0 / 1 / 2), so
# each one is looked at; an edge is a score of 2, at most 2 per list as in W2.
TASK_V3 = """Statements about the video:
{fields}

The four moments are parts of one video, so a statement about the whole video often fits all of them loosely. Score each moment for each statement:
- 2 = its frame shows (its speech says) specifically what the statement describes: a person, object, action, setting or on-screen text the statement names;
- 1 = it fits only loosely, as any part of this video would;
- 0 = it does not show (say) it.

For each statement, first write "evidence": at most 15 words on what the frames and the speech actually show of it. Then give "frames": the scores of the frames of moments 1, 2, 3, 4, and "speech": the scores of the speech of moments 1, 2, 3, 4.

Rules:
- Give 2 to at most two moments per list, and no 2 at all when no moment shows the statement specifically, for example when it comes from text that is not visible or audible in these moments.
- A moment without speech always gets 0 under "speech".
- Judge each moment on its own frame and speech; earlier moments are not more important than later ones.

Answer with JSON only, in this form:
{{"TARGET": {{"evidence": "", "frames": [0, 0, 0, 0], "speech": [0, 0, 0, 0]}}, "MESSAGE": {{...}}, "SITUATION": {{...}}, "PEOPLE": {{...}}}}"""

# v3 gave 82 % of frame scores a 2 and no 1 at all: in most videos one person or
# one scene fills all four frames ("appears in all frames"), so all four do show
# the field. Its mean score was flat over positions (1.63 / 1.70 / 1.55 / 1.65);
# the bias towards moments 1-2 came only from forcing 2 of 4 equal moments.
# v4 drops the budget: every moment scored 2 is an edge, up to all four. A field
# that no moment shows still gets none (v3's evidence-first answers did this).
TASK_V4 = TASK_V3.replace(
    "- Give 2 to at most two moments per list, and no 2 at all when no moment shows the statement "
    "specifically, for example when it comes from text that is not visible or audible in these moments.",
    "- Give 2 to every moment that shows (says) the statement specifically, which may be all four when "
    "the same person or scene fills the video; give no 2 at all when no moment shows it specifically, "
    "for example when the statement comes from text that is not visible or audible in these moments.")
assert TASK_V4 != TASK_V3

# version -> (task text, max moments per list, answer kind: moment lists or per-moment scores)
PROMPTS = {"v1": (TASK_V1, 4, "lists"), "v2": (TASK_V2, 2, "lists"), "v3": (TASK_V3, 2, "scores"),
           "v4": (TASK_V4, N_WIN, "scores")}


def schema(max_items: int, kind: str = "lists") -> dict:
    """JSON schema for constrained decoding; properties in the order the model writes them."""
    ev = {"type": "string", "maxLength": 200}
    if kind == "scores":
        lst = {"type": "array", "items": {"type": "integer", "enum": [0, 1, 2]}, "minItems": N_WIN, "maxItems": N_WIN}
        props, order = {"evidence": ev, "frames": lst, "speech": lst}, ["evidence", "frames", "speech"]
    else:
        lst = {"type": "array", "items": {"type": "integer", "enum": [1, 2, 3, 4]}, "maxItems": max_items}
        props, order = {"frames": lst, "speech": lst, "evidence": ev}, ["frames", "speech", "evidence"]
    return {"type": "object", "properties": {f: {"type": "object", "properties": props, "required": order,
                                                 "additionalProperties": False} for f in FIELDS},
            "required": FIELDS, "additionalProperties": False}


# ------------------------------------------------------------------ inputs
class Inputs:
    def __init__(self, cfg):
        self.cfg = cfg
        self.recs = {str(r["id_EXIST"]): r for k in ("train_json_v3", "test_json_v3")
                     for r in load_records(cfg.path(k))}
        self.moments = load_moments(cfg)
        self.videos = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str, "id_Tiktok": str}) \
            .set_index("id_EXIST")
        self._h5 = {}

    def frames(self, k: str) -> list[bytes]:
        """Middle JPEG of each H5 clip: the keyframe SigLIP 2 grounds fields with in W2."""
        v = self.videos.loc[k]
        if v.split not in self._h5:
            self._h5[v.split] = h5py.File(self.cfg.path(f"{v.split}_h5"), "r")
        clips = self._h5[v.split]["clips"][v.id_Tiktok]
        require(clips.shape[0] == N_WIN, f"{k}: 4 H5 clips", expected=N_WIN, found=clips.shape)
        return [bytes(clips[c, clips.shape[1] // 2]) for c in range(N_WIN)]

    def fields(self, k: str) -> dict:
        sec = self.recs[k].get("vlm_sections") or {}
        return {f: (sec.get(f) or "").strip() for f in FIELDS}

    def has_speech(self, k: str) -> list[bool]:
        return [bool(s.strip()) for s in self.moments[k]["speech"]]


def conversation(inp: Inputs, k: str, prompt: str = "v4") -> list[dict]:
    m = inp.moments[k]
    content = []
    for i, (jpg, (a, b)) in enumerate(zip(inp.frames(k), m["windows"]), 1):
        content.append({"type": "text", "text": f"Moment {i} ({a:.1f}-{b:.1f} s), frame:"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}})
    speech = "\n".join(f'Moment {i}: "{s.strip()}"' if s.strip() else f"Moment {i}: (no speech)"
                       for i, s in enumerate(m["speech"], 1))
    fields = "\n".join(f"{f}: {t or '(empty)'}" for f, t in inp.fields(k).items())
    content.append({"type": "text", "text": "Speech in each moment (automatic transcript, may contain "
                                            f"errors):\n{speech}\n\n" + PROMPTS[prompt][0].format(fields=fields)})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


# ------------------------------------------------------------------ answers
def parse(text: str, has_speech: list[bool], max_items: int = N_WIN,
          kind: str = "lists") -> tuple[str, dict | None, str | None]:
    """(status, edges, error). status: ok | invalid_json | bad_schema | too_many | cites_empty_speech.
    Lists keep the model's order (v2: strongest first), duplicates dropped. Scores (v3) become
    lists of the moments scored 2, in time order; the scores are kept as frame_scores /
    speech_scores, and any score above 0 for a silent moment's speech counts as citing it."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        d = json.loads(t)
    except json.JSONDecodeError as e:
        return "invalid_json", None, str(e)[:120]
    if not isinstance(d, dict) or set(d) != set(FIELDS):
        return "bad_schema", None, f"keys {sorted(d) if isinstance(d, dict) else type(d).__name__}"
    edges, empty, too_many = {}, [], []
    for f in FIELDS:
        v = d[f]
        if not isinstance(v, dict) or not {"frames", "speech"} <= set(v):
            return "bad_schema", None, f"{f}: {v!r}"[:120]
        lists, scores = {}, {}
        for part in ("frames", "speech"):
            x = v[part]
            if kind == "scores":
                if not (isinstance(x, list) and len(x) == N_WIN and all(i in (0, 1, 2) for i in x)):
                    return "bad_schema", None, f"{f}.{part}: {x!r}"[:120]
                scores[f"{part[:-1] if part == 'frames' else part}_scores"] = list(x)
                lists[part] = [i + 1 for i, sc in enumerate(x) if sc == 2]
                if part == "speech":
                    empty += [f"{f}:{i + 1}" for i, sc in enumerate(x) if sc > 0 and not has_speech[i]]
            else:
                if not isinstance(x, list) or not all(isinstance(i, int) and 1 <= i <= N_WIN for i in x):
                    return "bad_schema", None, f"{f}.{part}: {x!r}"[:120]
                lists[part] = list(dict.fromkeys(x))
                if part == "speech":
                    empty += [f"{f}:{i}" for i in lists["speech"] if not has_speech[i - 1]]
            if len(lists[part]) > max_items:
                too_many.append(f"{f}.{part}")
        edges[f] = {**lists, "evidence": str(v.get("evidence", ""))[:200], **scores}
    if too_many:
        return "too_many", edges, ",".join(too_many)
    if empty:
        return "cites_empty_speech", edges, ",".join(empty)
    return "ok", edges, None


def clean(edges: dict, has_speech: list[bool], max_items: int = N_WIN) -> dict:
    """Drop citations of silent moments, keep the first max_items of each list."""
    out = {}
    for f, e in edges.items():
        c = {**e, "frames": e["frames"][:max_items],
             "speech": [i for i in e["speech"] if has_speech[i - 1]][:max_items]}
        if "speech_scores" in e:
            c["speech_scores"] = [sc if has_speech[i] else 0 for i, sc in enumerate(e["speech_scores"])]
        out[f] = c
    return out


def sampling_params(SamplingParams, guided: bool, max_items: int, kind: str, **kw):
    """vLLM's JSON-constrained decoding: StructuredOutputsParams (0.11+) or GuidedDecodingParams."""
    if not guided:
        return SamplingParams(**kw)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(structured_outputs=StructuredOutputsParams(json=schema(max_items, kind)), **kw)
    except (ImportError, TypeError):
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(guided_decoding=GuidedDecodingParams(json=schema(max_items, kind)), **kw)


def pick_sample(inp: Inputs, n: int, seed: int = 0) -> list[str]:
    """n train videos, balanced over language x reliable speech."""
    v = inp.videos[inp.videos.split == "train"].copy()
    v["reliable"] = [inp.moments[k]["reliable"] for k in v.index]
    per = max(1, n // 4)
    out = []
    for (_, _), g in v.groupby(["lang", "reliable"]):
        out += list(g.sample(min(per, len(g)), random_state=seed).index)
    return sorted(out)[:n]


def generate(cfg, a) -> None:
    from vllm import LLM, SamplingParams
    from graph_stats import cost_log

    inp = Inputs(cfg)
    out = Path(a.out) if a.out else cfg.out("w3/edges.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        prev = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
        versions = {d.get("prompt_version") for d in prev}
        require(versions <= {f"w3_{a.prompt}"}, f"{out.name} holds only prompt {a.prompt} (resume)",
                expected=f"w3_{a.prompt}", found=sorted(map(str, versions)))
        done = {d["id_EXIST"] for d in prev}
    ids = pick_sample(inp, a.sample) if a.sample else sorted(inp.videos.index)
    todo = [k for k in ids if k not in done]
    missing = [k for k in todo if k not in inp.recs or k not in inp.moments]
    require(not missing, "every video has a v3 record and moments", expected=0, found=missing[:5])
    print(f"{len(ids)} videos, {len(ids) - len(todo)} done, {len(todo)} to go -> {out}", flush=True)
    if not todo:
        return

    version, (_, max_items, kind) = f"w3_{a.prompt}", PROMPTS[a.prompt]
    with cost_log(cfg.out("costs.jsonl"), "W3", "smoke" if a.sample else "all", "build", gpus=1,
                  model=MODEL, prompt_version=version) as cost:
        llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=a.max_model_len,
                  limit_mm_per_prompt={"image": N_WIN, "video": 0},
                  gpu_memory_utilization=0.9, seed=BASE_SEED)
        t0, n_done = time.time(), 0
        with open(out, "a", encoding="utf-8", newline="\n") as fout:
            for start in range(0, len(todo), a.batch):
                batch = todo[start:start + a.batch]
                res, pending = {}, list(batch)
                for attempt in range(a.max_retries + 1):
                    if not pending:
                        break
                    gen = GREEDY if attempt == 0 else RETRY
                    seeds = {k: BASE_SEED + 1000 * attempt + int(k) for k in pending}
                    outs = llm.chat([conversation(inp, k, a.prompt) for k in pending],
                                    [sampling_params(SamplingParams, not a.no_guided, max_items, kind, seed=seeds[k], **gen)
                                     for k in pending], use_tqdm=False)
                    still = []
                    for k, o in zip(pending, outs):
                        text = o.outputs[0].text or ""
                        hs = inp.has_speech(k)
                        status, edges, err = parse(text, hs, max_items, kind)
                        cost["vlm_calls"] += 1
                        cost["vlm_tokens"] += len(o.prompt_token_ids or []) + len(o.outputs[0].token_ids)
                        res[k] = {"id_EXIST": k, "status": status, "error": err, "attempt": attempt,
                                  "edges": edges, "raw": text, "has_speech": hs,
                                  "finish_reason": o.outputs[0].finish_reason,
                                  "n_prompt_tokens": len(o.prompt_token_ids or []),
                                  "n_output_tokens": len(o.outputs[0].token_ids),
                                  "model": MODEL, "prompt_version": version,
                                  "generation": {**gen, "seed": seeds[k], "guided": not a.no_guided}}
                        if status != "ok":
                            still.append(k)
                    pending = still
                for k in batch:
                    r = res[k]
                    if r["status"] in ("cites_empty_speech", "too_many"):   # still wrong after retries
                        r["edges"], r["status"] = clean(r["edges"], r["has_speech"], max_items), "ok_cleaned"
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += len(batch)
                el = time.time() - t0
                bad = sum(res[k]["status"] not in ("ok", "ok_cleaned") for k in batch)
                print(f"  {n_done}/{len(todo)}  {el / n_done:.2f} s/video  failed in batch: {bad}  "
                      f"eta {el / n_done * (len(todo) - n_done) / 60:.0f} min", flush=True)
    summarise(out)


def summarise(path: Path) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    st = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    ok = [r for r in rows if r["edges"]]
    n_f = np.array([[len(r["edges"][f]["frames"]) for f in FIELDS] for r in ok])
    n_s = np.array([[len(r["edges"][f]["speech"]) for f in FIELDS] for r in ok])
    sp = np.array([sum(r["has_speech"]) for r in ok])
    print(f"{len(rows)} videos: {st}; retried: {sum(r['attempt'] > 0 for r in rows)}")
    if len(ok):
        print(f"per field: frames {n_f.mean():.2f} (none {np.mean(n_f == 0):.0%}), speech {n_s.mean():.2f}"
              f" (none {np.mean(n_s == 0):.0%}; among videos with speech {n_s[sp > 0].mean() if (sp > 0).any() else 0:.2f})")
        for i, f in enumerate(FIELDS):
            print(f"  {f:9s} frames {n_f[:, i].mean():.2f}  speech {n_s[:, i].mean():.2f}")
        # informative edges pick some moments, differ between fields and are not
        # always the same position; a list of all 4 says nothing a hub does not
        same = np.mean([len({tuple(sorted(r["edges"][f]["frames"])) for f in FIELDS}) == 1 for r in ok])
        pos = np.bincount([i for r in ok for f in FIELDS for i in r["edges"][f]["frames"]], minlength=N_WIN + 1)[1:]
        first = np.bincount([r["edges"][f]["frames"][0] for r in ok for f in FIELDS if r["edges"][f]["frames"]],
                            minlength=N_WIN + 1)[1:]
        scored = all("frame_scores" in r["edges"][f] for r in ok for f in FIELDS)
        print(f"informativeness: fields linked to all 4 frames {np.mean(n_f == N_WIN):.0%}; videos where all "
              f"4 fields get the same frames {same:.0%}; frame chosen by position {pos.tolist()}"
              + ("" if scored else f"; first choice by position {first.tolist()}"))
        if scored:   # v3: mean frame score per position, and how often a field has no specific frame
            fs = np.array([r["edges"][f]["frame_scores"] for r in ok for f in FIELDS])
            print(f"frame scores by position (mean of 0/1/2): {fs.mean(0).round(2).tolist()}; "
                  f"share of 0 / 1 / 2: {[round(float(np.mean(fs == v)), 2) for v in (0, 1, 2)]}")


# ------------------------------------------------------------------ review
def review(cfg, path: Path, compare: Path | None = None) -> Path:
    """Hand-check page for `path`; with `compare`, that file's links are shown alongside (grey)."""
    inp = Inputs(cfg)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    other = {}
    if compare:
        other = {d["id_EXIST"]: d for d in (json.loads(l) for l in open(compare, encoding="utf-8") if l.strip())}
    oname = html.escape(compare.stem.replace("edges_", "")) if compare else ""
    esc = html.escape
    parts = []
    for r in rows:
        k = r["id_EXIST"]
        m, flds = inp.moments[k], inp.fields(k)
        cells = []
        for i, (jpg, (a, b), s) in enumerate(zip(inp.frames(k), m["windows"], m["speech"]), 1):
            cells.append(f'<div class="mom"><div class="mh">Moment {i} · {a:.1f}–{b:.1f} s</div>'
                         f'<img src="data:image/jpeg;base64,{base64.b64encode(jpg).decode()}">'
                         f'<div class="sp">{esc(s.strip()) or "<i>(no speech)</i>"}</div></div>')
        trs = []
        for f in FIELDS:
            e = (r["edges"] or {}).get(f, {"frames": [], "speech": [], "evidence": ""})
            sel = "".join(f'<option value="{v}">{v}</option>' for v in ("", "right", "partly", "wrong"))
            o = ((other.get(k) or {}).get("edges") or {}).get(f) if compare else None
            oc = (f'<td class="o">{", ".join(map(str, o["frames"])) or "—"} / '
                  f'{", ".join(map(str, o["speech"])) or "—"}</td>') if compare else ""
            sc = {p: (f'<div class="o">[{" ".join(map(str, e[p + "_scores"]))}]</div>'
                      if p + "_scores" in e else "") for p in ("frame", "speech")}
            trs.append(f'<tr><td><b>{f}</b><div class="ft">{esc(flds[f])}</div></td>'
                       f'<td class="n">{", ".join(map(str, e["frames"])) or "—"}{sc["frame"]}</td>'
                       f'<td class="n">{", ".join(map(str, e["speech"])) or "—"}{sc["speech"]}</td>{oc}'
                       f'<td>{esc(e.get("evidence", ""))}</td>'
                       f'<td><select data-id="{k}" data-field="{f}">{sel}</select></td></tr>')
        v = inp.videos.loc[k]
        parts.append(f'<section><h2>{k} · {v.lang} · status {r["status"]}'
                     f'{" · attempt " + str(r["attempt"]) if r["attempt"] else ""}</h2>'
                     f'<div class="moms">{"".join(cells)}</div>'
                     f'<table><tr><th>Field</th><th>frames</th><th>speech</th>'
                     f'{f"<th class=o>{oname}<br>frames / speech</th>" if compare else ""}<th>evidence</th>'
                     f'<th>verdict</th></tr>{"".join(trs)}</table></section>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>W3 edge review</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{{font:14px/1.4 system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:0 0 20px}}
h2{{font-size:15px;margin:0 0 8px}} .moms{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
.mom img{{width:100%;height:220px;object-fit:contain;background:#111;border-radius:4px}} .mh{{font-weight:600;font-size:12px}} .sp{{font-size:12px;color:#444}}
table{{width:100%;border-collapse:collapse;margin-top:10px}} td,th{{border-top:1px solid #eee;padding:6px;vertical-align:top;text-align:left}}
.ft{{font-size:12px;color:#555;max-width:520px}} .n{{font-weight:600;white-space:nowrap}}
.o{{color:#999;white-space:nowrap;font-size:12px}}
#bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;z-index:1}}
@media(max-width:700px){{.moms{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div id="bar"><b>W3 review</b> · {len(rows)} videos ({esc(path.name)}) ·
<span id="cnt"></span> <button onclick="dl()">Download verdicts (CSV)</button></div>
<p>For each field: are the listed moments' frames and speech really what the field describes, and is anything clearly missing?
<b>right</b> = links correct and nothing important missing; <b>partly</b> = some wrong or missing; <b>wrong</b> = mostly wrong.</p>
{"".join(parts)}
<script>
const KEY="w3review:{esc(path.name)}";let st={{}};try{{st=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
const sels=[...document.querySelectorAll("select")];
function upd(){{const d=sels.filter(s=>s.value).length;document.getElementById("cnt").textContent=d+" / "+sels.length+" fields judged"}}
sels.forEach(s=>{{const k=s.dataset.id+"|"+s.dataset.field;if(st[k])s.value=st[k];
s.onchange=()=>{{st[k]=s.value;try{{localStorage.setItem(KEY,JSON.stringify(st))}}catch(e){{}};upd()}}}});upd();
function dl(){{const rows=["id_EXIST,field,verdict"].concat(sels.map(s=>s.dataset.id+","+s.dataset.field+","+s.value));
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\\n")],{{type:"text/csv"}}));
a.download="w3_verdicts.csv";a.click()}}
</script></body></html>"""
    dst = path.with_name(path.stem + (f"_vs_{compare.stem.replace('edges_', '')}" if compare else "") + "_review.html")
    dst.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {dst} ({len(rows)} videos)")
    return dst


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None, help="default: outputs/w3/edges.jsonl")
    ap.add_argument("--sample", type=int, default=0, help="n train videos balanced by language x speech")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-guided", action="store_true", help="no JSON-constrained decoding")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="v4")
    ap.add_argument("--compare", default=None, help="with --review: a second edges jsonl shown alongside")
    ap.add_argument("--review", default=None, help="edges jsonl -> HTML page for the hand check")
    ap.add_argument("--summary", default=None, help="edges jsonl -> print counts")
    ap.add_argument("--dry-run", default=None, metavar="ID", help="print one video's prompt (no GPU)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.review:
        review(cfg, Path(a.review), Path(a.compare) if a.compare else None)
    elif a.summary:
        summarise(Path(a.summary))
    elif a.dry_run:
        conv = conversation(Inputs(cfg), a.dry_run, a.prompt)
        print("SYSTEM:", conv[0]["content"])
        for c in conv[1]["content"]:
            print("[image]" if c["type"] == "image_url" else c["text"])
    else:
        generate(cfg, a)


if __name__ == "__main__":
    main()
