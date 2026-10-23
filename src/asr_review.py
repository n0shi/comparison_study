"""Human check of ASR and timing on 40 videos, in one pass.

  sample  the 30 bake-off videos + 10 more with ordinary speech (5 EN, 5 ES)
          -> outputs/asr/review_ids.txt
  page    after s04_asr.py (on these ids) and asr_align_diag.py: a local HTML page
          with, per video, the clip, a timeline (VAD speech, kept and dropped
          ASR segments, the 4 H5 clips and moment windows, the organisers'
          sentences placed in time) and three questions. Clicking a sentence
          or segment plays the video from there. Serve the folder over http.

Usage:
    python src/asr_review.py sample --config config.yaml
    python src/asr_review.py page   --config config.yaml \\
        --asr outputs/asr/review_whisper.jsonl --align outputs/align/review_align.jsonl
"""
from __future__ import annotations

import argparse
import html
import json
import shutil

import numpy as np
import pandas as pd

from common import load_config, require, setup_stdout

SEED = 1


def cmd_sample(cfg, a) -> None:
    out = cfg.path("out_dir") / "asr"
    base = pd.read_csv(out / "bakeoff_sample.csv", dtype={"id_EXIST": str})
    v = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str})
    v = v[(v.split == "train") & ~v.id_EXIST.isin(base.id_EXIST)].copy()
    v["wps"] = v.transcript_words / v.duration_s
    rng = np.random.default_rng(SEED)
    extra = []
    for lang in ("en", "es"):
        pool = v[(v.lang == lang) & v.wps.between(1.5, 3.5) & (v.duration_s <= 60)]
        extra.append(pool.iloc[rng.choice(len(pool), 5, replace=False)])
    ids = list(base.id_EXIST) + list(pd.concat(extra).id_EXIST)
    require(len(ids) == 40 and len(set(ids)) == 40, "40 distinct videos", expected=40, found=len(set(ids)))
    (out / "review_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"wrote {out / 'review_ids.txt'}: 30 bake-off + 10 ordinary-speech videos")


def _span(s: dict) -> str:
    return "" if s["t0"] is None else f'{s["t0"]:.1f}–{s["t1"]:.1f}s'


def _svg(d: dict, a: dict, dur: float) -> str:
    W, rowh = 1000, 16
    x = lambda t: 0 if t is None else min(W, max(0, t / dur * W))
    rows = ["VAD speech", "ASR kept", "ASR dropped", "H5 clips", "sentences"]
    els = []
    for k0, k1 in a["windows"][1:]:
        els.append(f'<line x1="{x(k0):.1f}" x2="{x(k0):.1f}" y1="0" y2="{len(rows) * rowh + 4}" class="win"/>')

    def bar(r, t0, t1, cls, tip, click=None):
        c = f' onclick="seek(this,{t0})"' if click else ""
        return (f'<rect x="{x(t0):.1f}" y="{r * rowh + 3}" width="{max(1.5, x(t1) - x(t0)):.1f}" '
                f'height="{rowh - 6}" class="{cls}"{c}><title>{html.escape(tip)}</title></rect>')
    for s, e in d.get("vad", []):
        els.append(bar(0, s, e, "vad", f"speech {s:.1f}-{e:.1f}s"))
    for sg in d.get("segments", []):
        els.append(bar(1, sg["start"], sg["end"], "kept", f'{sg["start"]:.1f}s {sg["text"]}', True))
    for sg in d.get("dropped", []):
        els.append(bar(2, sg["start"], sg["end"], "drop", f'{sg["reason"]}: {sg["text"]}', True))
    for k, (s, e) in enumerate(a["clips"]):
        els.append(bar(3, s, e, "clip", f"H5 clip {k + 1}: {s:.1f}-{e:.1f}s"))
    for s in a["sentences"]:
        if s["kind"] == "speech":
            els.append(bar(4, s["t0"], s["t1"], "sent", f'[{s["t0"]:.1f}s] {s["text"]}', True))
    labels = "".join(f'<text x="-6" y="{r * rowh + 12}" class="lab">{n}</text>' for r, n in enumerate(rows))
    return (f'<svg viewBox="-90 0 {W + 90} {len(rows) * rowh + 6}" class="tl">{labels}'
            f'<rect x="0" y="0" width="{W}" height="{len(rows) * rowh}" class="bg"/>{"".join(els)}'
            f'<line x1="0" x2="0" y1="0" y2="{len(rows) * rowh}" class="cursor"/></svg>')


def cmd_page(cfg, a) -> None:
    ids = [l.strip() for l in open(cfg.path("out_dir") / "asr" / "review_ids.txt", encoding="utf-8") if l.strip()]
    load = lambda p: {d["id_EXIST"]: d for d in map(json.loads, open(p, encoding="utf-8")) if d}
    asr, ali = load(a.asr), load(a.align)
    require(set(ids) <= set(asr) and set(ids) <= set(ali), "ASR and alignment cover the 40 videos",
            expected=40, found=f"asr {len(set(ids) & set(asr))}, align {len(set(ids) & set(ali))}")
    v = pd.read_csv(cfg.out("videos.csv"), dtype={"id_EXIST": str, "id_Tiktok": str}).set_index("id_EXIST")
    dirs = {"train": cfg.path("train_videos_dir"), "test": cfg.path("test_videos_dir")}
    root = cfg.path("out_dir") / "asr" / "review"
    (root / "media").mkdir(parents=True, exist_ok=True)

    cards = []
    for n, k in enumerate(ids, 1):
        d, al, r = asr[k], ali[k], v.loc[k]
        name = f"{r.id_Tiktok}.mp4"
        if not (root / "media" / name).is_file():
            shutil.copy2(dirs[r.split] / name, root / "media" / name)
        dur = float(r.duration_s)
        lang = (f'record {d.get("lang_record")} · detected {d.get("lang_detected")} '
                f'({d.get("lang_prob")}) · used <b>{d.get("lang_used")}</b>')
        segs = "".join(f'<li onclick="seek(this,{s["start"]})"><span class="t">{s["start"]:.1f}s</span> '
                       f'{html.escape(s["text"])}</li>' for s in d.get("segments", [])) or "<li><i>no speech kept</i></li>"
        drops = "".join(f'<li onclick="seek(this,{s["start"]})"><span class="t">{s["start"]:.1f}s</span> '
                        f'<span class="why">{s["reason"]}</span> {html.escape(s["text"])}</li>'
                        for s in d.get("dropped", []))
        sents = "".join(
            f'<tr class="{s["kind"]}"' + (f' onclick="seek(this,{s["t0"]})"' if s["kind"] == "speech" else "") +
            f'><td>{s["kind"]}</td><td>{_span(s)}</td>'
            f'<td>{"" if s["window"] is None else s["window"] + 1}</td><td>{s["match"]:.0%}</td>'
            f'<td>{html.escape(s["text"])}</td></tr>' for s in al["sentences"])

        def q(name, opts):
            return "".join(f'<label><input type="radio" name="{name}{k}" value="{o}">{o.replace("_", " ")}</label>'
                           for o in opts)
        cards.append(f"""
<section class="card" data-id="{k}">
<h2>{n}. {k} · {r.lang} · {dur:.0f}s · speech {d.get("speech_ratio")} · {lang} · words: {d.get("word_source")}</h2>
<video controls preload="metadata" src="media/{name}"></video>
{_svg(d, al, dur)}
<div class="cols">
 <div><h3>ASR (kept segments)</h3><ul>{segs}</ul>{f'<h3>Dropped by filters</h3><ul class="drops">{drops}</ul>' if drops else ''}</div>
 <div><h3>Organisers' sentences: placed in time (speech) or not found in audio (overlay)</h3>
 <table><tr><th>kind</th><th>time</th><th>window</th><th>match</th><th>text</th></tr>{sents}</table></div>
</div>
<div class="qs">
 <p><b>1. ASR:</b> {q("asr", ["good", "minor_errors", "bad", "no_speech"])}</p>
 <p><b>2. Timing</b> (click a green sentence: does it start there?): {q("time", ["correct", "off_by_a_second", "wrong", "nothing_timed"])}</p>
 <p><b>3. Speech vs on-screen split:</b> {q("split", ["correct", "some_wrong", "mostly_wrong"])}</p>
 <p><b>4. Overlay rows</b> (really on screen?): {q("overlay", ["on_screen", "some_not_on_screen", "invented_text", "no_overlay"])}</p>
 <p><input class="note" placeholder="note (optional)"></p>
</div></section>""")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ASR and timing check</title>
<style>
:root{{--bg:#fff;--fg:#1b1b1b;--card:#f4f4f6;--mut:#666;--line:#ccc}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--fg:#e8e8ea;--card:#23252a;--mut:#9a9aa0;--line:#444}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;max-width:1250px;margin:auto;padding:16px}}
.card{{background:var(--card);border-radius:10px;padding:12px 16px;margin:16px 0}}
h2{{font-size:14px;margin:0 0 8px}} h3{{font-size:12px;color:var(--mut);margin:8px 0 4px}}
video{{width:100%;max-height:340px;background:#000;border-radius:6px}}
.tl{{width:100%;margin:6px 0}} .bg{{fill:none;stroke:var(--line)}} .lab{{font-size:10px;fill:var(--mut);text-anchor:end}}
.vad{{fill:#9aa}} .kept{{fill:#3b82f6;cursor:pointer}} .drop{{fill:#ef4444;cursor:pointer}} .clip{{fill:#f59e0b}}
.sent{{fill:#22c55e;cursor:pointer}} .win{{stroke:var(--mut);stroke-dasharray:3 3}} .cursor{{stroke:#e11d48;stroke-width:2}}
.cols{{display:grid;grid-template-columns:1fr 1.4fr;gap:14px}} @media (max-width:800px){{.cols{{grid-template-columns:1fr}}}}
ul{{margin:0;padding-left:16px}} li{{cursor:pointer;margin:2px 0}} .t{{color:var(--mut);font-variant-numeric:tabular-nums}}
.why{{color:#ef4444;font-size:11px}} table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border-bottom:1px solid var(--line);padding:3px 5px;vertical-align:top;text-align:left}}
tr.speech{{cursor:pointer}} tr.speech td:first-child{{color:#16a34a}} tr.overlay td:first-child{{color:var(--mut)}}
.qs p{{margin:4px 0}} label{{margin-right:10px;white-space:nowrap}} .note{{width:100%}}
textarea{{width:100%;height:140px}}
</style></head><body>
<h1>ASR and timing check (40 videos)</h1>
<p>Timeline rows: grey = speech found by VAD, blue = ASR segments kept, red = segments dropped by the filters
(hover for the reason), orange = the 4 H5 video clips, dashed lines = boundaries of the 4 moment windows,
green = the organisers' sentences placed in time. Click blue, red or green to play from there.</p>
{''.join(cards)}
<button onclick="collect()">Collect my answers</button><textarea id="out" readonly></textarea>
<script>
function seek(el,t){{const v=el.closest('.card').querySelector('video');v.currentTime=Math.max(0,t-0.3);v.play();}}
document.querySelectorAll('.card').forEach(c=>{{const v=c.querySelector('video'),cur=c.querySelector('.cursor'),
 bg=c.querySelector('.bg');v.addEventListener('timeupdate',()=>{{const x=v.currentTime/(v.duration||1)*bg.width.baseVal.value;
 cur.setAttribute('x1',x);cur.setAttribute('x2',x);}});}});
function collect(){{const r={{}};document.querySelectorAll('.card').forEach(c=>{{const k=c.dataset.id,a={{}};
 c.querySelectorAll('input[type=radio]:checked').forEach(i=>a[i.name.replace(k,'')]=i.value);
 const n=c.querySelector('.note').value.trim(); if(n)a.note=n; if(Object.keys(a).length)r[k]=a;}});
 document.getElementById('out').value=JSON.stringify(r,null,1);}}
</script></body></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {root / 'index.html'} ({len(ids)} videos)")


def main() -> None:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "page"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--asr")
    ap.add_argument("--align")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"sample": cmd_sample, "page": cmd_page}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
