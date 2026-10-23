"""Minimal cleaning of the organisers' `text` field, one rule set for every video.

The field mixes speech, on-screen text and the residue of an LLM cleaning
step (data_note.md, problem 2). Two things are removed, nothing else:

1. leaked instructions and meta-comments of that cleaning step, in English
   and Spanish ("clean the previous detected text by removing repetitions,
   ...", "limpia el texto anterior detectado eliminando repeticiones, ...",
   markers such as "cleaned text" / "texto limpio");
2. degenerate repetition loops (ASR looping on music): a word or short
   phrase repeated more than three times in a row is cut to three.

On-screen text, emoji names, handles and ordinary sentences stay. Every
pattern requires an instruction-like combination (a cleaning verb together
with "repetitions"/"readable"), so sentences such as "cuando haya que
limpiarla" are not touched; `python src/text_clean.py` prints every removed
span for review.
"""
from __future__ import annotations

import re

_F = re.I | re.U

# leaked instructions / meta-comments; each consumes up to the end of its sentence
LEAK_PATTERNS = [
    # EN: "clean the previous(ly)/above detected text by removing repetitions, leaving ..."
    re.compile(r"\bclean\b[^.\n]{0,60}?\btext\b[^.\n]{0,40}?\bremoving repetitions\b[^.\n]*\.?", _F),
    # ES: "limpia/elimina ... repeticiones ..." (instruction form)
    re.compile(r"\b(?:limpia|elimina)\b[^.\n]{0,80}?\brepeticiones\b[^.\n]*\.?", _F),
    # ES infinitive form: "luego, limpiar el texto anterior."
    re.compile(r"\blimpiar\s+el\s+texto\b[^.\n]*\.?", _F),
    # ES: "deja(ndo) el texto breve y esencial, muy limpio y completamente legible."
    re.compile(r"\bdeja(?:ndo)?\s+el\s+texto\b[^.\n]{0,60}?\b(?:esencial|legible)\b[^.\n]*\.?", _F),
    # meta-comments on the text itself
    re.compile(r"\bel texto contiene\b[^.\n]*\brepeticiones\b[^.\n]*\.?", _F),
    re.compile(r"\bes un texto limpio y legible\b\.?", _F),
    # section markers left by the cleaning step
    re.compile(r"\bcleaned text\b:?", _F),
    re.compile(r"\btexto limpio\b:?", _F),
    # a bare "text"/"texto" is a marker only when two spaces follow (in running
    # text, e.g. code shown on screen, it ends a line or takes one space)
    re.compile(r"(?:(?<=\s)|^)text(?= {2,})", _F),
    re.compile(r"(?:(?<=\s)|^)texto(?= {2,})", _F),
]

# a token (or 2-3 token phrase) followed by >= 3 more copies, separated by
# spaces or light punctuation -> keep three copies
_SEP = r"[\s,.;:!?¡¿…-]+"
LOOP_PATTERNS = [
    re.compile(rf"\b((?:\w+{_SEP}){{{n - 1}}}\w+)(?:{_SEP}\1\b){{3,}}", _F) for n in (1, 2, 3)
]


def _collapse(m: re.Match) -> str:
    unit = m.group(1)
    return f"{unit} {unit} {unit}"


def clean_text(text: str | None) -> tuple[str, dict]:
    """Return (cleaned text, report with the removed spans)."""
    s = text or ""
    report = {"leaks": [], "loops": 0}
    for p in LEAK_PATTERNS:
        report["leaks"] += [m.group(0).strip() for m in p.finditer(s) if m.group(0).strip()]
        s = p.sub(" ", s)
    for p in LOOP_PATTERNS:
        s, n = p.subn(_collapse, s)
        report["loops"] += n
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s).strip()
    return s, report


if __name__ == "__main__":
    # Review: every removed span and loop collapse over both splits.
    import json
    import sys
    from collections import Counter

    sys.stdout.reconfigure(encoding="utf-8")
    paths = sys.argv[1:] or ["../data/text_llms/training_vlm.json", "../data/text_llms/test_vlm.json"]
    spans, n_leak_videos, n_loop_videos, n = Counter(), 0, 0, 0
    examples = []
    for p in paths:
        for k, r in json.load(open(p, encoding="utf-8")).items():
            n += 1
            new, rep = clean_text(r.get("text"))
            spans.update(x.lower()[:110] for x in rep["leaks"])
            n_leak_videos += bool(rep["leaks"])
            n_loop_videos += bool(rep["loops"])
            if rep["loops"] and len(examples) < 4:
                examples.append((k, (r.get("text") or "")[:160], new[:160]))
    print(f"{n} texts; with removed leaks/markers: {n_leak_videos}; with collapsed loops: {n_loop_videos}")
    for s, c in spans.most_common():
        print(f"{c:4d} | {s}")
    for k, a, b in examples:
        print(f"\n[{k}] before: {a}\n        after:  {b}")
