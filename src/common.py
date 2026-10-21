"""Shared helpers: config loading, fail-loud checks, task metadata, gold labels."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_CATEGORIES = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
]

# Hierarchies follow the EXIST official evaluation (sexist subclasses under a
# virtual YES node). Part 1 evaluated 3.2 / 3.3 flat; its CV numbers for those
# tasks are therefore not comparable with the leaderboard or with ours.
TASKS = {
    "3_1": {"type": "binary", "classes": ["NO", "YES"], "hierarchy": None},
    "3_2": {"type": "multiclass", "classes": ["NO", "DIRECT", "JUDGEMENTAL"],
            "hierarchy": {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}},
    "3_3": {"type": "multilabel", "classes": ["NO"] + _CATEGORIES,
            "hierarchy": {"YES": list(_CATEGORIES), "NO": []}},
}


class CheckFailed(RuntimeError):
    pass


def require(cond: bool, what: str, expected: Any = None, found: Any = None) -> None:
    """Raise with a message naming what was expected and what was found."""
    if cond:
        return
    msg = f"CHECK FAILED: {what}"
    if expected is not None or found is not None:
        msg += f"\n  expected: {expected}\n  found:    {found}"
    raise CheckFailed(msg)


@dataclass
class Config:
    root: Path
    raw: dict

    def path(self, key: str) -> Path:
        require(key in self.raw["paths"], f"config.paths has key '{key}'",
                expected=key, found=sorted(self.raw["paths"]))
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else (self.root / p).resolve()

    def gold_path(self, task: str, kind: str) -> Path:
        name = self.raw["paths"]["gold_pattern"].format(task=task, kind=kind)
        return self.path("gold_dir") / name

    def out(self, name: str) -> Path:
        d = self.path("out_dir")
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    require(path.is_file(), "config file exists", expected=str(path), found="missing")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(root=path.parent, raw=raw)


def load_json(path: Path) -> Any:
    require(path.is_file(), "input file exists", expected=str(path), found="missing")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_records(path: Path) -> list[dict]:
    """Records in source-file order (row order feeds the stratified split)."""
    obj = load_json(path)
    require(isinstance(obj, dict), f"{path.name} is a dict keyed by id_EXIST",
            expected="dict", found=type(obj).__name__)
    recs = []
    for key, rec in obj.items():
        require(str(rec.get("id_EXIST")) == str(key), f"{path.name}: key equals id_EXIST",
                expected=key, found=rec.get("id_EXIST"))
        recs.append(rec)
    return recs


def load_gold(cfg: Config, task: str) -> tuple[dict, dict]:
    """Official gold as-is: (soft {id: {class: p}}, hard {id: label or list})."""
    soft = {str(r["id"]): r["value"] for r in load_json(cfg.gold_path(task, "soft"))}
    hard = {str(r["id"]): r["value"] for r in load_json(cfg.gold_path(task, "hard"))}
    classes = set(TASKS[task]["classes"])
    for vid, v in soft.items():
        require(set(v) <= classes, f"task{task} soft gold classes are known",
                expected=sorted(classes), found=sorted(v))
    return soft, hard


def setup_stdout() -> None:
    # Windows consoles default to a legacy codepage; transcripts are ES/EN.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
