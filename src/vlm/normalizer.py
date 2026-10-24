# Copied verbatim from vlm_provider/normalizer.py (sha256 bcca758fd9fed82b...), the code that
# produced the part-1 descriptions. Only change: `from .prompt` (package import).
# normalizer.py
from __future__ import annotations

import re

from .prompt import EXPECTED_SECTIONS


_HEADER_RE = re.compile(r"^\s*\[([A-Z_]+)\]\s*$", re.MULTILINE)


def parse_sections(description: str) -> dict[str, str]:
    if not description:
        return {}

    matches = list(_HEADER_RE.finditer(description))
    if not matches:
        return {"_unstructured": description.strip()}

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(description)
        sections[name] = description[start:end].strip()
    return sections


def validate_sections(sections: dict[str, str]) -> list[str]:
    """Return a list of expected sections that are missing or empty."""
    return [s for s in EXPECTED_SECTIONS if not sections.get(s, "").strip()]


def is_truncated(description: str) -> bool:
    """
    Heuristic: output is considered truncated if:
      - the last expected section header is missing, OR
      - the text ends mid-sentence (no terminal punctuation on the last line).
    """
    if not description:
        return True

    last_expected = EXPECTED_SECTIONS[-1]
    if f"[{last_expected}]" not in description:
        return True

    tail = description.rstrip()
    if not tail:
        return True
    if tail[-1] not in ".!?\"')]":
        return True

    return False


def to_train_text(
    analysis: dict,
    transcript: str | None = None,
    ocr_text: str | None = None,
    lang: str | None = None,
) -> str:
    lang = (lang or analysis.get("lang_used") or "en").lower()
    description = (analysis.get("description") or "").strip()
    transcript = (transcript or "").strip()
    ocr_text = (ocr_text or "").strip()

    parts = [f"[LANG] {lang}"]
    if transcript:
        parts.append(f"[TRANSCRIPT] {transcript}")
    if ocr_text:
        parts.append(f"[OCR] {ocr_text}")
    parts.append(f"[DESCRIPTION]\n{description}")
    return "\n\n".join(parts)