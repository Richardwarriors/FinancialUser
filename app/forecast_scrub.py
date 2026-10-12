"""Keep the rule engine's next-action forecast out of published prose.

Stored profiles retain the deterministic action forecast for audit and baseline
comparison. Public profiles must not expose that answer, including profiler prose
that paraphrases it. Detection is intentionally scoped to action-forecast
mechanisms: ordinary market forecasts and observed historical actions are valid.
"""
from __future__ import annotations

import re

_ENGINE = (
    r"action[_\s-]?forecast(?:[_\s-]?digest)?"
    r"|next[_\s-]?action[_\s-]?forecast"
    r"|(?:model|behaviou?ral|behaviou?r(?:[_\s-]?based)?|forecast)\s+engine"
    r"|action\s+engine"
)
_ACTION = (
    r"trim(?:ming|med)?(?:_position)?|add(?:ing)?(?:_to_existing)?"
    r"|start(?:ing)?(?:_new_position)?|open(?:ing)?(?:\s+a\s+new\s+position)?"
    r"|exit(?:ing)?(?:_position)?"
)
FORECAST_MARKERS = re.compile(
    rf"(?:{_ENGINE})"
    rf"|model\s+forecast(?=[^.;]{{0,100}}(?:action|position|{_ACTION}))"
    rf"|forecast(?:ed)?\s+(?:targets?\s+for\s+)?(?:{_ACTION})"
    rf"|forecast[^.;]{{0,45}}(?:ranks?|identifies?|targets?)[^.;]{{0,35}}(?:{_ACTION})"
    rf"|next[_\s-]?action\s+(?:item|inference|output)"
    # The engine's target basket is as copyable as its action.
    rf"|forecast(?:ed)?[_\s-]?targets?",
    re.IGNORECASE,
)

# A sentence can name the engine's answer without using any phrase above:
# "The system-generated next-action output favors an exit action, but this is a
# model forecast rather than observed behavior" carried the answer in 97 of 1760
# profile versions and matched none of the patterns. What it does have is a
# forecast reference AND an action in the same sentence, so that pair is the
# second rule. A market forecast with no action ("Analyst market forecasts remain
# uncertain") and an observed action with no forecast ("the user trimmed after
# gains") each match only one side and are kept -- which is the distinction that
# matters, since observed history is the evidence the dataset exists to publish.
_FORECAST_REF = re.compile(
    r"forecast|predict(?:s|ed|ion)?|next[_\s-]?action|likely[_\s-]next"
    r"|system[_\s-]?generated|model[_\s-]?generated"
    # "ranked first" is how the engine's output is phrased when the word
    # "forecast" is in the previous clause.
    r"|rank(?:s|ed|ing)?[_\s-]+(?:first|second|highest|top)"
    r"|(?:first|second|highest|top)[_\s-]?ranked", re.IGNORECASE)
_ACTION_REF = re.compile(
    r"\b(?:exit|exits|exiting|trim|trims|trimmed|trimming|add|adds|adding"
    r"|hold|holds|holding|open|opens|opening|close|closes|closing)\b"
    r"|\bnew\s+position\b"
    r"|\b(?:exit_position|start_new_position|add_to_existing|trim_position)\b",
    re.IGNORECASE)


def _sentence_leaks(piece: str) -> bool:
    return bool(FORECAST_MARKERS.search(piece)
                or (_FORECAST_REF.search(piece) and _ACTION_REF.search(piece)))

# Decimal points in scores do not end a sentence. A colon does not end one
# either: "Model forecast, not observed behavior: an exit-position action was
# ranked first" is ONE statement, and splitting it there put the forecast
# reference in one half and the action in the other, so the co-occurrence rule
# below saw neither -- it survived in 5 profile versions until the colon was
# taken out of this pattern.
_SPLIT = re.compile(r"(?<=;)\s+|(?<!\d)\.\s+|\n+")
WITHHELD = "No publishable conclusion for this field at this cutoff."
_WORD = re.compile(r"[A-Za-z]{2,}")
_ENUM = re.compile(r"^[a-z][a-z0-9_]*$")


def leaks(text: str) -> bool:
    """Whether any single sentence in `text` states the action forecast.

    Sentence by sentence, because the second rule is a co-occurrence: judging a
    whole paragraph would flag one that merely happens to contain a market
    forecast in one sentence and an observed trim in another.
    """
    if not isinstance(text, str):
        return False
    return any(_sentence_leaks(p) for p in _SPLIT.split(text) if p and p.strip())


def scrub_text(text: str) -> str:
    """Return text with action-forecast sentences removed."""
    if not leaks(text):
        return text
    kept = []
    for part in _SPLIT.split(text):
        piece = (part or "").strip()
        if not piece or _sentence_leaks(piece):
            continue
        kept.append(piece if piece[-1] in ".;:!?" else piece + ".")
    return " ".join(kept) if kept else WITHHELD


def is_prose(text: str) -> bool:
    """Distinguish reader-facing prose from enums and identifiers."""
    return isinstance(text, str) and len(_WORD.findall(text)) >= 4


def scrub_prose(node, key: str = "", *, prose: bool = False):
    """Recursively remove action-forecast prose while retaining observations."""
    if isinstance(node, dict):
        out = {}
        for child_key, value in node.items():
            cleaned = scrub_prose(value, str(child_key))
            if cleaned not in ({}, []):
                out[child_key] = cleaned
        return out
    if isinstance(node, list):
        out = []
        for item in node:
            if isinstance(item, str) and not _ENUM.match(item) and leaks(item):
                continue
            cleaned = scrub_prose(item, key)
            if cleaned not in ({}, []):
                out.append(cleaned)
        return out
    if isinstance(node, str):
        return scrub_text(node) if is_prose(node) else node
    return node


def find_leaks(node, path: str = "") -> list[str]:
    """Return ``path=excerpt`` entries for residual semantic contamination."""
    out = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            out.extend(find_leaks(value, child))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(find_leaks(value, f"{path}[{i}]"))
    elif leaks(node):
        out.append(f"{path}={node[:90]!r}")
    return out
