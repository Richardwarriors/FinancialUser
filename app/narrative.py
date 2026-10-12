"""Portrait: the human-readable layer of a profile.

The five profilers each write a short narrative for their own dimension (inside
their existing structured call, so it costs nothing extra). This module adds the
one thing no single dimension can write: the overall portrait that ties them
together.

Design rules, consistent with the rest of the harness:
  * Grounded   — the model sees the final profile (labels + signals + the five
                 dimension narratives) and the data-quality numbers, nothing else.
  * Honest     — unknown / unverified areas must be stated, never smoothed over.
  * Non-blocking — a failure here returns {"enabled": False} and the profile is
                 still written; narration must never break the pipeline.
  * No advice  — no suitability conclusions, no recommendations, no forecasts.
"""
import json
import os

from pydantic import BaseModel, Field

# v2: outputs are passed through the PII scrubber. The version moves so that stored
# portraits are rebuilt and cleaned — a portrait is reused within its TTL, so without
# this the scrub would only ever apply to users whose dimensions happened to move.
PORTRAIT_PROMPT_VERSION = "portrait-v3-english"
MAX_TOKENS = int(os.environ.get("LUNA_MAX_TOKENS", "4000"))

SYSTEM = (
    "You are a senior investment-advisory profile analyst. Based only on the supplied "
    "structured profile (five dimensions, evidence references, dimension narratives, "
    "and data coverage), write an English overview for an adviser. Requirements:\n"
    "1. Use plain English rather than field-name lists or number dumps; cite at most "
    "one to three decision-relevant figures.\n"
    "2. Distinguish observed behavior (revealed), user statements (stated), and facts "
    "that the data cannot establish.\n"
    "3. Say explicitly when evidence is insufficient. Never fill a gap by speculation "
    "or present stated information as revealed behavior.\n"
    "4. Do not provide advice, suitability conclusions, security recommendations, or forecasts.\n"
    "5. Write a concise headline, a three-to-five-sentence summary, three to five "
    "highlights, and caveats covering the key data gaps.\n"
    "6. Every natural-language output value must be in English. Return only JSON that "
    "conforms to the schema."
)

DIM_TITLES = {
    "risk": "Risk behavior",
    "investment_logic": "Investment logic",
    "preferences": "Preferences and horizon",
    "emotion_and_psychology": "Emotion and psychology",
    "investor_assessment": "Investor assessment",
}


class Portrait(BaseModel):
    headline: str = Field(description="One concise English headline")
    summary: str = Field(description="A three-to-five-sentence English profile overview")
    highlights: list[str] = Field(default_factory=list, description="Three to five English highlights")
    caveats: list[str] = Field(default_factory=list, description="Material data gaps in English")


def _shrink(profile, data_quality):
    """Send labels + narratives + signals, not the whole raw evidence pack."""
    dims = {}
    for key, title in DIM_TITLES.items():
        d = profile.get(key) or {}
        labels = {k: v for k, v in d.items()
                  if k not in ("key_signals", "narrative", "notes", "unknown_fields")}
        dims[title] = {
            "labels": labels,
            "narrative": d.get("narrative") or "",
            "unknown": d.get("unknown_fields") or [],
            "key_signals": (d.get("key_signals") or [])[:5],
        }
    return {
        "dimensions": dims,
        "conflicts": profile.get("conflicts") or [],
        "limitations": profile.get("limitations") or [],
        "data_quality": data_quality or profile.get("data_quality") or {},
        "agent_sources": profile.get("agent_sources") or {},
    }


def _fallback_portrait(profile, reason):
    """Deterministic portrait assembled from the dimension narratives.

    Used when the LLM is unavailable, so a caller always gets a usable text
    block rather than an empty field.
    """
    lines = []
    for key, title in DIM_TITLES.items():
        n = ((profile.get(key) or {}).get("narrative") or "").strip()
        if n:
            lines.append(f"{title}: {n}")
    headline = "Composite profile (LLM overview unavailable)"
    summary = "; ".join(lines) if lines else "The available data is insufficient to form a profile description."
    return {"enabled": False, "reason": reason, "model": None,
            "headline": headline, "summary": summary,
            "highlights": [], "caveats": profile.get("limitations") or []}


def build_portrait(profile, data_quality=None, timeout_note=""):
    """Return the portrait dict attached to the final profile as `portrait`."""
    from .profilers import _client

    client = _client()
    if client is None:
        return _fallback_portrait(profile, "no_api_key")
    payload = _shrink(profile, data_quality)
    try:
        from .profilers import resolve_model
        from .token_meter import record
        model = resolve_model("logic")     # summarising 5 dimensions is a reasoning task
        raw_run = client.with_structured_output(Portrait, include_raw=True)
        res = raw_run.invoke([
            ("system", SYSTEM),
            ("user", json.dumps(payload, ensure_ascii=False) + timeout_note),
        ])
        record("portrait", model, getattr(res.get("raw"), "usage_metadata", None))
        out = res.get("parsed")
        if out is None:
            return _fallback_portrait(profile, "parse_failed")
        d = out.model_dump()
        # The portrait summarises the five dimensions, so it can repeat person data
        # that a dimension narrative still carries (the channel itself is removed —
        # see app/pii.py). Like the tree summaries, it is scrubbed sentence-wise so
        # the analytical content survives and only the offending sentence is lost.
        from .pii import scrub as _pii_scrub
        return {
            "enabled": True,
            "model": model,
            "prompt_version": PORTRAIT_PROMPT_VERSION,
            "headline": _pii_scrub((d.get("headline") or "").strip()),
            "summary": _pii_scrub((d.get("summary") or "").strip()),
            "highlights": [s for s in (_pii_scrub(str(h).strip())
                                       for h in (d.get("highlights") or [])) if s][:6],
            "caveats": [s for s in (_pii_scrub(str(c).strip())
                                    for c in (d.get("caveats") or [])) if s][:6],
        }
    except Exception as e:                                        # noqa: BLE001
        return _fallback_portrait(profile, f"{type(e).__name__}")
