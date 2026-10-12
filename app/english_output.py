"""English-only boundary helpers for persisted profile artefacts.

Internal metric keys and enums are already English.  Some legacy deterministic
explainers and labels are Chinese; these helpers keep their machine-readable facts
while preventing untranslated prose from leaking into the public dataset.
"""
from __future__ import annotations

import hashlib
import json
import re

_CJK = re.compile(r"[㐀-鿿豈-﫿]")
_CJK_RUN = re.compile(r"[㐀-鿿豈-﫿]+")

_EXACT = {
    "现有证据不足以判断": "Insufficient evidence to determine this feature.",
    "现有数据无法支持间接推断": "Available data does not support a proxy inference.",
    "现有数据无法判断，需补充采集": "Unavailable from current data; additional collection is required.",
    "现有数据不足以计算": "Insufficient data to compute this feature.",
    "需补充采集": "Additional data collection is required.",
}


def humanize(key: str) -> str:
    return str(key or "feature").replace("_", " ").strip().title()


def contains_cjk(value) -> bool:
    return isinstance(value, str) and bool(_CJK.search(value))


def english_text(value: str, fallback: str = "") -> str:
    """Preserve citations/numbers from mixed strings, otherwise use a safe fallback."""
    if not contains_cjk(value):
        return value
    if value in _EXACT:
        return _EXACT[value]
    # Evidence citations often mix an English field path/value with Chinese prose.
    # Preserve the auditable machine fragment instead of translating semantics that
    # were never independently represented.
    cleaned = _CJK_RUN.sub(" ", value)
    cleaned = cleaned.translate(str.maketrans({"，": ",", "。": ".", "：": ":",
                                               "；": ";", "（": "(", "）": ")",
                                               "、": ","}))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:.()")
    if re.search(r"[A-Za-z0-9_]", cleaned) and len(cleaned) >= 3:
        return cleaned
    return fallback or "English explanation unavailable; refer to the structured value."


def englishize(obj, *, key_hint: str = ""):
    """Recursively enforce an English-only natural-language boundary.

    Keys, numbers, booleans and nulls are unchanged.  For a Chinese-only label the
    stable machine key becomes the English label; prose receives an explicit,
    non-invented fallback rather than a guessed translation.
    """
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            clean_key = str(key)
            if contains_cjk(clean_key):
                # Dynamic labels occasionally become mapping keys. Preserve identity
                # deterministically without leaking source-language prose.
                clean_key = "translated_key_" + hashlib.sha1(
                    clean_key.encode("utf-8")).hexdigest()[:10]
            if isinstance(value, str) and contains_cjk(value):
                if clean_key == "label":
                    out[clean_key] = humanize(key_hint)
                elif clean_key in ("display", "narrative", "summary", "note"):
                    out[clean_key] = english_text(value, "Insufficient English evidence summary available.")
                elif clean_key in ("caveat", "reason"):
                    out[clean_key] = english_text(value, "Interpret with caution because source coverage is limited.")
                else:
                    out[clean_key] = english_text(value)
            else:
                out[clean_key] = englishize(value, key_hint=clean_key)
        return out
    if isinstance(obj, list):
        return [englishize(value, key_hint=key_hint) for value in obj]
    if isinstance(obj, str):
        return english_text(obj)
    return obj


def assert_english(obj) -> None:
    blob = json.dumps(obj, ensure_ascii=False, default=str)
    match = _CJK.search(blob)
    if match:
        raise ValueError(f"non-English CJK content remains at offset {match.start()}")
