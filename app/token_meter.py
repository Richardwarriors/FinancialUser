"""Token accounting for LLM calls.

The harness already counted agent *calls* but not tokens, so cost was invisible:
"3 agents ran" says nothing about whether that was 4k or 40k tokens. This module
records real usage from the provider response, grouped by stage, so a refresh can
be attributed (orchestrator vs dimensions vs portrait vs KYC) rather than only
totalled.

Usage is a context variable, so a batch worker accumulates per user and a nested
call is attributed to the same accumulator. Nothing here raises: metering must
never be able to fail a profile.
"""
import contextvars

_ACC = contextvars.ContextVar("token_acc", default=None)

# stages, ordered by when they run in a refresh
STAGES = ("orchestrator", "dimension", "portrait", "kyc")


def start():
    """Begin accounting for one unit of work (one user refresh)."""
    acc = {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "by_stage": {},
        "by_model": {},
    }
    _ACC.set(acc)
    return acc


def current():
    return _ACC.get()


def record(stage, model, usage):
    """Add one call's usage. `usage` is AIMessage.usage_metadata (or None)."""
    acc = _ACC.get()
    if acc is None or not usage:
        return
    it = usage.get("input_tokens") or 0
    ot = usage.get("output_tokens") or 0
    rt = ((usage.get("output_token_details") or {}).get("reasoning")) or 0
    acc["calls"] += 1
    acc["input_tokens"] += it
    acc["output_tokens"] += ot
    acc["reasoning_tokens"] += rt
    for bucket, key in ((acc["by_stage"], stage or "unknown"),
                        (acc["by_model"], model or "unknown")):
        b = bucket.setdefault(key, {"calls": 0, "input_tokens": 0,
                                    "output_tokens": 0, "reasoning_tokens": 0})
        b["calls"] += 1
        b["input_tokens"] += it
        b["output_tokens"] += ot
        b["reasoning_tokens"] += rt


def report():
    """Snapshot of the current accumulator, with totals + per-stage breakdown."""
    acc = _ACC.get()
    if acc is None:
        return None
    out = dict(acc)
    out["total_tokens"] = acc["input_tokens"] + acc["output_tokens"]
    for b in list(out["by_stage"].values()) + list(out["by_model"].values()):
        b["total_tokens"] = b["input_tokens"] + b["output_tokens"]
    return out


def merge_into(target, source):
    """Fold one accumulator into a running total (for fleet-level averages)."""
    if not source:
        return target
    target["users"] = target.get("users", 0) + 1
    target["calls"] = target.get("calls", 0) + source.get("calls", 0)
    target["input_tokens"] = target.get("input_tokens", 0) + source.get("input_tokens", 0)
    target["output_tokens"] = target.get("output_tokens", 0) + source.get("output_tokens", 0)
    target["total_tokens"] = target["input_tokens"] + target["output_tokens"]
    for stage, b in (source.get("by_stage") or {}).items():
        t = target.setdefault("by_stage", {}).setdefault(
            stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        for k in ("calls", "input_tokens", "output_tokens", "total_tokens"):
            t[k] = t.get(k, 0) + b.get(k, 0)
    return target
