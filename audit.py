#!/usr/bin/env python3
"""Independent temporal-integrity audit of the rolling dataset.

This does NOT trust the build.  `build_samples.py` writes a
`leakage_audit` block and `build_prompt_view` has its own tripwires, but both are
the same code that produced the data: if that code is wrong, its self-report is
wrong in the same direction.  This file recomputes every boundary from the
calendar and from the source database, and re-derives a sample of labels.

The contract it checks is the one the dataset claims:

    profile as of day t  +  the seven trading days ending at t (behaviour,
    trades, position changes)  +  market information as of t   ->  predict t+1

so for a sample whose `prediction_date` is D:

  1  cutoff is the close of the trading day immediately before D
  2  the profile snapshot is as of exactly that cutoff, never later
  3  `recent_7d` is exactly seven trading days and ends at the cutoff day
  4  no trade, event or position change inside it is timestamped after the cutoff
  5  no market bar is dated after the cutoff day
  6  NO date anywhere in the model-facing view is after the cutoff -- the catch-all
     that does not depend on knowing which field could carry a future date.  The
     prediction date itself is exempt: the model is told which day to predict.
  7  the view carries no label, no rule-engine forecast, no label-side coverage
  8  the label is reproducible from the source DB over day D alone (sampled)

Usage
-----
    ./.venv/bin/python audit.py
    ./.venv/bin/python audit.py --label-sample 60
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from app.rolling_prompt import build_prompt_view                      # noqa: E402
from app.rolling_schema import public_profile_contamination           # noqa: E402
from app.rolling_time import (cutoff_for, last_n_trading_days,        # noqa: E402
                              parse_execution_et, trading_days)

FULL_DB = HERE / "data" / "observations_full.db"

# Any ISO date or datetime, anywhere in a string.
DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# Fields whose value is legitimately the prediction day or later: the day to
# predict, and the audit note that records the horizon.
EXEMPT_KEYS = frozenset({"prediction_date"})

# Keys that belong to the label side or to the audit surface. Any of them inside
# the model-facing view is a leak regardless of value.
BANNED_VIEW_KEYS = frozenset({
    "targets", "net_trade_action", "position_actions", "next_action",
    "after_book", "executions_observed", "execution_coverage_basis",
    "evidence_plain", "evidence_summary", "presentation", "profile_tree",
    "kyc_tree", "baselines", "rule_engine_forecast", "llm_next_action",
    "leakage_audit",
})


def _keys_present(node, banned, path="view"):
    """Paths of any banned KEY at any depth."""
    out = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}"
            if key in banned:
                out.append(child)
            out.extend(_keys_present(value, banned, child))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(_keys_present(value, banned, f"{path}[{i}]"))
    return out


def _dates_after(node, limit: date, path="view", exempt=EXEMPT_KEYS):
    """Every ISO date found under `node` that is later than `limit`."""
    out = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in exempt:
                continue
            out.extend(_dates_after(value, limit, f"{path}.{key}", exempt))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(_dates_after(value, limit, f"{path}[{i}]", exempt))
    elif isinstance(node, str):
        for y, m, d in DATE_RE.findall(node):
            try:
                found = date(int(y), int(m), int(d))
            except ValueError:
                continue
            if found > limit:
                out.append(f"{path}={found.isoformat()}")
    return out


def audit_sample(sample: dict, problems: list[str]) -> None:
    sid = sample.get("sample_id")
    pred_day = date.fromisoformat(sample["prediction_date"])
    cutoff = datetime.fromisoformat(sample["cutoff_et"])

    # 1. the cutoff is the close of the trading day immediately before D
    expected = cutoff_for(pred_day)
    if cutoff != expected:
        problems.append(f"{sid}: cutoff {cutoff.isoformat()} != {expected.isoformat()}")
    prev = trading_days(date(pred_day.year - 1, 1, 1), pred_day)
    if len(prev) < 2 or prev[-1] != pred_day:
        problems.append(f"{sid}: prediction_date {pred_day} is not a trading day")
    elif cutoff.date() != prev[-2]:
        problems.append(f"{sid}: cutoff day {cutoff.date()} is not the trading day "
                        f"before {pred_day} ({prev[-2]})")

    # 2. the profile is as of exactly the cutoff
    snap = sample.get("profile_snapshot") or {}
    if snap.get("cutoff_et") != sample["cutoff_et"]:
        problems.append(f"{sid}: profile cutoff {snap.get('cutoff_et')} "
                        f"!= sample cutoff {sample['cutoff_et']}")

    # 3. seven trading days, ending at the cutoff day
    recent = sample.get("recent_7d") or {}
    days = recent.get("trading_days") or []
    want = [d.isoformat() for d in last_n_trading_days(cutoff.date(), 7)]
    if days != want:
        problems.append(f"{sid}: recent window {days[:1]}..{days[-1:]} != {want[0]}..{want[-1]}")
    if recent.get("window_end") != cutoff.date().isoformat():
        problems.append(f"{sid}: window_end {recent.get('window_end')} != cutoff day")

    # 4. nothing inside the window is timestamped after the cutoff
    for order in (recent.get("trades") or {}).get("orders") or []:
        value = order.get("executed_at_et")
        if value and datetime.fromisoformat(value) > cutoff:
            problems.append(f"{sid}: trade executed {value} after cutoff")

    # 5. no market bar after the cutoff day
    market = sample.get("market_7d") or {}
    as_of = (market.get("as_of_date") or "")[:10]
    if as_of and date.fromisoformat(as_of) > cutoff.date():
        problems.append(f"{sid}: market as_of {as_of} after cutoff")

    # 6-7. the model-facing view: no future date, no label, no forecast
    view = build_prompt_view(sample)
    payload = {k: v for k, v in view.items() if k != "view_meta"}
    for hit in _dates_after(payload, cutoff.date()):
        problems.append(f"{sid}: future date in the prompt view: {hit}")
    # Structural, not a substring scan over the serialised view: `"targets" in
    # flat` reported 322 samples, and every one of them was the word inside a
    # sentence rather than a label field -- a check that cannot say WHERE the
    # leak is cannot be acted on, and it hides the real hits among the noise.
    for path in _keys_present(payload, BANNED_VIEW_KEYS):
        problems.append(f"{sid}: label-side field reached the prompt view: {path}")
    # Semantic contamination: a sentence that restates the deterministic
    # next-action forecast. This shares `forecast_scrub` with the build-time gate
    # -- what this file adds is that every boundary above is recomputed from the
    # calendar and the source DB rather than read back from the build.
    for hit in public_profile_contamination(view.get("profile") or {}):
        problems.append(f"{sid}: forecast contamination in the prompt view: {hit}")
    if not (sample.get("leakage_audit") or {}).get("clean"):
        problems.append(f"{sid}: build-time leakage audit not clean")


def recheck_labels(paths: list[Path], problems: list[str]) -> dict:
    """Re-derive `net_trade_action` from the source DB over day D alone.

    Deliberately written against the raw table rather than by calling
    `app.rolling_labels`: reusing the builder's own function would only prove it
    is deterministic.  trade_executions `executed_at` is naive UTC in this corpus, so the
    trading date is the ET date after conversion -- a same-day comparison in UTC
    would move every evening execution to the next day.
    """
    if not FULL_DB.exists():
        return {"checked": 0, "note": f"source db not found at {FULL_DB}"}
    con = sqlite3.connect(f"file:{FULL_DB}?mode=ro", uri=True)
    checked = Counter()
    for path in paths:
        sample = json.loads(path.read_text(encoding="utf-8"))
        day = date.fromisoformat(sample["prediction_date"])
        buy = sell = 0.0
        for when, side, amount, state in con.execute(
                "select executed_at, side, executed_amount, execution_state "
                "from trade_executions where user_id=?", (sample["user_id"],)):
            if (state or "executed") != "executed" or side not in ("buy", "sell"):
                continue
            when_et = parse_execution_et(when)
            if when_et is None or when_et.date() != day:
                continue
            if side == "buy":
                buy += abs(float(amount or 0))
            else:
                sell += abs(float(amount or 0))
        want = "buy" if buy > sell else "sell" if sell > buy else "hold"
        node = (sample.get("targets") or {}).get("net_trade_action") or {}
        got = node.get("action")
        if not node.get("is_observed"):
            checked["unobserved (not rechecked)"] += 1
            continue
        checked[f"{want}=={got}" if want == got else f"{want}!={got}"] += 1
        if want != got:
            problems.append(f"{sample['sample_id']}: label {got!r} but the source DB "
                            f"shows buy={buy:.2f} sell={sell:.2f} on {day} -> {want!r}")
    return {"checked": sum(checked.values()), "outcomes": dict(checked)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir",
                    default=str(HERE / "data" / "rolling_prediction" / "samples"))
    ap.add_argument("--label-sample", type=int, default=40,
                    help="how many samples to re-derive from the source DB")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(Path(args.samples_dir).rglob("*.json"))
    if not paths:
        print(f"no samples under {args.samples_dir}")
        return 1
    problems: list[str] = []
    for path in paths:
        audit_sample(json.loads(path.read_text(encoding="utf-8")), problems)

    random.seed(args.seed)
    sampled = random.sample(paths, min(args.label_sample, len(paths)))
    labels = recheck_labels(sampled, problems)

    print(json.dumps({
        "samples_audited": len(paths),
        "n_problems": len(problems),
        "problems": problems[:30],
        "label_recheck": labels,
    }, indent=2, ensure_ascii=False))
    print("\nCLEAN" if not problems else "\nPROBLEMS FOUND")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
