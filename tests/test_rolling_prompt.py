"""Run: .venv/bin/python tests/test_rolling_prompt.py

The prompt view must stay a projection: compact, traceable, and free of both the
audit layers and the target.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.rolling_prompt import (COVERAGE_SHOWN, TOKEN_BUDGET, build_prompt_view,
                                _tokens)  # noqa: E402
from app.rolling_schema import (SAMPLE_SCHEMA_VERSION,
                                public_profile_contamination)  # noqa: E402

SAMPLES = REPO / "data" / "rolling_prediction" / "v4" / "samples"


def main():
    # Defense in depth is testable before a corpus at the newest schema exists.
    synthetic = {
        "sample_id": "u1_2026-08-12", "user_id": "u1",
        "cutoff_et": "2026-08-11T16:00:00-04:00", "prediction_date": "2026-08-12",
        "profile_snapshot": {
            "cutoff_et": "2026-08-11T16:00:00-04:00", "profile_version": 1,
            "base_profile": {"headline": "Active investor",
                             "summary": "Observed behavior was active.",
                             "dimensions": {"risk_preference": "Observed."}},
            "structured": {"investment_logic": {"narrative": "Observed trades."}},
            "data_quality": {},
        },
        "recent_7d": {"window_start": "2026-08-03", "window_end": "2026-08-11",
                      "trading_days": ["2026-08-03"]},
        "market_7d": {"regime": {"SPY": {"status": "available"}}},
        "observation_coverage": {},
    }
    contaminated = json.loads(json.dumps(synthetic))
    contaminated["profile_snapshot"]["base_profile"]["summary"] = (
        "Observed behavior was active. The action forecast ranked trim_position first.")
    try:
        build_prompt_view(contaminated)
    except ValueError as exc:
        assert "contamination" in str(exc) and "profile.summary" in str(exc)
    else:
        raise AssertionError("semantic action-forecast prose reached the prompt")
    assert not public_profile_contamination(build_prompt_view(synthetic)["profile"])

    # Only samples at the current schema: an older sample can carry a field the
    # view now refuses, and asserting against it would test history.
    paths = [p for p in sorted(SAMPLES.rglob("*.json"))
             if json.loads(p.read_text(encoding="utf-8")).get("schema_version")
             == SAMPLE_SCHEMA_VERSION]
    if not paths:
        print(f"PASS prompt semantic guard; no {SAMPLE_SCHEMA_VERSION} corpus yet")
        return
    sample = json.loads(paths[0].read_text(encoding="utf-8"))
    view = build_prompt_view(sample)
    # `view_meta` documents what was dropped, so it names the very fields these
    # checks scan for. Scan the model-facing payload, not the audit note.
    payload = {k: v for k, v in view.items() if k != "view_meta"}
    flat = json.dumps(payload, ensure_ascii=False)

    # 1. the audit layers are gone
    for banned in ("evidence_plain", "profile_tree", "kyc_tree"):
        assert banned not in flat, f"{banned} leaked into the prompt view"

    # 2. the signal layers survive
    assert view["profile"]["summary"], "profile summary missing"
    assert view["recent_7d"]["window_start"], "recent window missing"
    assert view["market_7d"], "market block missing"
    assert view["cutoff_et"] and view["prediction_date"]

    # 3. the ground-truth label never reaches the model. The structural check is
    # against the sample's `targets` block rather than a substring scan, because a
    # label value can legitimately coincide with profile prose.
    assert "targets" not in view, "the label block was copied into the view"
    label = sample.get("targets") or {}
    assert label, "sample has no targets block to guard against"
    assert json.dumps(label, ensure_ascii=False, sort_keys=True) not in \
        json.dumps(payload, ensure_ascii=False, sort_keys=True), "label leaked"
    for family in ("trade_decision", "net_trade_action", "position_actions"):
        node = label.get(family)
        if isinstance(node, dict) and node.get("label") is not None:
            # the label value may coincide with profile text, so check placement
            assert family not in flat, f"{family} leaked into the prompt view"

    # 4. it is a copy, not a rewrite
    assert view["market_7d"] == sample["market_7d"]
    assert view["profile"]["traits"] == sample["profile_snapshot"]["structured"]

    # 5. the budget is a real tripwire
    try:
        build_prompt_view(sample, budget=10)
    except ValueError:
        pass
    else:
        raise AssertionError("budget was not enforced")

    # 6. every built sample fits
    worst, worst_path = 0, None
    for path in paths:
        v = build_prompt_view(json.loads(path.read_text(encoding="utf-8")))
        if v["view_meta"]["approx_tokens"] > worst:
            worst, worst_path = v["view_meta"]["approx_tokens"], path
    assert worst <= TOKEN_BUDGET, f"{worst_path} needs {worst} tokens"

    # 7. the rule engine's own next-action forecast is not shown. It is pre-cutoff,
    # so it is not temporal leakage -- but it states an answer in the label
    # vocabulary, so a predictor could copy it and score the engine instead.
    assert "next_action" not in flat, "the rule-engine forecast reached the view"
    baseline = (sample.get("baselines") or {}).get("rule_engine_forecast")
    assert baseline is not None, "the forecast must still be archived as a baseline"

    # 8. the built view also passes the shared semantic scan.
    assert not public_profile_contamination(view["profile"])

    # 9. coverage is label provenance: only the cutoff-side facts may be shown.
    # `after_book_as_of` is the prediction date itself, and the execution-coverage
    # basis names a date weeks past the cutoff.
    shown = set(view["observation_coverage"])
    assert shown <= set(COVERAGE_SHOWN), f"label-side coverage shown: {shown}"
    for banned in ("after_book", "executions_observed", "execution_coverage_basis"):
        assert banned not in flat, f"{banned} leaked into the prompt view"
    pred = sample["prediction_date"]
    assert pred not in json.dumps(view["observation_coverage"]), \
        "coverage dates the prediction day"

    print(f"PASS rolling prompt view ({len(paths)} samples, worst {worst} tok, "
          f"budget {TOKEN_BUDGET}, full sample ~{_tokens(sample)} tok)")


if __name__ == "__main__":
    main()
