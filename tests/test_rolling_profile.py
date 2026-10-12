"""Run: .venv/bin/python tests/test_rolling_profile.py"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.forecast_scrub import find_leaks, leaks, scrub_prose, scrub_text  # noqa: E402
from app.rolling_profile import (forecast_baseline, llm_restatement,
                                 public_profile)  # noqa: E402
from app.rolling_schema import (PROFILE_DIMENSIONS, PROFILE_SCHEMA_VERSION,
                                public_profile_contamination)  # noqa: E402


def _doc():
    dims = {dim: "Supported historical observation." for dim in PROFILE_DIMENSIONS}
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "user_id": "u1", "profile_version": 3, "parent_version": 2,
        "cutoff_et": "2026-08-11T16:00:00-04:00", "profile_date": "2026-08-11",
        "base_profile": {
            "summary": ("The user previously trimmed positions after gains. "
                        "The action engine forecasts a trim as the next action."),
            "dimensions": dims,
            "highlights": ["Observed quote views increased.",
                           "The model forecast ranks trimming first."],
        },
        "structured": {"investment_logic": {
            "narrative": ("Observed position changes were incremental. "
                          "The action forecast ranked trim_position first."),
            "nested": {"prediction_detail": {"primary": "trim_position"}},
            "next_action": {"primary": "trim_position"},
            "key_signals": ["The user previously trimmed positions after gains.",
                            "action_forecast_digest.primary=trim_position with confidence=0.8."],
        }},
        "evidence_summary": {"action_forecast": {
            "available": True, "version": "v1", "confidence": 0.8,
            "margin": 0.2, "candidates": [
                {"action": "trim_position", "score": 0.8, "targets": []},
                {"action": "add_to_existing", "score": 0.6, "targets": []}],
        }},
        "data_quality": {}, "provenance": {},
    }


def main():
    mixed = ("The user previously trimmed positions after gains. "
             "The action forecast ranked trim_position first.")
    cleaned = scrub_text(mixed)
    assert cleaned == "The user previously trimmed positions after gains."
    assert not leaks("Analyst market forecasts remain uncertain.")
    assert not leaks("The user previously trimmed positions after gains.")
    hits = find_leaks({"x": mixed})
    assert len(hits) == 1 and hits[0].startswith("x='The user previously trimmed")

    list_cleaned = scrub_prose(["Observed quote views increased.",
                                "The model forecast ranks trimming first."])
    assert list_cleaned == ["Observed quote views increased."]

    doc = _doc()
    baseline = forecast_baseline(doc)
    restatement = llm_restatement(doc)
    public = public_profile(doc)
    assert baseline["primary_action"] == "trim_position"
    assert restatement["investment_logic.next_action"]["primary"] == "trim_position"
    assert not public_profile_contamination(public)
    assert public["base_profile"]["summary"] == \
        "The user previously trimmed positions after gains."
    assert public["base_profile"]["highlights"] == ["Observed quote views increased."]
    assert public["structured"]["investment_logic"]["key_signals"] == [
        "The user previously trimmed positions after gains."]
    assert "next_action" not in public["structured"]["investment_logic"]
    assert "nested" not in public["structured"]["investment_logic"]
    assert all(public["base_profile"]["dimensions"].get(dim)
               for dim in PROFILE_DIMENSIONS)
    print("PASS rolling public-profile forecast sanitizer")


if __name__ == "__main__":
    main()
