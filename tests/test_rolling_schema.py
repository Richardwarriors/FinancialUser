"""Run: .venv/bin/python tests/test_rolling_schema.py"""
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.rolling_schema import (PROFILE_DIMENSIONS, PROFILE_SCHEMA_VERSION,
                                PUBLIC_PROFILE_SCHEMA_VERSION,
                                public_profile_contamination, validate_profile,
                                validate_public_profile)  # noqa: E402


def _public():
    return {
        "schema_version": PUBLIC_PROFILE_SCHEMA_VERSION,
        "user_id": "u1", "profile_version": 1,
        "cutoff_et": "2026-08-01T16:00:00-04:00",
        "profile_date": "2026-08-01",
        "base_profile": {
            "summary": "The user previously trimmed positions after gains.",
            "dimensions": {dim: "Supported historical observation."
                           for dim in PROFILE_DIMENSIONS},
            "caveats": ["Market forecasts are uncertain."],
        },
        "structured": {"investment_logic": {
            "narrative": "The user previously trimmed positions after gains."}},
        "data_quality": {}, "provenance": {},
    }


def _rejected(doc, expected):
    try:
        validate_public_profile(doc)
    except ValueError as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError("contaminated public profile was accepted")


def main():
    valid = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "user_id": "u1", "profile_version": 1,
        "cutoff_et": "2026-08-01T16:00:00-04:00", "profile_date": "2026-08-01",
        "base_profile": {"summary": "observed"},
        "structured": {"risk": {"confidence": 0.5}},
        "data_quality": {}, "provenance": {},
    }
    assert validate_profile(valid) is valid
    bad = {**valid, "schema_version": "financial_base_profile_v1"}
    try:
        validate_profile(bad)
    except ValueError as exc:
        assert "unsupported profile schema" in str(exc)
    else:
        raise AssertionError("legacy schema must not be accepted silently")

    clean = _public()
    assert validate_public_profile(clean) is clean
    assert not public_profile_contamination(clean)

    nested_key = deepcopy(clean)
    nested_key["structured"]["investment_logic"]["nested"] = {
        "action_forecast_digest": {"primary": "trim_position"}}
    _rejected(nested_key, "action_forecast_digest")

    prose = deepcopy(clean)
    prose["base_profile"]["summary"] = (
        "Observed trading was active. The action forecast ranked trim_position first.")
    _rejected(prose, "base_profile.summary")

    highlight = deepcopy(clean)
    highlight["base_profile"]["highlights"] = [
        "The model forecast identifies trimming as the most likely next action."]
    _rejected(highlight, "base_profile.highlights[0]")

    print("PASS rolling schema contract and semantic contamination checks")


if __name__ == "__main__":
    main()
