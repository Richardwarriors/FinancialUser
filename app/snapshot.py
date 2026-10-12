"""Canonical UserFinancialSnapshot: the data contract boundary.

Everything upstream (link now / Hive later) ends here; everything downstream
(evidence builder, agents, harness) only ever sees this structure.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

SCHEMA_VERSION = "v2"


def _now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class UserFinancialSnapshot:
    user_id: str
    as_of: str = field(default_factory=_now)
    behaviors_6m: list = field(default_factory=list)     # behavior_events
    trades_6m: list = field(default_factory=list)        # trade_executions
    account_daily: list = field(default_factory=list)    # account_snapshots (replaces weekly_holdings + equity_summary)
    conversations_1m: list = field(default_factory=list)  # conversations
    # NOTE `users.base_profile_summary` is deliberately NOT carried into the
    # snapshot. It is a demographic self-description (国籍 94/157, 语言 96/157,
    # 年龄 39/157 users, often literally "Chinese national"), rewritten into first
    # person by an upstream LLM from third-person facts that included a full
    # locality. Person data has no analytical role in an investment profile, so the
    # channel is cut here rather than filtered later — see app/pii.py for the
    # boundary guard that covers every OTHER source.
    source_version: str = ""
    extraction_time: str = field(default_factory=_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self):
        return asdict(self)


def build_snapshot(ds, user_id, as_of=None):
    """Pull all streams through the datasource interface into one snapshot.

    ``as_of`` describes the evidence boundary, not the wall-clock build time.  A
    rolling caller must pass it explicitly so lifecycle age and leakage checks use
    the prediction cutoff rather than the date on which the backtest was run.
    """
    values = {
        "user_id": user_id,
        "behaviors_6m": ds.get_behaviors(user_id),
        "trades_6m": ds.get_trades(user_id),
        "account_daily": ds.get_account_daily(user_id),
        "conversations_1m": ds.get_conversations(user_id),
        "source_version": ds.source_version(),
    }
    if as_of is not None:
        values["as_of"] = str(as_of)
    return UserFinancialSnapshot(**values)
