"""Financial data gateway with point-in-time source adapters.

Streams:
    behavior_events   -> get_behaviors
    trade_executions  -> get_trades
    account_snapshots -> get_account_daily
    conversations     -> get_conversations

SQLiteDataSource reads normalized local records prepared by load_naacl_db.py.
HiveDataSource reserves the same interface for a future Hive adapter.
Agents consume the interface independently of the underlying connector.
"""
import json
import sqlite3
from abc import ABC, abstractmethod

from .rolling_time import parse_cutoff, parse_event_et, parse_execution_et


class FinancialDataSource(ABC):
    """Canonical read interface for all financial observation streams."""

    name = "abstract"

    @abstractmethod
    def get_behaviors(self, user_id): ...

    @abstractmethod
    def get_trades(self, user_id): ...

    @abstractmethod
    def get_account_daily(self, user_id): ...

    @abstractmethod
    def get_conversations(self, user_id): ...

    @abstractmethod
    def get_user_summary(self, user_id): ...

    @abstractmethod
    def list_users(self): ...

    def source_version(self):
        return f"{self.name}:local-snapshot"


class SQLiteDataSource(FinancialDataSource):
    """Local SQLite adapter: periodic pull persisted into local SQLite."""

    name = "link"

    def __init__(self, db_path, cutoff_et=None, include_conversations=True):
        self.db_path = str(db_path)
        self.cutoff_et = parse_cutoff(cutoff_et)
        self.include_conversations = include_conversations

    def at(self, cutoff_et, include_conversations=False):
        """Return a point-in-time view over the same database.

        Conversations have no timestamps, so rolling views exclude them by default.
        They remain available to the fixed T0 build where the database itself is
        physically truncated at the boundary.
        """
        return type(self)(self.db_path, cutoff_et=cutoff_et,
                          include_conversations=include_conversations)

    def _con(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _has_table(self, name):
        c = self._con()
        return c.execute("select 1 from sqlite_master where type='table' and name=?",
                         (name,)).fetchone() is not None

    def list_users(self):
        """Every user id that appears in ANY input stream.

        This used to read only the `users` table. That silently hid anyone who
        shows up in a financial data stream without a row there — which is exactly what
        happens at scale: a user's first appearance may be a single behavior_events event,
        a conversation, or one daily snapshot. The contract is now the UNION, so
        such a user still gets a profile built from whatever exists and is filled
        in later as more streams arrive.

        Output-only tables (profile_versions, audit_log, profile_updates) are
        deliberately excluded: having a profile is not a source of user identity.
        """
        streams = ("events", "trades", "trade_executions", "equity_summaries",
                   "account_snapshots", "conversations", "users")
        c = self._con()
        found = set()
        for table in streams:
            try:
                rows = c.execute(
                    f"select distinct user_id from {table} where user_id is not null")
            except sqlite3.OperationalError:
                continue                     # table not present in this database
            found.update(r["user_id"] for r in rows)
        return sorted(found)

    def get_user_summary(self, user_id):
        """Stated background, or "" when absent.

        A missing row is a data gap, not an error: with the union user universe a
        user can legitimately exist in a financial data stream while having no `users` row,
        and refusing to build their snapshot would defeat the point.
        """
        c = self._con()
        r = c.execute("select base_profile_summary from users where user_id=?", (user_id,)).fetchone()
        if not r:
            return ""
        return r["base_profile_summary"] or ""

    def get_behaviors(self, user_id):
        c = self._con()
        if self._has_table("events"):
            sql = ("select event_date,timestamp_et,action_text,source from events "
                   "where user_id=? order by timestamp_et")
        elif self._has_table("behavior_events"):
            sql = ("select event_date,timestamp_et,action_text,null as source "
                   "from behavior_events where user_id=? order by timestamp_et")
        else:
            return []
        rows = [dict(r) for r in c.execute(sql, (user_id,))]
        if self.cutoff_et is None:
            return rows
        return [r for r in rows
                if (parse_event_et(r.get("timestamp_et")) is not None
                    and parse_event_et(r.get("timestamp_et")) <= self.cutoff_et)]

    def get_trades(self, user_id):
        """trade_executions trades.

        `executed_at` is the real execution timestamp recovered from the raw
        payload. The `trade_date` column is NOT the trade date — it is the label
        extraction date, and differs from the real date on every row observed,
        so it must never be used for time-windowed metrics.
        """
        c = self._con()
        out = []
        cols = {r[1] for r in c.execute("pragma table_info(trade_executions)")}
        if "executed_quantity" in cols:
            sql = ("select trade_date,ticker,side,executed_quantity,executed_amount,"
                   "execution_state,payload,executed_at,order_at,venue "
                   "from trade_executions where user_id=? order by coalesce(executed_at,trade_date)")
        else:
            sql = ("select executed_date as trade_date,ticker,side,"
                   "executed_qty as executed_quantity,executed_amount,execution_state,"
                   "null as payload,executed_at,null as order_at,null as venue "
                   "from trade_executions where user_id=? order by executed_at")
        for r in c.execute(sql, (user_id,)):
            t = dict(r)
            # recover price from raw payload when available: bits[8] is price
            try:
                bits = (t.get("payload") or "").split("#")
                t["price"] = float(bits[8]) if len(bits) > 8 and bits[8] else None
            except Exception:
                t["price"] = None
            executed_et = parse_execution_et(t.get("executed_at"))
            if self.cutoff_et is not None and (executed_et is None
                                               or executed_et > self.cutoff_et):
                continue
            t["executed_at_et"] = executed_et.isoformat() if executed_et else None
            t["trade_day"] = executed_et.date().isoformat() if executed_et else None
            out.append(t)
        return out

    def get_account_daily(self, user_id):
        """account_snapshots daily account snapshot: asset/stock/cash totals + holdings.

        This single stream replaces the legacy weekly_holdings (weekly holdings) and
        equity_summary (6-month equity summary) channels: it is daily, per-account and
        carries the full position book, so both the position and the asset
        evidence are derived from it.
        """
        c = self._con()
        out = []
        cols = {r[1] for r in c.execute("pragma table_info(account_snapshots)")}
        n_holdings = "n_holdings" if "n_holdings" in cols else "null as n_holdings"
        for r in c.execute(
                f"select snapshot_date,asset_total,stock_total,money_remain,"
                f"stock_profit_daily,holdings_json,{n_holdings} from account_snapshots "
                "where user_id=? order by snapshot_date", (user_id,)):
            d = dict(r)
            try:
                d["holdings"] = json.loads(d.pop("holdings_json") or "{}")
            except Exception:
                d["holdings"] = {}
            if (self.cutoff_et is None
                    or d.get("snapshot_date") <= self.cutoff_et.date().isoformat()):
                out.append(d)
        return out

    def get_conversations(self, user_id):
        """conversations conversation-title stream (user questions asked to Aime).

        conversations has no timestamp.  It is therefore permitted for a database-level T0
        snapshot, but excluded from rolling point-in-time views by default.
        """
        if not self.include_conversations or not self._has_table("conversations"):
            return []
        c = self._con()
        return [dict(r) for r in c.execute(
            "select seq,title from conversations where user_id=? order by seq",
            (user_id,))]


class HiveDataSource(FinancialDataSource):
    """Future Hive connector. Same contract, zero downstream change."""

    name = "hive"

    def __getattr__(self, item):
        raise NotImplementedError(
            "HiveDataSource is reserved for the future Hive rollout; "
            "swap it in behind the same FinancialDataSource interface.")


def get_datasource(kind="link", db_path="data/financial.db"):
    if kind == "link":
        return SQLiteDataSource(db_path)
    if kind == "hive":
        return HiveDataSource()
    raise ValueError(f"unknown datasource kind: {kind}")
