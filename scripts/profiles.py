"""Build seed profiles from a physically truncated T0 database."""
import argparse
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.harness_graph import refresh
from app.rolling_config import PROFILE_CUTOFF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--users-file', required=True)
    args = parser.parse_args()
    # Fail before any model call if the purported seed database contains future rows.
    with sqlite3.connect(f'file:{args.db}?mode=ro', uri=True) as conn:
        for table, field in [('events', 'event_date'), ('trade_executions', 'trade_date'),
                             ('account_snapshots', 'snapshot_date')]:
            if conn.execute(f'SELECT COUNT(*) FROM {table} WHERE {field} > ?',
                            (PROFILE_CUTOFF,)).fetchone()[0]:
                raise SystemExit(f'{table} contains rows after PROFILE_CUTOFF')
    users = Path(args.users_file).read_text().split()
    if not users:
        raise SystemExit('users-file is empty')
    for uid in users:
        result = refresh(args.db, uid, cutoff_et=f'{PROFILE_CUTOFF}T16:00:00-04:00')
        profile = result.get('final_profile') or {}
        if not profile.get('version'):
            raise SystemExit(f'No persisted profile for {uid}; check observation coverage')
        print(f'{uid}: version {profile["version"]}')


if __name__ == '__main__':
    main()
