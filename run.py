"""Run the experiment in three stages: prepare, build, predict."""
from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env():
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, sep, value = line.removeprefix('export ').partition('=')
            if sep:
                os.environ.setdefault(key.strip(), value.strip().strip('\"\''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'build', 'predict'])
    parser.add_argument('--end', default='2026-10-07', help='last source/target date')
    parser.add_argument('--sessions', type=int, default=40, help='prediction trading sessions')
    parser.add_argument('--coverage-through', help='last date of verified complete execution data')
    parser.add_argument('--model', help='override prediction model')
    parser.add_argument('--limit', type=int, default=0, help='prediction samples per user; 0 means all')
    parser.add_argument('--out', help='prediction output directory')
    parser.add_argument('--resume', action='store_true', help='resume the same prediction configuration')
    parser.add_argument('--dry-run', action='store_true', help='prepare: preview commands; build/predict: validate without model calls')
    args = parser.parse_args()
    load_env()
    users = ROOT / 'data/users.txt'
    if not users.is_file():
        parser.error('create data/users.txt with one anonymous user ID per line')
    ids = users.read_text().split()
    if not ids:
        parser.error('data/users.txt is empty')
    if args.sessions < 1 or args.limit < 0:
        parser.error('--sessions must be positive and --limit cannot be negative')
    if args.stage == 'prepare':
        if not (ROOT / 'data/raw_records').is_dir() or not (ROOT / 'data/symbols.txt').is_file():
            parser.error('prepare requires data/raw_records/ and data/symbols.txt')
    if not args.dry_run:
        if not (os.environ.get('AIME_LITELLM_API_KEY') and os.environ.get('LUNA_BASE_URL')
                and ((args.model if args.stage == 'predict' else None) or os.environ.get('LUNA_MODEL') or os.environ.get('LUNA_MODEL_STRONG'))):
            parser.error('set the API key, endpoint and model in .env')
    # Read the shared experiment boundary without importing the profile engine.
    from app.rolling_config import PROFILE_CUTOFF
    commands = []
    if args.stage == 'prepare':
        for database, cutoff in [('observations_t0.db', PROFILE_CUTOFF), ('observations_full.db', args.end)]:
            commands.append(['load_data.py', '--raw', 'data/raw_records', '--out', f'data/{database}',
                             '--cutoff', cutoff, '--users-file', 'data/users.txt'])
        commands += [
            ['scripts/profiles.py', '--db', 'data/observations_t0.db', '--users-file', 'data/users.txt'],
            ['export_profiles.py', '--db', 'data/observations_t0.db', '--users-file', 'data/users.txt',
             '--expected-users', str(len(set(ids))), '--out', 'data/rolling_prediction/base_t0'],
            ['scripts/market.py', '--symbols-file', 'data/symbols.txt', '--cache-dir', 'data/market_raw',
             '--start', PROFILE_CUTOFF, '--end', args.end],
        ]
    elif args.stage == 'build':
        command = ['build_samples.py', '--t0-db', 'data/observations_t0.db',
                   '--full-db', 'data/observations_full.db', '--users-file', 'data/users.txt',
                   '--market-cache', 'data/market_raw', '--end', args.end,
                   '--prediction-sessions', str(args.sessions)]
        if args.coverage_through:
            command += ['--execution-coverage-through', args.coverage_through]
        if args.dry_run:
            command += ['--dry-run']
        commands.append(command)
        if not args.dry_run:
            commands.append(['check_samples.py', '--out', 'data/rolling_prediction',
                             '--users-file', 'data/users.txt', '--end', args.end,
                             '--prediction-sessions', str(args.sessions)])
    else:
        command = ['predict.py', '--users-file', 'data/users.txt', '--limit', str(args.limit)]
        if args.model:
            command += ['--model', args.model]
        if args.out:
            command += ['--out', args.out]
        if args.resume:
            command += ['--resume']
        if args.dry_run:
            command += ['--dry-run']
        commands.append(command)
    for command in commands:
        print(shlex.join(['python'] + command), flush=True)
        if args.stage == 'prepare' and args.dry_run:
            continue
        result = subprocess.run([sys.executable] + command, cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
