"""Explicitly fetch Yahoo daily bars for a locally supplied list of symbols."""
import argparse
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rolling_market import YahooMarketCache, underlying_symbol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols-file', required=True)
    parser.add_argument('--cache-dir', default='data/market_raw')
    parser.add_argument('--start', required=True, type=date.fromisoformat)
    parser.add_argument('--end', required=True, type=date.fromisoformat)
    args = parser.parse_args()
    cache = YahooMarketCache(args.cache_dir)
    symbols = {'SPY', 'QQQ'} | {
        underlying_symbol(s)[0] for s in Path(args.symbols_file).read_text().split()}
    for symbol in sorted(symbols):
        rows = cache.fetch(symbol, args.start, args.end)
        print(f'{symbol}: {len(rows)} bars')


if __name__ == '__main__':
    main()
