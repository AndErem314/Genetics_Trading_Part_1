"""
Crypto Data Downloader — Binance Spot 1h OHLCV
================================================
Downloads historical 1-hour candle data for multiple crypto pairs from
Binance spot market using the ccxt library (no API key required).

Usage:
    python crypto_data_downloader.py                # download all symbols
    python crypto_data_downloader.py --update       # append new bars only
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "LTC/USDT"]
TIMEFRAME = "1h"
DATA_DIR = Path(__file__).parent / "data"
START_DATE = "2021-07-01T00:00:00Z"      # earliest date for all symbols
BATCH_LIMIT = 1000                        # Binance max candles per request
RATE_LIMIT_SLEEP = 0.5                    # seconds between requests


def symbol_to_filename(symbol: str) -> str:
    """Convert 'BTC/USDT' → 'BTCUSDT_1h.csv'."""
    return f"{symbol.replace('/', '')}_{TIMEFRAME}.csv"


def fetch_ohlcv_all(exchange: ccxt.Exchange, symbol: str,
                    since_ms: int) -> pd.DataFrame:
    """Fetch all 1h OHLCV candles from *since_ms* to now, paginating."""
    all_candles = []
    fetch_since = since_ms
    now_ms = exchange.milliseconds()

    print(f"  Fetching {symbol} from {exchange.iso8601(fetch_since)} ...")

    while fetch_since < now_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, TIMEFRAME, since=fetch_since, limit=BATCH_LIMIT
            )
        except ccxt.NetworkError as e:
            print(f"  ⚠ Network error, retrying in 5s: {e}")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            print(f"  ✖ Exchange error: {e}")
            break

        if not candles:
            break

        all_candles.extend(candles)

        # Move past the last candle timestamp
        last_ts = candles[-1][0]
        if last_ts == fetch_since:
            # No progress — we've reached the end
            break
        fetch_since = last_ts + 1

        # Progress
        pct = min(100, (fetch_since - since_ms) / max(1, now_ms - since_ms) * 100)
        print(f"    {len(all_candles):>8,} candles  "
              f"({pct:5.1f}%)  last: {exchange.iso8601(last_ts)}", end="\r")

        time.sleep(RATE_LIMIT_SLEEP)

    print()  # newline after progress

    if not all_candles:
        return pd.DataFrame(columns=["timestamp", "open", "high",
                                     "low", "close", "volume"])

    df = pd.DataFrame(all_candles,
                      columns=["timestamp", "open", "high",
                               "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)
    return df


def download_symbol(exchange: ccxt.Exchange, symbol: str,
                    update: bool = False) -> None:
    """Download (or update) a single symbol and save to CSV."""
    csv_path = DATA_DIR / symbol_to_filename(symbol)
    since_ms = exchange.parse8601(START_DATE)

    if update and csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["timestamp"])
        last_ts = existing["timestamp"].max()
        since_ms = int(last_ts.timestamp() * 1000) + 1
        print(f"  Updating {symbol} from {last_ts} ...")
    else:
        existing = None

    new_data = fetch_ohlcv_all(exchange, symbol, since_ms)

    if existing is not None and not new_data.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        combined.to_csv(csv_path, index=False)
        print(f"  ✔ {symbol}: {len(combined):,} total bars → {csv_path.name}")
    elif not new_data.empty:
        new_data.to_csv(csv_path, index=False)
        print(f"  ✔ {symbol}: {len(new_data):,} bars → {csv_path.name}")
    else:
        print(f"  ⚠ {symbol}: no new data fetched")


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance spot 1h OHLCV data for GP crypto strategy"
    )
    parser.add_argument("--update", action="store_true",
                        help="Only fetch new candles (append to existing CSVs)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override default symbol list (e.g. BTC/USDT ETH/USDT)")
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Crypto Data Downloader — Binance Spot 1h OHLCV")
    print(f"Symbols: {symbols}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Output: {DATA_DIR.resolve()}")
    print("=" * 60)

    exchange = ccxt.binance({"enableRateLimit": True})

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        download_symbol(exchange, symbol, update=args.update)

    print(f"\n✔ All done. Data saved to {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
