"""
Historical Data Backfill Script

Fetches historical OHLCV data from Bybit and stores in SQLite.
Handles pagination and rate limits.
"""

import ccxt
import time
import argparse
from datetime import datetime, timedelta
from typing import Optional

from .database import init_db, upsert_ohlcv, get_latest_timestamp, get_data_range


class HistoricalDataFetcher:
    """Fetches historical OHLCV data from Bybit"""

    def __init__(self):
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        units = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000,
        }

        unit = timeframe[-1]
        value = int(timeframe[:-1])

        return value * units.get(unit, 60 * 1000)

    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
        progress_callback=None
    ) -> list:
        """
        Fetch OHLCV data between start and end timestamps.

        Bybit limit: 1000 candles per request
        For 1h timeframe: 1000 candles = ~41.6 days
        1 year = ~365 days = ~9 requests needed
        """
        all_data = []
        current_start = start_ts
        tf_ms = self._timeframe_to_ms(timeframe)

        request_count = 0
        while current_start < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_start,
                    limit=1000
                )

                if not ohlcv:
                    break

                # Filter out data beyond end_ts
                ohlcv = [candle for candle in ohlcv if candle[0] <= end_ts]

                all_data.extend(ohlcv)

                # Move to next batch
                last_ts = ohlcv[-1][0] if ohlcv else current_start
                current_start = last_ts + tf_ms

                request_count += 1

                if progress_callback:
                    progress_callback(len(all_data), current_start, end_ts)
                else:
                    # Progress output
                    progress = min(100, (current_start - start_ts) / (end_ts - start_ts) * 100)
                    print(f"  Fetched {len(all_data)} candles ({progress:.1f}%)")

                # Rate limiting - be conservative
                time.sleep(0.3)

            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(1)
                continue

        return all_data

    def backfill(
        self,
        symbol: str = 'BTC/USDT:USDT',
        timeframe: str = '1h',
        days: int = 365
    ) -> int:
        """
        Backfill historical data for specified period.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days to fetch

        Returns:
            Number of candles stored
        """
        # Initialize database
        init_db()

        # Calculate time range
        end_ts = int(datetime.utcnow().timestamp() * 1000)
        start_ts = end_ts - (days * 24 * 60 * 60 * 1000)

        print(f"\nBackfilling {symbol} {timeframe}")
        print(f"  From: {datetime.fromtimestamp(start_ts/1000)}")
        print(f"  To:   {datetime.fromtimestamp(end_ts/1000)}")
        print(f"  Days: {days}")

        # Fetch data
        data = self.fetch_ohlcv_range(symbol, timeframe, start_ts, end_ts)

        if not data:
            print("No data fetched!")
            return 0

        # Store in database
        stored = upsert_ohlcv(symbol, timeframe, data)

        print(f"\nStored {stored} candles in database")

        # Show data range
        info = get_data_range(symbol, timeframe)
        print(f"  Total candles: {info['count']}")
        print(f"  Date range: {info['oldest_date']} to {info['latest_date']}")

        return stored

    def update_to_latest(self, symbol: str, timeframe: str) -> int:
        """
        Update database with latest candles (incremental update).

        Fetches only candles newer than the last stored timestamp.
        """
        init_db()

        last_ts = get_latest_timestamp(symbol, timeframe)

        if last_ts is None:
            print(f"No existing data for {symbol} {timeframe}, run full backfill first")
            return 0

        # Fetch from last timestamp to now
        end_ts = int(datetime.utcnow().timestamp() * 1000)

        # Start from last timestamp (will fetch that candle again + newer)
        start_ts = last_ts

        print(f"Updating {symbol} {timeframe} from {datetime.fromtimestamp(last_ts/1000)}")

        data = self.fetch_ohlcv_range(symbol, timeframe, start_ts, end_ts)

        if not data:
            print("No new data")
            return 0

        stored = upsert_ohlcv(symbol, timeframe, data)
        print(f"Updated {stored} candles")

        return stored


def main():
    parser = argparse.ArgumentParser(description='Backfill historical OHLCV data')
    parser.add_argument('--symbol', default='BTC/USDT:USDT', help='Trading pair')
    parser.add_argument('--timeframe', default='1h', help='Candle timeframe')
    parser.add_argument('--days', type=int, default=365, help='Number of days')
    parser.add_argument('--update', action='store_true', help='Incremental update only')

    args = parser.parse_args()

    fetcher = HistoricalDataFetcher()

    if args.update:
        fetcher.update_to_latest(args.symbol, args.timeframe)
    else:
        fetcher.backfill(args.symbol, args.timeframe, args.days)


if __name__ == '__main__':
    main()
