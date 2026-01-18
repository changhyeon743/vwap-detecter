"""
Background Data Collector

Periodically fetches new candles and stores them in SQLite.
Used by main.py to keep data up-to-date without blocking.
"""

import ccxt
import threading
import time
from datetime import datetime
from typing import List, Optional

from .database import init_db, upsert_ohlcv, get_latest_timestamp


class DataCollector:
    """
    Background data collector that periodically fetches new candles.

    Usage:
        collector = DataCollector()
        collector.add_symbol('BTC/USDT:USDT', '1h')
        collector.start(interval=60)  # Update every 60 seconds

        # Later...
        collector.stop()
    """

    def __init__(self):
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })
        self.symbols = []  # List of (symbol, timeframe) tuples
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Initialize database
        init_db()

    def add_symbol(self, symbol: str, timeframe: str):
        """Add a symbol/timeframe pair to collect"""
        with self._lock:
            pair = (symbol, timeframe)
            if pair not in self.symbols:
                self.symbols.append(pair)
                print(f"DataCollector: Added {symbol} {timeframe}")

    def remove_symbol(self, symbol: str, timeframe: str):
        """Remove a symbol/timeframe pair"""
        with self._lock:
            pair = (symbol, timeframe)
            if pair in self.symbols:
                self.symbols.remove(pair)
                print(f"DataCollector: Removed {symbol} {timeframe}")

    def collect_once(self, symbol: str, timeframe: str) -> int:
        """
        Fetch latest candles for a symbol and store in database.

        Returns number of candles stored.
        """
        try:
            # Get last timestamp in database
            last_ts = get_latest_timestamp(symbol, timeframe)

            if last_ts is None:
                # No data yet, fetch recent 100 candles
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            else:
                # Fetch from last timestamp (will include last + newer)
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe,
                    since=last_ts,
                    limit=100
                )

            if not ohlcv:
                return 0

            # Store in database
            stored = upsert_ohlcv(symbol, timeframe, ohlcv)
            return stored

        except Exception as e:
            print(f"DataCollector error for {symbol} {timeframe}: {e}")
            return 0

    def _collection_loop(self, interval: int):
        """Background thread loop"""
        print(f"DataCollector: Started (interval={interval}s)")

        while self._running:
            with self._lock:
                symbols_to_collect = list(self.symbols)

            for symbol, timeframe in symbols_to_collect:
                if not self._running:
                    break

                stored = self.collect_once(symbol, timeframe)
                if stored > 0:
                    print(f"DataCollector: {symbol} {timeframe} +{stored} candles")

                # Small delay between symbols
                time.sleep(0.3)

            # Wait for next collection cycle
            for _ in range(interval):
                if not self._running:
                    break
                time.sleep(1)

        print("DataCollector: Stopped")

    def start(self, interval: int = 60):
        """
        Start background collection.

        Args:
            interval: Seconds between collection cycles
        """
        if self._running:
            print("DataCollector: Already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._collection_loop,
            args=(interval,),
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop background collection"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def is_running(self) -> bool:
        """Check if collector is running"""
        return self._running

    def get_status(self) -> dict:
        """Get collector status"""
        with self._lock:
            return {
                'running': self._running,
                'symbols': list(self.symbols),
                'symbol_count': len(self.symbols),
            }


# Global collector instance for main.py integration
_global_collector: Optional[DataCollector] = None


def get_collector() -> DataCollector:
    """Get or create global collector instance"""
    global _global_collector
    if _global_collector is None:
        _global_collector = DataCollector()
    return _global_collector


def start_collector(symbols: List[tuple], interval: int = 60):
    """
    Convenience function to start global collector.

    Args:
        symbols: List of (symbol, timeframe) tuples
        interval: Collection interval in seconds
    """
    collector = get_collector()

    for symbol, timeframe in symbols:
        collector.add_symbol(symbol, timeframe)

    collector.start(interval)
    return collector


def stop_collector():
    """Stop global collector"""
    global _global_collector
    if _global_collector:
        _global_collector.stop()


if __name__ == '__main__':
    # Test collector
    collector = DataCollector()
    collector.add_symbol('BTC/USDT:USDT', '1h')

    print("Starting collector (press Ctrl+C to stop)...")
    collector.start(interval=10)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        collector.stop()
