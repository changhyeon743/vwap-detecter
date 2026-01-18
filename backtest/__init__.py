"""
VWAP Strategy Backtesting Module

Provides:
- SQLite database for OHLCV storage
- Historical data backfill
- Background data collector
- Backtesting engine
- Parameter optimization (Optuna)
"""

from .database import init_db, get_ohlcv, upsert_ohlcv, get_latest_timestamp
from .collector import DataCollector
from .backtester import Backtester
from .metrics import calculate_all_metrics, calculate_sharpe_ratio

__all__ = [
    'init_db',
    'get_ohlcv',
    'upsert_ohlcv',
    'get_latest_timestamp',
    'DataCollector',
    'Backtester',
    'calculate_all_metrics',
    'calculate_sharpe_ratio',
]
