"""
SQLite Database Management for OHLCV Data

Provides unified data storage for both backtesting and live trading.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager

# Default database path
DB_PATH = Path(__file__).parent.parent / 'data' / 'ohlcv.db'


@contextmanager
def get_connection(db_path: str = None):
    """Context manager for database connections"""
    path = db_path or str(DB_PATH)
    conn = sqlite3.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = None):
    """Initialize database with schema"""
    path = db_path or str(DB_PATH)

    # Ensure data directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with get_connection(path) as conn:
        cursor = conn.cursor()

        # OHLCV table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)

        # Index for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_ts
            ON ohlcv(symbol, timeframe, timestamp)
        """)

        conn.commit()
        print(f"Database initialized at {path}")


def upsert_ohlcv(symbol: str, timeframe: str, data: List, db_path: str = None):
    """
    Insert or replace OHLCV data

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT:USDT')
        timeframe: Candle timeframe (e.g., '1h')
        data: List of [timestamp, open, high, low, close, volume]
    """
    path = db_path or str(DB_PATH)

    with get_connection(path) as conn:
        cursor = conn.cursor()

        insert_sql = """
            INSERT OR REPLACE INTO ohlcv
            (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        rows = [
            (symbol, timeframe, d[0], d[1], d[2], d[3], d[4], d[5])
            for d in data
        ]

        cursor.executemany(insert_sql, rows)
        conn.commit()

        return len(rows)


def get_ohlcv(
    symbol: str,
    timeframe: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    limit: Optional[int] = None,
    db_path: str = None
) -> pd.DataFrame:
    """
    Get OHLCV data from database

    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        start_ts: Start timestamp (ms) - optional
        end_ts: End timestamp (ms) - optional
        limit: Max number of rows (returns most recent if set)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    path = db_path or str(DB_PATH)

    with get_connection(path) as conn:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(start_ts)

        if end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(end_ts)

        query += " ORDER BY timestamp ASC"

        if limit is not None:
            # Get last N rows
            query = f"""
                SELECT * FROM (
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv
                    WHERE symbol = ? AND timeframe = ?
                    {"AND timestamp >= ?" if start_ts else ""}
                    {"AND timestamp <= ?" if end_ts else ""}
                    ORDER BY timestamp DESC
                    LIMIT ?
                ) ORDER BY timestamp ASC
            """
            params_with_limit = [symbol, timeframe]
            if start_ts is not None:
                params_with_limit.append(start_ts)
            if end_ts is not None:
                params_with_limit.append(end_ts)
            params_with_limit.append(limit)
            params = params_with_limit

        df = pd.read_sql_query(query, conn, params=params)

        return df


def get_latest_timestamp(symbol: str, timeframe: str, db_path: str = None) -> Optional[int]:
    """
    Get the most recent timestamp for a symbol/timeframe

    Returns:
        Timestamp in milliseconds, or None if no data
    """
    path = db_path or str(DB_PATH)

    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(timestamp) FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """, (symbol, timeframe))

        result = cursor.fetchone()
        return result[0] if result and result[0] else None


def get_oldest_timestamp(symbol: str, timeframe: str, db_path: str = None) -> Optional[int]:
    """
    Get the oldest timestamp for a symbol/timeframe

    Returns:
        Timestamp in milliseconds, or None if no data
    """
    path = db_path or str(DB_PATH)

    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MIN(timestamp) FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """, (symbol, timeframe))

        result = cursor.fetchone()
        return result[0] if result and result[0] else None


def get_candle_count(symbol: str, timeframe: str, db_path: str = None) -> int:
    """Get total number of candles stored"""
    path = db_path or str(DB_PATH)

    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """, (symbol, timeframe))

        result = cursor.fetchone()
        return result[0] if result else 0


def get_data_range(symbol: str, timeframe: str, db_path: str = None) -> dict:
    """Get data range info for a symbol/timeframe"""
    path = db_path or str(DB_PATH)

    count = get_candle_count(symbol, timeframe, path)
    oldest = get_oldest_timestamp(symbol, timeframe, path)
    latest = get_latest_timestamp(symbol, timeframe, path)

    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'count': count,
        'oldest_ts': oldest,
        'latest_ts': latest,
        'oldest_date': pd.Timestamp(oldest, unit='ms') if oldest else None,
        'latest_date': pd.Timestamp(latest, unit='ms') if latest else None,
    }


def delete_old_data(symbol: str, timeframe: str, before_ts: int, db_path: str = None) -> int:
    """Delete data older than specified timestamp"""
    path = db_path or str(DB_PATH)

    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM ohlcv
            WHERE symbol = ? AND timeframe = ? AND timestamp < ?
        """, (symbol, timeframe, before_ts))

        deleted = cursor.rowcount
        conn.commit()

        return deleted


if __name__ == '__main__':
    # Test database operations
    init_db()

    # Check data range
    info = get_data_range('BTC/USDT:USDT', '1h')
    print(f"Data range: {info}")
