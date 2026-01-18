"""
Backtesting Engine

Simulates trading using historical data with the exact same
VWAPStrategy logic used in live trading.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Add parent directory to path for importing main.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from .database import get_ohlcv


@dataclass
class Position:
    """Represents an open position"""
    signal_type: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: int  # timestamp ms
    entry_bar_idx: int
    stop_loss: float
    take_profit: Optional[float]  # VWAP target (dynamic)
    quantity: float
    atr: float
    signal_low: float
    signal_high: float


@dataclass
class Trade:
    """Completed trade record"""
    signal_type: str
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    exit_reason: str  # 'TP', 'SL', 'SAFETY_EXIT', 'END_OF_DATA'
    stop_loss: float
    take_profit: Optional[float]
    quantity: float
    pnl: float
    pnl_pct: float


class BacktestSettings:
    """
    Mock settings for backtesting with custom parameters.

    Allows backtester to use different parameters without modifying
    the global settings.
    """

    def __init__(self, params: Dict[str, Any]):
        self._params = params

    # Strategy parameters (from optimization)
    @property
    def band_entry_mult(self):
        return self._params.get('band_entry_mult', 2.0)

    @property
    def min_strength(self):
        return self._params.get('min_strength', 0.7)

    @property
    def min_vol_ratio(self):
        return self._params.get('min_vol_ratio', 0.05)

    @property
    def sl_buffer_atr_mult(self):
        return self._params.get('sl_buffer_atr_mult', 0.5)

    @property
    def num_opposing_bars(self):
        return self._params.get('num_opposing_bars', 2)

    @property
    def enable_safety_exit(self):
        return self._params.get('enable_safety_exit', True)

    # Fixed settings for backtest
    @property
    def session_timezone(self):
        return self._params.get('session_timezone', 'UTC')

    @property
    def exit_mode_long(self):
        return self._params.get('exit_mode_long', 'VWAP')

    @property
    def exit_mode_short(self):
        return self._params.get('exit_mode_short', 'VWAP')

    @property
    def target_long_deviation(self):
        return self._params.get('target_long_deviation', 2.0)

    @property
    def target_short_deviation(self):
        return self._params.get('target_short_deviation', 2.0)

    @property
    def allow_longs(self):
        return self._params.get('allow_longs', True)

    @property
    def allow_shorts(self):
        return self._params.get('allow_shorts', True)

    @property
    def no_trade_around_0900(self):
        # Disable time filter for backtesting
        return False

    @property
    def debug_mode(self):
        return False


class Backtester:
    """
    Core backtesting engine.

    Reuses VWAPStrategy from main.py to ensure consistency
    between backtesting and live trading.
    """

    def __init__(self):
        self.strategy = None
        self._original_settings = None

    def _setup_strategy(self, params: Dict):
        """Initialize strategy with custom settings"""
        # Import here to avoid circular imports
        import main

        # Save original settings
        self._original_settings = main.settings

        # Create mock settings with backtest parameters
        main.settings = BacktestSettings(params)

        # Create strategy instance
        from main import VWAPStrategy
        self.strategy = VWAPStrategy()

    def _restore_settings(self):
        """Restore original settings after backtest"""
        if self._original_settings is not None:
            import main
            main.settings = self._original_settings

    def run(
        self,
        symbol: str,
        timeframe: str,
        params: Dict,
        initial_capital: float = 10000,
        position_size_pct: float = 0.1,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None
    ) -> Dict:
        """
        Run backtest simulation.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            timeframe: Candle timeframe (e.g., '1h')
            params: Strategy parameters dict
            initial_capital: Starting capital in USDT
            position_size_pct: Position size as percentage of capital
            start_ts: Start timestamp (ms) - optional
            end_ts: End timestamp (ms) - optional

        Returns:
            Dict with trades, equity_curve, metrics
        """
        try:
            self._setup_strategy(params)

            # Load data from SQLite
            df = get_ohlcv(symbol, timeframe, start_ts, end_ts)

            if len(df) < 50:
                raise ValueError(f"Not enough data: {len(df)} candles (need at least 50)")

            # State variables
            capital = initial_capital
            equity_curve = [initial_capital]
            trades: List[Trade] = []
            position: Optional[Position] = None

            # Lookback for VWAP calculation
            lookback = 50

            for i in range(lookback, len(df)):
                current_bar = df.iloc[i]
                # Data up to and including current bar
                historical_df = df.iloc[:i + 1].copy()

                # Update equity with current position value
                if position:
                    unrealized_pnl = self._calculate_unrealized_pnl(
                        position, current_bar['close']
                    )
                    equity_curve.append(capital + unrealized_pnl)
                else:
                    equity_curve.append(capital)

                # Check for exit conditions if in position
                if position:
                    exit_result = self._check_exit_conditions(
                        position, current_bar, historical_df, params
                    )

                    if exit_result:
                        # Close position
                        trade = self._close_position(
                            position,
                            exit_result['price'],
                            int(current_bar['timestamp']),
                            exit_result['reason']
                        )
                        trades.append(trade)
                        capital += trade.pnl
                        position = None
                        continue  # Don't look for new signal on exit bar

                # Check for entry signal (only if not in position)
                if not position:
                    signal = self.strategy.check_signal(
                        historical_df.copy(), symbol, timeframe
                    )

                    if signal and signal.get('type'):
                        # Calculate position size
                        position_value = capital * position_size_pct
                        quantity = position_value / current_bar['close']

                        position = Position(
                            signal_type=signal['type'],
                            entry_price=signal['price'],
                            entry_time=int(current_bar['timestamp']),
                            entry_bar_idx=i,
                            stop_loss=signal['stop_loss'],
                            take_profit=signal.get('target'),
                            quantity=quantity,
                            atr=signal.get('atr', 0),
                            signal_low=signal.get('signal_low', current_bar['low']),
                            signal_high=signal.get('signal_high', current_bar['high']),
                        )

            # Close any remaining position at end
            if position:
                final_bar = df.iloc[-1]
                trade = self._close_position(
                    position,
                    final_bar['close'],
                    int(final_bar['timestamp']),
                    'END_OF_DATA'
                )
                trades.append(trade)
                capital += trade.pnl

            # Convert trades to dicts for metrics
            trades_dicts = [self._trade_to_dict(t) for t in trades]

            return {
                'trades': trades_dicts,
                'equity_curve': equity_curve,
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_bars': len(df),
                'params': params,
            }

        finally:
            self._restore_settings()

    def _check_exit_conditions(
        self,
        position: Position,
        current_bar: pd.Series,
        historical_df: pd.DataFrame,
        params: Dict
    ) -> Optional[Dict]:
        """
        Check exit conditions.

        Returns dict with 'price' and 'reason' if exit triggered.
        """
        # 1. Check Stop Loss
        if position.signal_type == 'LONG':
            if current_bar['low'] <= position.stop_loss:
                return {'price': position.stop_loss, 'reason': 'SL'}
        else:  # SHORT
            if current_bar['high'] >= position.stop_loss:
                return {'price': position.stop_loss, 'reason': 'SL'}

        # 2. Check Take Profit (VWAP target - dynamic)
        # Recalculate VWAP for current bar
        vwap_df = self.strategy.calculate_vwap(historical_df.copy(), '', '')
        current_vwap = vwap_df.iloc[-1]['vwap']

        if pd.notna(current_vwap):
            if position.signal_type == 'LONG':
                # TP when price reaches VWAP from below
                if current_bar['high'] >= current_vwap:
                    return {'price': current_vwap, 'reason': 'TP'}
            else:  # SHORT
                # TP when price reaches VWAP from above
                if current_bar['low'] <= current_vwap:
                    return {'price': current_vwap, 'reason': 'TP'}

        # 3. Check Safety Exit (N consecutive opposing bars)
        if params.get('enable_safety_exit', True):
            n = params.get('num_opposing_bars', 2)
            if len(historical_df) >= n:
                # Get last N completed bars (exclude current)
                recent_bars = historical_df.tail(n + 1).head(n)

                if position.signal_type == 'LONG':
                    # Count bearish bars (close < open)
                    bearish_count = sum(
                        1 for _, row in recent_bars.iterrows()
                        if row['close'] < row['open']
                    )
                    if bearish_count == n:
                        return {'price': current_bar['close'], 'reason': 'SAFETY_EXIT'}
                else:  # SHORT
                    # Count bullish bars (close > open)
                    bullish_count = sum(
                        1 for _, row in recent_bars.iterrows()
                        if row['close'] > row['open']
                    )
                    if bullish_count == n:
                        return {'price': current_bar['close'], 'reason': 'SAFETY_EXIT'}

        return None

    def _calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """Calculate unrealized PnL for open position"""
        if position.signal_type == 'LONG':
            return (current_price - position.entry_price) * position.quantity
        else:  # SHORT
            return (position.entry_price - current_price) * position.quantity

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_time: int,
        reason: str
    ) -> Trade:
        """Close position and return Trade record"""
        if position.signal_type == 'LONG':
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.quantity

        position_value = position.entry_price * position.quantity
        pnl_pct = (pnl / position_value * 100) if position_value > 0 else 0

        return Trade(
            signal_type=position.signal_type,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=reason,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )

    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade dataclass to dict"""
        return {
            'signal_type': trade.signal_type,
            'entry_time': trade.entry_time,
            'entry_price': trade.entry_price,
            'exit_time': trade.exit_time,
            'exit_price': trade.exit_price,
            'exit_reason': trade.exit_reason,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'quantity': trade.quantity,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
        }


def run_single_backtest(
    symbol: str = 'BTC/USDT:USDT',
    timeframe: str = '1h',
    params: Optional[Dict] = None,
    initial_capital: float = 10000,
    position_size_pct: float = 0.1
) -> Dict:
    """
    Convenience function to run a single backtest.

    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        params: Strategy parameters (uses defaults if None)
        initial_capital: Starting capital
        position_size_pct: Position size as % of capital

    Returns:
        Backtest results dict
    """
    if params is None:
        params = {
            'band_entry_mult': 2.0,
            'min_strength': 0.7,
            'min_vol_ratio': 0.05,
            'sl_buffer_atr_mult': 0.5,
            'num_opposing_bars': 2,
        }

    backtester = Backtester()
    return backtester.run(
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
    )
