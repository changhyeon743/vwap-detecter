"""
Performance Metrics Calculator

Calculates key trading metrics including Sharpe Ratio (optimization objective).
"""

import numpy as np
import pandas as pd
from typing import List, Dict


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760  # 1h candles
) -> float:
    """
    Calculate annualized Sharpe Ratio.

    For 1h data: 8760 periods/year (365 * 24)

    Sharpe = (mean_return - risk_free) / std_return * sqrt(periods)

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) == 0:
        return 0.0

    std = np.std(returns)
    if std == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    return float(np.mean(excess_returns) / std * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760
) -> float:
    """
    Sortino Ratio - only considers downside volatility.

    Better than Sharpe for strategies with asymmetric returns.
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    return float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown percentage.

    Args:
        equity_curve: List of equity values over time

    Returns:
        Max drawdown as percentage (e.g., 15.5 for 15.5%)
    """
    if len(equity_curve) < 2:
        return 0.0

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)

    # Avoid division by zero
    peak = np.where(peak == 0, 1, peak)

    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown)

    return float(max_dd * 100)


def calculate_calmar_ratio(
    total_return_pct: float,
    max_drawdown_pct: float,
    years: float = 1.0
) -> float:
    """
    Calmar Ratio = Annualized Return / Max Drawdown

    Higher is better. Values > 1 are good.
    """
    if max_drawdown_pct == 0:
        return float('inf') if total_return_pct > 0 else 0.0

    annualized_return = total_return_pct / years
    return float(annualized_return / max_drawdown_pct)


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Calculate win rate percentage.

    Args:
        trades: List of trade dicts with 'pnl' field

    Returns:
        Win rate as percentage
    """
    if len(trades) == 0:
        return 0.0

    winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
    return float(winning_trades / len(trades) * 100)


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Profit Factor = Gross Profit / Gross Loss

    > 1.0 means profitable system
    > 1.5 is good
    > 2.0 is excellent
    """
    if len(trades) == 0:
        return 0.0

    gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
    gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def calculate_expectancy(trades: List[Dict]) -> float:
    """
    Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)

    Expected $ per trade.
    """
    if len(trades) == 0:
        return 0.0

    winning = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
    losing = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]

    win_rate = len(winning) / len(trades) if trades else 0
    loss_rate = len(losing) / len(trades) if trades else 0

    avg_win = np.mean(winning) if winning else 0
    avg_loss = abs(np.mean(losing)) if losing else 0

    return float((win_rate * avg_win) - (loss_rate * avg_loss))


def calculate_avg_trade_duration(trades: List[Dict]) -> float:
    """
    Calculate average trade duration in hours.
    """
    if len(trades) == 0:
        return 0.0

    durations = []
    for t in trades:
        entry_time = t.get('entry_time')
        exit_time = t.get('exit_time')
        if entry_time and exit_time:
            # Assuming timestamps in milliseconds
            duration_hours = (exit_time - entry_time) / (1000 * 60 * 60)
            durations.append(duration_hours)

    return float(np.mean(durations)) if durations else 0.0


def calculate_all_metrics(
    trades: List[Dict],
    equity_curve: List[float],
    initial_capital: float,
    final_capital: float
) -> Dict:
    """
    Calculate all performance metrics.

    Args:
        trades: List of trade dicts
        equity_curve: List of equity values
        initial_capital: Starting capital
        final_capital: Ending capital

    Returns:
        Dictionary of all metrics
    """
    # Calculate returns from equity curve
    if len(equity_curve) < 2:
        returns = np.array([])
    else:
        equity = np.array(equity_curve)
        # Avoid division by zero
        equity_shifted = np.where(equity[:-1] == 0, 1, equity[:-1])
        returns = np.diff(equity) / equity_shifted

    total_return = final_capital - initial_capital
    total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0

    return {
        # Return metrics
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'initial_capital': initial_capital,
        'final_capital': final_capital,

        # Risk-adjusted metrics
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_drawdown_pct': calculate_max_drawdown(equity_curve),
        'calmar_ratio': calculate_calmar_ratio(
            total_return_pct,
            calculate_max_drawdown(equity_curve)
        ),

        # Trade statistics
        'total_trades': len(trades),
        'winning_trades': sum(1 for t in trades if t.get('pnl', 0) > 0),
        'losing_trades': sum(1 for t in trades if t.get('pnl', 0) <= 0),
        'win_rate': calculate_win_rate(trades),
        'profit_factor': calculate_profit_factor(trades),
        'expectancy': calculate_expectancy(trades),

        # Average trade metrics
        'avg_trade_pnl': float(np.mean([t.get('pnl', 0) for t in trades])) if trades else 0,
        'avg_trade_pnl_pct': float(np.mean([t.get('pnl_pct', 0) for t in trades])) if trades else 0,
        'avg_trade_duration_hours': calculate_avg_trade_duration(trades),

        # Best/worst trades
        'best_trade_pnl': max([t.get('pnl', 0) for t in trades]) if trades else 0,
        'worst_trade_pnl': min([t.get('pnl', 0) for t in trades]) if trades else 0,
    }


def print_metrics_report(metrics: Dict):
    """Print formatted metrics report"""
    print("\n" + "=" * 60)
    print("                 PERFORMANCE REPORT")
    print("=" * 60)

    print("\n[RETURNS]")
    print(f"  {'Initial Capital':25s}: ${metrics['initial_capital']:,.2f}")
    print(f"  {'Final Capital':25s}: ${metrics['final_capital']:,.2f}")
    print(f"  {'Total Return':25s}: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")

    print("\n[RISK METRICS]")
    print(f"  {'Sharpe Ratio':25s}: {metrics['sharpe_ratio']:.4f}")
    print(f"  {'Sortino Ratio':25s}: {metrics['sortino_ratio']:.4f}")
    print(f"  {'Max Drawdown':25s}: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  {'Calmar Ratio':25s}: {metrics['calmar_ratio']:.4f}")

    print("\n[TRADE STATISTICS]")
    print(f"  {'Total Trades':25s}: {metrics['total_trades']}")
    print(f"  {'Winning Trades':25s}: {metrics['winning_trades']}")
    print(f"  {'Losing Trades':25s}: {metrics['losing_trades']}")
    print(f"  {'Win Rate':25s}: {metrics['win_rate']:.2f}%")
    print(f"  {'Profit Factor':25s}: {metrics['profit_factor']:.2f}")
    print(f"  {'Expectancy':25s}: ${metrics['expectancy']:.2f}")

    print("\n[TRADE AVERAGES]")
    print(f"  {'Avg Trade PnL':25s}: ${metrics['avg_trade_pnl']:.2f} ({metrics['avg_trade_pnl_pct']:.2f}%)")
    print(f"  {'Avg Duration':25s}: {metrics['avg_trade_duration_hours']:.1f} hours")
    print(f"  {'Best Trade':25s}: ${metrics['best_trade_pnl']:.2f}")
    print(f"  {'Worst Trade':25s}: ${metrics['worst_trade_pnl']:.2f}")

    print("=" * 60)
