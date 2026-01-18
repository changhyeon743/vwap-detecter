#!/usr/bin/env python3
"""
VWAP Strategy Backtester CLI

Usage:
    python -m backtest.run backfill --days 365
    python -m backtest.run backtest
    python -m backtest.run optimize --trials 100
    python -m backtest.run info
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .database import init_db, get_data_range
from .backfill import HistoricalDataFetcher
from .backtester import Backtester, run_single_backtest
from .optimizer import VWAPOptimizer
from .metrics import calculate_all_metrics, print_metrics_report


def cmd_backfill(args):
    """Backfill historical data"""
    print(f"\n{'='*60}")
    print("BACKFILL HISTORICAL DATA")
    print(f"{'='*60}")

    fetcher = HistoricalDataFetcher()

    if args.update:
        fetcher.update_to_latest(args.symbol, args.timeframe)
    else:
        fetcher.backfill(args.symbol, args.timeframe, args.days)

    # Show data info
    info = get_data_range(args.symbol, args.timeframe)
    print(f"\nData Summary:")
    print(f"  Candles: {info['count']}")
    print(f"  From: {info['oldest_date']}")
    print(f"  To: {info['latest_date']}")


def cmd_backtest(args):
    """Run single backtest"""
    print(f"\n{'='*60}")
    print("SINGLE BACKTEST")
    print(f"{'='*60}")

    params = {
        'band_entry_mult': args.band_entry_mult,
        'min_strength': args.min_strength,
        'min_vol_ratio': args.min_vol_ratio,
        'sl_buffer_atr_mult': args.sl_buffer_atr_mult,
        'num_opposing_bars': args.num_opposing_bars,
    }

    print(f"\nSymbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Position Size: {args.position_size * 100:.1f}%")
    print(f"\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Run backtest
    backtester = Backtester()
    result = backtester.run(
        symbol=args.symbol,
        timeframe=args.timeframe,
        params=params,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
    )

    # Calculate and print metrics
    metrics = calculate_all_metrics(
        result['trades'],
        result['equity_curve'],
        result['initial_capital'],
        result['final_capital'],
    )

    print_metrics_report(metrics)

    # Exit reason breakdown
    if result['trades']:
        print("\n[EXIT REASONS]")
        from collections import Counter
        reasons = Counter(t['exit_reason'] for t in result['trades'])
        for reason, count in reasons.most_common():
            pct = count / len(result['trades']) * 100
            print(f"  {reason:20s}: {count} ({pct:.1f}%)")

    # Save trade log
    if args.save:
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trades_file = results_dir / f'trades_{timestamp}.csv'

        import pandas as pd
        df = pd.DataFrame(result['trades'])
        if not df.empty:
            df['entry_date'] = pd.to_datetime(df['entry_time'], unit='ms')
            df['exit_date'] = pd.to_datetime(df['exit_time'], unit='ms')
        df.to_csv(trades_file, index=False)
        print(f"\nTrade log saved to: {trades_file}")


def cmd_optimize(args):
    """Run parameter optimization"""
    print(f"\n{'='*60}")
    print("PARAMETER OPTIMIZATION")
    print(f"{'='*60}")

    print(f"\nSymbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Mode: {'Grid Search' if args.grid else f'Optuna TPE ({args.trials} trials)'}")
    print(f"Initial Capital: ${args.capital:,.2f}")

    optimizer = VWAPOptimizer(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_capital=args.capital,
    )

    result = optimizer.optimize(
        n_trials=args.trials,
        use_grid_search=args.grid,
        show_progress=True,
    )

    # Show results
    result.summary()

    # Save results
    filepath = optimizer.save_results()

    # Show top parameters
    print("\nTop 10 Parameter Sets:")
    top_df = result.get_top_n(10)
    cols = ['trial', 'band_entry_mult', 'min_strength', 'min_vol_ratio',
            'sl_buffer_atr_mult', 'num_opposing_bars', 'sharpe_ratio',
            'total_return_pct', 'max_drawdown_pct', 'win_rate', 'total_trades']
    cols = [c for c in cols if c in top_df.columns]
    print(top_df[cols].to_string(index=False))

    # Save best params to JSON
    if result.best_params:
        results_dir = Path(__file__).parent.parent / 'results'
        best_params_file = results_dir / 'best_params.json'
        with open(best_params_file, 'w') as f:
            json.dump(result.best_params, f, indent=2)
        print(f"\nBest params saved to: {best_params_file}")


def cmd_info(args):
    """Show database info"""
    print(f"\n{'='*60}")
    print("DATABASE INFO")
    print(f"{'='*60}")

    init_db()

    info = get_data_range(args.symbol, args.timeframe)

    print(f"\nSymbol: {info['symbol']}")
    print(f"Timeframe: {info['timeframe']}")
    print(f"Candles: {info['count']}")

    if info['oldest_date']:
        print(f"From: {info['oldest_date']}")
        print(f"To: {info['latest_date']}")

        # Calculate date range
        days = (info['latest_date'] - info['oldest_date']).days
        print(f"Days: {days}")
    else:
        print("\nNo data found. Run: python -m backtest.run backfill")


def main():
    parser = argparse.ArgumentParser(
        description='VWAP Strategy Backtester',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backtest.run backfill --days 365
  python -m backtest.run backtest --band-entry-mult 2.0
  python -m backtest.run optimize --trials 100
  python -m backtest.run optimize --grid
  python -m backtest.run info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Backfill command
    p_backfill = subparsers.add_parser('backfill', help='Fetch historical data')
    p_backfill.add_argument('--symbol', default='BTC/USDT:USDT', help='Trading pair')
    p_backfill.add_argument('--timeframe', default='1h', help='Candle timeframe')
    p_backfill.add_argument('--days', type=int, default=365, help='Days to fetch')
    p_backfill.add_argument('--update', action='store_true', help='Incremental update only')

    # Backtest command
    p_backtest = subparsers.add_parser('backtest', help='Run single backtest')
    p_backtest.add_argument('--symbol', default='BTC/USDT:USDT')
    p_backtest.add_argument('--timeframe', default='1h')
    p_backtest.add_argument('--capital', type=float, default=10000, help='Initial capital')
    p_backtest.add_argument('--position-size', type=float, default=0.1, help='Position size (0-1)')
    p_backtest.add_argument('--band-entry-mult', type=float, default=2.0)
    p_backtest.add_argument('--min-strength', type=float, default=0.7)
    p_backtest.add_argument('--min-vol-ratio', type=float, default=0.05)
    p_backtest.add_argument('--sl-buffer-atr-mult', type=float, default=0.5)
    p_backtest.add_argument('--num-opposing-bars', type=int, default=2)
    p_backtest.add_argument('--save', action='store_true', help='Save trade log')

    # Optimize command
    p_optimize = subparsers.add_parser('optimize', help='Optimize parameters')
    p_optimize.add_argument('--symbol', default='BTC/USDT:USDT')
    p_optimize.add_argument('--timeframe', default='1h')
    p_optimize.add_argument('--trials', type=int, default=100, help='Number of trials')
    p_optimize.add_argument('--grid', action='store_true', help='Use grid search')
    p_optimize.add_argument('--capital', type=float, default=10000)

    # Info command
    p_info = subparsers.add_parser('info', help='Show database info')
    p_info.add_argument('--symbol', default='BTC/USDT:USDT')
    p_info.add_argument('--timeframe', default='1h')

    args = parser.parse_args()

    if args.command == 'backfill':
        cmd_backfill(args)
    elif args.command == 'backtest':
        cmd_backtest(args)
    elif args.command == 'optimize':
        cmd_optimize(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
