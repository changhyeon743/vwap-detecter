"""
Parameter Optimizer using Optuna

Optimizes VWAP strategy parameters using Sharpe Ratio as objective.
Optuna uses Bayesian optimization (TPE sampler) for efficient search.
"""

import optuna
from optuna.samplers import TPESampler, GridSampler
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .backtester import Backtester
from .metrics import calculate_all_metrics


# Suppress Optuna logs (show only warnings and errors)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class VWAPOptimizer:
    """
    Parameter optimizer for VWAP strategy.

    Uses Optuna for efficient hyperparameter search.
    Default objective is Sharpe Ratio maximization.
    """

    # Default parameter search space
    DEFAULT_PARAM_SPACE = {
        'band_entry_mult': [1.0, 1.5, 2.0, 2.5, 3.0],
        'min_strength': [0.5, 0.6, 0.7, 0.8, 0.9],
        'min_vol_ratio': [0.02, 0.05, 0.08, 0.1, 0.15],
        'sl_buffer_atr_mult': [0.3, 0.5, 0.7, 1.0, 1.5],
        'num_opposing_bars': [1, 2, 3],
    }

    def __init__(
        self,
        symbol: str = 'BTC/USDT:USDT',
        timeframe: str = '1h',
        initial_capital: float = 10000,
        position_size_pct: float = 0.1,
        param_space: Optional[Dict] = None
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.param_space = param_space or self.DEFAULT_PARAM_SPACE

        self.backtester = Backtester()
        self.results: List[Dict] = []
        self.study: Optional[optuna.Study] = None

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Returns Sharpe Ratio (higher is better).
        """
        # Sample parameters from search space
        params = {}
        for param_name, values in self.param_space.items():
            params[param_name] = trial.suggest_categorical(param_name, values)

        try:
            # Run backtest
            result = self.backtester.run(
                self.symbol,
                self.timeframe,
                params,
                self.initial_capital,
                self.position_size_pct,
            )

            # Calculate metrics
            metrics = calculate_all_metrics(
                result['trades'],
                result['equity_curve'],
                result['initial_capital'],
                result['final_capital'],
            )

            # Store result
            self.results.append({
                'trial': trial.number,
                **params,
                **metrics,
            })

            # Get Sharpe Ratio
            sharpe = metrics['sharpe_ratio']

            # Penalize if too few trades (might be overfitting)
            total_trades = metrics['total_trades']
            if total_trades < 5:
                sharpe *= 0.3  # Heavy penalty for very few trades
            elif total_trades < 10:
                sharpe *= 0.7  # Moderate penalty

            # Handle infinity/nan
            if not pd.notna(sharpe) or sharpe == float('inf'):
                sharpe = -100

            return sharpe

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return -100  # Return very bad score on failure

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        use_grid_search: bool = False,
        show_progress: bool = True
    ) -> 'OptimizationResult':
        """
        Run parameter optimization.

        Args:
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds (optional)
            use_grid_search: If True, use exhaustive grid search
            show_progress: Show progress bar

        Returns:
            OptimizationResult object
        """
        # Reset results
        self.results = []

        # Create sampler
        if use_grid_search:
            sampler = GridSampler(self.param_space)
            # Calculate total combinations
            total_combinations = 1
            for values in self.param_space.values():
                total_combinations *= len(values)
            n_trials = total_combinations
            print(f"Grid search: {total_combinations} combinations")
        else:
            sampler = TPESampler(seed=42)
            print(f"TPE optimization: {n_trials} trials")

        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'vwap_{self.symbol}_{self.timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
        )

        return OptimizationResult(self.study, self.results)

    def save_results(self, filepath: Optional[str] = None):
        """Save all trial results to CSV"""
        if not self.results:
            print("No results to save")
            return

        if filepath is None:
            results_dir = Path(__file__).parent.parent / 'results'
            results_dir.mkdir(exist_ok=True)
            filepath = str(results_dir / f'optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")

        return filepath


class OptimizationResult:
    """Container for optimization results"""

    def __init__(self, study: optuna.Study, all_results: List[Dict]):
        self.study = study
        self.all_results = all_results

        if study.best_trial:
            self.best_params = study.best_params
            self.best_sharpe = study.best_value
            self.best_trial_number = study.best_trial.number
        else:
            self.best_params = {}
            self.best_sharpe = -float('inf')
            self.best_trial_number = -1

    def summary(self):
        """Print optimization summary"""
        print("\n" + "=" * 70)
        print("                    OPTIMIZATION RESULTS")
        print("=" * 70)

        print(f"\nTotal Trials: {len(self.all_results)}")
        print(f"Best Trial: #{self.best_trial_number}")
        print(f"\nBest Sharpe Ratio: {self.best_sharpe:.4f}")

        print(f"\nBest Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key:25s}: {value}")

        # Find full metrics for best trial
        best_result = None
        for r in self.all_results:
            if r.get('trial') == self.best_trial_number:
                best_result = r
                break

        if best_result:
            print(f"\nPerformance Metrics:")
            print(f"  {'Total Return':25s}: {best_result.get('total_return_pct', 0):.2f}%")
            print(f"  {'Max Drawdown':25s}: {best_result.get('max_drawdown_pct', 0):.2f}%")
            print(f"  {'Win Rate':25s}: {best_result.get('win_rate', 0):.2f}%")
            print(f"  {'Profit Factor':25s}: {best_result.get('profit_factor', 0):.2f}")
            print(f"  {'Total Trades':25s}: {best_result.get('total_trades', 0)}")
            print(f"  {'Calmar Ratio':25s}: {best_result.get('calmar_ratio', 0):.4f}")

        print("=" * 70)

    def get_top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top N parameter sets by Sharpe Ratio"""
        df = pd.DataFrame(self.all_results)
        if 'sharpe_ratio' in df.columns:
            return df.nlargest(n, 'sharpe_ratio')
        return df.head(n)

    def get_best_params_json(self) -> str:
        """Get best parameters as JSON string"""
        return json.dumps(self.best_params, indent=2)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to DataFrame"""
        return pd.DataFrame(self.all_results)


def run_optimization(
    symbol: str = 'BTC/USDT:USDT',
    timeframe: str = '1h',
    n_trials: int = 100,
    use_grid_search: bool = False
) -> OptimizationResult:
    """
    Convenience function to run optimization.

    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        n_trials: Number of trials (ignored for grid search)
        use_grid_search: Use exhaustive grid search

    Returns:
        OptimizationResult
    """
    optimizer = VWAPOptimizer(symbol=symbol, timeframe=timeframe)
    result = optimizer.optimize(n_trials=n_trials, use_grid_search=use_grid_search)
    optimizer.save_results()
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Optimize VWAP Strategy Parameters')
    parser.add_argument('--symbol', default='BTC/USDT:USDT', help='Trading pair')
    parser.add_argument('--timeframe', default='1h', help='Candle timeframe')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--grid', action='store_true', help='Use grid search')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')

    args = parser.parse_args()

    optimizer = VWAPOptimizer(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_capital=args.capital,
    )

    result = optimizer.optimize(
        n_trials=args.trials,
        use_grid_search=args.grid,
    )

    result.summary()
    optimizer.save_results()

    print("\nTop 10 Parameter Sets:")
    print(result.get_top_n(10).to_string())
