"""
Cross-validation script to compare our backtester with Backtrader.

This script runs the same strategy on both our backtester and Backtrader
and compares the results to validate our implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Tuple

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our backtester
from backtester.data import CSVDataSource, DataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import BasicPortfolioManager
from backtester.core import Backtester
from backtester.utils.constants import SignalType

# Import backtrader if available
try:
    import backtrader as bt
except ImportError:
    logger = logging.getLogger("cross_validation")
    logger.error("Backtrader is not installed. Please install it with: pip install backtrader")
    logger.error("Cross-validation will be skipped.")
    
    # Create a summary report indicating that backtrader is not installed
    os.makedirs("../../output/results/cross_validation", exist_ok=True)
    with open("../../output/results/cross_validation/summary.txt", "w") as f:
        f.write("Cross-Validation Summary\n")
        f.write("=======================\n\n")
        f.write("SKIPPED: Backtrader is not installed.\n")
        f.write("Please install backtrader with: pip install backtrader\n")
    
    # Exit with success code to allow other validations to continue
    if __name__ == "__main__":
        sys.exit(0)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../output/logs/cross_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cross_validation")

# Create results directory if it doesn't exist
os.makedirs("../../output/results/cross_validation", exist_ok=True)


# Create a proper DataFrameSource class that implements the required abstract methods
class DataFrameSource(DataSource):
    def __init__(self, data, event_bus=None):
        super().__init__(event_bus)
        self._data = data
        
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from the DataFrame."""
        return self._data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """No preprocessing needed as data is already in the correct format."""
        return data


def create_synthetic_data(days: int = 252, trend: str = 'random', volatility: float = 0.01) -> pd.DataFrame:
    """
    Create synthetic price data for testing.
    
    Args:
        days: Number of days of data
        trend: Type of trend ('up', 'down', 'random', 'cycle')
        volatility: Daily volatility
        
    Returns:
        DataFrame with OHLCV data
    """
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(days)]
    
    # Generate price series based on trend
    if trend == 'up':
        drift = 0.0005  # Positive drift
        prices = 100 * np.cumprod(1 + np.random.normal(drift, volatility, days))
    elif trend == 'down':
        drift = -0.0005  # Negative drift
        prices = 100 * np.cumprod(1 + np.random.normal(drift, volatility, days))
    elif trend == 'cycle':
        # Create a cyclical pattern
        t = np.linspace(0, 4*np.pi, days)
        cycle = np.sin(t) * 20
        prices = 100 + cycle + np.cumsum(np.random.normal(0, volatility, days))
    else:  # random
        prices = 100 * np.cumprod(1 + np.random.normal(0, volatility, days))
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.uniform(100, 1000, days)
    })
    
    data = data.set_index('timestamp')
    return data


# Define a Backtrader strategy equivalent to our MovingAverageCrossover
class BTMovingAverageCrossover(bt.Strategy):
    params = (
        ('short_window', 10),
        ('long_window', 30),
    )
    
    def __init__(self):
        # Create moving averages
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_window
        )
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_window
        )
        
        # Create crossover indicator
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        
        # Initialize variables
        self.order = None
        self.trades = []
    
    def next(self):
        # Check if an order is pending
        if self.order:
            return
        
        # Check for buy signal
        if self.crossover > 0:  # Short MA crosses above long MA
            self.order = self.buy()
            self.trades.append({
                'timestamp': self.data.datetime.datetime(),
                'type': 'buy',
                'price': self.data.close[0]
            })
        
        # Check for sell signal
        elif self.crossover < 0:  # Short MA crosses below long MA
            self.order = self.sell()
            self.trades.append({
                'timestamp': self.data.datetime.datetime(),
                'type': 'sell',
                'price': self.data.close[0]
            })
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


def run_our_backtester(data: pd.DataFrame, short_window: int = 10, long_window: int = 30) -> Dict[str, Any]:
    """
    Run our backtester with the given data and parameters.
    
    Args:
        data: Price data
        short_window: Short moving average window
        long_window: Long moving average window
        
    Returns:
        Dictionary with backtest results
    """
    logger.info("Running our backtester")
    
    # Create strategy and portfolio manager
    strategy = MovingAverageCrossover(short_window=short_window, long_window=long_window)
    portfolio_manager = BasicPortfolioManager(initial_capital=10000)
    
    # Create a data source from the DataFrame
    data_source = DataFrameSource(data)
    
    # Create and run backtester
    backtester = Backtester(
        data_source=data_source,
        strategy=strategy,
        portfolio_manager=portfolio_manager,
        transaction_costs=0.001,
        slippage=0.0005
    )
    
    results = backtester.run()
    
    # Extract key metrics
    final_value = results.portfolio_values['portfolio_value'].iloc[-1]
    total_return = results.total_return
    num_trades = len(results.trades)
    
    # Save equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(results.portfolio_values['portfolio_value'])
    plt.title('Our Backtester Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.savefig('../../output/results/cross_validation/our_backtester_equity.png')
    
    return {
        "final_value": final_value,
        "total_return": total_return,
        "num_trades": num_trades,
        "equity_curve": results.portfolio_values['portfolio_value']
    }


def run_backtrader(data: pd.DataFrame, short_window: int = 10, long_window: int = 30) -> Dict[str, Any]:
    """
    Run Backtrader with the given data and parameters.
    
    Args:
        data: Price data
        short_window: Short moving average window
        long_window: Long moving average window
        
    Returns:
        Dictionary with backtest results
    """
    logger.info("Running Backtrader")
    
    # Create a Backtrader cerebro engine
    cerebro = bt.Cerebro()
    
    # Add the strategy
    cerebro.addstrategy(BTMovingAverageCrossover, 
                        short_window=short_window, 
                        long_window=long_window)
    
    # Create a Backtrader data feed from the DataFrame
    data_feed = bt.feeds.PandasData(
        dataname=data,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=None
    )
    
    # Add the data feed to cerebro
    cerebro.adddata(data_feed)
    
    # Set initial cash
    cerebro.broker.setcash(10000.0)
    
    # Set commission
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    
    # Run the backtest
    results = cerebro.run()
    strategy = results[0]
    
    # Extract key metrics
    final_value = cerebro.broker.getvalue()
    total_return = strategy.analyzers.returns.get_analysis()['rtot']
    trades_analysis = strategy.analyzers.trades.get_analysis()
    num_trades = trades_analysis.get('total', {}).get('total', 0)
    
    # Get equity curve
    time_return = strategy.analyzers.time_return.get_analysis()
    equity_curve = pd.Series([10000 * (1 + v) for v in time_return.values()], index=time_return.keys())
    
    # Save equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve)
    plt.title('Backtrader Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.savefig('../../output/results/cross_validation/backtrader_equity.png')
    
    return {
        "final_value": final_value,
        "total_return": total_return,
        "num_trades": num_trades,
        "equity_curve": equity_curve
    }


def compare_results(our_results: Dict[str, Any], bt_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare the results from our backtester and Backtrader.
    
    Args:
        our_results: Results from our backtester
        bt_results: Results from Backtrader
        
    Returns:
        Dictionary with comparison metrics
    """
    logger.info("Comparing results")
    
    # Calculate differences
    final_value_diff = abs(our_results['final_value'] - bt_results['final_value'])
    final_value_pct_diff = final_value_diff / bt_results['final_value'] * 100
    
    total_return_diff = abs(our_results['total_return'] - bt_results['total_return'])
    total_return_pct_diff = total_return_diff / abs(bt_results['total_return']) * 100 if bt_results['total_return'] != 0 else float('inf')
    
    num_trades_diff = abs(our_results['num_trades'] - bt_results['num_trades'])
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Align the equity curves if they have different indices
    our_equity = our_results['equity_curve']
    bt_equity = bt_results['equity_curve']
    
    # Resample to daily if needed
    if isinstance(our_equity.index[0], datetime) and isinstance(bt_equity.index[0], datetime):
        our_equity = our_equity.resample('D').last().dropna()
        bt_equity = bt_equity.resample('D').last().dropna()
    
    plt.plot(our_equity, label='Our Backtester')
    plt.plot(bt_equity, label='Backtrader')
    plt.title('Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.savefig('../../output/results/cross_validation/equity_comparison.png')
    
    # Log results
    logger.info(f"Our backtester final value: {our_results['final_value']:.2f}")
    logger.info(f"Backtrader final value: {bt_results['final_value']:.2f}")
    logger.info(f"Final value difference: {final_value_diff:.2f} ({final_value_pct_diff:.2f}%)")
    
    logger.info(f"Our backtester total return: {our_results['total_return']:.4f}")
    logger.info(f"Backtrader total return: {bt_results['total_return']:.4f}")
    logger.info(f"Total return difference: {total_return_diff:.4f} ({total_return_pct_diff:.2f}%)")
    
    logger.info(f"Our backtester number of trades: {our_results['num_trades']}")
    logger.info(f"Backtrader number of trades: {bt_results['num_trades']}")
    logger.info(f"Number of trades difference: {num_trades_diff}")
    
    # Determine if the results are close enough
    is_valid = (
        final_value_pct_diff < 5 and  # Less than 5% difference in final value
        total_return_pct_diff < 10 and  # Less than 10% difference in total return
        num_trades_diff <= 2  # No more than 2 trades difference
    )
    
    return {
        "final_value_diff": final_value_diff,
        "final_value_pct_diff": final_value_pct_diff,
        "total_return_diff": total_return_diff,
        "total_return_pct_diff": total_return_pct_diff,
        "num_trades_diff": num_trades_diff,
        "is_valid": is_valid
    }


def run_cross_validation():
    """Run cross-validation between our backtester and Backtrader."""
    logger.info("Starting cross-validation")
    
    # Create synthetic data for testing
    data = create_synthetic_data(days=252, trend='random')
    
    # Save the data for reference
    data.to_csv('../../output/results/cross_validation/test_data.csv')
    
    # Run both backtesters
    our_results = run_our_backtester(data)
    bt_results = run_backtrader(data)
    
    # Compare results
    comparison = compare_results(our_results, bt_results)
    
    # Create summary report
    with open("../../output/results/cross_validation/summary.txt", "w") as f:
        f.write(f"Cross-Validation Summary\n")
        f.write(f"=======================\n\n")
        f.write(f"Validation {'PASSED' if comparison['is_valid'] else 'FAILED'}\n\n")
        
        f.write(f"Our Backtester:\n")
        f.write(f"  Final Value: {our_results['final_value']:.2f}\n")
        f.write(f"  Total Return: {our_results['total_return']:.4f}\n")
        f.write(f"  Number of Trades: {our_results['num_trades']}\n\n")
        
        f.write(f"Backtrader:\n")
        f.write(f"  Final Value: {bt_results['final_value']:.2f}\n")
        f.write(f"  Total Return: {bt_results['total_return']:.4f}\n")
        f.write(f"  Number of Trades: {bt_results['num_trades']}\n\n")
        
        f.write(f"Differences:\n")
        f.write(f"  Final Value: {comparison['final_value_diff']:.2f} ({comparison['final_value_pct_diff']:.2f}%)\n")
        f.write(f"  Total Return: {comparison['total_return_diff']:.4f} ({comparison['total_return_pct_diff']:.2f}%)\n")
        f.write(f"  Number of Trades: {comparison['num_trades_diff']}\n")
    
    logger.info(f"Cross-validation {'PASSED' if comparison['is_valid'] else 'FAILED'}")
    logger.info("Summary saved to ../../output/results/cross_validation/summary.txt")
    
    return comparison['is_valid']


if __name__ == "__main__":
    success = run_cross_validation()
    sys.exit(0 if success else 1) 