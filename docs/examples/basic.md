# Basic Backtest Example

This example demonstrates how to run a basic backtest using the Modular Backtesting System with a Moving Average Crossover strategy.

## Overview

The Moving Average Crossover strategy is a simple trend-following strategy that generates buy signals when a short-term moving average crosses above a long-term moving average, and sell signals when it crosses below.

## Code Example

```python
import pandas as pd
import matplotlib.pyplot as plt
from backtester.data import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import SimplePortfolioManager
from backtester.backtester import Backtester

# Load data
data_source = CSVDataSource("data/sample_data.csv")

# Create strategy
strategy = MovingAverageCrossover(short_window=10, long_window=30)

# Create portfolio manager
portfolio_manager = SimplePortfolioManager(initial_capital=10000)

# Create and run backtester
backtester = Backtester(data_source, strategy, portfolio_manager)
results = backtester.run()

# Print performance metrics
print(f"Total Return: {results.total_return:.2%}")
print(f"Annualized Return: {results.annualized_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
print(f"Profit Factor: {results.profit_factor:.2f}")

# Plot results
results.plot_equity_curve()
plt.title("Equity Curve")
plt.savefig("results/equity_curve.png")
plt.close()

results.plot_drawdown()
plt.title("Drawdown")
plt.savefig("results/drawdown.png")
plt.close()

# Plot trades
results.plot_trades()
plt.title("Trades")
plt.savefig("results/trades.png")
plt.close()
```

## Step-by-Step Explanation

### 1. Import Required Modules

```python
import pandas as pd
import matplotlib.pyplot as plt
from backtester.data import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import SimplePortfolioManager
from backtester.backtester import Backtester
```

We import the necessary modules from the Modular Backtesting System, as well as pandas for data manipulation and matplotlib for visualization.

### 2. Load Market Data

```python
data_source = CSVDataSource("data/sample_data.csv")
```

We create a `CSVDataSource` object to load market data from a CSV file. The file should contain at least the following columns: date, open, high, low, close, and volume.

### 3. Create a Strategy

```python
strategy = MovingAverageCrossover(short_window=10, long_window=30)
```

We create a `MovingAverageCrossover` strategy with a short window of 10 periods and a long window of 30 periods. This strategy generates buy signals when the 10-period moving average crosses above the 30-period moving average, and sell signals when it crosses below.

### 4. Create a Portfolio Manager

```python
portfolio_manager = SimplePortfolioManager(initial_capital=10000)
```

We create a `SimplePortfolioManager` with an initial capital of $10,000. This portfolio manager handles position sizing, risk management, and trade execution simulation.

### 5. Create and Run the Backtester

```python
backtester = Backtester(data_source, strategy, portfolio_manager)
results = backtester.run()
```

We create a `Backtester` object with our data source, strategy, and portfolio manager, and then run the backtest. The `run` method returns a `BacktestResults` object containing the trade history and portfolio performance.

### 6. Analyze Results

```python
print(f"Total Return: {results.total_return:.2%}")
print(f"Annualized Return: {results.annualized_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
print(f"Profit Factor: {results.profit_factor:.2f}")
```

We print various performance metrics to evaluate the strategy's performance.

### 7. Visualize Results

```python
results.plot_equity_curve()
plt.title("Equity Curve")
plt.savefig("results/equity_curve.png")
plt.close()

results.plot_drawdown()
plt.title("Drawdown")
plt.savefig("results/drawdown.png")
plt.close()

results.plot_trades()
plt.title("Trades")
plt.savefig("results/trades.png")
plt.close()
```

We generate visualizations of the backtest results, including an equity curve, drawdown chart, and trade chart, and save them as PNG files.

## Expected Output

### Performance Metrics

```
Total Return: 15.23%
Annualized Return: 7.61%
Sharpe Ratio: 1.25
Max Drawdown: 8.45%
Win Rate: 55.00%
Profit Factor: 1.75
```

### Equity Curve

![Equity Curve](../assets/equity_curve.png)

### Drawdown

![Drawdown](../assets/drawdown.png)

### Trades

![Trades](../assets/trades.png)

## Variations

### Different Moving Average Periods

You can experiment with different moving average periods to see how they affect the strategy's performance:

```python
# Short-term crossover
strategy = MovingAverageCrossover(short_window=5, long_window=10)

# Medium-term crossover
strategy = MovingAverageCrossover(short_window=20, long_window=50)

# Long-term crossover
strategy = MovingAverageCrossover(short_window=50, long_window=200)
```

### Different Position Sizing

You can experiment with different position sizing methods:

```python
# Fixed position size (10% of capital per trade)
portfolio_manager = SimplePortfolioManager(initial_capital=10000, position_size=0.1)

# Fixed dollar amount per trade
portfolio_manager = SimplePortfolioManager(initial_capital=10000, position_size=1000)
```

### Different Data Sources

You can use different data sources:

```python
# Load data from a different CSV file
data_source = CSVDataSource("data/other_data.csv")

# Load data from an exchange API
from backtester.data import APIDataSource
data_source = APIDataSource(
    exchange="binance",
    symbol="BTC/USDT",
    timeframe="1d",
    start_date="2020-01-01",
    end_date="2020-12-31"
)
```

## Next Steps

Now that you've run a basic backtest, you can:

1. Try different strategies, such as [RSI](../api/strategy.md#rsi) or [MACD](../api/strategy.md#macd)
2. Combine multiple strategies into a [Strategy Ensemble](ensemble.md)
3. Use [configuration files](config.md) for more complex backtests
4. Create your own custom strategy by inheriting from the [Strategy](../api/strategy.md#strategy) class 