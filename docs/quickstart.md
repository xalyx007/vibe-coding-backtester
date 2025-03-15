# Quick Start Guide

This guide will help you get started with the Modular Backtesting System quickly. We'll cover installation, basic usage, and a simple example.

## Installation

### From PyPI

The easiest way to install the Modular Backtesting System is via pip:

```bash
pip install backtester
```

### From Source

For the latest development version or if you want to contribute to the project:

```bash
git clone https://github.com/yourusername/backtester.git
cd backtester
pip install -e ".[dev]"
```

## Basic Usage

Here's a simple example of how to use the Modular Backtesting System:

```python
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

# Analyze results
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")

# Plot results
results.plot_equity_curve()
results.plot_drawdown()
```

## Step-by-Step Walkthrough

Let's break down the example above:

### 1. Import Required Modules

```python
from backtester.data import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import SimplePortfolioManager
from backtester.backtester import Backtester
```

### 2. Load Market Data

The `CSVDataSource` class loads market data from a CSV file. The file should contain at least the following columns: date, open, high, low, close, and volume.

```python
data_source = CSVDataSource("data/sample_data.csv")
```

<div class="tip">
You can also use other data sources like <code>APIDataSource</code> to fetch data from exchange APIs.
</div>

### 3. Create a Strategy

The `MovingAverageCrossover` strategy generates buy signals when the short-term moving average crosses above the long-term moving average, and sell signals when it crosses below.

```python
strategy = MovingAverageCrossover(short_window=10, long_window=30)
```

<div class="key-concept">
The <code>short_window</code> and <code>long_window</code> parameters determine the number of periods used to calculate the moving averages.
</div>

### 4. Create a Portfolio Manager

The `SimplePortfolioManager` manages positions and applies trading rules. It handles position sizing, risk management, and trade execution simulation.

```python
portfolio_manager = SimplePortfolioManager(initial_capital=10000)
```

### 5. Create and Run the Backtester

The `Backtester` class coordinates the interaction between all components and simulates trading based on signals from the strategy and portfolio updates from the portfolio manager.

```python
backtester = Backtester(data_source, strategy, portfolio_manager)
results = backtester.run()
```

### 6. Analyze Results

The `BacktestResults` class provides methods for analyzing backtest results and generating performance metrics.

```python
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

### 7. Visualize Results

The `BacktestResults` class also provides methods for generating visualizations.

```python
results.plot_equity_curve()
results.plot_drawdown()
```

## Using Configuration Files

For more complex backtests, you can use configuration files to set up the backtesting system:

```python
from backtester.utils.config import load_config
from backtester.utils.factory import create_data_source, create_strategy, create_portfolio_manager
from backtester.backtester import Backtester

# Load configuration
config = load_config("examples/config.yaml")

# Create components from configuration
data_source = create_data_source(config["data_source"])
strategy = create_strategy(config["strategy"])
portfolio_manager = create_portfolio_manager(config["portfolio_manager"])

# Create and run backtester
backtester = Backtester(data_source, strategy, portfolio_manager, config["backtest_params"])
results = backtester.run()

# Analyze and visualize results
results.analyze()
results.visualize()
```

<div class="code-example">
Example configuration file (YAML):

```yaml
data_source:
  type: csv
  params:
    file_path: data/sample_data.csv

strategy:
  type: moving_average_crossover
  params:
    short_window: 10
    long_window: 30

portfolio_manager:
  type: simple
  params:
    initial_capital: 10000
    position_size: 0.1

backtest_params:
  start_date: 2020-01-01
  end_date: 2020-12-31
  commission: 0.001
```
</div>

## Next Steps

Now that you have a basic understanding of how to use the Modular Backtesting System, you can:

1. Explore the [Examples](examples/basic.md) to see more complex use cases
2. Read the [Architecture](architecture.md) documentation to understand the system design
3. Check out the [API Reference](api/data.md) for detailed information about the available classes and methods
4. Learn how to [create custom components](architecture.md#extensibility) to extend the system 