# Modular Backtesting System Examples

This directory contains examples of how to use the Modular Backtesting System.

## Examples

### Basic Backtest

[basic_backtest.py](basic_backtest.py) demonstrates how to run a simple backtest using the Moving Average Crossover strategy.

```bash
python examples/basic_backtest.py
```

### Bitcoin Backtest

[btc_backtest.py](btc_backtest.py) demonstrates how to run a backtest using Bitcoin data from Yahoo Finance. The script automatically downloads the data if it doesn't exist.

```bash
python examples/btc_backtest.py
```

### Strategy Ensemble

[strategy_ensemble.py](strategy_ensemble.py) demonstrates how to combine multiple strategies using the StrategyEnsemble class.

```bash
python examples/strategy_ensemble.py
```

### Configuration-Based Backtest

[config.yaml](config.yaml) is a sample configuration file that can be used with the CLI.

```bash
python -m backtester.cli -c examples/config.yaml
```

## Creating Your Own Examples

To create your own examples, you can use the existing examples as a starting point. Here's a basic template:

```python
from backtester.inputs import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import BasicPortfolioManager
from backtester.backtest import Backtester

# Create components
data_source = CSVDataSource("data/btc_daily.csv")
strategy = MovingAverageCrossover(short_window=20, long_window=50)
portfolio_manager = BasicPortfolioManager(initial_capital=10000)

# Create backtester
backtester = Backtester(
    data_source=data_source,
    strategy=strategy,
    portfolio_manager=portfolio_manager
)

# Run backtest
results = backtester.run()

# Analyze results
print(f"Total Return: {results.total_return:.2%}")
```

## Data

The examples use sample data that is generated automatically if it doesn't exist. You can replace this with your own data by placing CSV files in the `data` directory.

For real cryptocurrency data, you can use the data downloaders in the `scripts/data_downloaders` directory:

```bash
# Download Bitcoin data from Yahoo Finance
python scripts/data_downloaders/download_yahoo_crypto.py --btc-only
```

## Command-Line Interface

You can also run backtests using the command-line interface:

```bash
# Run a backtest with a configuration file
python -m backtester.cli -c examples/config.yaml

# Run a backtest with command-line arguments
python -m backtester.cli --data-source csv --data-path data/btc_daily.csv --strategy moving_average --short-window 20 --long-window 50 --initial-capital 10000 --save-plots --save-results
```

For more information on the command-line interface, run:

```bash
python -m backtester.cli --help
``` 