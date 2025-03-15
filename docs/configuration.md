# Configuration Guide

This document provides guidance on configuring the Modular Backtesting System for different use cases.

## Configuration Files

The system can be configured using YAML or JSON configuration files. These files allow you to specify the components to use and their parameters.

### Example Configuration

```yaml
# config.yaml
data_source:
  type: "CSVDataSource"
  params:
    file_path: "data/btc_daily.csv"
    date_column: "timestamp"
    date_format: "%Y-%m-%d %H:%M:%S"

strategy:
  type: "MovingAverageCrossover"
  params:
    short_window: 20
    long_window: 50
    price_column: "close"

portfolio_manager:
  type: "BasicPortfolioManager"
  params:
    initial_capital: 10000
    position_sizer:
      type: "PercentageSizer"
      params:
        percentage: 0.1

backtest:
  transaction_costs: 0.001
  slippage: 0.0005
  start_date: "2020-01-01"
  end_date: "2021-01-01"
  symbol: "BTC-USD"

events:
  use_redis: false
  redis_url: "redis://localhost:6379/0"
```

## Loading Configuration

You can load the configuration using the `load_config` function from the `utils` module:

```python
from backtester.utils import load_config

# Load configuration from a file
config = load_config("config.yaml")

# Or specify the configuration directly
config = {
    "data_source": {
        "type": "CSVDataSource",
        "params": {
            "file_path": "data/btc_daily.csv",
            "date_column": "timestamp",
            "date_format": "%Y-%m-%d %H:%M:%S"
        }
    },
    # ...
}
```

## Creating Components from Configuration

You can create components from the configuration using factory functions:

```python
from backtester.utils.factory import create_data_source, create_strategy, create_portfolio_manager

# Create components
data_source = create_data_source(config["data_source"])
strategy = create_strategy(config["strategy"])
portfolio_manager = create_portfolio_manager(config["portfolio_manager"])

# Create backtester
from backtester.backtest import Backtester

backtester = Backtester(
    data_source=data_source,
    strategy=strategy,
    portfolio_manager=portfolio_manager,
    transaction_costs=config["backtest"]["transaction_costs"],
    slippage=config["backtest"]["slippage"]
)

# Run backtest
results = backtester.run(
    start_date=config["backtest"]["start_date"],
    end_date=config["backtest"]["end_date"],
    symbol=config["backtest"]["symbol"]
)
```

## Configuration Options

### Data Source Options

#### CSVDataSource

```yaml
data_source:
  type: "CSVDataSource"
  params:
    file_path: "data/btc_daily.csv"  # Path to the CSV file
    date_column: "timestamp"         # Name of the date/timestamp column
    date_format: "%Y-%m-%d %H:%M:%S" # Format of the date/timestamp column
```

#### ExcelDataSource

```yaml
data_source:
  type: "ExcelDataSource"
  params:
    file_path: "data/btc_daily.xlsx" # Path to the Excel file
    sheet_name: "Sheet1"             # Name of the sheet to load
    date_column: "timestamp"         # Name of the date/timestamp column
    date_format: "%Y-%m-%d %H:%M:%S" # Format of the date/timestamp column
```

#### ExchangeDataSource

```yaml
data_source:
  type: "ExchangeDataSource"
  params:
    exchange_id: "binance"           # ID of the exchange
    symbol: "BTC/USD"                # Trading symbol
    timeframe: "1d"                  # Timeframe for the data
    api_key: "your-api-key"          # API key for the exchange (optional)
    api_secret: "your-api-secret"    # API secret for the exchange (optional)
```

### Strategy Options

#### MovingAverageCrossover

```yaml
strategy:
  type: "MovingAverageCrossover"
  params:
    short_window: 20                 # Window size for the short moving average
    long_window: 50                  # Window size for the long moving average
    price_column: "close"            # Column to use for price data
```

#### RSIStrategy

```yaml
strategy:
  type: "RSIStrategy"
  params:
    period: 14                       # Period for RSI calculation
    overbought: 70                   # Threshold for overbought condition
    oversold: 30                     # Threshold for oversold condition
    price_column: "close"            # Column to use for price data
```

#### BollingerBandsStrategy

```yaml
strategy:
  type: "BollingerBandsStrategy"
  params:
    window: 20                       # Window size for the moving average
    num_std: 2                       # Number of standard deviations for the bands
    price_column: "close"            # Column to use for price data
```

#### MLStrategy

```yaml
strategy:
  type: "MLStrategy"
  params:
    model_path: "models/model.pkl"   # Path to the trained model
    features:                        # List of feature columns to use
      - "feature1"
      - "feature2"
    threshold: 0.5                   # Threshold for signal generation
```

#### StrategyEnsemble

```yaml
strategy:
  type: "StrategyEnsemble"
  params:
    strategies:                      # List of strategies to combine
      - type: "MovingAverageCrossover"
        params:
          short_window: 20
          long_window: 50
      - type: "RSIStrategy"
        params:
          period: 14
          overbought: 70
          oversold: 30
    weights: [0.5, 0.5]              # Weights for each strategy
    combination_method: "weighted"   # Method for combining signals
```

### Portfolio Manager Options

#### BasicPortfolioManager

```yaml
portfolio_manager:
  type: "BasicPortfolioManager"
  params:
    initial_capital: 10000           # Initial capital for the portfolio
    position_sizer:                  # Position sizer configuration
      type: "PercentageSizer"
      params:
        percentage: 0.1
```

#### FixedAmountSizer

```yaml
position_sizer:
  type: "FixedAmountSizer"
  params:
    amount: 1000                     # Fixed amount to trade
```

#### PercentageSizer

```yaml
position_sizer:
  type: "PercentageSizer"
  params:
    percentage: 0.1                  # Percentage of available capital to trade
```

### Backtest Options

```yaml
backtest:
  transaction_costs: 0.001           # Transaction costs as a fraction of trade value
  slippage: 0.0005                   # Slippage as a fraction of price
  start_date: "2020-01-01"           # Start date for the backtest
  end_date: "2021-01-01"             # End date for the backtest
  symbol: "BTC-USD"                  # Trading symbol
```

### Events Options

```yaml
events:
  use_redis: false                   # Whether to use Redis for event distribution
  redis_url: "redis://localhost:6379/0" # Redis URL if using Redis
```

## Parameter Sweeps

You can perform parameter sweeps by creating multiple configurations with different parameter values:

```python
from backtester.utils import load_config
from backtester.utils.factory import create_data_source, create_strategy, create_portfolio_manager
from backtester.backtest import Backtester

# Load base configuration
base_config = load_config("config.yaml")

# Define parameter sweep
short_windows = [10, 20, 30]
long_windows = [50, 100, 150]

results = []

# Perform parameter sweep
for short_window in short_windows:
    for long_window in long_windows:
        # Skip invalid combinations
        if short_window >= long_window:
            continue
            
        # Create a copy of the base configuration
        config = base_config.copy()
        
        # Update strategy parameters
        config["strategy"]["params"]["short_window"] = short_window
        config["strategy"]["params"]["long_window"] = long_window
        
        # Create components
        data_source = create_data_source(config["data_source"])
        strategy = create_strategy(config["strategy"])
        portfolio_manager = create_portfolio_manager(config["portfolio_manager"])
        
        # Create backtester
        backtester = Backtester(
            data_source=data_source,
            strategy=strategy,
            portfolio_manager=portfolio_manager,
            transaction_costs=config["backtest"]["transaction_costs"],
            slippage=config["backtest"]["slippage"]
        )
        
        # Run backtest
        result = backtester.run(
            start_date=config["backtest"]["start_date"],
            end_date=config["backtest"]["end_date"],
            symbol=config["backtest"]["symbol"]
        )
        
        # Store result
        results.append({
            "short_window": short_window,
            "long_window": long_window,
            "total_return": result.total_return,
            "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
            "max_drawdown": result.metrics.get("max_drawdown", 0)
        })

# Find best parameters
best_result = max(results, key=lambda x: x["sharpe_ratio"])
print(f"Best parameters: short_window={best_result['short_window']}, long_window={best_result['long_window']}")
print(f"Sharpe ratio: {best_result['sharpe_ratio']:.2f}")
print(f"Total return: {best_result['total_return']:.2%}")
print(f"Max drawdown: {best_result['max_drawdown']:.2%}") 