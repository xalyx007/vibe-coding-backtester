# Modular Backtesting System

<div align="center">
    <img src="assets/architecture_diagram.png" alt="Modular Backtesting System" width="600">
</div>

## Overview

The Modular Backtesting System is a flexible, extensible framework for backtesting trading strategies. It is designed with modularity in mind, allowing for easy customization and extension of its components.

<div class="key-concept">
<strong>Key Features:</strong>
<ul>
<li>Modular architecture with decoupled components</li>
<li>Event-driven design for flexible communication</li>
<li>Support for multiple data sources</li>
<li>Extensible strategy framework</li>
<li>Customizable portfolio management</li>
<li>Comprehensive performance metrics and visualization</li>
<li>Configuration-based setup for easy parameter tuning</li>
</ul>
</div>

## Quick Start

### Installation

```bash
pip install backtester
```

For development:

```bash
git clone https://github.com/yourusername/backtester.git
cd backtester
pip install -e ".[dev]"
```

### Basic Usage

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

## Architecture

The system is built around several core modules:

- **Data Module**: Handles loading and preprocessing market data
- **Strategy Module**: Generates buy/sell signals based on market data
- **Portfolio Module**: Manages positions and applies trading rules
- **Event System**: Facilitates communication between components
- **Backtesting Module**: Simulates trading based on signals and portfolio updates
- **Analysis Module**: Analyzes backtesting results and generates visualizations

For more details, see the [Architecture](architecture.md) documentation.

## Examples

The system comes with several examples to help you get started:

- [Basic Backtest](examples/basic.md): A simple moving average crossover strategy
- [Strategy Ensemble](examples/ensemble.md): Combining multiple strategies
- [Configuration-Based](examples/config.md): Using configuration files for backtesting

## Documentation

- [Installation](installation.md): Detailed installation instructions
- [Quick Start](quickstart.md): Get up and running quickly
- [User Guide](architecture.md): Comprehensive guide to using the system
- [API Reference](api/data.md): Detailed API documentation
- [Examples](examples/basic.md): Example usage scenarios

## Contributing

Contributions are welcome! Please see the [Contributing](contributing.md) guide for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/yourusername/backtester/blob/main/LICENSE) file for details. 