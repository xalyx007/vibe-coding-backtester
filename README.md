# Modular Backtesting System for Trading Strategies

A flexible, modular backend system for backtesting trading strategies with a focus on cryptocurrencies, extensible to other asset classes.

[![Tests](https://github.com/yourusername/backtester/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/backtester/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/backtester/badge/?version=latest)](https://backtester.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/backtester.svg)](https://badge.fury.io/py/backtester)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This system provides a comprehensive framework for backtesting trading strategies with the following key features:

- **Modular Architecture**: Decoupled components for data inputs, strategy generation, portfolio management, and backtesting
- **Multiple Data Sources**: Support for CSV, spreadsheets, and exchange APIs
- **Flexible Strategy Implementation**: Simple heuristics, machine learning models, and strategy combinations
- **Event-Based Communication**: Designed to integrate with a separate frontend via events
- **Comprehensive Performance Metrics**: Detailed analysis of backtesting results
- **Visualization Tools**: Generate charts and reports for strategy performance
- **Command-Line Interface**: Run backtests from the command line
- **Configuration-Based Setup**: Define backtests using YAML or JSON configuration files

## Mandatory Validation

This project enforces mandatory validation before committing changes. All tests must pass before commits are allowed. See [VALIDATION.md](VALIDATION.md) for details.

## Installation

### From PyPI (Recommended)

```bash
pip install backtester
```

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/backtester.git
cd backtester

# Set up the validation process
python setup_validation.py

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Basic Backtesting

```python
from backtester.data import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import SimplePortfolioManager
from backtester.backtester import Backtester
from backtester.utils.metrics import calculate_metrics
from backtester.utils.visualization import plot_equity_curve

# Initialize components
data_source = CSVDataSource("data/btc_daily.csv")
data = data_source.load_data()
strategy = MovingAverageCrossover(short_window=20, long_window=50)
portfolio = SimplePortfolioManager(initial_capital=10000, position_size=0.1)
backtester = Backtester(data=data, strategy=strategy, portfolio_manager=portfolio)

# Run backtest
results = backtester.run()

# Analyze results
metrics = calculate_metrics(results)
print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
plot_equity_curve(results)
```

### Using the Command Line Interface

```bash
# Run a backtest using a configuration file
backtester run --config examples/config.yaml

# Run a backtest with command-line arguments
backtester run --data-source data/btc_daily.csv --strategy moving_average_crossover --short-window 20 --long-window 50 --initial-capital 10000 --position-size 0.1 --output results.json
```

## Architecture

The system consists of four main modules:

1. **Data Module**: Gathers and preprocesses market data
2. **Strategy Module**: Generates buy/sell signals based on input data
3. **Portfolio Manager Module**: Manages positions and applies trading rules
4. **Backtesting Module**: Simulates trading based on signals and portfolio updates

These modules communicate through an event-driven architecture, allowing for flexible and extensible backtesting.

![Architecture Diagram](docs/assets/architecture_diagram.png)

## Key Components

### Data Sources

- **CSVDataSource**: Load data from CSV files
- **APIDataSource**: Fetch data from exchange APIs
- **Custom Data Sources**: Create your own data sources by inheriting from the base class

### Strategies

- **MovingAverageCrossover**: Simple moving average crossover strategy
- **StrategyEnsemble**: Combine multiple strategies with custom weighting
- **Custom Strategies**: Create your own strategies by inheriting from the base class

### Portfolio Management

- **SimplePortfolioManager**: Basic portfolio management with fixed position sizing
- **Custom Portfolio Managers**: Create your own portfolio managers by inheriting from the base class

### Event System

- **EventBus**: Central event dispatcher for communication between components
- **EventTypes**: Predefined event types for system communication

## Documentation

For detailed documentation, see the [docs](./docs) directory or visit the [online documentation](https://backtester.readthedocs.io/):

- [Architecture Overview](./docs/architecture.md)
- [Module Specifications](./docs/module_specs.md)
- [API Reference](./docs/api_reference.md)
- [Configuration Guide](./docs/configuration.md)

## Examples

The [examples](./examples) directory contains sample scripts and configuration files for various backtesting scenarios:

- [Basic Moving Average Crossover](./examples/basic_backtest.py)
- [Strategy Ensemble](./examples/strategy_ensemble.py)
- [Configuration-Based Backtest](./examples/config.yaml)

## Notebooks

The [notebooks](./notebooks) directory contains Jupyter notebooks for interactive examples and tutorials:

- [Basic Backtest](./notebooks/basic_backtest.py)
- [Parameter Optimization](./notebooks/parameter_optimization.py)
- [Strategy Ensemble](./notebooks/strategy_ensemble.py)
- [Visualization](./notebooks/visualization.py)

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check code style
flake8 backtester tests
black --check backtester tests
isort --check-only --profile black backtester tests
```

### Project Structure

The project is organized into the following main directories:

- `backtester/`: Main package with the backtesting system code
  - `core/`: Core functionality (engine, results, config)
  - `data/`: Data handling and preprocessing
  - `strategy/`: Trading strategy implementations
  - `portfolio/`: Portfolio management
  - `analysis/`: Analysis and visualization
  - `validation/`: Strategy validation techniques
  - `events/`: Event system
  - `utils/`: Utilities
  - `cli/`: Command-line interface

- `docs/`: Documentation
  - `api/`: API documentation
  - `user_guide/`: User guide and tutorials
  - `examples/`: Documentation for examples
  - `CHANGELOG.md`: Project changelog
  - `CODE_OF_CONDUCT.md`: Code of conduct
  - `SECURITY.md`: Security policy
  - `MIGRATION_GUIDE.md`: Migration guide

- `output/`: Output files
  - `logs/`: Log files
  - `results/`: Results from backtests and validations
  - `reports/`: Generated reports

- `tools/`: Development tools
  - `scripts/`: Utility scripts
  - `docker/`: Docker configuration
  - `ci/`: CI/CD configuration
  - `config/`: Copies of configuration files for reference
  - `package/`: Copies of package-related files for reference

**Note on Configuration and Package Files**: Configuration files (like `.flake8`, `pytest.ini`, etc.) and package files (like `setup.py`, `pyproject.toml`, etc.) remain in the root directory for functionality but are copied to `tools/config/` and `tools/package/` respectively for organization and reference.

See [docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md) for a detailed overview of the project structure.

### Contributing

Contributions are welcome! Please see [docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details. 