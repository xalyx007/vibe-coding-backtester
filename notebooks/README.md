# Jupyter Notebooks

This directory contains Jupyter notebooks that demonstrate the usage of the Modular Backtesting System.

## Notebooks

- `basic_backtest.ipynb`: A basic example of backtesting a moving average crossover strategy.
- `btc_analysis.ipynb`: Analysis and backtesting of Bitcoin data from Yahoo Finance.
- `strategy_ensemble.ipynb`: An example of combining multiple strategies.
- `parameter_optimization.ipynb`: An example of optimizing strategy parameters.
- `visualization.ipynb`: Examples of visualizing backtest results.

## Running the Notebooks

You can run these notebooks in several ways:

### Local Installation

1. Install Jupyter:
   ```bash
   pip install jupyter
   ```

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Navigate to the notebook you want to run.

### Using Docker

You can also use the provided Docker configuration:

```bash
docker-compose up jupyter
```

This will start a Jupyter server accessible at http://localhost:8888.

## Requirements

These notebooks require the Modular Backtesting System to be installed, along with Jupyter and other dependencies:

```bash
pip install -e ".[dev]"
pip install jupyter matplotlib seaborn yfinance
```

## Data

Some notebooks use sample data that is generated automatically. The `btc_analysis.ipynb` notebook downloads Bitcoin data from Yahoo Finance using the `download_yahoo_crypto.py` script in the `scripts/data_downloaders` directory.

To manually download the data before running the notebook:

```bash
python scripts/data_downloaders/download_yahoo_crypto.py --btc-only
``` 