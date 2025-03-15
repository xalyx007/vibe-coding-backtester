# Backtest Results

This directory is used to store the results of backtests run with the Modular Backtesting System.

## Directory Structure

Results are typically organized by strategy and date:

```
results/
├── moving_average/
│   ├── 2023-01-01_123456/
│   │   ├── metrics.json       # Performance metrics
│   │   ├── trades.csv         # Trade history
│   │   ├── equity_curve.png   # Equity curve visualization
│   │   └── drawdown.png       # Drawdown visualization
│   └── ...
├── strategy_ensemble/
│   └── ...
└── ...
```

## Result Files

Each backtest run typically produces the following files:

- **metrics.json**: A JSON file containing performance metrics such as total return, Sharpe ratio, and maximum drawdown.
- **trades.csv**: A CSV file containing the trade history, including entry and exit prices, position sizes, and profits/losses.
- **equity_curve.png**: A visualization of the equity curve over time.
- **drawdown.png**: A visualization of the drawdown over time.
- **results.pkl**: A pickled DataFrame containing the full backtest results for further analysis.

## Using Results

You can load and analyze results using the following code:

```python
import pandas as pd
import json
import matplotlib.pyplot as plt
from backtester.utils.visualization import plot_equity_curve, plot_drawdown

# Load metrics
with open('results/moving_average/2023-01-01_123456/metrics.json', 'r') as f:
    metrics = json.load(f)
    
# Print metrics
for key, value in metrics.items():
    print(f"{key}: {value}")
    
# Load trades
trades = pd.read_csv('results/moving_average/2023-01-01_123456/trades.csv')

# Load full results
results = pd.read_pickle('results/moving_average/2023-01-01_123456/results.pkl')

# Plot equity curve
plot_equity_curve(results)
plt.show()

# Plot drawdown
plot_drawdown(results)
plt.show()
```

## Note

This directory is included in `.gitignore` to prevent large result files from being committed to the repository. You should back up important results separately if needed. 