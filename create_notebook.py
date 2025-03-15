import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create markdown cell with title
title_cell = nbf.v4.new_markdown_cell("# Basic Backtest Example\n\nThis notebook demonstrates how to use the Modular Backtesting System to run a simple backtest using a moving average crossover strategy.")

# Create code cells
import_cell = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)""")

modules_cell = nbf.v4.new_code_cell("""# Import backtester modules
from backtester.data import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import SimplePortfolioManager
from backtester.core.engine import Backtester
from backtester.validation.metrics import calculate_metrics
from backtester.analysis.visualization import plot_equity_curve, plot_drawdown""")

data_markdown = nbf.v4.new_markdown_cell("## Loading Data\n\nFirst, we'll load the data from a CSV file.")

data_cell = nbf.v4.new_code_cell("""# Create a data source
print("Loading data...")
data_source = CSVDataSource(
    file_path="../data/yahoo/1d/BTC_USD.csv",
    date_format="%Y-%m-%d",
    timestamp_column="Date",
    open_column="Open",
    high_column="High",
    low_column="Low",
    close_column="Close",
    volume_column="Volume"
)

# Load the data
data = data_source.load_data()
print(f"Loaded {len(data)} data points.")

# Display the first few rows of data
data.head()""")

strategy_markdown = nbf.v4.new_markdown_cell("## Creating a Strategy\n\nNext, we'll create a moving average crossover strategy.")

strategy_cell = nbf.v4.new_code_cell("""# Create a strategy
print("Creating strategy...")
strategy = MovingAverageCrossover(
    short_window=20,  # 20-day short-term moving average
    long_window=50    # 50-day long-term moving average
)

# Generate signals
signals = strategy.generate_signals(data)
print(f"Generated {len(signals)} signals.")

# Display the first few signals
signals.head()""")

portfolio_markdown = nbf.v4.new_markdown_cell("## Setting Up Portfolio Management\n\nNow, we'll create a portfolio manager to handle our trades.")

portfolio_cell = nbf.v4.new_code_cell("""# Create a portfolio manager
print("Creating portfolio manager...")
portfolio_manager = SimplePortfolioManager(
    initial_capital=10000,  # Starting with $10,000
    position_size=0.1       # Invest 10% of capital in each position
)""")

backtest_markdown = nbf.v4.new_markdown_cell("## Running the Backtest\n\nNow we'll run the backtest using our data, strategy, and portfolio manager.")

backtest_cell = nbf.v4.new_code_cell("""# Create a backtester
print("Running backtest...")
backtester = Backtester(
    data=data,
    strategy=strategy,
    portfolio_manager=portfolio_manager
)

# Run the backtest
results = backtester.run()
print(f"Backtest completed with {len(results.portfolio_values)} results.")""")

metrics_markdown = nbf.v4.new_markdown_cell("## Analyzing the Results\n\nLet's calculate and display the performance metrics.")

metrics_cell = nbf.v4.new_code_cell("""# Calculate performance metrics
print("Calculating performance metrics...")
metrics = calculate_metrics(
    returns=results.returns,
    positions=results.positions,
    trades=results.trades
)

# Display the metrics
print("\\nPerformance Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")""")

viz_markdown = nbf.v4.new_markdown_cell("## Visualizing the Results\n\nFinally, let's visualize the equity curve and drawdown.")

equity_cell = nbf.v4.new_code_cell("""# Plot the equity curve
plt.figure(figsize=(12, 6))
plt.title('Equity Curve')
plt.plot(results.portfolio_values)
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.show()""")

drawdown_cell = nbf.v4.new_code_cell("""# Calculate drawdown
peak = results.portfolio_values.cummax()
drawdown = (results.portfolio_values - peak) / peak

# Plot the drawdown
plt.figure(figsize=(12, 6))
plt.title('Drawdown')
plt.plot(drawdown)
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.grid(True)
plt.show()""")

conclusion_markdown = nbf.v4.new_markdown_cell("## Conclusion\n\nIn this notebook, we've demonstrated how to use the Modular Backtesting System to run a simple backtest using a moving average crossover strategy. We've loaded data, created a strategy, set up portfolio management, run the backtest, and analyzed the results.")

# Add cells to notebook
nb.cells = [
    title_cell,
    import_cell,
    modules_cell,
    data_markdown,
    data_cell,
    strategy_markdown,
    strategy_cell,
    portfolio_markdown,
    portfolio_cell,
    backtest_markdown,
    backtest_cell,
    metrics_markdown,
    metrics_cell,
    viz_markdown,
    equity_cell,
    drawdown_cell,
    conclusion_markdown
]

# Write the notebook to a file
with open('notebooks/basic_backtest.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully at notebooks/basic_backtest.ipynb") 