#!/usr/bin/env python
"""
Visualization Example

This script demonstrates the visualization capabilities of the Modular Backtesting System.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set up plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Import backtester modules
from backtester.data import CSVDataSource
from backtester.strategy import MovingAverageCrossover
from backtester.portfolio import SimplePortfolioManager
from backtester.backtester import Backtester
from backtester.utils.metrics import calculate_metrics
from backtester.utils.visualization import (
    plot_equity_curve, 
    plot_drawdown, 
    plot_returns_distribution,
    plot_monthly_returns_heatmap,
    plot_underwater_chart
)

def main():
    """Run a visualization example."""
    print("Running visualization example...")
    
    # Create a data source
    print("Loading data...")
    data_source = CSVDataSource(
        file_path="data/BTC-USD.csv",
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
    
    # Create a strategy
    print("Creating strategy...")
    strategy = MovingAverageCrossover(
        short_window=20,
        long_window=50
    )
    
    # Create a portfolio manager
    print("Creating portfolio manager...")
    portfolio_manager = SimplePortfolioManager(
        initial_capital=10000,
        position_size=0.1
    )
    
    # Create a backtester
    print("Running backtest...")
    backtester = Backtester(
        data=data,
        strategy=strategy,
        portfolio_manager=portfolio_manager
    )
    
    # Run the backtest
    results = backtester.run()
    print(f"Backtest completed with {len(results)} results.")
    
    # Calculate performance metrics
    print("Calculating performance metrics...")
    metrics = calculate_metrics(results)
    
    # Display the metrics
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Create a directory for visualizations
    print("\nGenerating visualizations...")
    
    # 1. Equity Curve
    print("Plotting equity curve...")
    plt.figure(figsize=(12, 6))
    plot_equity_curve(results)
    plt.savefig("notebooks/assets/equity_curve_visualization.png")
    plt.close()
    
    # 2. Drawdown
    print("Plotting drawdown...")
    plt.figure(figsize=(12, 6))
    plot_drawdown(results)
    plt.savefig("notebooks/assets/drawdown_visualization.png")
    plt.close()
    
    # 3. Returns Distribution
    print("Plotting returns distribution...")
    plt.figure(figsize=(12, 6))
    plot_returns_distribution(results)
    plt.savefig("notebooks/assets/returns_distribution.png")
    plt.close()
    
    # 4. Monthly Returns Heatmap
    print("Plotting monthly returns heatmap...")
    plt.figure(figsize=(12, 8))
    plot_monthly_returns_heatmap(results)
    plt.savefig("notebooks/assets/monthly_returns_heatmap.png")
    plt.close()
    
    # 5. Underwater Chart
    print("Plotting underwater chart...")
    plt.figure(figsize=(12, 6))
    plot_underwater_chart(results)
    plt.savefig("notebooks/assets/underwater_chart.png")
    plt.close()
    
    # 6. Combined Dashboard
    print("Creating combined dashboard...")
    plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2)
    
    # Equity Curve
    ax1 = plt.subplot(gs[0, :])
    plot_equity_curve(results, ax=ax1)
    ax1.set_title('Equity Curve')
    
    # Drawdown
    ax2 = plt.subplot(gs[1, 0])
    plot_drawdown(results, ax=ax2)
    ax2.set_title('Drawdown')
    
    # Returns Distribution
    ax3 = plt.subplot(gs[1, 1])
    plot_returns_distribution(results, ax=ax3)
    ax3.set_title('Returns Distribution')
    
    # Underwater Chart
    ax4 = plt.subplot(gs[2, :])
    plot_underwater_chart(results, ax=ax4)
    ax4.set_title('Underwater Chart')
    
    plt.tight_layout()
    plt.savefig("notebooks/assets/dashboard.png")
    plt.close()
    
    # 7. Create a metrics table image
    print("Creating metrics table...")
    plt.figure(figsize=(8, 6))
    
    # Convert metrics to a DataFrame for display
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    
    # Create a table
    table = plt.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center'
    )
    
    # Modify table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Hide axes
    plt.axis('off')
    plt.title('Performance Metrics')
    
    plt.savefig("notebooks/assets/metrics_table.png", bbox_inches='tight')
    plt.close()
    
    print("\nVisualization example completed. Results saved to notebooks/assets/")

if __name__ == "__main__":
    main() 