#!/usr/bin/env python
"""
Test script for the validation module.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtester.validation import (
    run_cross_validation,
    run_monte_carlo,
    run_walk_forward,
    calculate_metrics
)


def create_sample_data():
    """Create sample price data for testing."""
    days = 252
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(days)]
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, days))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.uniform(100, 1000, days)
    })
    
    data = data.set_index('timestamp')
    return data


def create_sample_trades():
    """Create sample trades for testing."""
    return [
        {
            'entry_time': datetime(2020, 1, 10),
            'exit_time': datetime(2020, 1, 15),
            'entry_price': 100.0,
            'exit_price': 105.0,
            'quantity': 1.0,
            'direction': 'long',
            'profit': 5.0,
            'profit_pct': 0.05
        },
        {
            'entry_time': datetime(2020, 2, 10),
            'exit_time': datetime(2020, 2, 15),
            'entry_price': 110.0,
            'exit_price': 108.0,
            'quantity': 1.0,
            'direction': 'long',
            'profit': -2.0,
            'profit_pct': -0.018
        },
        {
            'entry_time': datetime(2020, 3, 10),
            'exit_time': datetime(2020, 3, 15),
            'entry_price': 105.0,
            'exit_price': 115.0,
            'quantity': 1.0,
            'direction': 'long',
            'profit': 10.0,
            'profit_pct': 0.095
        }
    ]


def main():
    """Run the validation tests."""
    print("Testing validation module...\n")
    
    # Create sample data
    sample_data = create_sample_data()
    sample_trades = create_sample_trades()
    output_dir = "output/results/validation/test"
    
    # Run cross-validation
    print("Running cross-validation...")
    cv_results = run_cross_validation(
        data_source=None,  # Will use synthetic data
        strategy=None,     # Will use default strategy
        portfolio_manager=None,  # Will use default portfolio manager
        folds=5,
        output_dir=output_dir
    )
    print(json.dumps(cv_results, indent=2))
    
    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    mc_results = run_monte_carlo(
        data_source=None,  # Will use synthetic data
        strategy=None,     # Will use default strategy
        portfolio_manager=None,  # Will use default portfolio manager
        simulations=100,
        output_dir=output_dir
    )
    print(json.dumps(mc_results, indent=2))
    
    # Run walk-forward optimization
    print("\nRunning walk-forward optimization...")
    wf_results = run_walk_forward(
        data_source=None,  # Will use synthetic data
        strategy=None,     # Will use default strategy
        portfolio_manager=None,  # Will use default portfolio manager
        window_size=60,
        step_size=20,
        output_dir=output_dir
    )
    print(json.dumps(wf_results, indent=2))
    
    # Calculate validation metrics
    print("\nCalculating validation metrics...")
    # Convert sample_data['close'] to returns for the calculate_metrics function
    returns = sample_data['close'].pct_change().dropna().tolist()
    # Create dummy positions (1 for each period)
    positions = [1] * len(returns)
    # Convert sample_trades to the format expected by calculate_metrics
    # The calculate_metrics function expects trades as tuples of (direction, size, date)
    # where direction * size represents the profit/loss
    trades_tuples = [(t['profit'], t['quantity'], t['exit_time']) for t in sample_trades]
    
    metrics = calculate_metrics(
        returns=returns,
        positions=positions,
        trades=trades_tuples
    )
    print(json.dumps(metrics, indent=2))
    
    print("\nAll validation tests completed successfully!")


if __name__ == "__main__":
    main() 