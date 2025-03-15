# Validation Scripts Migration Guide

## Overview

The backtester validation functionality has been migrated from standalone scripts in the `tools/scripts` directory to a dedicated validation module in `backtester/validation`. This migration provides a more modular, reusable, and maintainable approach to validation.

## Changes

### Old Validation Scripts (Deprecated)

The following scripts are now deprecated and should no longer be used:

- `validate_backtester.py`: Basic validation tests
- `cross_validate_with_backtrader.py`: Cross-validation with Backtrader
- `monte_carlo_validation.py`: Monte Carlo validation
- `run_all_validations.py`: Comprehensive validation runner

### New Validation Module

The validation functionality is now available in the `backtester/validation` module:

- `backtester/validation/cross_validation.py`: Cross-validation functionality
- `backtester/validation/monte_carlo.py`: Monte Carlo simulation functionality
- `backtester/validation/walk_forward.py`: Walk-forward optimization functionality
- `backtester/validation/metrics.py`: Validation metrics calculation

### New Validation Script

A new script has been created to replace the old `run_all_validations.py`:

- `run_comprehensive_validation.py`: Runs all validation processes using the new validation module

## Usage

### Running Comprehensive Validation

To run a comprehensive validation of the backtester, use the new script:

```bash
cd tools/scripts
python run_comprehensive_validation.py
```

This will:
1. Run cross-validation with 5 folds
2. Run Monte Carlo simulation with 100 simulations
3. Run walk-forward optimization with parameter grid search
4. Generate a comprehensive report at `output/results/validation/comprehensive_report.md`

### Using the Validation Module Directly

You can also use the validation module directly in your Python code:

```python
from backtester.validation import (
    run_cross_validation,
    run_monte_carlo,
    run_walk_forward,
    calculate_validation_metrics
)

# Run cross-validation
cross_val_results = run_cross_validation(
    data_source=data_source,  # Optional, will use synthetic data if None
    strategy=strategy,        # Optional, will use MovingAverageCrossover if None
    portfolio_manager=portfolio_manager,  # Optional, will use BasicPortfolioManager if None
    folds=5,
    output_dir="output/results/validation/cross_validation"  # Optional, to save results
)

# Run Monte Carlo simulation
monte_carlo_results = run_monte_carlo(
    data_source=data_source,  # Optional, will use synthetic data if None
    strategy=strategy,        # Optional, will use MovingAverageCrossover if None
    portfolio_manager=portfolio_manager,  # Optional, will use BasicPortfolioManager if None
    simulations=100,
    output_dir="output/results/validation/monte_carlo"  # Optional, to save results
)

# Run walk-forward optimization
walk_forward_results = run_walk_forward(
    data_source=data_source,  # Optional, will use synthetic data if None
    strategy=strategy,        # Optional, will use MovingAverageCrossover if None
    portfolio_manager=portfolio_manager,  # Optional, will use BasicPortfolioManager if None
    window_size=60,
    step_size=20,
    parameter_grid={  # Optional, for parameter optimization
        'short_window': [5, 10, 20],
        'long_window': [30, 50, 100]
    },
    output_dir="output/results/validation/walk_forward"  # Optional, to save results
)

# Calculate validation metrics
metrics = calculate_validation_metrics(
    prices=prices,  # Series of prices
    trades=trades,  # List of trade dictionaries
    risk_free_rate=0.0,  # Optional, annualized risk-free rate
    annualization_factor=252  # Optional, annualization factor (252 for daily data)
)
```

## Benefits of the New Approach

1. **Modularity**: Each validation technique is implemented in a separate module, making it easier to maintain and extend.
2. **Reusability**: The validation functionality can be used directly from Python code, not just from command-line scripts.
3. **Flexibility**: Each validation function accepts optional parameters, allowing for customization of the validation process.
4. **Consistency**: All validation functions follow a consistent interface, making them easier to use and understand.
5. **Documentation**: The validation module includes comprehensive docstrings and a detailed README.

## Cleanup

The old validation scripts and directories will be kept for reference until the new validation module has been thoroughly tested. Once the new module is confirmed to be working correctly, the old scripts and directories can be removed.

## Questions and Support

If you have any questions or need support with the new validation module, please contact the backtester development team. 