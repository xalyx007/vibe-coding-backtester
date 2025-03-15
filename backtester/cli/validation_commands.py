"""
Validation command implementation for the CLI.
"""

import os
import logging
from typing import Optional
from datetime import datetime

from backtester.core import BacktesterConfig
from backtester.validation import run_cross_validation, run_monte_carlo, run_walk_forward
from backtester.utils.factory import create_data_source, create_strategy, create_portfolio_manager


def run_validation(args, config: Optional[BacktesterConfig], logger: logging.Logger):
    """
    Run validation tests based on command-line arguments or configuration.
    
    Args:
        args: Command-line arguments
        config: Optional configuration object
        logger: Logger instance
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    validation_dir = os.path.join(args.output_dir, 'results', 'validation')
    
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    
    # Get components from config or args
    if config:
        config_dict = config.to_dict()
        data_source = create_data_source(config_dict["data_source"])
        strategy = create_strategy(config_dict["strategy"])
        portfolio_manager = create_portfolio_manager(config_dict["portfolio_manager"])
        
        # Get backtest parameters
        transaction_costs = config_dict["backtest"].get("transaction_costs", 0.001)
        slippage = config_dict["backtest"].get("slippage", 0.0005)
        start_date = config_dict["backtest"].get("start_date")
        end_date = config_dict["backtest"].get("end_date")
        symbol = config_dict["backtest"].get("symbol", "BTC-USD")
    else:
        # Check if required arguments are provided
        if not hasattr(args, 'data_source') or not args.data_source:
            logger.warning("No data source provided. Using placeholder components for validation.")
            data_source = None
            strategy = None
            portfolio_manager = None
            
            # Get backtest parameters
            transaction_costs = getattr(args, 'transaction_costs', 0.001)
            slippage = getattr(args, 'slippage', 0.0005)
            start_date = getattr(args, 'start_date', None)
            end_date = getattr(args, 'end_date', None)
            symbol = getattr(args, 'symbol', 'BTC-USD')
        else:
            # Use the backtest_commands functions to create components
            from backtester.cli.backtest_commands import (
                create_data_source_from_args,
                create_strategy_from_args,
                create_portfolio_manager_from_args
            )
            
            data_source = create_data_source_from_args(args)
            strategy = create_strategy_from_args(args)
            portfolio_manager = create_portfolio_manager_from_args(args)
            
            # Get backtest parameters
            transaction_costs = getattr(args, 'transaction_costs', 0.001)
            slippage = getattr(args, 'slippage', 0.0005)
            start_date = getattr(args, 'start_date', None)
            end_date = getattr(args, 'end_date', None)
            symbol = getattr(args, 'symbol', 'BTC-USD')
    
    # Run the appropriate validation
    validation_type = args.type
    
    if validation_type == 'cross':
        # Create specific output directory for this validation
        cross_val_dir = os.path.join(validation_dir, 'cross_validation')
        os.makedirs(cross_val_dir, exist_ok=True)
        
        logger.info(f"Running cross-validation with {args.folds} folds")
        results = run_cross_validation(
            data_source=data_source,
            strategy=strategy,
            portfolio_manager=portfolio_manager,
            folds=args.folds,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            transaction_costs=transaction_costs,
            slippage=slippage,
            output_dir=cross_val_dir
        )
        
        # Save results
        results_path = os.path.join(
            validation_dir, 
            f"cross_validation_{strategy.__class__.__name__ if strategy else 'default'}_{timestamp}.json"
        )
        
    elif validation_type == 'monte-carlo':
        # Create specific output directory for this validation
        monte_carlo_dir = os.path.join(validation_dir, 'monte_carlo')
        os.makedirs(monte_carlo_dir, exist_ok=True)
        
        logger.info(f"Running Monte Carlo validation with {args.simulations} simulations")
        results = run_monte_carlo(
            data_source=data_source,
            strategy=strategy,
            portfolio_manager=portfolio_manager,
            simulations=args.simulations,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            transaction_costs=transaction_costs,
            slippage=slippage,
            output_dir=monte_carlo_dir
        )
        
        # Save results
        results_path = os.path.join(
            validation_dir, 
            f"monte_carlo_{strategy.__class__.__name__ if strategy else 'default'}_{timestamp}.json"
        )
        
    elif validation_type == 'walk-forward':
        # Create specific output directory for this validation
        walk_forward_dir = os.path.join(validation_dir, 'walk_forward')
        os.makedirs(walk_forward_dir, exist_ok=True)
        
        # Check if parameter grid is provided
        parameter_grid = None
        if hasattr(args, 'parameter_grid') and args.parameter_grid:
            import json
            parameter_grid = json.loads(args.parameter_grid)
        
        logger.info(f"Running walk-forward validation with window size {args.window_size}")
        results = run_walk_forward(
            data_source=data_source,
            strategy=strategy,
            portfolio_manager=portfolio_manager,
            window_size=args.window_size,
            step_size=args.step_size,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            transaction_costs=transaction_costs,
            slippage=slippage,
            parameter_grid=parameter_grid,
            output_dir=walk_forward_dir
        )
        
        # Save results
        results_path = os.path.join(
            validation_dir, 
            f"walk_forward_{strategy.__class__.__name__ if strategy else 'default'}_{timestamp}.json"
        )
    
    # Save results
    with open(results_path, 'w') as f:
        import json
        json.dump(results, f, indent=4, default=str)
    
    logger.info(f"Validation results saved to {results_path}")
    
    # Print summary
    logger.info("Validation Summary:")
    if validation_type == 'cross':
        logger.info(f"  Average Return: {results['average_return']:.4f}")
        logger.info(f"  Standard Deviation: {results['std_return']:.4f}")
        logger.info(f"  Max Return: {results['max_return']:.4f}")
        logger.info(f"  Min Return: {results['min_return']:.4f}")
    elif validation_type == 'monte-carlo':
        logger.info(f"  Average Return: {results['average_return']:.4f}")
        logger.info(f"  95% VaR: {results['var_95']:.4f}")
        logger.info(f"  99% VaR: {results['var_99']:.4f}")
        logger.info(f"  Max Return: {results['max_return']:.4f}")
        logger.info(f"  Min Return: {results['min_return']:.4f}")
    elif validation_type == 'walk-forward':
        logger.info(f"  Average Return: {results['average_return']:.4f}")
        logger.info(f"  Best Window Return: {results['best_window_return']:.4f}")
        logger.info(f"  Worst Window Return: {results['worst_window_return']:.4f}")
        logger.info(f"  Number of Windows: {len(results['window_results'])}")
    
    return results 