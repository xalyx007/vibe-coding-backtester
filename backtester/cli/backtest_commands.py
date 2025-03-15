"""
Backtest command implementation for the CLI.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from backtester.core import Backtester, BacktesterConfig
from backtester.events import EventBus
from backtester.utils.factory import create_data_source, create_strategy, create_portfolio_manager


def run_backtest(args, config: Optional[BacktesterConfig], logger: logging.Logger):
    """
    Run a backtest based on command-line arguments or configuration.
    
    Args:
        args: Command-line arguments
        config: Optional configuration object
        logger: Logger instance
    """
    # Create event bus
    event_bus = EventBus()
    
    # Subscribe to events
    event_bus.subscribe("backtest_started", lambda event: logger.info(f"Backtest started: {event['data']}"))
    event_bus.subscribe("backtest_completed", lambda event: logger.info(f"Backtest completed: {event['data']}"))
    event_bus.subscribe("error", lambda event: logger.error(f"Error: {event['data']}"))
    
    # Create components
    if config:
        # Create components from configuration
        config_dict = config.to_dict()
        data_source = create_data_source(config_dict["data_source"])
        strategy = create_strategy(config_dict["strategy"])
        portfolio_manager = create_portfolio_manager(config_dict["portfolio_manager"])
        
        # Set event bus
        data_source.event_bus = event_bus
        strategy.event_bus = event_bus
        portfolio_manager.event_bus = event_bus
        
        # Get backtest parameters
        transaction_costs = config_dict["backtest"].get("transaction_costs", 0.001)
        slippage = config_dict["backtest"].get("slippage", 0.0005)
        start_date = config_dict["backtest"].get("start_date")
        end_date = config_dict["backtest"].get("end_date")
        symbol = config_dict["backtest"].get("symbol", "BTC-USD")
    else:
        # Create components from arguments
        data_source = create_data_source_from_args(args)
        strategy = create_strategy_from_args(args)
        portfolio_manager = create_portfolio_manager_from_args(args)
        
        # Set event bus
        data_source.event_bus = event_bus
        strategy.event_bus = event_bus
        portfolio_manager.event_bus = event_bus
        
        # Get backtest parameters
        transaction_costs = args.transaction_costs
        slippage = args.slippage
        start_date = args.start_date
        end_date = args.end_date
        symbol = args.symbol
    
    # Create backtester
    backtester = Backtester(
        data_source=data_source,
        strategy=strategy,
        portfolio_manager=portfolio_manager,
        event_bus=event_bus,
        transaction_costs=transaction_costs,
        slippage=slippage
    )
    
    # Run backtest
    logger.info(f"Running backtest with {strategy.__class__.__name__} strategy")
    results = backtester.run(
        start_date=start_date,
        end_date=end_date,
        symbol=symbol
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.output_dir, 'backtest_results')
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_path = os.path.join(
        results_dir, 
        f"backtest_{strategy.__class__.__name__}_{timestamp}.json"
    )
    
    results.to_json(results_path)
    logger.info(f"Results saved to {results_path}")
    
    # Print summary
    logger.info("Backtest Summary:")
    for key, value in results.summary().items():
        logger.info(f"  {key}: {value}")
    
    return results


def create_data_source_from_args(args):
    """Create a data source from command-line arguments."""
    data_source_config = {
        "type": args.data_source
    }
    
    if args.data_source == 'csv':
        data_source_config.update({
            "path": args.data_path,
            "date_column": args.date_column
        })
    elif args.data_source == 'excel':
        data_source_config.update({
            "path": args.data_path,
            "date_column": args.date_column
        })
    elif args.data_source == 'exchange':
        data_source_config.update({
            "exchange_id": args.exchange_id,
            "timeframe": args.timeframe
        })
    
    return create_data_source(data_source_config)


def create_strategy_from_args(args):
    """Create a strategy from command-line arguments."""
    strategy_config = {
        "type": args.strategy
    }
    
    # Parse strategy parameters if provided
    if args.strategy_params:
        try:
            params = json.loads(args.strategy_params)
            strategy_config.update({"parameters": params})
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON for strategy parameters")
    
    return create_strategy(strategy_config)


def create_portfolio_manager_from_args(args):
    """Create a portfolio manager from command-line arguments."""
    portfolio_config = {
        "type": "simple",  # Default to simple portfolio manager
        "initial_capital": args.initial_capital,
        "position_size": args.position_size
    }
    
    return create_portfolio_manager(portfolio_config) 