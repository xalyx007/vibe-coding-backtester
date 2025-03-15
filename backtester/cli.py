#!/usr/bin/env python
"""
Command-line interface for the Modular Backtesting System.
"""

import argparse
import os
import sys
import yaml
import json
import logging
from datetime import datetime

from backtester.utils import load_config
from backtester.utils.factory import create_data_source, create_strategy, create_portfolio_manager
from backtester.backtest import Backtester
from backtester.events import EventBus


def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"backtester_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    return logging.getLogger('backtester')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Modular Backtesting System')
    
    # Config file
    parser.add_argument('-c', '--config', type=str, help='Path to configuration file (YAML or JSON)')
    
    # Data source options
    data_group = parser.add_argument_group('Data Source')
    data_group.add_argument('--data-source', type=str, choices=['csv', 'excel', 'exchange'], 
                           help='Type of data source')
    data_group.add_argument('--data-path', type=str, help='Path to data file')
    data_group.add_argument('--date-column', type=str, default='timestamp', 
                           help='Name of date/timestamp column')
    data_group.add_argument('--exchange-id', type=str, help='Exchange ID (for exchange data source)')
    data_group.add_argument('--timeframe', type=str, help='Timeframe (for exchange data source)')
    
    # Strategy options
    strategy_group = parser.add_argument_group('Strategy')
    strategy_group.add_argument('--strategy', type=str, 
                               choices=['moving_average', 'rsi', 'bollinger_bands', 'ensemble'], 
                               help='Type of strategy')
    strategy_group.add_argument('--short-window', type=int, default=20, 
                               help='Short window for moving average strategy')
    strategy_group.add_argument('--long-window', type=int, default=50, 
                               help='Long window for moving average strategy')
    strategy_group.add_argument('--rsi-period', type=int, default=14, 
                               help='Period for RSI strategy')
    strategy_group.add_argument('--rsi-overbought', type=int, default=70, 
                               help='Overbought threshold for RSI strategy')
    strategy_group.add_argument('--rsi-oversold', type=int, default=30, 
                               help='Oversold threshold for RSI strategy')
    
    # Portfolio options
    portfolio_group = parser.add_argument_group('Portfolio')
    portfolio_group.add_argument('--initial-capital', type=float, default=10000, 
                                help='Initial capital')
    portfolio_group.add_argument('--position-sizing', type=str, 
                                choices=['fixed', 'percentage'], default='percentage', 
                                help='Position sizing method')
    portfolio_group.add_argument('--position-size', type=float, default=0.1, 
                                help='Position size (amount or percentage)')
    
    # Backtest options
    backtest_group = parser.add_argument_group('Backtest')
    backtest_group.add_argument('--symbol', type=str, default='BTC-USD', 
                               help='Trading symbol')
    backtest_group.add_argument('--start-date', type=str, 
                               help='Start date (YYYY-MM-DD)')
    backtest_group.add_argument('--end-date', type=str, 
                               help='End date (YYYY-MM-DD)')
    backtest_group.add_argument('--transaction-costs', type=float, default=0.001, 
                               help='Transaction costs as a fraction of trade value')
    backtest_group.add_argument('--slippage', type=float, default=0.0005, 
                               help='Slippage as a fraction of price')
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', type=str, default='results', 
                             help='Output directory for results and plots')
    output_group.add_argument('--save-plots', action='store_true', 
                             help='Save plots to output directory')
    output_group.add_argument('--show-plots', action='store_true', 
                             help='Show plots')
    output_group.add_argument('--save-results', action='store_true', 
                             help='Save results to output directory')
    
    # Other options
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    return parser.parse_args()


def create_components_from_args(args, logger):
    """Create components from command-line arguments."""
    # Create data source
    if args.data_source == 'csv':
        if not args.data_path:
            logger.error("Data path is required for CSV data source")
            sys.exit(1)
        
        from backtester.inputs import CSVDataSource
        data_source = CSVDataSource(
            file_path=args.data_path,
            date_column=args.date_column
        )
    elif args.data_source == 'excel':
        if not args.data_path:
            logger.error("Data path is required for Excel data source")
            sys.exit(1)
        
        from backtester.inputs import ExcelDataSource
        data_source = ExcelDataSource(
            file_path=args.data_path,
            date_column=args.date_column
        )
    elif args.data_source == 'exchange':
        if not args.exchange_id:
            logger.error("Exchange ID is required for exchange data source")
            sys.exit(1)
        
        from backtester.inputs import ExchangeDataSource
        data_source = ExchangeDataSource(
            exchange_id=args.exchange_id,
            symbol=args.symbol,
            timeframe=args.timeframe or '1d'
        )
    else:
        logger.error(f"Unknown data source: {args.data_source}")
        sys.exit(1)
    
    # Create strategy
    if args.strategy == 'moving_average':
        from backtester.strategy import MovingAverageCrossover
        strategy = MovingAverageCrossover(
            short_window=args.short_window,
            long_window=args.long_window
        )
    elif args.strategy == 'rsi':
        from backtester.strategy import RSIStrategy
        strategy = RSIStrategy(
            period=args.rsi_period,
            overbought=args.rsi_overbought,
            oversold=args.rsi_oversold
        )
    elif args.strategy == 'bollinger_bands':
        from backtester.strategy import BollingerBandsStrategy
        strategy = BollingerBandsStrategy(
            window=args.short_window,
            num_std=2
        )
    elif args.strategy == 'ensemble':
        from backtester.strategy import StrategyEnsemble, MovingAverageCrossover, RSIStrategy
        
        # Create individual strategies
        ma_strategy = MovingAverageCrossover(
            short_window=args.short_window,
            long_window=args.long_window
        )
        
        rsi_strategy = RSIStrategy(
            period=args.rsi_period,
            overbought=args.rsi_overbought,
            oversold=args.rsi_oversold
        )
        
        strategy = StrategyEnsemble(
            strategies=[ma_strategy, rsi_strategy],
            weights=[0.5, 0.5]
        )
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        sys.exit(1)
    
    # Create portfolio manager
    if args.position_sizing == 'fixed':
        from backtester.portfolio import BasicPortfolioManager, FixedAmountSizer
        position_sizer = FixedAmountSizer(amount=args.position_size)
    else:  # percentage
        from backtester.portfolio import BasicPortfolioManager, PercentageSizer
        position_sizer = PercentageSizer(percentage=args.position_size)
    
    portfolio_manager = BasicPortfolioManager(
        initial_capital=args.initial_capital,
        position_sizer=position_sizer
    )
    
    return data_source, strategy, portfolio_manager


def main():
    """Main entry point for the CLI."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    logger.info("Starting backtester")
    
    # Create event bus
    event_bus = EventBus()
    
    # Subscribe to events
    event_bus.subscribe("backtest_started", lambda event: logger.info(f"Backtest started: {event['data']}"))
    event_bus.subscribe("backtest_completed", lambda event: logger.info(f"Backtest completed: {event['data']}"))
    event_bus.subscribe("error", lambda event: logger.error(f"Error: {event['data']}"))
    
    # Create components
    if args.config:
        # Load configuration from file
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Create components from configuration
        data_source = create_data_source(config["data_source"])
        strategy = create_strategy(config["strategy"])
        portfolio_manager = create_portfolio_manager(config["portfolio_manager"])
        
        # Set event bus
        data_source.event_bus = event_bus
        strategy.event_bus = event_bus
        portfolio_manager.event_bus = event_bus
        
        # Get backtest parameters
        transaction_costs = config["backtest"].get("transaction_costs", 0.001)
        slippage = config["backtest"].get("slippage", 0.0005)
        start_date = config["backtest"].get("start_date")
        end_date = config["backtest"].get("end_date")
        symbol = config["backtest"].get("symbol", "BTC-USD")
    else:
        # Create components from arguments
        data_source, strategy, portfolio_manager = create_components_from_args(args, logger)
        
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
    logger.info("Running backtest...")
    results = backtester.run(
        start_date=start_date,
        end_date=end_date,
        symbol=symbol
    )
    
    # Print results summary
    summary = results.summary()
    logger.info("\nBacktest Results Summary:")
    logger.info(f"Initial Capital: ${summary['initial_capital']:.2f}")
    logger.info(f"Final Capital: ${summary['final_capital']:.2f}")
    logger.info(f"Total Return: {summary['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max Drawdown: {summary.get('max_drawdown', 0):.2%}")
    logger.info(f"Number of Trades: {summary['total_trades']}")
    
    # Create output directory if needed
    output_dir = args.output_dir
    if (args.save_plots or args.save_results) and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save plots
    if args.save_plots:
        logger.info(f"Saving plots to {output_dir}...")
        fig1 = results.plot_equity_curve()
        fig2 = results.plot_drawdown()
        fig3 = results.plot_trades()
        
        fig1.savefig(os.path.join(output_dir, "equity_curve.png"))
        fig2.savefig(os.path.join(output_dir, "drawdown.png"))
        fig3.savefig(os.path.join(output_dir, "trades.png"))
    
    # Save results
    if args.save_results:
        results_path = os.path.join(output_dir, "backtest_results.json")
        logger.info(f"Saving results to {results_path}...")
        results.to_json(results_path)
    
    # Show plots
    if args.show_plots:
        import matplotlib.pyplot as plt
        logger.info("Showing plots...")
        results.plot_equity_curve()
        results.plot_drawdown()
        results.plot_trades()
        plt.show()
    
    logger.info("Backtest completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 