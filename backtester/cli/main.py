#!/usr/bin/env python
"""
Main entry point for the backtesting system CLI.
"""

import argparse
import os
import sys
import logging
from datetime import datetime

from backtester.core import BacktesterConfig
from backtester.utils.logging import setup_logging
from backtester.cli.backtest_commands import run_backtest
from backtester.cli.validation_commands import run_validation
from backtester.cli.analysis_commands import run_analysis


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Modular Backtesting System')
    
    # Global options
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-c', '--config', type=str, help='Path to configuration file (YAML or JSON)')
    parser.add_argument('-o', '--output-dir', type=str, default='output', 
                       help='Directory to store output files')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest')
    
    # Data source options
    data_group = backtest_parser.add_argument_group('Data Source')
    data_group.add_argument('--data-source', type=str, choices=['csv', 'excel', 'exchange'], 
                           help='Type of data source')
    data_group.add_argument('--data-path', type=str, help='Path to data file')
    data_group.add_argument('--date-column', type=str, default='timestamp', 
                           help='Name of date/timestamp column')
    data_group.add_argument('--exchange-id', type=str, help='Exchange ID (for exchange data source)')
    data_group.add_argument('--timeframe', type=str, help='Timeframe (for exchange data source)')
    
    # Strategy options
    strategy_group = backtest_parser.add_argument_group('Strategy')
    strategy_group.add_argument('--strategy', type=str, 
                               choices=['moving_average', 'bollinger_bands', 'rsi', 'ml', 'ensemble', 'combined'], 
                               help='Type of strategy')
    strategy_group.add_argument('--strategy-params', type=str, 
                               help='Strategy parameters as JSON string')
    
    # Portfolio options
    portfolio_group = backtest_parser.add_argument_group('Portfolio')
    portfolio_group.add_argument('--initial-capital', type=float, default=10000.0, 
                                help='Initial capital')
    portfolio_group.add_argument('--position-size', type=float, default=0.1, 
                                help='Position size as fraction of portfolio')
    
    # Backtest options
    backtest_group = backtest_parser.add_argument_group('Backtest')
    backtest_group.add_argument('--start-date', type=str, help='Start date for backtest')
    backtest_group.add_argument('--end-date', type=str, help='End date for backtest')
    backtest_group.add_argument('--symbol', type=str, default='BTC-USD', help='Trading symbol')
    backtest_group.add_argument('--transaction-costs', type=float, default=0.001, 
                               help='Transaction costs as fraction of trade value')
    backtest_group.add_argument('--slippage', type=float, default=0.0005, 
                               help='Slippage as fraction of price')
    
    # Validation command
    validation_parser = subparsers.add_parser('validation', help='Run validation tests')
    validation_parser.add_argument('--type', type=str, 
                                 choices=['cross', 'monte-carlo', 'walk-forward'], 
                                 required=True,
                                 help='Type of validation to run')
    validation_parser.add_argument('--folds', type=int, default=5, 
                                 help='Number of folds for cross-validation')
    validation_parser.add_argument('--simulations', type=int, default=100, 
                                 help='Number of simulations for Monte Carlo validation')
    validation_parser.add_argument('--window-size', type=int, default=252, 
                                 help='Window size for walk-forward validation')
    validation_parser.add_argument('--step-size', type=int, default=63, 
                                 help='Step size for walk-forward validation')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analysis', help='Run analysis on results')
    analysis_parser.add_argument('--results-path', type=str, required=True, 
                               help='Path to results file or directory')
    analysis_parser.add_argument('--type', type=str, 
                               choices=['metrics', 'visualization', 'report'], 
                               required=True,
                               help='Type of analysis to run')
    analysis_parser.add_argument('--report-format', type=str, 
                               choices=['html', 'pdf', 'json'], 
                               default='html',
                               help='Format for report output')
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(verbose=args.verbose)
    logger.info("Starting backtester")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output directory: {args.output_dir}")
    
    # Load configuration if provided
    config = None
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = BacktesterConfig(args.config)
    
    # Run the appropriate command
    if args.command == 'backtest':
        run_backtest(args, config, logger)
    elif args.command == 'validation':
        run_validation(args, config, logger)
    elif args.command == 'analysis':
        run_analysis(args, config, logger)
    else:
        logger.error("No command specified")
        sys.exit(1)


if __name__ == '__main__':
    main() 