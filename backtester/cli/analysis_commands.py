"""
Analysis command implementation for the CLI.
"""

import os
import json
import logging
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime

from backtester.core import BacktesterConfig
from backtester.analysis.metrics import calculate_metrics
from backtester.analysis.visualization import plot_equity_curve, plot_drawdown, plot_trades


def run_analysis(args, config: Optional[BacktesterConfig], logger: logging.Logger):
    """
    Run analysis on backtest results based on command-line arguments or configuration.
    
    Args:
        args: Command-line arguments
        config: Optional configuration object
        logger: Logger instance
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    reports_dir = os.path.join(args.output_dir, 'reports')
    
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Load results
    results_path = args.results_path
    
    if not os.path.exists(results_path):
        logger.error(f"Results file not found: {results_path}")
        return
    
    logger.info(f"Loading results from {results_path}")
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in results file: {results_path}")
        return
    
    # Run the appropriate analysis
    analysis_type = args.type
    
    if analysis_type == 'metrics':
        logger.info("Calculating performance metrics")
        metrics = calculate_metrics_from_results(results)
        
        # Save metrics
        metrics_path = os.path.join(
            reports_dir, 
            f"metrics_{os.path.basename(results_path).split('.')[0]}_{timestamp}.json"
        )
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Print metrics
        logger.info("Performance Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        return metrics
        
    elif analysis_type == 'visualization':
        logger.info("Generating visualizations")
        
        # Create visualizations
        figures = generate_visualizations(results)
        
        # Save visualizations
        base_name = os.path.basename(results_path).split('.')[0]
        
        for name, fig in figures.items():
            fig_path = os.path.join(
                reports_dir, 
                f"{name}_{base_name}_{timestamp}.png"
            )
            
            fig.savefig(fig_path)
            logger.info(f"Visualization saved to {fig_path}")
        
        return figures
        
    elif analysis_type == 'report':
        logger.info(f"Generating {args.report_format} report")
        
        # Generate report
        report_format = args.report_format
        
        if report_format == 'html':
            report_path = os.path.join(
                reports_dir, 
                f"report_{os.path.basename(results_path).split('.')[0]}_{timestamp}.html"
            )
            
            # Import here to avoid circular imports
            from backtester.analysis.reporting.html_report import generate_html_report
            generate_html_report(results, report_path)
            
        elif report_format == 'pdf':
            report_path = os.path.join(
                reports_dir, 
                f"report_{os.path.basename(results_path).split('.')[0]}_{timestamp}.pdf"
            )
            
            # Import here to avoid circular imports
            from backtester.analysis.reporting.pdf_report import generate_pdf_report
            generate_pdf_report(results, report_path)
            
        elif report_format == 'json':
            report_path = os.path.join(
                reports_dir, 
                f"report_{os.path.basename(results_path).split('.')[0]}_{timestamp}.json"
            )
            
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=4)
        
        logger.info(f"Report saved to {report_path}")
        
        return report_path


def calculate_metrics_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        results: Dictionary with backtest results
        
    Returns:
        Dictionary with performance metrics
    """
    # Extract data from results
    portfolio_values = pd.DataFrame(results["portfolio_values"])
    trades = pd.DataFrame(results["trades"])
    initial_capital = results["summary"]["initial_capital"]
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio_values, trades, initial_capital)
    
    return metrics


def generate_visualizations(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate visualizations from backtest results.
    
    Args:
        results: Dictionary with backtest results
        
    Returns:
        Dictionary with visualization figures
    """
    # Extract data from results
    portfolio_values = pd.DataFrame(results["portfolio_values"])
    trades = pd.DataFrame(results["trades"])
    
    # Convert timestamp strings to datetime
    portfolio_values["timestamp"] = pd.to_datetime(portfolio_values["timestamp"])
    portfolio_values.set_index("timestamp", inplace=True)
    
    if "timestamp" in trades.columns:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        trades.set_index("timestamp", inplace=True)
    
    # Generate visualizations
    figures = {}
    
    # Equity curve
    figures["equity_curve"] = plot_equity_curve(portfolio_values)
    
    # Drawdown
    figures["drawdown"] = plot_drawdown(portfolio_values)
    
    # Trades
    if "signals" in results:
        signals = pd.DataFrame(results["signals"])
        signals["timestamp"] = pd.to_datetime(signals["timestamp"])
        signals.set_index("timestamp", inplace=True)
        
        symbol = results["summary"]["symbol"]
        figures["trades"] = plot_trades(trades, signals, symbol)
    
    return figures 