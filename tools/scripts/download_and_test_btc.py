#!/usr/bin/env python
"""
Download and Test Bitcoin Data

This script downloads Bitcoin data from Yahoo Finance and runs a simple backtest
to verify that the data is valid and the backtesting system is working correctly.
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the download script
from scripts.data_downloaders.download_yahoo_crypto import download_btc_data

# Try to import backtesting components
try:
    # First try the standard import path
    from backtester.inputs import CSVDataSource
    from backtester.strategy import MovingAverageCrossover
    from backtester.portfolio import BasicPortfolioManager
    from backtester.backtest import Backtester
    logger.info("Successfully imported backtesting components using standard paths")
except ImportError:
    try:
        # Try alternative import paths based on the actual project structure
        from backtester.data import CSVDataSource
        from backtester.strategy import MovingAverageCrossover
        from backtester.portfolio import SimplePortfolioManager
        from backtester.backtester import Backtester
        logger.info("Successfully imported backtesting components using alternative paths")
    except ImportError as e:
        logger.error(f"Failed to import backtesting components: {e}")
        logger.error("Please make sure the backtester package is installed or in the Python path")
        sys.exit(1)


def download_data():
    """Download Bitcoin data from Yahoo Finance."""
    logger.info("Downloading Bitcoin data...")
    btc_data = download_btc_data()
    
    # Check if data was downloaded successfully
    btc_data_path = "data/yahoo/1d/BTC_USD.csv"
    if os.path.exists(btc_data_path):
        logger.info(f"Bitcoin data downloaded successfully to {btc_data_path}")
        return btc_data_path
    else:
        logger.error(f"Failed to download Bitcoin data to {btc_data_path}")
        return None


def verify_data(data_path):
    """Verify that the downloaded data is valid."""
    logger.info(f"Verifying data at {data_path}...")
    
    try:
        # Load the data
        data = pd.read_csv(data_path)
        
        # Check if the data has the required columns
        required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for column in required_columns:
            if column not in data.columns:
                logger.error(f"Missing required column: {column}")
                return False
        
        # Check if the data has enough rows
        if len(data) < 100:
            logger.warning(f"Data has only {len(data)} rows, which may not be enough for reliable backtesting")
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Data contains {missing_values} missing values")
        
        logger.info(f"Data verification completed. Data has {len(data)} rows and {len(data.columns)} columns.")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying data: {e}")
        return False


def run_backtest(data_path):
    """Run a simple backtest to verify that the backtesting system is working correctly."""
    logger.info("Running backtest...")
    
    try:
        # Create data source
        data_source = CSVDataSource(data_path)
        
        # Create strategy
        strategy = MovingAverageCrossover(short_window=20, long_window=50)
        
        # Create portfolio manager (use the correct class based on what was imported)
        if 'SimplePortfolioManager' in globals():
            portfolio_manager = SimplePortfolioManager(initial_capital=10000, position_size=0.1)
        else:
            portfolio_manager = BasicPortfolioManager(initial_capital=10000)
        
        # Create and run backtester
        backtester = Backtester(data_source, strategy, portfolio_manager)
        results = backtester.run()
        
        # Print performance metrics
        logger.info("Backtest completed successfully.")
        logger.info(f"Total Return: {results.total_return:.2%}")
        logger.info(f"Annualized Return: {results.annualized_return:.2%}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Save equity curve
        results.plot_equity_curve()
        plt.title("Bitcoin Strategy - Equity Curve")
        plt.savefig("results/btc_test_equity_curve.png")
        plt.close()
        
        logger.info("Equity curve saved to results/btc_test_equity_curve.png")
        
        return True
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return False


def main():
    """Main function."""
    logger.info("Starting download and test process...")
    
    # Download data
    data_path = download_data()
    if not data_path:
        logger.error("Failed to download data. Exiting.")
        return False
    
    # Verify data
    if not verify_data(data_path):
        logger.error("Data verification failed. Exiting.")
        return False
    
    # Run backtest
    if not run_backtest(data_path):
        logger.error("Backtest failed. Exiting.")
        return False
    
    logger.info("Download and test process completed successfully.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 