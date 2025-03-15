#!/usr/bin/env python
"""
Migration script to help users migrate to the new project structure.

This script updates import statements in Python files to reflect the new
project structure. It should be run after the new structure has been set up.
"""

import os
import re
import sys
from pathlib import Path


# Define import mappings
IMPORT_MAPPINGS = {
    # Core module
    r'from backtester\.backtest import Backtester': 'from backtester.core import Backtester',
    r'from backtester\.backtest\.backtester import Backtester': 'from backtester.core import Backtester',
    r'from backtester\.backtest import BacktestResults': 'from backtester.core import BacktestResults',
    r'from backtester\.backtest\.results import BacktestResults': 'from backtester.core import BacktestResults',
    
    # Data module (renamed from inputs)
    r'from backtester\.inputs import DataSource': 'from backtester.data import DataSource',
    r'from backtester\.inputs\.base import DataSource': 'from backtester.data.base import DataSource',
    r'from backtester\.inputs import CSVDataSource': 'from backtester.data import CSVDataSource',
    r'from backtester\.inputs\.csv_source import CSVDataSource': 'from backtester.data.csv_source import CSVDataSource',
    r'from backtester\.inputs import ExcelDataSource': 'from backtester.data import ExcelDataSource',
    r'from backtester\.inputs\.excel_source import ExcelDataSource': 'from backtester.data.excel_source import ExcelDataSource',
    r'from backtester\.inputs import ExchangeDataSource': 'from backtester.data import ExchangeDataSource',
    r'from backtester\.inputs\.exchange_source import ExchangeDataSource': 'from backtester.data.exchange_source import ExchangeDataSource',
    
    # CLI module
    r'from backtester import cli': 'from backtester.cli import main',
    r'from backtester\.cli import ': 'from backtester.cli.main import ',
    
    # Validation module
    r'from backtester\.validation_master import ': 'from backtester.validation import ',
}


def update_imports(file_path):
    """
    Update import statements in a Python file.
    
    Args:
        file_path: Path to the Python file
    
    Returns:
        bool: True if the file was modified, False otherwise
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Apply import mappings
    for old_import, new_import in IMPORT_MAPPINGS.items():
        content = re.sub(old_import, new_import, content)
    
    # Write the updated content back to the file
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    
    return False


def process_directory(directory):
    """
    Process all Python files in a directory and its subdirectories.
    
    Args:
        directory: Path to the directory
    
    Returns:
        tuple: (total_files, modified_files)
    """
    total_files = 0
    modified_files = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                total_files += 1
                
                if update_imports(file_path):
                    modified_files += 1
                    print(f"Updated imports in {file_path}")
    
    return total_files, modified_files


def main():
    """Main entry point for the script."""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = '.'
    
    print(f"Migrating Python files in {directory} to the new project structure...")
    
    total_files, modified_files = process_directory(directory)
    
    print(f"Migration complete. Processed {total_files} files, modified {modified_files} files.")
    
    if modified_files > 0:
        print("\nPlease review the changes and run your tests to ensure everything works correctly.")
        print("You may need to manually update some imports that were not caught by this script.")


if __name__ == '__main__':
    main() 