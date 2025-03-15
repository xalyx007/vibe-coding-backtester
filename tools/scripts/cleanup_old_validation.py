#!/usr/bin/env python
"""
Cleanup script for old validation files.

This script moves the old validation scripts and directories to a backup directory.
"""

import os
import shutil
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../output/logs/cleanup_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cleanup_validation")

# Create backup directory
backup_dir = f"../../backup/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(backup_dir, exist_ok=True)

# List of old validation scripts and directories to move
old_validation_files = [
    "validate_backtester.py",
    "cross_validate_with_backtrader.py",
    "monte_carlo_validation.py",
    "run_all_validations.py",
    "monte_carlo_validation.log",
    "validation_results.log",
    "validation_master.log",
    "VALIDATION_README.md"
]

old_validation_dirs = [
    "monte_carlo_results",
    "cross_validation_results",
    "validation_results",
    "validation_master"
]

def move_file(file_path, backup_path):
    """Move a file to the backup directory."""
    try:
        if os.path.exists(file_path):
            shutil.move(file_path, backup_path)
            logger.info(f"Moved {file_path} to {backup_path}")
        else:
            logger.warning(f"File {file_path} does not exist, skipping")
    except Exception as e:
        logger.error(f"Error moving {file_path}: {str(e)}")

def move_directory(dir_path, backup_path):
    """Move a directory to the backup directory."""
    try:
        if os.path.exists(dir_path):
            shutil.move(dir_path, backup_path)
            logger.info(f"Moved {dir_path} to {backup_path}")
        else:
            logger.warning(f"Directory {dir_path} does not exist, skipping")
    except Exception as e:
        logger.error(f"Error moving {dir_path}: {str(e)}")

def main():
    """Move old validation scripts and directories to the backup directory."""
    logger.info(f"Starting cleanup of old validation files and directories")
    logger.info(f"Backup directory: {backup_dir}")
    
    # Move files
    for file_name in old_validation_files:
        file_path = os.path.join(".", file_name)
        backup_path = os.path.join(backup_dir, file_name)
        move_file(file_path, backup_path)
    
    # Move directories
    for dir_name in old_validation_dirs:
        dir_path = os.path.join(".", dir_name)
        backup_path = os.path.join(backup_dir, dir_name)
        move_directory(dir_path, backup_path)
    
    logger.info(f"Cleanup completed")
    logger.info(f"Old validation files and directories have been moved to {backup_dir}")
    logger.info(f"Please verify that the new validation module is working correctly before deleting the backup")


if __name__ == "__main__":
    main() 