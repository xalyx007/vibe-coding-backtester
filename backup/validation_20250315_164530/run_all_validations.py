#!/usr/bin/env python
"""
Comprehensive Backtester Validation Script

This script runs all validation processes and generates a comprehensive report.
"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../output/logs/validation_master.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("validation_master")

# Create results directory
os.makedirs("../../output/results/validation", exist_ok=True)


def run_command(command, description):
    """Run a command and log the output."""
    logger.info(f"Running {description}...")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise an exception on non-zero exit code
        )
        end_time = time.time()
        
        # Combine stdout and stderr for analysis
        output = result.stdout + result.stderr
        
        # For basic validation, check if all tests passed
        if description == "Basic validation tests":
            # Look for the line that indicates how many tests passed
            match = re.search(r"Validation complete: (\d+)/(\d+) tests passed", output)
            if match:
                passed = int(match.group(1))
                total = int(match.group(2))
                success = passed == total
                if success:
                    logger.info(f"{description} completed successfully with {passed}/{total} tests passed")
                    return True, output
                else:
                    logger.error(f"{description} failed with {passed}/{total} tests passed")
                    return False, output
        
        # For other commands, use the exit code
        if result.returncode == 0:
            logger.info(f"{description} completed successfully in {end_time - start_time:.2f} seconds")
            return True, output
        else:
            logger.error(f"{description} failed with error code {result.returncode}")
            logger.error(f"Error output: {output}")
            return False, output
    except Exception as e:
        logger.error(f"{description} failed with exception: {str(e)}")
        return False, str(e)


def run_all_validations():
    """Run all validation processes."""
    logger.info("Starting comprehensive backtester validation")
    
    validation_results = {}
    
    # 1. Run basic validation tests
    basic_success, basic_output = run_command(
        ["python", "validate_backtester.py"],
        "Basic validation tests"
    )
    validation_results["basic_validation"] = {
        "success": basic_success,
        "output": basic_output
    }
    
    # 2. Run cross-validation with Backtrader
    cross_success, cross_output = run_command(
        ["python", "cross_validate_with_backtrader.py"],
        "Cross-validation with Backtrader"
    )
    validation_results["cross_validation"] = {
        "success": cross_success,
        "output": cross_output
    }
    
    # 3. Run Monte Carlo validation
    monte_carlo_success, monte_carlo_output = run_command(
        ["python", "monte_carlo_validation.py"],
        "Monte Carlo validation"
    )
    validation_results["monte_carlo_validation"] = {
        "success": monte_carlo_success,
        "output": monte_carlo_output
    }
    
    # Generate comprehensive report
    generate_comprehensive_report(validation_results)
    
    # Overall success
    overall_success = all([
        validation_results["basic_validation"]["success"],
        validation_results["cross_validation"]["success"],
        validation_results["monte_carlo_validation"]["success"]
    ])
    
    if overall_success:
        logger.info("All validation processes completed successfully!")
    else:
        logger.warning("Some validation processes failed. See the report for details.")
    
    return overall_success


def generate_comprehensive_report(validation_results):
    """Generate a comprehensive validation report."""
    logger.info("Generating comprehensive validation report")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open("../../output/results/validation/comprehensive_report.md", "w") as f:
        f.write(f"# Comprehensive Backtester Validation Report\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        # Overall status
        overall_success = all([
            validation_results["basic_validation"]["success"],
            validation_results["cross_validation"]["success"],
            validation_results["monte_carlo_validation"]["success"]
        ])
        
        f.write(f"## Overall Status: {'PASSED' if overall_success else 'FAILED'}\n\n")
        
        # Summary table
        f.write("| Validation Process | Status | Details |\n")
        f.write("|-------------------|--------|--------|\n")
        
        for process, result in validation_results.items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            details_link = f"[View Details](#{process.replace('_', '-')})"
            f.write(f"| {process.replace('_', ' ').title()} | {status} | {details_link} |\n")
        
        f.write("\n")
        
        # Detailed results
        for process, result in validation_results.items():
            f.write(f"## {process.replace('_', ' ').title()} {'{#' + process.replace('_', '-') + '}'}\n\n")
            f.write(f"Status: {'PASSED' if result['success'] else 'FAILED'}\n\n")
            
            # Include links to detailed reports
            if process == "basic_validation":
                f.write("### Detailed Reports\n\n")
                f.write("- [Validation Summary](summary.txt)\n")
                f.write("- [Buy and Hold Equity Curve](buy_and_hold_equity.png)\n")
                f.write("- [Perfect Foresight Equity Curve](perfect_foresight_equity.png)\n")
                f.write("- [Transaction Costs Impact](transaction_costs.png)\n")
                f.write("- [Slippage Impact](slippage.png)\n")
                f.write("- [Look-Ahead Bias Test](look_ahead_bias.png)\n")
            elif process == "cross_validation":
                f.write("### Detailed Reports\n\n")
                f.write("- [Cross-Validation Summary](../cross_validation/summary.txt)\n")
                f.write("- [Our Backtester Equity Curve](../cross_validation/our_backtester_equity.png)\n")
                f.write("- [Backtrader Equity Curve](../cross_validation/backtrader_equity.png)\n")
                f.write("- [Equity Curve Comparison](../cross_validation/equity_comparison.png)\n")
            elif process == "monte_carlo_validation":
                f.write("### Detailed Reports\n\n")
                f.write("- [Monte Carlo Summary](../monte_carlo/summary.txt)\n")
                f.write("- [Moving Average Returns Distribution](../monte_carlo/MovingAverageCrossover_returns_histogram.png)\n")
                f.write("- [RSI Returns Distribution](../monte_carlo/RSIStrategy_returns_histogram.png)\n")
                f.write("- [Bollinger Bands Returns Distribution](../monte_carlo/BollingerBandsStrategy_returns_histogram.png)\n")
                f.write("- [Parameter Sensitivity Analysis](../monte_carlo/MovingAverageCrossover_parameter_sensitivity.csv)\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if overall_success:
            f.write("All validation tests have passed, indicating that the backtester is functioning correctly. Here are some recommendations for ongoing validation:\n\n")
            f.write("1. **Regular Revalidation**: Run this comprehensive validation process after any significant changes to the backtester code.\n")
            f.write("2. **Expand Test Coverage**: Consider adding more test cases, especially for edge cases and complex strategies.\n")
            f.write("3. **Real-World Validation**: Compare backtest results with real-world trading performance when possible.\n")
        else:
            f.write("Some validation tests have failed. Here are recommendations for addressing the issues:\n\n")
            
            if not validation_results["basic_validation"]["success"]:
                f.write("- **Basic Validation Issues**: Review the basic validation results to identify specific tests that failed. These often point to fundamental issues in the backtester implementation.\n")
            
            if not validation_results["cross_validation"]["success"]:
                f.write("- **Cross-Validation Issues**: The discrepancies between our backtester and Backtrader indicate potential issues in trade execution, position sizing, or performance calculation. Review the detailed comparison to identify specific areas of difference.\n")
            
            if not validation_results["monte_carlo_validation"]["success"]:
                f.write("- **Monte Carlo Validation Issues**: The Monte Carlo simulations may have revealed instability or sensitivity to small data variations. Review the distribution of results to identify potential robustness issues.\n")
        
        f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        if overall_success:
            f.write("The backtester has passed all validation tests and can be considered reliable for strategy development and evaluation. Continue to monitor performance and revalidate regularly to maintain confidence in the results.\n")
        else:
            f.write("The backtester has failed some validation tests and requires attention before it can be considered fully reliable. Address the issues identified in the detailed reports and rerun the validation process.\n")
    
    logger.info("Comprehensive report generated at output/results/validation/comprehensive_report.md")


if __name__ == "__main__":
    success = run_all_validations()
    sys.exit(0 if success else 1) 