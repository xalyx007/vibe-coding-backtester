# Migration Guide

This guide helps you migrate your code to the new project structure.

## Directory Structure Changes

The project structure has been reorganized to improve maintainability and scalability:

### Main Changes

1. **Core Module**: Created a dedicated `core/` module for central components:
   - `engine.py`: Main backtester engine (moved from backtester.py)
   - `results.py`: Results handling (moved from backtest/results.py)
   - `config.py`: Centralized configuration management

2. **Modular CLI**: Split the CLI into multiple files:
   - `main.py`: Entry point and argument parsing
   - `backtest_commands.py`: Backtest command implementation
   - `validation_commands.py`: Validation command implementation
   - `analysis_commands.py`: Analysis command implementation

3. **Data Module**: Renamed `inputs/` to `data/` for clarity and added a `processors/` submodule.

4. **Validation Module**: Consolidated validation code into a dedicated `validation/` module.

5. **Analysis Reporting**: Added a dedicated `reporting/` submodule for report generation.

6. **Consolidated Output**: Created an `output/` directory with subdirectories:
   - `logs/`: Log files
   - `results/`: Results from backtests and validations
   - `reports/`: Generated reports

7. **Development Tools**: Created a `tools/` directory for development-related files:
   - `scripts/`: Utility scripts
   - `docker/`: Docker-related files
   - `ci/`: CI/CD configuration
   - `config/`: Copies of configuration files for reference
   - `package/`: Copies of package-related files for reference

8. **Documentation**: Moved documentation to a structured `docs/` directory:
   - `api/`: API documentation
   - `user_guide/`: User guide
   - `examples/`: Example documentation
   - `CHANGELOG.md`: Project changelog
   - `CODE_OF_CONDUCT.md`: Code of conduct
   - `SECURITY.md`: Security policy
   - `MIGRATION_GUIDE.md`: This guide

### Note on Configuration and Package Files

Configuration files (like `.flake8`, `pytest.ini`, etc.) and package files (like `setup.py`, `pyproject.toml`, etc.) remain in the root directory for functionality but are copied to `tools/config/` and `tools/package/` respectively for organization and reference.

## Updating Your Code

### Import Statements

Import statements need to be updated to reflect the new structure. You can use the provided script to automatically update most import statements:

```bash
python tools/scripts/update_imports.py path/to/your/code
```

Common import changes:

```python
# Old imports
from backtester.backtest import Backtester
from backtester.backtest.results import BacktestResults
from backtester.inputs import CSVDataSource
from backtester.validation_master import run_cross_validation

# New imports
from backtester.core import Backtester
from backtester.core import BacktestResults
from backtester.data import CSVDataSource
from backtester.validation import run_cross_validation
```

### File Paths

If your code references files in the old directory structure, update the paths:

```python
# Old paths
log_file = "validation_master.log"
results_file = "results/backtest_results.json"

# New paths
log_file = "output/logs/validation_master.log"
results_file = "output/results/backtest/backtest_results.json"
```

### Configuration Files

If you have configuration files that reference the old directory structure, update them:

```yaml
# Old configuration
output_dir: results
log_file: validation_master.log

# New configuration
output_dir: output/results/backtest
log_file: output/logs/validation_master.log
```

## Running Tests

After updating your code, run the tests to ensure everything works correctly:

```bash
pytest
```

## Getting Help

If you encounter any issues with the migration, please open an issue on GitHub or contact the maintainers. 