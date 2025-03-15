# Modular Backtesting System Tests

This directory contains tests for the Modular Backtesting System.

## Test Structure

The tests are organized into two main categories:

- **Unit Tests**: Tests for individual components in isolation
- **Integration Tests**: Tests for multiple components working together

## Running Tests

You can run the tests using pytest:

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit

# Run integration tests only
pytest tests/integration

# Run a specific test file
pytest tests/unit/test_moving_average.py

# Run a specific test
pytest tests/unit/test_moving_average.py::TestMovingAverageCrossover::test_generate_signals
```

## Test Coverage

You can generate a test coverage report using pytest-cov:

```bash
# Generate coverage report
pytest --cov=backtester

# Generate HTML coverage report
pytest --cov=backtester --cov-report=html
```

## Writing Tests

When writing tests, follow these guidelines:

1. Use pytest for all tests
2. Use descriptive test names that indicate what is being tested
3. Use fixtures for common setup
4. Use assertions to verify expected behavior
5. Use mocks when appropriate to isolate components

### Example Test

```python
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtester.strategy import MovingAverageCrossover
from backtester.utils.constants import SignalType


class TestMovingAverageCrossover(unittest.TestCase):
    def setUp(self):
        # Create sample data
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        prices = np.zeros(100)
        prices[:50] = np.linspace(100, 50, 50)  # Downtrend
        prices[50:] = np.linspace(50, 150, 50)  # Uptrend
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        self.data.set_index('timestamp', inplace=True)
        
        self.strategy = MovingAverageCrossover(
            short_window=10,
            long_window=30
        )
    
    def test_generate_signals(self):
        signals = self.strategy.generate_signals(self.data)
        
        # Check that signals DataFrame has the correct shape
        self.assertEqual(signals.shape[0], self.data.shape[0])
        self.assertIn('signal', signals.columns)
        
        # Check that signals are of the correct type
        for signal in signals['signal']:
            self.assertIsInstance(signal, SignalType)
```

## Continuous Integration

The tests are run automatically on each pull request and push to the main branch using GitHub Actions. See the [GitHub Actions workflow file](../.github/workflows/tests.yml) for details. 