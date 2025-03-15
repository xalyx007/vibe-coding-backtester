# Data Module API Reference

The Data Module is responsible for loading and preprocessing market data from various sources. It provides a standardized interface for accessing market data, regardless of the source.

## DataSource

::: backtester.data.DataSource
    options:
      show_root_heading: true
      show_source: true

## CSVDataSource

::: backtester.data.CSVDataSource
    options:
      show_root_heading: true
      show_source: true

## APIDataSource

::: backtester.data.APIDataSource
    options:
      show_root_heading: true
      show_source: true

## Data Utilities

### Preprocessing Functions

::: backtester.data.preprocessing
    options:
      show_root_heading: true
      show_source: true

### Resampling Functions

::: backtester.data.resampling
    options:
      show_root_heading: true
      show_source: true

## Examples

### Loading Data from CSV

```python
from backtester.data import CSVDataSource

# Load data from a CSV file
data_source = CSVDataSource(
    file_path="data/sample_data.csv",
    date_column="date",
    open_column="open",
    high_column="high",
    low_column="low",
    close_column="close",
    volume_column="volume",
    date_format="%Y-%m-%d"
)

# Get the data as a pandas DataFrame
data = data_source.get_data()

# Get a specific range of data
start_date = "2020-01-01"
end_date = "2020-12-31"
filtered_data = data_source.get_data(start_date=start_date, end_date=end_date)
```

### Loading Data from an API

```python
from backtester.data import APIDataSource

# Load data from an exchange API
data_source = APIDataSource(
    exchange="binance",
    symbol="BTC/USDT",
    timeframe="1d",
    start_date="2020-01-01",
    end_date="2020-12-31"
)

# Get the data as a pandas DataFrame
data = data_source.get_data()
```

### Preprocessing Data

```python
from backtester.data import CSVDataSource
from backtester.data.preprocessing import add_indicators

# Load data
data_source = CSVDataSource("data/sample_data.csv")
data = data_source.get_data()

# Add technical indicators
indicators = {
    "sma_10": {"func": "sma", "window": 10},
    "sma_30": {"func": "sma", "window": 30},
    "rsi": {"func": "rsi", "window": 14},
    "macd": {"func": "macd", "fast": 12, "slow": 26, "signal": 9}
}
data_with_indicators = add_indicators(data, indicators)
```

## Best Practices

<div class="best-practice">
<strong>Data Handling Best Practices:</strong>
<ul>
<li>Always check for missing values in your data and handle them appropriately</li>
<li>Ensure that your data is sorted by date in ascending order</li>
<li>Use the appropriate date format for your data source</li>
<li>Consider resampling your data to a lower frequency for faster backtesting</li>
<li>Add relevant technical indicators to your data before passing it to the strategy</li>
</ul>
</div>

## Custom Data Sources

You can create custom data sources by inheriting from the `DataSource` class and implementing the required methods:

```python
from backtester.data import DataSource
import pandas as pd

class CustomDataSource(DataSource):
    def __init__(self, custom_param):
        super().__init__()
        self.custom_param = custom_param
        
    def load_data(self):
        # Implement your custom data loading logic here
        # ...
        
        # Return a pandas DataFrame with the required columns
        return pd.DataFrame({
            "date": [...],
            "open": [...],
            "high": [...],
            "low": [...],
            "close": [...],
            "volume": [...]
        })
        
    def get_data(self, start_date=None, end_date=None):
        # Get the data and filter by date range
        data = self.load_data()
        
        if start_date:
            data = data[data["date"] >= start_date]
        if end_date:
            data = data[data["date"] <= end_date]
            
        return data
``` 