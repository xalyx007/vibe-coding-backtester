# API Reference

This document provides a reference for the main APIs of the Modular Backtesting System.

## Inputs Module

### DataSource

```python
class DataSource(ABC):
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the data source.
        
        Args:
            event_bus: Optional event bus for emitting events
        """
        
    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load data from the source.
        
        Args:
            **kwargs: Source-specific parameters
            
        Returns:
            DataFrame containing the loaded data
        """
        
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            data: Raw data to preprocess
            
        Returns:
            Preprocessed data
        """
        
    def get_data(self, 
                timeframe: Optional[TimeFrame] = None, 
                start_date: Optional[str] = None, 
                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get data for the specified timeframe and date range.
        
        Args:
            timeframe: Optional timeframe to resample data to
            start_date: Optional start date for filtering data
            end_date: Optional end date for filtering data
            
        Returns:
            DataFrame containing the requested data
        """
```

### CSVDataSource

```python
class CSVDataSource(DataSource):
    def __init__(self, 
                file_path: str, 
                date_column: str = 'timestamp', 
                date_format: Optional[str] = None,
                event_bus: Optional[EventBus] = None):
        """
        Initialize the CSV data source.
        
        Args:
            file_path: Path to the CSV file
            date_column: Name of the date/timestamp column
            date_format: Format of the date/timestamp column
            event_bus: Optional event bus for emitting events
        """
        
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load data from the CSV file.
        
        Args:
            **kwargs: Additional parameters for pd.read_csv
            
        Returns:
            DataFrame containing the loaded data
        """
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            data: Raw data to preprocess
            
        Returns:
            Preprocessed data
        """
```

## Strategy Module

### Strategy

```python
class Strategy(ABC):
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the strategy.
        
        Args:
            event_bus: Optional event bus for emitting events
        """
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the input data.
        
        Args:
            data: Market data to analyze
            
        Returns:
            DataFrame with signals (BUY, SELL, HOLD) for each timestamp
        """
        
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the strategy parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
```

### MovingAverageCrossover

```python
class MovingAverageCrossover(Strategy):
    def __init__(self, 
                short_window: int, 
                long_window: int, 
                price_column: str = 'close',
                event_bus: Optional[EventBus] = None):
        """
        Initialize the moving average crossover strategy.
        
        Args:
            short_window: Window size for the short moving average
            long_window: Window size for the long moving average
            price_column: Column to use for price data
            event_bus: Optional event bus for emitting events
        """
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data: Market data to analyze
            
        Returns:
            DataFrame with signals (BUY, SELL, HOLD) for each timestamp
        """
```

## Portfolio Module

### PortfolioManager

```python
class PortfolioManager(ABC):
    def __init__(self, initial_capital: float, event_bus: Optional[EventBus] = None):
        """
        Initialize the portfolio manager.
        
        Args:
            initial_capital: Initial capital for the portfolio
            event_bus: Optional event bus for emitting events
        """
        
    @abstractmethod
    def update_position(self, timestamp, symbol: str, signal_type: SignalType, 
                       price: float, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update portfolio positions based on a signal.
        
        Args:
            timestamp: Timestamp of the signal
            symbol: Trading symbol (e.g., 'BTC-USD')
            signal_type: Type of signal (BUY, SELL, HOLD)
            price: Current price of the asset
            metadata: Additional information about the signal
            
        Returns:
            Dictionary with details of the executed trade
        """
        
    @abstractmethod
    def calculate_position_size(self, symbol: str, price: float, 
                               signal_type: SignalType) -> float:
        """
        Calculate the position size for a trade.
        
        Args:
            symbol: Trading symbol
            price: Current price of the asset
            signal_type: Type of signal
            
        Returns:
            Quantity to trade
        """
        
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate the current portfolio value.
        
        Args:
            prices: Dictionary mapping symbols to current prices
            
        Returns:
            Total portfolio value (cash + holdings)
        """
        
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get the portfolio history.
        
        Returns:
            DataFrame with portfolio history
        """
```

## Backtest Module

### Backtester

```python
class Backtester:
    def __init__(self, 
                data_source: DataSource, 
                strategy: Strategy, 
                portfolio_manager: PortfolioManager,
                event_bus: Optional[EventBus] = None,
                transaction_costs: float = 0.001,
                slippage: float = 0.0005):
        """
        Initialize the backtester.
        
        Args:
            data_source: Data source for market data
            strategy: Strategy for generating signals
            portfolio_manager: Portfolio manager for executing trades
            event_bus: Optional event bus for emitting events
            transaction_costs: Transaction costs as a fraction of trade value
            slippage: Slippage as a fraction of price
        """
        
    def run(self, 
           start_date: Optional[str] = None, 
           end_date: Optional[str] = None,
           symbol: str = "BTC-USD") -> BacktestResults:
        """
        Run the backtest.
        
        Args:
            start_date: Optional start date for the backtest
            end_date: Optional end date for the backtest
            symbol: Trading symbol
            
        Returns:
            BacktestResults object with the results of the backtest
        """
```

### BacktestResults

```python
class BacktestResults:
    def __init__(self, 
                portfolio_values: pd.DataFrame, 
                trades: pd.DataFrame,
                signals: pd.DataFrame,
                strategy_parameters: Dict[str, Any],
                initial_capital: float,
                symbol: str):
        """
        Initialize the backtest results.
        
        Args:
            portfolio_values: DataFrame with portfolio values over time
            trades: DataFrame with trade details
            signals: DataFrame with strategy signals
            strategy_parameters: Dictionary of strategy parameters
            initial_capital: Initial capital for the backtest
            symbol: Trading symbol
        """
        
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the backtest results.
        
        Returns:
            Dictionary with summary statistics
        """
        
    def plot_equity_curve(self, figsize=(12, 6)):
        """
        Plot the equity curve.
        
        Args:
            figsize: Figure size
        """
        
    def plot_drawdown(self, figsize=(12, 6)):
        """
        Plot the drawdown.
        
        Args:
            figsize: Figure size
        """
        
    def plot_trades(self, figsize=(12, 6)):
        """
        Plot the trades.
        
        Args:
            figsize: Figure size
        """
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the results to a dictionary.
        
        Returns:
            Dictionary representation of the results
        """
        
    def to_json(self, path: str):
        """
        Save the results to a JSON file.
        
        Args:
            path: Path to save the JSON file
        """
```

## Events Module

### EventBus

```python
class EventBus:
    def __init__(self, use_redis: bool = False, redis_url: Optional[str] = None):
        """
        Initialize the event bus.
        
        Args:
            use_redis: Whether to use Redis for event distribution
            redis_url: Redis URL if using Redis
        """
        
    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when the event occurs
        """
        
    def emit(self, event_type: str, data: Dict[str, Any]):
        """
        Emit an event.
        
        Args:
            event_type: Type of event to emit
            data: Data associated with the event
        """
        
    def start_listening(self, event_types: List[str] = None):
        """
        Start listening for events from Redis.
        
        Args:
            event_types: List of event types to listen for (None for all)
        """
        
    def stop(self):
        """Stop the event bus and clean up resources."""
```

### Event

```python
class Event:
    def __init__(self, 
                event_type: EventType, 
                data: Dict[str, Any],
                timestamp: Optional[datetime] = None):
        """
        Initialize the event.
        
        Args:
            event_type: Type of event
            data: Data associated with the event
            timestamp: Optional timestamp (defaults to current time)
        """
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        
    @classmethod
    def from_dict(cls, event_dict: Dict[str, Any]) -> 'Event':
        """
        Create an event from a dictionary.
        
        Args:
            event_dict: Dictionary representation of the event
            
        Returns:
            Event object
        """
```

## Analysis Module

### Metrics Functions

```python
def calculate_metrics(portfolio_values: pd.DataFrame, 
                     trades: pd.DataFrame, 
                     initial_capital: float) -> Dict[str, Any]:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        trades: DataFrame with trade details
        initial_capital: Initial capital for the backtest
        
    Returns:
        Dictionary with performance metrics
    """
    
def calculate_drawdowns(portfolio_values: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate drawdowns from portfolio values.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        
    Returns:
        DataFrame with drawdown information
    """
    
def calculate_monthly_returns(portfolio_values: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly returns from portfolio values.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        
    Returns:
        DataFrame with monthly returns
    """
```

### Visualization Functions

```python
def plot_equity_curve(portfolio_values: pd.DataFrame, 
                     figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Plot the equity curve from portfolio values.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    
def plot_drawdown(portfolio_values: pd.DataFrame, 
                 figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Plot the drawdown from portfolio values.
    
    Args:
        portfolio_values: DataFrame with portfolio values over time
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    
def plot_trades(trades: pd.DataFrame, 
               signals: pd.DataFrame, 
               symbol: str,
               figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Plot trades on top of price data.
    
    Args:
        trades: DataFrame with trade details
        signals: DataFrame with strategy signals and price data
        symbol: Trading symbol
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
``` 