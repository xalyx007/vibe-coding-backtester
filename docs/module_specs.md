# Module Specifications

This document provides detailed specifications for each module in the Modular Backtesting System.

## 1. Inputs Module

### Purpose
The Inputs Module is responsible for gathering and preprocessing market data from various sources and providing it to the Strategy Module.

### Components

#### DataSource (Abstract Base Class)
- **Interface**:
  - `load_data(**kwargs) -> pd.DataFrame`: Load data from the source
  - `preprocess_data(data: pd.DataFrame) -> pd.DataFrame`: Preprocess the loaded data
  - `get_data(timeframe=None, start_date=None, end_date=None) -> pd.DataFrame`: Get data for the specified timeframe and date range

#### CSVDataSource
- **Description**: Loads data from CSV files
- **Parameters**:
  - `file_path`: Path to the CSV file
  - `date_column`: Name of the date/timestamp column
  - `date_format`: Format of the date/timestamp column

#### ExcelDataSource
- **Description**: Loads data from Excel files
- **Parameters**:
  - `file_path`: Path to the Excel file
  - `sheet_name`: Name of the sheet to load
  - `date_column`: Name of the date/timestamp column
  - `date_format`: Format of the date/timestamp column

#### ExchangeDataSource
- **Description**: Loads data from cryptocurrency exchanges via CCXT
- **Parameters**:
  - `exchange_id`: ID of the exchange (e.g., 'binance', 'coinbase')
  - `symbol`: Trading symbol (e.g., 'BTC/USD')
  - `timeframe`: Timeframe for the data (e.g., '1m', '1h', '1d')
  - `api_key`: API key for the exchange (optional)
  - `api_secret`: API secret for the exchange (optional)

### Data Format
- **Required Fields**:
  - `timestamp`: Date and time of the candlestick
  - `open`: Opening price
  - `high`: Highest price
  - `low`: Lowest price
  - `close`: Closing price
  - `volume`: Trading volume
- **Optional Fields**:
  - Technical indicators
  - Order book data
  - External signals

## 2. Strategy Module

### Purpose
The Strategy Module is responsible for generating buy/sell signals based on input data, fully decoupled from backtesting logic.

### Components

#### Strategy (Abstract Base Class)
- **Interface**:
  - `generate_signals(data: pd.DataFrame) -> pd.DataFrame`: Generate trading signals based on the input data
  - `get_parameters() -> Dict[str, Any]`: Get the strategy parameters

#### MovingAverageCrossover
- **Description**: Generates signals based on moving average crossovers
- **Parameters**:
  - `short_window`: Window size for the short moving average
  - `long_window`: Window size for the long moving average
  - `price_column`: Column to use for price data (default: 'close')

#### RSIStrategy
- **Description**: Generates signals based on the Relative Strength Index (RSI)
- **Parameters**:
  - `period`: Period for RSI calculation
  - `overbought`: Threshold for overbought condition
  - `oversold`: Threshold for oversold condition
  - `price_column`: Column to use for price data (default: 'close')

#### BollingerBandsStrategy
- **Description**: Generates signals based on Bollinger Bands
- **Parameters**:
  - `window`: Window size for the moving average
  - `num_std`: Number of standard deviations for the bands
  - `price_column`: Column to use for price data (default: 'close')

#### MLStrategy
- **Description**: Generates signals based on machine learning models
- **Parameters**:
  - `model`: Trained machine learning model
  - `features`: List of feature columns to use
  - `threshold`: Threshold for signal generation

#### StrategyEnsemble
- **Description**: Combines signals from multiple strategies
- **Parameters**:
  - `strategies`: List of strategy instances
  - `weights`: List of weights for each strategy
  - `combination_method`: Method for combining signals ('vote', 'weighted', 'meta')

### Signal Format
- **Required Fields**:
  - `timestamp`: Date and time of the signal
  - `signal`: Signal type (BUY, SELL, HOLD)
- **Optional Fields**:
  - `confidence`: Confidence level of the signal
  - `target_price`: Target price for the trade
  - `stop_loss`: Stop loss price for the trade

## 3. Portfolio Manager Module

### Purpose
The Portfolio Manager Module is responsible for managing positions and trades, decoupled from strategy and backtesting logic.

### Components

#### PortfolioManager (Abstract Base Class)
- **Interface**:
  - `update_position(timestamp, symbol, signal_type, price, metadata=None) -> Dict[str, Any]`: Update portfolio positions based on a signal
  - `calculate_position_size(symbol, price, signal_type) -> float`: Calculate the position size for a trade
  - `get_portfolio_value(prices) -> float`: Calculate the current portfolio value
  - `get_portfolio_history() -> pd.DataFrame`: Get the portfolio history

#### BasicPortfolioManager
- **Description**: Basic implementation of portfolio management
- **Parameters**:
  - `initial_capital`: Initial capital for the portfolio
  - `position_sizer`: Position sizer instance (optional)

#### FixedAmountSizer
- **Description**: Calculates position size based on a fixed amount
- **Parameters**:
  - `amount`: Fixed amount to trade

#### PercentageSizer
- **Description**: Calculates position size based on a percentage of available capital
- **Parameters**:
  - `percentage`: Percentage of available capital to trade

### Trade Format
- **Required Fields**:
  - `timestamp`: Date and time of the trade
  - `symbol`: Trading symbol
  - `type`: Trade type (BUY, SELL)
  - `price`: Execution price
  - `quantity`: Quantity traded
  - `value`: Value of the trade
- **Optional Fields**:
  - `fees`: Trading fees
  - `slippage`: Price slippage
  - `profit`: Profit/loss from the trade

## 4. Backtesting Module

### Purpose
The Backtesting Module is responsible for simulating trading based on signals from the Strategy Module and portfolio updates from the Portfolio Manager.

### Components

#### Backtester
- **Description**: Main class for backtesting trading strategies
- **Parameters**:
  - `data_source`: Data source for market data
  - `strategy`: Strategy for generating signals
  - `portfolio_manager`: Portfolio manager for executing trades
  - `transaction_costs`: Transaction costs as a fraction of trade value
  - `slippage`: Slippage as a fraction of price

#### BacktestResults
- **Description**: Class for storing and analyzing backtest results
- **Methods**:
  - `summary() -> Dict[str, Any]`: Get a summary of the backtest results
  - `plot_equity_curve()`: Plot the equity curve
  - `plot_drawdown()`: Plot the drawdown
  - `plot_trades()`: Plot the trades
  - `to_dict() -> Dict[str, Any]`: Convert the results to a dictionary
  - `to_json(path)`: Save the results to a JSON file

### Simulation Parameters
- **Transaction Costs**: Fixed or percentage-based costs for each trade
- **Slippage**: Difference between expected and actual execution price
- **Time Granularity**: Timeframe for the simulation (e.g., minute, hourly, daily)

## 5. Events Module

### Purpose
The Events Module facilitates event-based communication between the backend and a separate frontend, as well as between internal modules.

### Components

#### EventBus
- **Description**: Central hub for event distribution
- **Methods**:
  - `subscribe(event_type, callback)`: Subscribe to an event type
  - `emit(event_type, data)`: Emit an event
  - `start_listening(event_types=None)`: Start listening for events from Redis
  - `stop()`: Stop the event bus and clean up resources

#### EventType
- **Description**: Enum of event types
- **Values**:
  - `DATA_LOADED`: Data has been loaded
  - `SIGNAL_GENERATED`: A signal has been generated
  - `TRADE_EXECUTED`: A trade has been executed
  - `BACKTEST_STARTED`: A backtest has started
  - `BACKTEST_COMPLETED`: A backtest has completed
  - `PORTFOLIO_UPDATED`: The portfolio has been updated
  - `ERROR`: An error has occurred
  - `CONFIG_UPDATED`: The configuration has been updated

#### Event
- **Description**: Class representing an event
- **Fields**:
  - `event_type`: Type of event
  - `data`: Data associated with the event
  - `timestamp`: Timestamp of the event

## 6. Analysis Module

### Purpose
The Analysis Module is responsible for analyzing backtesting results and generating performance metrics and visualizations.

### Components

#### Metrics Functions
- `calculate_metrics(portfolio_values, trades, initial_capital) -> Dict[str, Any]`: Calculate performance metrics
- `calculate_drawdowns(portfolio_values) -> pd.DataFrame`: Calculate drawdowns
- `calculate_monthly_returns(portfolio_values) -> pd.DataFrame`: Calculate monthly returns

#### Visualization Functions
- `plot_equity_curve(portfolio_values, figsize=(12, 6)) -> Figure`: Plot the equity curve
- `plot_drawdown(portfolio_values, figsize=(12, 6)) -> Figure`: Plot the drawdown
- `plot_trades(trades, signals, symbol, figsize=(12, 6)) -> Figure`: Plot trades on top of price data
- `plot_monthly_returns(portfolio_values, figsize=(12, 6)) -> Figure`: Plot monthly returns as a heatmap

### Performance Metrics
- **Return Metrics**:
  - Total return
  - Annualized return
- **Risk Metrics**:
  - Volatility
  - Sharpe ratio
  - Maximum drawdown
- **Trade Metrics**:
  - Number of trades
  - Win rate
  - Profit factor
  - Average profit per trade 