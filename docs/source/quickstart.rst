Quickstart
==========

This guide will help you get started with the Modular Backtesting System by walking through a simple example.

Basic Example
------------

Let's create a simple backtest using a moving average crossover strategy.

1. First, import the necessary modules:

   .. code-block:: python

       from backtester.data import CSVDataSource
       from backtester.strategy import MovingAverageCrossover
       from backtester.portfolio import SimplePortfolioManager
       from backtester.backtester import Backtester
       from backtester.utils.metrics import calculate_metrics
       from backtester.utils.visualization import plot_equity_curve, plot_drawdown

2. Load the market data:

   .. code-block:: python

       data_source = CSVDataSource(
           file_path="data/AAPL.csv",
           date_format="%Y-%m-%d",
           timestamp_column="Date",
           open_column="Open",
           high_column="High",
           low_column="Low",
           close_column="Close",
           volume_column="Volume"
       )
       data = data_source.load_data()

3. Create a strategy:

   .. code-block:: python

       strategy = MovingAverageCrossover(
           short_window=10,
           long_window=30
       )

4. Create a portfolio manager:

   .. code-block:: python

       portfolio_manager = SimplePortfolioManager(
           initial_capital=10000,
           position_size=0.1
       )

5. Create and run the backtest:

   .. code-block:: python

       backtester = Backtester(
           data=data,
           strategy=strategy,
           portfolio_manager=portfolio_manager
       )
       results = backtester.run()

6. Analyze the results:

   .. code-block:: python

       # Calculate performance metrics
       metrics = calculate_metrics(results)
       print(metrics)

       # Plot equity curve
       plot_equity_curve(results)

       # Plot drawdown
       plot_drawdown(results)

Using the Command Line Interface
-------------------------------

The Modular Backtesting System also provides a command-line interface for running backtests:

.. code-block:: bash

    backtester run --data-source data/AAPL.csv --strategy moving_average_crossover --short-window 10 --long-window 30 --initial-capital 10000 --position-size 0.1 --output results.json

Using Configuration Files
-----------------------

For more complex backtests, you can use configuration files:

1. Create a configuration file (config.yaml):

   .. code-block:: yaml

       data_source:
         type: csv
         params:
           file_path: data/AAPL.csv
           date_format: "%Y-%m-%d"
           timestamp_column: Date
           open_column: Open
           high_column: High
           low_column: Low
           close_column: Close
           volume_column: Volume

       strategy:
         type: moving_average_crossover
         params:
           short_window: 10
           long_window: 30

       portfolio_manager:
         type: simple
         params:
           initial_capital: 10000
           position_size: 0.1

       backtest:
         start_date: 2020-01-01
         end_date: 2021-01-01

       output:
         metrics: true
         plots: true
         save_results: true
         results_file: results.json

2. Run the backtest using the configuration file:

   .. code-block:: bash

       backtester run --config config.yaml

Next Steps
---------

Now that you've run your first backtest, you can:

* Explore different strategies in the strategies section
* Learn how to create your own strategy in the custom_strategies section
* Understand the architecture of the system in the architecture section
* Explore the API reference in the api section 