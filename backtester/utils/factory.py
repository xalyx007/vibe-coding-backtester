"""
Factory functions for creating components from configuration.
"""

from typing import Dict, Any, List, Optional
import importlib


def create_data_source(config: Dict[str, Any]):
    """
    Create a data source from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataSource instance
    """
    data_source_type = config["type"]
    params = config.get("params", {})
    
    # Import the appropriate class
    module_name = f"backtester.inputs"
    try:
        module = importlib.import_module(module_name)
        data_source_class = getattr(module, data_source_type)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Invalid data source type: {data_source_type}") from e
    
    # Create the data source
    return data_source_class(**params)


def create_strategy(config: Dict[str, Any]):
    """
    Create a strategy from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Strategy instance
    """
    strategy_type = config["type"]
    params = config.get("params", {})
    
    # Handle special case for StrategyEnsemble
    if strategy_type == "StrategyEnsemble":
        return _create_strategy_ensemble(params)
    
    # Import the appropriate class
    module_name = f"backtester.strategy"
    try:
        module = importlib.import_module(module_name)
        strategy_class = getattr(module, strategy_type)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Invalid strategy type: {strategy_type}") from e
    
    # Create the strategy
    return strategy_class(**params)


def _create_strategy_ensemble(params: Dict[str, Any]):
    """
    Create a strategy ensemble from configuration.
    
    Args:
        params: Configuration parameters
        
    Returns:
        StrategyEnsemble instance
    """
    from backtester.strategy import StrategyEnsemble
    
    # Create individual strategies
    strategies = []
    for strategy_config in params["strategies"]:
        strategy = create_strategy(strategy_config)
        strategies.append(strategy)
    
    # Create the ensemble
    return StrategyEnsemble(
        strategies=strategies,
        weights=params.get("weights"),
        combination_method=params.get("combination_method", "weighted")
    )


def create_portfolio_manager(config: Dict[str, Any]):
    """
    Create a portfolio manager from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PortfolioManager instance
    """
    portfolio_manager_type = config["type"]
    params = config.get("params", {})
    
    # Handle position sizer if present
    if "position_sizer" in params:
        position_sizer_config = params.pop("position_sizer")
        position_sizer = create_position_sizer(position_sizer_config)
        params["position_sizer"] = position_sizer
    
    # Import the appropriate class
    module_name = f"backtester.portfolio"
    try:
        module = importlib.import_module(module_name)
        portfolio_manager_class = getattr(module, portfolio_manager_type)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Invalid portfolio manager type: {portfolio_manager_type}") from e
    
    # Create the portfolio manager
    return portfolio_manager_class(**params)


def create_position_sizer(config: Dict[str, Any]):
    """
    Create a position sizer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Position sizer instance
    """
    position_sizer_type = config["type"]
    params = config.get("params", {})
    
    # Import the appropriate class
    module_name = f"backtester.portfolio"
    try:
        module = importlib.import_module(module_name)
        position_sizer_class = getattr(module, position_sizer_type)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Invalid position sizer type: {position_sizer_type}") from e
    
    # Create the position sizer
    return position_sizer_class(**params) 