"""
Strategy Ensemble for the Backtesting System.

This module implements an ensemble of multiple strategies.
"""

import pandas as pd
import numpy as np
from backtester.strategy.base import Strategy

class StrategyEnsemble(Strategy):
    """
    A strategy that combines signals from multiple strategies.
    
    Attributes:
        strategies (list): The list of strategies in the ensemble.
        weights (list): The weights for each strategy.
        threshold (float): The threshold for generating a signal.
    """
    
    def __init__(self, strategies=None, weights=None, threshold=0.5):
        """
        Initialize the strategy ensemble.
        
        Args:
            strategies (list): The list of strategies in the ensemble.
            weights (list): The weights for each strategy.
            threshold (float): The threshold for generating a signal.
        """
        super().__init__()
        self.strategies = strategies or []
        
        # If weights are not provided, use equal weights
        if weights is None:
            self.weights = [1.0 / len(self.strategies)] * len(self.strategies) if self.strategies else []
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
            
        self.threshold = threshold
        
    def add_strategy(self, strategy, weight=1.0):
        """
        Add a strategy to the ensemble.
        
        Args:
            strategy (Strategy): The strategy to add.
            weight (float): The weight for the strategy.
        """
        self.strategies.append(strategy)
        
        # Recalculate weights
        total_weight = sum(self.weights) + weight
        self.weights = [w / total_weight for w in self.weights]
        self.weights.append(weight / total_weight)
        
    def generate_signals(self, data):
        """
        Generate trading signals based on the ensemble of strategies.
        
        Args:
            data (pd.DataFrame): The market data.
            
        Returns:
            pd.DataFrame: The input data with additional signal columns.
        """
        if not self.strategies:
            return data.copy()
            
        # Generate signals for each strategy
        all_signals = []
        for i, strategy in enumerate(self.strategies):
            signals = strategy.generate_signals(data)
            all_signals.append(signals['signal'] * self.weights[i])
            
        # Make a copy of the data to avoid modifying the original
        ensemble_signals = data.copy()
        
        # Combine signals
        ensemble_signals['signal'] = sum(all_signals)
        
        # Apply threshold
        ensemble_signals['signal'] = np.where(ensemble_signals['signal'] > self.threshold, 1.0, 0.0)
        ensemble_signals['signal'] = np.where(ensemble_signals['signal'] < -self.threshold, -1.0, ensemble_signals['signal'])
        
        # Generate positions (1 for long, -1 for short, 0 for no position)
        ensemble_signals['position'] = ensemble_signals['signal'].diff()
        
        # Replace NaN values with 0
        ensemble_signals.fillna(0, inplace=True)
        
        return ensemble_signals 