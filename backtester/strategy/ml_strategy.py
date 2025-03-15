"""
Machine Learning Strategy for the Backtesting System.

This module implements a strategy based on machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from backtester.strategy.base import Strategy

class MLStrategy(Strategy):
    """
    A strategy that generates signals based on machine learning predictions.
    
    Attributes:
        model (object): The machine learning model.
        features (list): The list of features to use for prediction.
        target (str): The target column for prediction.
        train_size (float): The proportion of data to use for training.
        price_column (str): The column name for price data.
        scaler (object): The scaler for feature normalization.
    """
    
    def __init__(self, features=None, target='return', train_size=0.7, price_column='close'):
        """
        Initialize the machine learning strategy.
        
        Args:
            features (list): The list of features to use for prediction.
            target (str): The target column for prediction.
            train_size (float): The proportion of data to use for training.
            price_column (str): The column name for price data.
        """
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = features or []
        self.target = target
        self.train_size = train_size
        self.price_column = price_column
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """
        Prepare features for the machine learning model.
        
        Args:
            data (pd.DataFrame): The market data.
            
        Returns:
            pd.DataFrame: The prepared features.
        """
        # Calculate returns
        data['return'] = data[self.price_column].pct_change()
        
        # Create binary target (1 for positive return, 0 for negative return)
        data['target'] = np.where(data['return'].shift(-1) > 0, 1, 0)
        
        # Add technical indicators as features if no features are specified
        if not self.features:
            # Add moving averages
            data['sma_10'] = data[self.price_column].rolling(window=10).mean()
            data['sma_30'] = data[self.price_column].rolling(window=30).mean()
            
            # Add RSI
            delta = data[self.price_column].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Add Bollinger Bands
            data['bb_middle'] = data[self.price_column].rolling(window=20).mean()
            data['bb_std'] = data[self.price_column].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
            data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
            
            # Set features
            self.features = ['sma_10', 'sma_30', 'rsi', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower']
        
        return data
        
    def train_model(self, data):
        """
        Train the machine learning model.
        
        Args:
            data (pd.DataFrame): The market data with features.
            
        Returns:
            object: The trained model.
        """
        # Prepare data
        data = self.prepare_features(data)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Split data into training and testing sets
        train_size = int(len(data) * self.train_size)
        train_data = data.iloc[:train_size]
        
        # Scale features
        X_train = self.scaler.fit_transform(train_data[self.features])
        y_train = train_data['target']
        
        # Train model
        self.model.fit(X_train, y_train)
        
        return self.model
        
    def generate_signals(self, data):
        """
        Generate trading signals based on machine learning predictions.
        
        Args:
            data (pd.DataFrame): The market data.
            
        Returns:
            pd.DataFrame: The input data with additional signal columns.
        """
        # Make a copy of the data to avoid modifying the original
        signals = data.copy()
        
        # Prepare features
        signals = self.prepare_features(signals)
        
        # Train model if not already trained
        if not hasattr(self.model, 'classes_'):
            self.train_model(signals)
        
        # Drop rows with NaN values
        signals_clean = signals.dropna()
        
        # Scale features
        X = self.scaler.transform(signals_clean[self.features])
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create signal column (1 for buy, 0 for hold)
        signals['signal'] = 0.0
        signals.loc[signals_clean.index, 'signal'] = predictions
        
        # Generate positions (1 for long, -1 for short, 0 for no position)
        signals['position'] = signals['signal'].diff()
        
        # Replace NaN values with 0
        signals.fillna(0, inplace=True)
        
        return signals 