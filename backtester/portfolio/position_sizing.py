"""
Position Sizing for the Backtesting System.

This module implements various position sizing strategies.
"""

class FixedAmountSizer:
    """
    A position sizer that allocates a fixed amount of capital to each trade.
    
    Attributes:
        amount (float): The fixed amount to allocate to each trade.
    """
    
    def __init__(self, amount=1000.0):
        """
        Initialize the fixed amount sizer.
        
        Args:
            amount (float): The fixed amount to allocate to each trade.
        """
        self.amount = amount
        
    def calculate_position_size(self, price, capital):
        """
        Calculate the position size for a trade.
        
        Args:
            price (float): The current price of the asset.
            capital (float): The current capital in the portfolio.
            
        Returns:
            float: The number of units to trade.
        """
        return self.amount / price
        
class PercentageSizer:
    """
    A position sizer that allocates a percentage of capital to each trade.
    
    Attributes:
        percentage (float): The percentage of capital to allocate to each trade.
    """
    
    def __init__(self, percentage=0.1):
        """
        Initialize the percentage sizer.
        
        Args:
            percentage (float): The percentage of capital to allocate to each trade.
        """
        self.percentage = percentage
        
    def calculate_position_size(self, price, capital):
        """
        Calculate the position size for a trade.
        
        Args:
            price (float): The current price of the asset.
            capital (float): The current capital in the portfolio.
            
        Returns:
            float: The number of units to trade.
        """
        trade_value = capital * self.percentage
        return trade_value / price 