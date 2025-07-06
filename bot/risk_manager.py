class RiskManager:
    def __init__(self, settings):
        self.settings = settings
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.open_positions = 0

    def calculate_position_size(self, balance):
        risk_amount = (self.settings.risk_per_trade_percent / 100) * balance
        return risk_amount
        
    def validate_trade(self, order_data):
        """Validate if a trade meets risk management criteria"""
        
        # Check maximum concurrent trades
        if self.open_positions >= self.settings.max_concurrent_trades:
            print(f"❌ Risk check failed: Too many open positions ({self.open_positions})")
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -abs(self.settings.max_daily_loss_percent):
            print(f"❌ Risk check failed: Daily loss limit reached ({self.daily_pnl}%)")
            return False
        
        # Check position size
        if not self._validate_position_size(order_data):
            return False
        
        # All checks passed
        return True
    
    def _validate_position_size(self, order_data):
        """Validate position size against risk parameters"""
        quantity = order_data.get('quantity', 0)
        
        if quantity <= 0:
            print(f"❌ Risk check failed: Invalid quantity ({quantity})")
            return False
        
        # Add more position size validation logic here
        return True
    
    def update_daily_stats(self, pnl_change):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl_change
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new day)"""
        self.daily_trades = 0
        self.daily_pnl = 0.0
    
    def update_position_count(self, count):
        """Update the number of open positions"""
        self.open_positions = count
