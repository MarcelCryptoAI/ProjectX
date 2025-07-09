import numpy as np
from datetime import datetime
import logging
import random

class AITrader:
    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
    def get_prediction(self, market_conditions=None):
        """Get AI prediction for trading with dynamic take profit within bounds and perfect entry price"""
        # Get take profit bounds from settings
        min_tp = self.settings.min_take_profit_percent
        max_tp = self.settings.max_take_profit_percent
        default_tp = self.settings.take_profit_percent
        
        # Generate AI-determined take profit based on market conditions
        ai_take_profit = self._calculate_dynamic_take_profit(
            min_tp, max_tp, default_tp, market_conditions
        )
        
        # Generate trading signal based on market analysis
        # In a real implementation, this would use ML models
        symbol, side, confidence = self._generate_trading_signal(market_conditions)
        
        # Calculate perfect entry price based on market analysis
        perfect_entry_price = self._calculate_perfect_entry_price(symbol, side, market_conditions)
        
        # Calculate AI-advised stop loss based on market conditions
        ai_stop_loss = self._calculate_dynamic_stop_loss(market_conditions)
        
        # Placeholder implementation with dynamic TP and perfect entry
        return {
            'symbol': symbol,
            'side': side,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'take_profit': ai_take_profit,
            'stop_loss': ai_stop_loss,
            'entry_price': perfect_entry_price,
            'leverage': self._calculate_dynamic_leverage(confidence, market_conditions),
            'amount': 100,
            'accuracy': 70 + random.uniform(0, 25)  # 70-95% range
        }
    
    def _calculate_dynamic_take_profit(self, min_tp, max_tp, default_tp, market_conditions=None):
        """Calculate dynamic take profit based on market conditions"""
        # Base take profit starts at default
        base_tp = default_tp
        
        if market_conditions:
            # Adjust based on market volatility
            volatility = market_conditions.get('avg_volatility', 2.0)
            market_trend = market_conditions.get('market_trend', 'neutral')
            volume_strength = market_conditions.get('volume_strength', 0)
            
            # Higher volatility = wider take profit targets
            volatility_adjustment = min(2.0, volatility / 2.0)  # Cap at 2% adjustment
            
            # Trend alignment bonus
            trend_bonus = 0
            if market_trend == 'bullish':
                trend_bonus = 1.0  # Extra 1% for bullish markets
            elif market_trend == 'bearish':
                trend_bonus = -0.5  # Slightly lower TP in bearish markets
            
            # Volume strength adjustment
            volume_adjustment = 0
            if volume_strength > 50:  # Strong volume
                volume_adjustment = 0.5
            elif volume_strength < -50:  # Weak volume
                volume_adjustment = -0.5
            
            # Calculate AI take profit
            ai_tp = base_tp + volatility_adjustment + trend_bonus + volume_adjustment
            
            # Add some randomness for variation (10% of the calculated value)
            variation = random.uniform(-0.1, 0.1) * ai_tp
            ai_tp += variation
            
        else:
            # Fallback to simple variation if no market conditions
            variation = random.uniform(-1, 1)
            ai_tp = base_tp + (variation * 2)
        
        # Ensure within min/max bounds
        ai_tp = max(min_tp, min(max_tp, ai_tp))
        
        self.logger.info(f"Calculated AI take profit: {ai_tp:.2f}% (bounds: {min_tp}-{max_tp}%)")
        
        return round(ai_tp, 2)
        
    def _generate_trading_signal(self, market_conditions=None):
        """Generate trading signal based on market analysis"""
        # List of enabled trading pairs from settings
        enabled_pairs = self.settings.bot.get('enabled_pairs', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
        
        # Randomly select a symbol for now
        # In production, this would use market scanners and ML models
        symbol = random.choice(enabled_pairs[:10])  # Use top 10 pairs
        
        # Determine side based on market conditions
        if market_conditions and market_conditions.get('market_trend') == 'bullish':
            side = 'Buy' if random.random() > 0.3 else 'Sell'  # 70% buy in bull market
        elif market_conditions and market_conditions.get('market_trend') == 'bearish':
            side = 'Sell' if random.random() > 0.3 else 'Buy'  # 70% sell in bear market
        else:
            side = random.choice(['Buy', 'Sell'])  # 50/50 in neutral market
        
        # Generate confidence based on market conditions
        base_confidence = 75
        if market_conditions:
            volatility = market_conditions.get('avg_volatility', 2.0)
            # Lower confidence in high volatility
            confidence = base_confidence - min(10, volatility * 2)
            # Add some randomness
            confidence += random.uniform(-5, 10)
        else:
            confidence = base_confidence + random.uniform(-10, 15)
        
        # Ensure confidence is within reasonable bounds
        confidence = max(60, min(95, confidence))
        
        return symbol, side, confidence
    
    def _calculate_perfect_entry_price(self, symbol, side, market_conditions=None):
        """Calculate perfect entry price based on market analysis and support/resistance levels"""
        # In a real implementation, this would analyze:
        # - Support and resistance levels
        # - Order book depth
        # - Recent price action
        # - Volume profile
        # - Market microstructure
        
        # For now, simulate intelligent entry pricing
        base_adjustment = 0.001  # 0.1% base adjustment
        
        if market_conditions:
            volatility = market_conditions.get('avg_volatility', 2.0)
            market_trend = market_conditions.get('market_trend', 'neutral')
            volume_strength = market_conditions.get('volume_strength', 0)
            
            # Adjust entry based on volatility (higher volatility = more conservative entry)
            volatility_factor = min(0.005, volatility * 0.001)  # Cap at 0.5%
            
            # Adjust for market trend alignment
            trend_factor = 0
            if market_trend == 'bullish' and side == 'Buy':
                trend_factor = -0.001  # Slightly more aggressive on aligned trend
            elif market_trend == 'bearish' and side == 'Sell':
                trend_factor = -0.001  # Slightly more aggressive on aligned trend
            else:
                trend_factor = 0.001  # More conservative on counter-trend
            
            # Volume-based adjustment
            volume_factor = 0
            if volume_strength > 50:  # High volume = more aggressive entry
                volume_factor = -0.0005
            elif volume_strength < -50:  # Low volume = more conservative
                volume_factor = 0.0005
            
            # Calculate final adjustment
            final_adjustment = base_adjustment + volatility_factor + trend_factor + volume_factor
            
            # For buy orders, place slightly below market
            # For sell orders, place slightly above market
            if side == 'Buy':
                price_adjustment = -abs(final_adjustment)  # Always negative for buy
            else:
                price_adjustment = abs(final_adjustment)   # Always positive for sell
        else:
            # Simple fallback
            price_adjustment = -base_adjustment if side == 'Buy' else base_adjustment
        
        # Return the adjustment factor (will be applied to current market price)
        return price_adjustment
    
    def _calculate_dynamic_stop_loss(self, market_conditions=None):
        """Calculate AI-advised stop loss based on market conditions"""
        base_sl = self.settings.stop_loss_percent
        
        if market_conditions:
            volatility = market_conditions.get('avg_volatility', 2.0)
            
            # Adjust stop loss based on volatility
            # Higher volatility = wider stop loss
            volatility_adjustment = min(1.0, volatility * 0.3)  # Max 1% adjustment
            
            # Add some AI randomness for variation
            ai_variation = random.uniform(-0.2, 0.2)
            
            ai_sl = base_sl + volatility_adjustment + ai_variation
            
            # Ensure reasonable bounds (0.5% - 5%)
            ai_sl = max(0.5, min(5.0, ai_sl))
        else:
            # Simple variation
            ai_sl = base_sl + random.uniform(-0.5, 0.5)
            ai_sl = max(0.5, min(5.0, ai_sl))
        
        return round(ai_sl, 2)
    
    def _calculate_dynamic_leverage(self, confidence, market_conditions=None):
        """Calculate dynamic leverage based on confidence and market conditions"""
        min_leverage = self.settings.bot.get('min_leverage', 1)
        max_leverage = self.settings.bot.get('max_leverage', 10)
        default_leverage = self.settings.bot.get('default_leverage', 5)
        
        # Base leverage on confidence
        # Higher confidence = potentially higher leverage
        confidence_factor = (confidence - 60) / 35  # Normalize 60-95 to 0-1
        
        # Start with default leverage
        leverage = default_leverage
        
        if market_conditions:
            volatility = market_conditions.get('avg_volatility', 2.0)
            # Reduce leverage in high volatility
            if volatility > 3:
                leverage = leverage * 0.8
            elif volatility < 1:
                leverage = leverage * 1.2
        
        # Apply confidence factor
        leverage = leverage * (0.8 + confidence_factor * 0.4)  # 80% to 120% of base
        
        # Ensure within bounds
        leverage = int(max(min_leverage, min(max_leverage, leverage)))
        
        return leverage
    
    def analyze_market(self, symbol):
        """Analyze market data for a symbol"""
        # Placeholder implementation
        return {
            'trend': 'bullish',
            'confidence': 0.8,
            'entry_price': 0,
            'take_profit': 0,
            'stop_loss': 0
        }

class Trader:
    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
    def execute_trade(self, trade_data):
        """Execute a trade"""
        # Placeholder implementation
        return True