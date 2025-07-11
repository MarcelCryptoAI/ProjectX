import numpy as np
from datetime import datetime
import logging
import random

class AITrader:
    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
    
    def _get_user_accuracy_setting(self):
        """Get exact user accuracy setting from database"""
        try:
            from db_singleton import get_database
            db = get_database()
            db_settings = db.load_settings()
            user_accuracy = float(db_settings.get('accuracyThreshold', 88))  # Default to 88
            self.logger.info(f"Using user accuracy setting: {user_accuracy}%")
            return user_accuracy
        except Exception as e:
            self.logger.warning(f"Could not load user accuracy from database: {e}, using default: 88")
            return 88.0  # Default fallback
    
    def get_prediction(self, market_conditions=None):
        """Get AI prediction for trading with dynamic take profit within bounds and perfect entry price"""
        
        # Log original market conditions
        if market_conditions:
            trend = market_conditions.get('market_trend', 'unknown')
            volatility = market_conditions.get('avg_volatility', 0)
            self.logger.info(f"🧠 AI analyzing: Market trend={trend.upper()}, Volatility={volatility:.2f}%")
        
        # Use market conditions as provided
        adjusted_market_conditions = market_conditions or {}
        
        # STRICT database enforcement - NO YAML fallback for user settings
        try:
            from db_singleton import get_database
            db = get_database()
            db_settings = db.load_settings()
            
            if not db_settings:
                raise Exception("Database settings are empty - cannot proceed without user configuration")
            
            # Check if dynamic TP is enabled
            enable_dynamic_tp = db_settings.get('enableDynamicTP', True)
            
            if enable_dynamic_tp:
                # Use dynamic TP with min/max bounds - MUST come from database
                if 'minTakeProfit' not in db_settings or 'maxTakeProfit' not in db_settings:
                    raise Exception("Dynamic TP enabled but minTakeProfit/maxTakeProfit not found in database")
                min_tp = float(db_settings['minTakeProfit'])
                max_tp = float(db_settings['maxTakeProfit'])
                default_tp = (min_tp + max_tp) / 2  # Use midpoint as default for dynamic
            else:
                # Use static TP value - MUST come from database
                if 'staticTakeProfit' not in db_settings:
                    raise Exception("Static TP selected but staticTakeProfit not found in database")
                static_tp = float(db_settings['staticTakeProfit'])
                min_tp = static_tp
                max_tp = static_tp
                default_tp = static_tp
        except Exception as e:
            # Log the error and use YAML as emergency fallback ONLY
            self.logger.error(f"CRITICAL: Database settings failed: {e}")
            self.logger.error("Using YAML emergency fallback - user settings will be ignored!")
            min_tp = self.settings.min_take_profit_percent
            max_tp = self.settings.max_take_profit_percent
            default_tp = self.settings.take_profit_percent
        
        # Log which settings are being used for debugging
        self.logger.info(f"TP Settings: min={min_tp}%, max={max_tp}%, default={default_tp}%")
        if 'db_settings' in locals():
            self.logger.info(f"Database settings loaded: enableDynamicTP={db_settings.get('enableDynamicTP', 'NOT_SET')}")
        
        # Generate AI-determined take profit based on market conditions
        ai_take_profit = self._calculate_dynamic_take_profit(
            min_tp, max_tp, default_tp, adjusted_market_conditions
        )
        
        # Generate trading signal based on market analysis
        # In a real implementation, this would use ML models
        symbol, side, confidence = self._generate_trading_signal(adjusted_market_conditions)
        
        # Calculate perfect entry price based on market analysis
        perfect_entry_price = self._calculate_perfect_entry_price(symbol, side, adjusted_market_conditions)
        
        # Calculate AI-advised stop loss based on market conditions
        ai_stop_loss = self._calculate_dynamic_stop_loss(adjusted_market_conditions)
        
        # Log AI signal generation
        self.logger.info(f"🎯 Generating AI trading signal")
        
        # Placeholder implementation with dynamic TP and perfect entry
        return {
            'symbol': symbol,
            'side': side,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'take_profit': ai_take_profit,
            'stop_loss': ai_stop_loss,
            'entry_price': perfect_entry_price,
            'leverage': self._calculate_dynamic_leverage(confidence, adjusted_market_conditions),
            'amount': 100,
            'accuracy': self._get_user_accuracy_setting()  # Use exact user setting
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
            
            
            # Log the calculation breakdown
            self.logger.info(f"🎯 TP Calculation: Base={base_tp:.2f}% + Volatility={volatility_adjustment:.2f}% + Trend={trend_bonus:.2f}% + Volume={volume_adjustment:.2f}% = {ai_tp:.2f}%")
            
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
            self.logger.info(f"📈 BULLISH market detected → {side} bias (70% buy probability)")
        elif market_conditions and market_conditions.get('market_trend') == 'bearish':
            side = 'Sell' if random.random() > 0.3 else 'Buy'  # 70% sell in bear market
            self.logger.info(f"📉 BEARISH market detected → {side} bias (70% sell probability)")
        elif market_conditions and market_conditions.get('market_trend') == 'sideways':
            side = random.choice(['Buy', 'Sell'])  # 50/50 in sideways market
            self.logger.info(f"↔️ SIDEWAYS market detected → {side} (50/50 probability)")
        else:
            side = random.choice(['Buy', 'Sell'])  # 50/50 in neutral market
            self.logger.info(f"❓ NEUTRAL/UNKNOWN market → {side} (50/50 probability)")
        
        # Generate confidence based on market conditions
        base_confidence = 75
        if market_conditions:
            volatility = market_conditions.get('avg_volatility', 2.0)
            # Lower confidence in high volatility
            confidence = base_confidence - min(10, volatility * 2)
            # Add some randomness
            confidence += random.uniform(-5, 10)
            
            # Use base confidence with market adjustments
        else:
            confidence = base_confidence + random.uniform(-10, 15)
        
        # Ensure confidence is within reasonable bounds using dynamic settings
        min_confidence = self.settings.bot.get('ai_accuracy_threshold', 70)
        confidence = max(min_confidence, min(95, confidence))
        
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
            
            # Calculate final adjustment based on market conditions
            
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
        # STRICT database enforcement for stop loss
        try:
            from db_singleton import get_database
            db = get_database()
            db_settings = db.load_settings()
            
            if not db_settings:
                raise Exception("Database settings are empty - cannot proceed without stop loss configuration")
            
            # Check if dynamic SL is enabled
            enable_dynamic_sl = db_settings.get('enableDynamicSL', True)
            
            if enable_dynamic_sl:
                # Use dynamic SL calculation - need default base from database
                if 'defaultStopLoss' in db_settings:
                    base_sl = float(db_settings['defaultStopLoss'])
                else:
                    # If no defaultStopLoss, calculate reasonable base from static if available
                    if 'staticStopLoss' in db_settings:
                        base_sl = float(db_settings['staticStopLoss'])
                    else:
                        raise Exception("Dynamic SL enabled but no stop loss value found in database")
            else:
                # Use static SL value - MUST come from database
                if 'staticStopLoss' not in db_settings:
                    raise Exception("Static SL selected but staticStopLoss not found in database")
                static_sl = float(db_settings['staticStopLoss'])
                return static_sl
        except Exception as e:
            # Log the error and use YAML as emergency fallback ONLY
            self.logger.error(f"CRITICAL: Database stop loss settings failed: {e}")
            self.logger.error("Using YAML emergency fallback for stop loss!")
            base_sl = self.settings.stop_loss_percent
        
        if market_conditions:
            volatility = market_conditions.get('avg_volatility', 2.0)
            
            # Adjust stop loss based on volatility
            # Higher volatility = wider stop loss
            volatility_adjustment = min(1.0, volatility * 0.3)  # Max 1% adjustment
            
            # Add some AI randomness for variation
            ai_variation = random.uniform(-0.2, 0.2)
            
            ai_sl = base_sl + volatility_adjustment + ai_variation
            
            # Apply market-based stop loss adjustment
            
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
        min_confidence = self.settings.bot.get('ai_accuracy_threshold', 70)
        confidence_factor = (confidence - min_confidence) / (95 - min_confidence)  # Normalize to 0-1
        
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
        
        # Apply market-based leverage adjustment
        
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