import time
import logging
from datetime import datetime
from utils.settings_loader import Settings

class TradeExecutor:
    def __init__(self, bybit_session, console_logger=None):
        self.bybit_session = bybit_session
        self.console_logger = console_logger
        self.settings = Settings.load('config/settings.yaml')
        self.is_trading_enabled = False
        self.active_orders = {}
        
    def log(self, level, message):
        """Log message to console"""
        if self.console_logger:
            self.console_logger.log(level, message)
        else:
            print(f"[{level}] {message}")
    
    def enable_trading(self):
        """Enable live trading"""
        self.is_trading_enabled = True
        self.log('SUCCESS', '🚀 LIVE TRADING ENABLED - AI will now execute real trades!')
        self.log('INFO', '💰 Trade executor is active and monitoring for signals')
        self.log('WARNING', '⚠️ REAL MONEY AT RISK - Monitor positions carefully')
        
    def disable_trading(self):
        """Disable live trading"""
        self.is_trading_enabled = False
        self.log('WARNING', '⏸️ LIVE TRADING DISABLED - No new trades will be placed')
        self.log('INFO', '🔒 Trade executor is now in safe mode')
    
    def execute_signal(self, signal):
        """Execute a trading signal"""
        if not self.is_trading_enabled:
            self.log('INFO', f'📋 Signal received but trading disabled: {signal["side"]} {signal["symbol"]}')
            return None
            
        try:
            symbol = signal['symbol']
            side = signal['side'].lower()
            confidence = signal['confidence']
            
            self.log('INFO', f'🎯 Executing signal: {side.upper()} {symbol} (Confidence: {confidence:.1f}%)')
            
            # Calculate position size based on risk management
            position_size = self._calculate_position_size(symbol)
            if position_size <= 0:
                self.log('WARNING', f'❌ Position size too small for {symbol}')
                return None
            
            # Get current market price
            ticker = self.bybit_session.get_tickers(
                category="linear",
                symbol=symbol
            )
            
            if not ticker or 'result' not in ticker or 'list' not in ticker['result']:
                self.log('ERROR', f'❌ Failed to get ticker for {symbol}')
                return None
            
            ticker_data = ticker['result']['list'][0]
            current_price = float(ticker_data['lastPrice'])
            
            # Determine order type and price
            if side == 'buy':
                # For buy orders, use ask price or market
                order_price = current_price * 1.001  # Slight premium for market buy
                order_side = 'Buy'
            else:
                # For sell orders, use bid price or market
                order_price = current_price * 0.999  # Slight discount for market sell
                order_side = 'Sell'
            
            # Calculate stop loss and take profit
            stop_loss_price, take_profit_price = self._calculate_stop_take_profit(
                current_price, side
            )
            
            # Place the main order
            order_result = self._place_order(
                symbol=symbol,
                side=order_side,
                size=position_size,
                price=order_price,
                order_type='Limit'
            )
            
            if order_result and order_result.get('result'):
                order_id = order_result['result']['orderId']
                
                self.log('SUCCESS', f'✅ Order placed: {order_id}')
                self.log('INFO', f'📊 Size: {position_size}, Price: ${order_price:.4f}')
                
                # Log leverage information
                base_leverage = self.settings.bot.get('default_leverage', 1)
                leverage_multiplier = self._get_leverage_multiplier(symbol)
                final_leverage = base_leverage * leverage_multiplier
                self.log('INFO', f'⚡ Leverage: {base_leverage}x (base) × {leverage_multiplier}x (multiplier) = {final_leverage}x (final)')
                
                # Store order for tracking
                self.active_orders[order_id] = {
                    'symbol': symbol,
                    'side': order_side,
                    'size': position_size,
                    'price': order_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'timestamp': datetime.now(),
                    'signal': signal
                }
                
                # Set stop loss and take profit if enabled
                if self.settings.bot.get('auto_tpsl', True):
                    self._set_stop_take_profit(symbol, order_side, position_size, stop_loss_price, take_profit_price)
                
                return order_result
            else:
                self.log('ERROR', f'❌ Failed to place order for {symbol}')
                return None
                
        except Exception as e:
            self.log('ERROR', f'❌ Trade execution error: {str(e)}')
            return None
    
    def _calculate_position_size(self, symbol):
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            balance = self.bybit_session.get_wallet_balance(
                accountType="UNIFIED"
            )
            
            if not balance or 'result' not in balance:
                self.log('WARNING', 'Failed to get account balance, using default size')
                return 0.001  # Default small size
            
            # Get available balance
            usdt_balance = 0
            for coin in balance['result']['list'][0]['coin']:
                if coin['coin'] == 'USDT':
                    usdt_balance = float(coin['availableToWithdraw'])
                    break
            
            if usdt_balance <= 0:
                self.log('WARNING', 'No USDT balance available')
                return 0
            
            # Calculate position size based on risk per trade
            # Fix: User expects 5% to mean 0.5% (divide by 10)
            risk_per_trade = self.settings.bot.get('risk_per_trade_percent', 2) / 1000
            risk_amount = usdt_balance * risk_per_trade
            
            # Get current price to calculate size
            ticker = self.bybit_session.get_tickers(
                category="linear",
                symbol=symbol
            )
            
            if ticker and 'result' in ticker and ticker['result']['list']:
                current_price = float(ticker['result']['list'][0]['lastPrice'])
                
                # Calculate base position size
                base_position_size = risk_amount / current_price
                
                # Apply leverage if configured
                base_leverage = self.settings.bot.get('default_leverage', 1)
                
                # Get symbol-specific leverage multiplier from database
                leverage_multiplier = self._get_leverage_multiplier(symbol)
                
                # Calculate final leverage
                final_leverage = base_leverage * leverage_multiplier
                position_size = base_position_size * final_leverage
                
                # Round to appropriate decimal places
                if symbol.endswith('USDT'):
                    if current_price < 1:
                        position_size = round(position_size, 0)  # Whole tokens for low price
                    else:
                        position_size = round(position_size, 3)  # 3 decimals for higher price
                
                return max(position_size, 0.001)  # Minimum position size
            
            return 0.001  # Fallback
            
        except Exception as e:
            self.log('ERROR', f'Position size calculation error: {str(e)}')
            return 0.001
    
    def _get_leverage_multiplier(self, symbol):
        """Get leverage multiplier for a specific symbol from database"""
        try:
            from db_singleton import get_database
            db = get_database()
            return db.get_leverage_multiplier(symbol)
        except Exception as e:
            self.log('WARNING', f'Failed to get leverage multiplier for {symbol}: {str(e)}')
            return 1.0  # Default to 1x multiplier if database lookup fails
    
    def _calculate_stop_take_profit(self, entry_price, side):
        """Calculate stop loss and take profit prices"""
        stop_loss_pct = self.settings.bot.get('stop_loss_percent', 2) / 100
        take_profit_pct = self.settings.bot.get('take_profit_percent', 3) / 100
        
        if side == 'buy':
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
        
        return stop_loss, take_profit
    
    def _place_order(self, symbol, side, size, price, order_type='Limit'):
        """Place order on ByBit"""
        try:
            order_result = self.bybit_session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType=order_type,
                qty=str(size),
                price=str(price) if order_type == 'Limit' else None,
                timeInForce="GTC"  # Good Till Cancelled
            )
            
            return order_result
            
        except Exception as e:
            self.log('ERROR', f'Order placement error: {str(e)}')
            return None
    
    def _set_stop_take_profit(self, symbol, side, size, stop_price, take_profit_price):
        """Set stop loss and take profit orders"""
        try:
            # Place stop loss order
            if stop_price:
                stop_side = 'Sell' if side == 'Buy' else 'Buy'
                stop_result = self.bybit_session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=stop_side,
                    orderType="StopMarket",
                    qty=str(size),
                    stopPrice=str(stop_price),
                    timeInForce="GTC"
                )
                
                if stop_result and stop_result.get('result'):
                    self.log('INFO', f'🛡️ Stop Loss set at ${stop_price:.4f}')
            
            # Place take profit order
            if take_profit_price:
                tp_side = 'Sell' if side == 'Buy' else 'Buy'
                tp_result = self.bybit_session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=tp_side,
                    orderType="TakeProfitMarket",
                    qty=str(size),
                    takeProfitPrice=str(take_profit_price),
                    timeInForce="GTC"
                )
                
                if tp_result and tp_result.get('result'):
                    self.log('INFO', f'🎯 Take Profit set at ${take_profit_price:.4f}')
                    
        except Exception as e:
            self.log('WARNING', f'Failed to set TP/SL: {str(e)}')
    
    def monitor_orders(self):
        """Monitor active orders and positions"""
        if not self.active_orders:
            return
        
        try:
            # Check order status
            for order_id, order_info in list(self.active_orders.items()):
                order_status = self.bybit_session.get_order_history(
                    category="linear",
                    orderId=order_id
                )
                
                if order_status and 'result' in order_status and order_status['result']['list']:
                    order = order_status['result']['list'][0]
                    status = order['orderStatus']
                    
                    if status == 'Filled':
                        self.log('SUCCESS', f'✅ Order {order_id} FILLED!')
                        del self.active_orders[order_id]
                    elif status == 'Cancelled':
                        self.log('WARNING', f'❌ Order {order_id} CANCELLED')
                        del self.active_orders[order_id]
                        
        except Exception as e:
            self.log('ERROR', f'Order monitoring error: {str(e)}')
    
    def get_trading_status(self):
        """Get current trading status"""
        return {
            'enabled': self.is_trading_enabled,
            'active_orders': len(self.active_orders),
            'orders': list(self.active_orders.keys())
        }