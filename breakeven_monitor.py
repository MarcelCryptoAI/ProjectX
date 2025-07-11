import time
import logging
from datetime import datetime
from db_singleton import get_database
from pybit.unified_trading import HTTP
import os

class BreakevenMonitor:
    def __init__(self, ai_worker=None):
        self.logger = logging.getLogger(__name__)
        self.database = get_database()
        self.bybit_session = None
        self.monitored_positions = {}  # Track which positions we're monitoring
        self.ai_worker = ai_worker  # Reference to AI worker for updating active_trades
        
    def initialize_bybit(self):
        """Initialize ByBit session"""
        try:
            api_key = os.getenv('BYBIT_API_KEY')
            api_secret = os.getenv('BYBIT_API_SECRET')
            testnet = os.getenv('BYBIT_TESTNET', 'False').lower() == 'true'
            
            if not api_key or not api_secret:
                self.logger.error("ByBit API credentials not found")
                return False
                
            self.bybit_session = HTTP(
                testnet=testnet,
                api_key=api_key,
                api_secret=api_secret,
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize ByBit session: {e}")
            return False
    
    def get_breakeven_settings(self):
        """Get breakeven settings from database"""
        try:
            settings = self.database.load_settings()
            move_to_breakeven = settings.get('moveStopLossOnPartialTP', True)
            breakeven_percentage = float(settings.get('breakevenPercentage', 0.2))
            return move_to_breakeven, breakeven_percentage
        except Exception as e:
            self.logger.error(f"Failed to load breakeven settings: {e}")
            return True, 0.2  # Default values
    
    def get_active_positions(self):
        """Get all active positions from ByBit"""
        try:
            if not self.bybit_session:
                if not self.initialize_bybit():
                    return []
                    
            response = self.bybit_session.get_positions(category="linear", limit=200)
            if response.get('retCode') == 0:
                positions = []
                for position in response['result']['list']:
                    if float(position['size']) > 0:  # Only active positions
                        positions.append({
                            'symbol': position['symbol'],
                            'side': position['side'],
                            'size': float(position['size']),
                            'avgPrice': float(position['avgPrice']),
                            'markPrice': float(position['markPrice']),
                            'unrealisedPnl': float(position['unrealisedPnl']),
                            'leverage': position['leverage'],
                            'stopLoss': float(position['stopLoss']) if position['stopLoss'] else None,
                            'takeProfit': float(position['takeProfit']) if position['takeProfit'] else None
                        })
                return positions
            else:
                self.logger.error(f"Failed to get positions: {response}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting active positions: {e}")
            return []
    
    def get_position_orders(self, symbol):
        """Get all orders for a position"""
        try:
            response = self.bybit_session.get_open_orders(category="linear", symbol=symbol, limit=50)
            if response.get('retCode') == 0:
                return response['result']['list']
            return []
        except Exception as e:
            self.logger.error(f"Error getting orders for {symbol}: {e}")
            return []
    
    def check_tp1_hit(self, position, orders):
        """Check if TP1 has been hit for a position"""
        try:
            symbol = position['symbol']
            side = position['side']
            avg_price = position['avgPrice']
            mark_price = position['markPrice']
            
            # Get take profit orders
            tp_orders = [order for order in orders if 'TP' in order.get('orderType', '')]
            
            if not tp_orders:
                return False
                
            # Sort TP orders by price (closest to entry first)
            if side == 'Buy':
                tp_orders.sort(key=lambda x: float(x['price']))  # Lowest TP first for long
                tp1_price = float(tp_orders[0]['price'])
                tp1_hit = mark_price >= tp1_price
            else:
                tp_orders.sort(key=lambda x: float(x['price']), reverse=True)  # Highest TP first for short
                tp1_price = float(tp_orders[0]['price'])
                tp1_hit = mark_price <= tp1_price
            
            if tp1_hit:
                self.logger.info(f"🎯 TP1 HIT for {symbol} {side}: Mark={mark_price}, TP1={tp1_price}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking TP1 for {position['symbol']}: {e}")
            return False
    
    def calculate_breakeven_price(self, position, breakeven_percentage):
        """Calculate the new breakeven stop loss price"""
        try:
            avg_price = position['avgPrice']
            side = position['side']
            
            if side == 'Buy':
                # For long positions: entry + percentage
                breakeven_price = avg_price * (1 + breakeven_percentage / 100)
            else:
                # For short positions: entry - percentage  
                breakeven_price = avg_price * (1 - breakeven_percentage / 100)
            
            self.logger.info(f"💡 Calculated breakeven price for {position['symbol']}: {breakeven_price:.6f} (Entry: {avg_price}, +{breakeven_percentage}%)")
            return breakeven_price
            
        except Exception as e:
            self.logger.error(f"Error calculating breakeven price: {e}")
            return None
    
    def move_stop_loss_to_breakeven(self, position, breakeven_price):
        """Move stop loss to breakeven + percentage"""
        try:
            symbol = position['symbol']
            side = position['side']
            current_sl = position['stopLoss']
            
            # Check if we need to update
            if current_sl and abs(current_sl - breakeven_price) < 0.000001:
                self.logger.info(f"🔄 Stop loss already at breakeven for {symbol}")
                return True
            
            # Set new stop loss
            response = self.bybit_session.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(breakeven_price)
            )
            
            if response.get('retCode') == 0:
                self.logger.info(f"✅ Moved stop loss to breakeven for {symbol}: {current_sl} → {breakeven_price}")
                
                # Update AI worker's active trades if available
                if self.ai_worker and hasattr(self.ai_worker, 'active_trades'):
                    for order_id, trade_data in self.ai_worker.active_trades.items():
                        if trade_data.get('symbol') == symbol and trade_data.get('side') == side:
                            # Mark TP1 as hit and SL moved to breakeven
                            trade_data['tp1_hit'] = True
                            trade_data['sl_moved_to_breakeven'] = True
                            trade_data['stop_loss']['price'] = breakeven_price
                            
                            # Update TP1 status
                            if len(trade_data.get('take_profit_levels', [])) > 0:
                                trade_data['take_profit_levels'][0]['status'] = 'hit'
                                trade_data['take_profit_levels'][0]['hit_time'] = datetime.now().isoformat()
                            
                            self.logger.info(f"📝 Updated active_trades for {symbol}: TP1 hit, SL moved to BE")
                            break
                
                return True
            else:
                self.logger.error(f"❌ Failed to move stop loss for {symbol}: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error moving stop loss for {position['symbol']}: {e}")
            return False
    
    def monitor_positions(self):
        """Main monitoring loop - check all positions for TP1 hits"""
        try:
            move_to_breakeven, breakeven_percentage = self.get_breakeven_settings()
            
            if not move_to_breakeven:
                self.logger.debug("Breakeven monitoring disabled in settings")
                return
            
            positions = self.get_active_positions()
            self.logger.debug(f"Monitoring {len(positions)} active positions for TP1 hits")
            
            for position in positions:
                symbol = position['symbol']
                
                # Get position orders
                orders = self.get_position_orders(symbol)
                
                # Check if TP1 has been hit
                if self.check_tp1_hit(position, orders):
                    position_key = f"{symbol}_{position['side']}"
                    
                    # Only process each position once
                    if position_key not in self.monitored_positions:
                        self.monitored_positions[position_key] = True
                        
                        # Calculate breakeven price
                        breakeven_price = self.calculate_breakeven_price(position, breakeven_percentage)
                        
                        if breakeven_price:
                            # Move stop loss to breakeven
                            success = self.move_stop_loss_to_breakeven(position, breakeven_price)
                            
                            if success:
                                self.logger.info(f"🏆 Successfully moved {symbol} stop loss to breakeven +{breakeven_percentage}%")
                            else:
                                self.logger.error(f"💥 Failed to move {symbol} stop loss to breakeven")
                
        except Exception as e:
            self.logger.error(f"Error in position monitoring: {e}")
    
    def start_monitoring(self, interval=30):
        """Start the monitoring loop"""
        self.logger.info(f"🚀 Starting breakeven monitor (interval: {interval}s)")
        
        while True:
            try:
                self.monitor_positions()
                time.sleep(interval)
            except KeyboardInterrupt:
                self.logger.info("⏹️ Breakeven monitor stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in monitoring loop: {e}")
                time.sleep(interval)

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = BreakevenMonitor()
    monitor.start_monitoring()