import json
import csv
from datetime import datetime
import os

class TradeLogger:
    def __init__(self, log_file='logs/trades.csv'):
        self.log_file = log_file
        self.ensure_log_directory()
        self.ensure_log_file()
    
    def ensure_log_directory(self):
        """Create logs directory if it doesn't exist"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def ensure_log_file(self):
        """Create log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price', 
                    'order_type', 'stop_loss', 'take_profit', 'pnl', 'fee',
                    'status', 'order_id', 'execution_id'
                ])
    
    def log_trade(self, order_data, execution_result):
        """Log a trade to the CSV file"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Extract data from order and execution result
            symbol = order_data.get('symbol', '')
            side = order_data.get('side', '')
            quantity = order_data.get('quantity', 0)
            price = order_data.get('price', 0)
            order_type = order_data.get('orderType', 'Market')
            stop_loss = order_data.get('stopLoss', 0)
            take_profit = order_data.get('takeProfit', 0)
            
            # Extract execution details
            order_id = execution_result.get('result', {}).get('orderId', '')
            status = execution_result.get('retMsg', 'Success')
            
            # Calculate PnL and fee (simplified)
            pnl = 0.0  # Will be updated later when position is closed
            fee = 0.0  # Will be calculated based on execution
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, symbol, side, quantity, price,
                    order_type, stop_loss, take_profit, pnl, fee,
                    status, order_id, ''
                ])
                
        except Exception as e:
            print(f"Error logging trade: {e}")
    
    def get_recent_trades(self, days=30):
        """Get recent trades from the log file"""
        trades = []
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        trades.append({
                            'timestamp': row['timestamp'],
                            'symbol': row['symbol'],
                            'side': row['side'],
                            'quantity': float(row['quantity']) if row['quantity'] else 0,
                            'price': float(row['price']) if row['price'] else 0,
                            'pnl': float(row['pnl']) if row['pnl'] else 0,
                            'fee': float(row['fee']) if row['fee'] else 0,
                            'status': row['status']
                        })
        except Exception as e:
            print(f"Error reading trades: {e}")
        
        return trades
    
    def get_trade_statistics(self):
        """Calculate trade statistics from logged trades"""
        trades = self.get_recent_trades()
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'win_rate': 0.0
            }
        
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        
        total_pnl = sum(t['pnl'] for t in trades)
        
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_pnl': total_pnl,
            'largest_win': max(wins) if wins else 0.0,
            'largest_loss': min(losses) if losses else 0.0,
            'average_win': sum(wins) / len(wins) if wins else 0.0,
            'average_loss': sum(losses) / len(losses) if losses else 0.0,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        }
    
    def calculate_stats(self):
        """Legacy method for backward compatibility"""
        stats = self.get_trade_statistics()
        return {
            "total_trades": stats['total_trades'],
            "wins": stats['winning_trades'],
            "losses": stats['losing_trades'],
            "win_rate": stats['win_rate'],
            "total_profit": stats['total_pnl']
        }
