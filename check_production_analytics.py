#!/usr/bin/env python3
"""
Check production analytics data - can be run via web endpoint
"""
import os
import sys
import json
from datetime import datetime, timedelta

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_production_analytics():
    """Check production analytics data and return results"""
    results = {
        'database_type': None,
        'total_signals': 0,
        'status_breakdown': {},
        'completed_signals': 0,
        'completed_with_pnl': 0,
        'analytics_data': {},
        'sample_signals': [],
        'issues': [],
        'recommendations': []
    }
    
    try:
        from database import TradingDatabase
        
        # Initialize database
        db = TradingDatabase()
        results['database_type'] = 'PostgreSQL' if db.use_postgres else 'SQLite'
        
        # Get all signals
        signals = db.get_trading_signals()
        results['total_signals'] = len(signals)
        
        if len(signals) == 0:
            results['issues'].append("No signals found in database")
            results['recommendations'].append("Check if AI worker is running")
            return results
        
        # Analyze signal statuses
        for signal in signals:
            status = signal.get('status', 'unknown')
            results['status_breakdown'][status] = results['status_breakdown'].get(status, 0) + 1
        
        # Count completed signals
        completed_signals = [s for s in signals if s.get('status') == 'completed']
        completed_with_pnl = [s for s in completed_signals if s.get('realized_pnl') is not None]
        
        results['completed_signals'] = len(completed_signals)
        results['completed_with_pnl'] = len(completed_with_pnl)
        
        # Replicate analytics calculation
        if completed_with_pnl:
            total_trades = len(completed_with_pnl)
            winning_trades = 0
            losing_trades = 0
            total_profit = 0
            total_loss = 0
            largest_win = 0
            largest_loss = 0
            realized_pnl_24h = 0
            realized_pnl_all_time = 0
            
            now = datetime.utcnow()
            yesterday = now - timedelta(days=1)
            
            for signal in completed_with_pnl:
                pnl = float(signal.get('realized_pnl', 0))
                realized_pnl_all_time += pnl
                
                if pnl > 0:
                    winning_trades += 1
                    total_profit += pnl
                    largest_win = max(largest_win, pnl)
                elif pnl < 0:
                    losing_trades += 1
                    total_loss += abs(pnl)
                    largest_loss = min(largest_loss, pnl)
                
                # Check 24h P&L
                if signal.get('exit_time'):
                    try:
                        exit_time_str = str(signal.get('exit_time'))
                        if 'T' in exit_time_str:
                            exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                        else:
                            exit_time = datetime.strptime(exit_time_str, '%Y-%m-%d %H:%M:%S')
                        
                        if exit_time.tzinfo is not None:
                            exit_time = exit_time.replace(tzinfo=None)
                        
                        if exit_time >= yesterday:
                            realized_pnl_24h += pnl
                    except:
                        pass
            
            results['analytics_data'] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'realized_pnl_24h': realized_pnl_24h,
                'realized_pnl_all_time': realized_pnl_all_time
            }
        
        # Get sample signals
        for signal in signals[:5]:
            results['sample_signals'].append({
                'symbol': signal['symbol'],
                'side': signal['side'],
                'status': signal['status'],
                'entry_price': signal.get('entry_price'),
                'exit_price': signal.get('exit_price'),
                'realized_pnl': signal.get('realized_pnl'),
                'created_at': str(signal.get('created_at', '')),
                'exit_time': str(signal.get('exit_time', ''))
            })
        
        # Analyze issues
        if results['completed_signals'] == 0:
            results['issues'].append("No completed signals found")
            results['recommendations'].append("Check AI worker trade monitoring")
        elif results['completed_with_pnl'] == 0:
            results['issues'].append("Completed signals exist but no P&L data")
            results['recommendations'].append("Check P&L calculation in monitor_trades_and_move_sl()")
        
        if results['analytics_data'].get('total_trades', 0) == 0:
            results['issues'].append("Analytics would show 0 trades")
        
    except Exception as e:
        results['issues'].append(f"Error: {str(e)}")
        import traceback
        results['error_traceback'] = traceback.format_exc()
    
    return results

def main():
    """Main function for command line usage"""
    results = check_production_analytics()
    
    print("=" * 80)
    print("PRODUCTION ANALYTICS CHECK")
    print("=" * 80)
    
    print(f"Database Type: {results['database_type']}")
    print(f"Total Signals: {results['total_signals']}")
    print(f"Completed Signals: {results['completed_signals']}")
    print(f"Completed with P&L: {results['completed_with_pnl']}")
    
    if results['status_breakdown']:
        print("\nStatus Breakdown:")
        for status, count in results['status_breakdown'].items():
            print(f"  {status}: {count}")
    
    if results['analytics_data']:
        print("\nAnalytics Data:")
        analytics = results['analytics_data']
        print(f"  Total Trades: {analytics['total_trades']}")
        print(f"  Winning Trades: {analytics['winning_trades']}")
        print(f"  Losing Trades: {analytics['losing_trades']}")
        print(f"  Win Rate: {analytics['win_rate']:.1f}%")
        print(f"  Realized P&L 24h: {analytics['realized_pnl_24h']:.4f} USDT")
        print(f"  Realized P&L All-time: {analytics['realized_pnl_all_time']:.4f} USDT")
    
    if results['issues']:
        print("\nIssues Found:")
        for issue in results['issues']:
            print(f"  ❌ {issue}")
    
    if results['recommendations']:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  📝 {rec}")
    
    return results

if __name__ == "__main__":
    main()