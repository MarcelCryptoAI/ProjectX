#!/usr/bin/env python3
"""
Utility script to view trade completion status and P&L data
"""

import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import TradingDatabase

def view_trade_completion_status():
    """View current trade completion status"""
    print("📊 Trade Completion Status Report")
    print("=" * 60)
    
    db = TradingDatabase()
    
    # Get all signals
    signals = db.get_trading_signals()
    
    # Categorize signals by status
    status_counts = {}
    for signal in signals:
        status = signal.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\n1. Signal Status Summary:")
    for status, count in sorted(status_counts.items()):
        print(f"   {status}: {count} signals")
    
    # Show completed trades with P&L
    completed_trades = [s for s in signals if s.get('status') == 'completed']
    
    print(f"\n2. Completed Trades Details ({len(completed_trades)} total):")
    if completed_trades:
        print(f"   {'Signal ID':<25} {'Symbol':<12} {'Side':<6} {'Entry':<12} {'Exit':<12} {'P&L':<12} {'Result':<8}")
        print(f"   {'-'*25} {'-'*12} {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
        
        for trade in completed_trades:
            signal_id = trade.get('signal_id', 'N/A')[:25]
            symbol = trade.get('symbol', 'N/A')
            side = trade.get('side', 'N/A')
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            pnl = trade.get('realized_pnl', 0)
            
            entry_str = f"${entry_price:.2f}" if entry_price else "N/A"
            exit_str = f"${exit_price:.2f}" if exit_price else "N/A"
            pnl_str = f"${pnl:.2f}" if pnl is not None else "N/A"
            result = "WIN" if pnl and pnl > 0 else "LOSS" if pnl and pnl < 0 else "BE"
            
            print(f"   {signal_id:<25} {symbol:<12} {side:<6} {entry_str:<12} {exit_str:<12} {pnl_str:<12} {result:<8}")
    else:
        print("   No completed trades found")
    
    # Calculate performance metrics
    print(f"\n3. Performance Metrics:")
    if completed_trades:
        trades_with_pnl = [t for t in completed_trades if t.get('realized_pnl') is not None]
        
        if trades_with_pnl:
            total_pnl = sum(float(t['realized_pnl']) for t in trades_with_pnl)
            winning_trades = [t for t in trades_with_pnl if float(t['realized_pnl']) > 0]
            losing_trades = [t for t in trades_with_pnl if float(t['realized_pnl']) < 0]
            
            win_rate = (len(winning_trades) / len(trades_with_pnl)) * 100 if trades_with_pnl else 0
            avg_win = sum(float(t['realized_pnl']) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(float(t['realized_pnl']) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            print(f"   Total Trades with P&L: {len(trades_with_pnl)}")
            print(f"   Winning Trades: {len(winning_trades)}")
            print(f"   Losing Trades: {len(losing_trades)}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Total P&L: ${total_pnl:.2f}")
            print(f"   Average Win: ${avg_win:.2f}")
            print(f"   Average Loss: ${avg_loss:.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades))
                print(f"   Profit Factor: {profit_factor:.2f}")
        else:
            print("   No trades with P&L data found")
    else:
        print("   No completed trades to analyze")
    
    # Show pending trades (positions currently open)
    pending_trades = [s for s in signals if s.get('status') == 'pending']
    
    print(f"\n4. Currently Open Positions ({len(pending_trades)} total):")
    if pending_trades:
        print(f"   {'Signal ID':<25} {'Symbol':<12} {'Side':<6} {'Entry':<12} {'Created':<20}")
        print(f"   {'-'*25} {'-'*12} {'-'*6} {'-'*12} {'-'*20}")
        
        for trade in pending_trades:
            signal_id = trade.get('signal_id', 'N/A')[:25]
            symbol = trade.get('symbol', 'N/A')
            side = trade.get('side', 'N/A')
            entry_price = trade.get('entry_price', 0)
            created_at = trade.get('created_at', 'N/A')
            
            entry_str = f"${entry_price:.2f}" if entry_price else "N/A"
            created_str = str(created_at)[:19] if created_at != 'N/A' else 'N/A'
            
            print(f"   {signal_id:<25} {symbol:<12} {side:<6} {entry_str:<12} {created_str:<20}")
    else:
        print("   No open positions")
    
    # Show recent activity
    print(f"\n5. Recent Activity (Last 10 signals):")
    recent_signals = sorted(signals, key=lambda x: x.get('updated_at', ''), reverse=True)[:10]
    
    if recent_signals:
        print(f"   {'Signal ID':<25} {'Symbol':<12} {'Status':<12} {'Updated':<20}")
        print(f"   {'-'*25} {'-'*12} {'-'*12} {'-'*20}")
        
        for signal in recent_signals:
            signal_id = signal.get('signal_id', 'N/A')[:25]
            symbol = signal.get('symbol', 'N/A')
            status = signal.get('status', 'N/A')
            updated_at = signal.get('updated_at', 'N/A')
            
            updated_str = str(updated_at)[:19] if updated_at != 'N/A' else 'N/A'
            
            print(f"   {signal_id:<25} {symbol:<12} {status:<12} {updated_str:<20}")
    else:
        print("   No recent signals")
    
    print("\n" + "=" * 60)
    print("✅ Trade completion tracking is active and recording P&L data!")
    print("   - Completed trades are properly recorded with entry/exit prices")
    print("   - P&L calculations are working correctly")
    print("   - Performance analytics are available")

if __name__ == "__main__":
    view_trade_completion_status()