#\!/usr/bin/env python3
"""
Script to recalculate and fix realized P&L values in the database
"""

import os
from database import TradingDatabase

def fix_realized_pnl():
    """Recalculate and fix all realized P&L values"""
    db = TradingDatabase()
    
    # Get all completed signals with P&L data
    signals = db.get_trading_signals()
    
    fixed_count = 0
    
    for signal in signals:
        if signal['status'] == 'completed' and signal.get('entry_price') and signal.get('exit_price'):
            # Calculate correct P&L
            entry_price = float(signal['entry_price'])
            exit_price = float(signal['exit_price'])
            amount = float(signal['amount'])
            
            if signal['side'] == 'Buy':
                # For long positions: profit when price goes up
                correct_pnl = (exit_price - entry_price) * amount
            else:
                # For short positions: profit when price goes down
                correct_pnl = (entry_price - exit_price) * amount
            
            # Update if different from stored value
            current_pnl = signal.get('realized_pnl', 0)
            if current_pnl is None or abs(float(current_pnl) - correct_pnl) > 0.01:
                print(f"Fixing {signal['symbol']} {signal['side']}: "
                      f"Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}, "
                      f"Old P&L: ${current_pnl:.2f}, "
                      f"New P&L: ${correct_pnl:.2f}")
                
                # Update in database
                db.update_signal_with_pnl(
                    signal['signal_id'],
                    entry_price,
                    exit_price,
                    correct_pnl
                )
                fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} P&L values")
    
    # Show current totals
    all_signals = db.get_trading_signals()
    total_pnl = sum(s.get('realized_pnl', 0) for s in all_signals if s['status'] == 'completed' and s.get('realized_pnl'))
    
    print(f"\n📊 Current P&L totals:")
    print(f"All-time realized P&L: ${total_pnl:.2f}")

if __name__ == '__main__':
    fix_realized_pnl()
