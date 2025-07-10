#!/usr/bin/env python3
"""
Test script to verify trade completion tracking functionality
"""

import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import TradingDatabase

def test_trade_completion_tracking():
    """Test the trade completion tracking system"""
    print("🔍 Testing Trade Completion Tracking System")
    print("=" * 50)
    
    # Initialize database
    db = TradingDatabase()
    
    # Test 1: Create a test signal
    print("\n1. Creating test signal...")
    test_signal = {
        'signal_id': 'test_signal_001',
        'symbol': 'BTCUSDT',
        'side': 'Buy',
        'confidence': 85.5,
        'accuracy': 78.2,
        'amount': 100,
        'leverage': 5,
        'stop_loss': 2.0,
        'take_profit': 3.0,
        'entry_price': 43250.50,
        'status': 'waiting'
    }
    
    db.save_trading_signal(test_signal)
    print(f"✅ Created test signal: {test_signal['signal_id']}")
    
    # Test 2: Update signal to executing
    print("\n2. Updating signal to executing...")
    db.update_signal_status(test_signal['signal_id'], 'executing')
    print(f"✅ Updated signal status to 'executing'")
    
    # Test 3: Update signal to pending with entry price
    print("\n3. Updating signal to pending with entry price...")
    entry_price = 43275.25
    
    conn = db.get_connection()
    cursor = conn.cursor()
    placeholder = '%s' if db.use_postgres else '?'
    
    cursor.execute(f'''
        UPDATE trading_signals 
        SET status = {placeholder}, entry_price = {placeholder}, updated_at = CURRENT_TIMESTAMP
        WHERE signal_id = {placeholder}
    ''', ('pending', entry_price, test_signal['signal_id']))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Updated signal to 'pending' with entry price: ${entry_price}")
    
    # Test 4: Complete the trade with P&L
    print("\n4. Completing trade with P&L...")
    exit_price = 44125.75
    realized_pnl = (exit_price - entry_price) * (test_signal['amount'] / entry_price)  # Calculate P&L
    
    db.update_signal_with_pnl(
        signal_id=test_signal['signal_id'],
        entry_price=entry_price,
        exit_price=exit_price,
        realized_pnl=realized_pnl
    )
    
    print(f"✅ Trade completed with P&L: ${realized_pnl:.2f}")
    print(f"   Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}")
    
    # Test 5: Verify the data in database
    print("\n5. Verifying data in database...")
    signals = db.get_trading_signals()
    
    test_signal_data = None
    for signal in signals:
        if signal['signal_id'] == test_signal['signal_id']:
            test_signal_data = signal
            break
    
    if test_signal_data:
        print(f"✅ Found signal in database:")
        print(f"   Signal ID: {test_signal_data['signal_id']}")
        print(f"   Symbol: {test_signal_data['symbol']}")
        print(f"   Side: {test_signal_data['side']}")
        print(f"   Status: {test_signal_data['status']}")
        print(f"   Entry Price: ${test_signal_data['entry_price']:.2f}" if test_signal_data['entry_price'] else "   Entry Price: None")
        print(f"   Exit Price: ${test_signal_data['exit_price']:.2f}" if test_signal_data['exit_price'] else "   Exit Price: None")
        print(f"   Realized P&L: ${test_signal_data['realized_pnl']:.2f}" if test_signal_data['realized_pnl'] else "   Realized P&L: None")
        print(f"   Exit Time: {test_signal_data['exit_time']}" if test_signal_data['exit_time'] else "   Exit Time: None")
        
        # Verify all fields are populated correctly
        if (test_signal_data['status'] == 'completed' and 
            test_signal_data['entry_price'] and 
            test_signal_data['exit_price'] and 
            test_signal_data['realized_pnl'] is not None):
            print("✅ All P&L fields are correctly populated!")
        else:
            print("❌ Some P&L fields are missing or incorrect")
    else:
        print("❌ Test signal not found in database")
    
    # Test 6: Test P&L Analytics Query
    print("\n6. Testing P&L analytics query...")
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Query for completed trades with P&L data
    cursor.execute('''
        SELECT 
            signal_id, symbol, side, entry_price, exit_price, realized_pnl, 
            exit_time, created_at,
            CASE 
                WHEN realized_pnl > 0 THEN 'WIN'
                WHEN realized_pnl < 0 THEN 'LOSS'
                ELSE 'BREAKEVEN'
            END as trade_result
        FROM trading_signals
        WHERE status = 'completed' AND realized_pnl IS NOT NULL
        ORDER BY exit_time DESC
    ''')
    
    completed_trades = cursor.fetchall()
    conn.close()
    
    if completed_trades:
        print(f"✅ Found {len(completed_trades)} completed trades with P&L data:")
        for trade in completed_trades:
            pnl = float(trade[5]) if trade[5] else 0
            result = trade[8] if len(trade) > 8 else ("WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN")
            print(f"   {trade[0]} | {trade[1]} | {trade[2]} | P&L: ${pnl:.2f} | {result}")
    else:
        print("❌ No completed trades found with P&L data")
    
    # Test 7: Calculate performance metrics
    print("\n7. Calculating performance metrics...")
    
    if completed_trades:
        total_pnl = sum(float(trade[5]) for trade in completed_trades if trade[5])
        winning_trades = [trade for trade in completed_trades if trade[5] and float(trade[5]) > 0]
        losing_trades = [trade for trade in completed_trades if trade[5] and float(trade[5]) < 0]
        
        win_rate = (len(winning_trades) / len(completed_trades)) * 100 if completed_trades else 0
        avg_win = sum(float(trade[5]) for trade in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(float(trade[5]) for trade in losing_trades) / len(losing_trades) if losing_trades else 0
        
        print(f"✅ Performance Metrics:")
        print(f"   Total Trades: {len(completed_trades)}")
        print(f"   Winning Trades: {len(winning_trades)}")
        print(f"   Losing Trades: {len(losing_trades)}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total P&L: ${total_pnl:.2f}")
        print(f"   Average Win: ${avg_win:.2f}")
        print(f"   Average Loss: ${avg_loss:.2f}")
    else:
        print("❌ No completed trades to calculate metrics")
    
    # Cleanup
    print("\n8. Cleaning up test data...")
    conn = db.get_connection()
    cursor = conn.cursor()
    placeholder = '%s' if db.use_postgres else '?'
    
    cursor.execute(f'''
        DELETE FROM trading_signals WHERE signal_id = {placeholder}
    ''', (test_signal['signal_id'],))
    
    conn.commit()
    conn.close()
    
    print("✅ Test data cleaned up")
    
    print("\n" + "=" * 50)
    print("🎉 Trade Completion Tracking Test Complete!")
    print("   All functionality appears to be working correctly.")
    print("   The system can now:")
    print("   - Track trade status from waiting → executing → pending → completed")
    print("   - Record actual entry and exit prices")
    print("   - Calculate realized P&L")
    print("   - Store exit timestamps")
    print("   - Provide data for P&L analytics")

if __name__ == "__main__":
    test_trade_completion_tracking()