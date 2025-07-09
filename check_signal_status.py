#!/usr/bin/env python3
"""
Quick script to check the status of trading signals in the database
"""

from database import TradingDatabase
from datetime import datetime

def check_signal_status():
    """Check and display trading signal statistics"""
    
    try:
        db = TradingDatabase()
        
        # Get all trading signals
        signals = db.get_trading_signals()
        
        print("=" * 60)
        print("TRADING SIGNALS DATABASE STATUS")
        print("=" * 60)
        print(f"Total signals in database: {len(signals)}")
        print()
        
        if signals:
            # Group by status
            status_counts = {}
            for signal in signals:
                status = signal.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print("Signals by status:")
            for status, count in sorted(status_counts.items()):
                print(f"  - {status}: {count}")
            
            print("\nMost recent 5 signals:")
            for i, signal in enumerate(signals[:5]):
                print(f"\n  Signal {i+1}:")
                print(f"    ID: {signal['signal_id']}")
                print(f"    Symbol: {signal['symbol']}")
                print(f"    Side: {signal['side']}")
                print(f"    Status: {signal['status']}")
                print(f"    Created: {signal['created_at']}")
        else:
            print("No trading signals found in the database.")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error checking signal status: {e}")

if __name__ == "__main__":
    check_signal_status()