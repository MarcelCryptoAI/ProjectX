#!/usr/bin/env python3
"""
Script to delete old trading signals from the database
"""

import os
import sys
from database import TradingDatabase

def main():
    """Delete the oldest 11 trading signals"""
    try:
        db = TradingDatabase()
        
        # Get current count
        current_signals = db.get_trading_signals()
        print(f"Current signals in database: {len(current_signals)}")
        
        # Delete oldest 11 signals
        deleted_count = db.delete_oldest_trading_signals(11)
        
        print(f"Deleted {deleted_count} old trading signals")
        
        # Get new count
        remaining_signals = db.get_trading_signals()
        print(f"Remaining signals in database: {len(remaining_signals)}")
        
        if len(remaining_signals) <= 5:
            print("✅ Database now has 5 or fewer signals for better analytics")
        else:
            print(f"⚠️  Still {len(remaining_signals)} signals remaining")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()