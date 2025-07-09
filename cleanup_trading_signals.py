#!/usr/bin/env python3
"""
Script to cleanup old trading signals from the database.
Keeps only the most recent 30 signals and deletes all older ones.
"""

from database import TradingDatabase
from datetime import datetime
import sys

def cleanup_trading_signals():
    """Delete all trading signals except the most recent 30"""
    
    print("=" * 60)
    print("TRADING SIGNALS CLEANUP SCRIPT")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize database connection
        db = TradingDatabase()
        
        # Get current trading signals count
        signals = db.get_trading_signals()
        total_signals = len(signals)
        
        print(f"Current number of trading signals in database: {total_signals}")
        
        if total_signals <= 30:
            print("\n✓ No cleanup needed. Database has 30 or fewer signals.")
            return
        
        # Calculate how many signals to delete
        signals_to_delete = total_signals - 30
        
        print(f"\nDeleting {signals_to_delete} oldest signals...")
        print("Keeping the 30 most recent signals.")
        
        # Delete the oldest signals
        deleted_count = db.delete_oldest_trading_signals(signals_to_delete)
        
        if deleted_count > 0:
            print(f"\n✓ Successfully deleted {deleted_count} old trading signals.")
            
            # Verify the cleanup
            remaining_signals = db.get_trading_signals()
            remaining_count = len(remaining_signals)
            
            print(f"\nRemaining trading signals in database: {remaining_count}")
            
            if remaining_count == 30:
                print("✓ Cleanup completed successfully! Exactly 30 signals remain.")
            else:
                print(f"⚠ Warning: Expected 30 signals but found {remaining_count}")
        else:
            print("\n✗ No signals were deleted. Please check the database.")
            
    except Exception as e:
        print(f"\n✗ Error during cleanup: {str(e)}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    cleanup_trading_signals()