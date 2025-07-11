#!/usr/bin/env python3
"""
Script to check database status and symbol counts
"""
import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_singleton import get_database
from utils.settings_loader import Settings

def check_database_status():
    """Check the status of symbols in the database"""
    try:
        db = get_database()
        
        # Get all symbols from database
        symbols = db.get_supported_symbols()
        
        print(f"=== Database Symbol Status ===")
        print(f"Total symbols in database: {len(symbols)}")
        
        # Count by status
        active_count = len([s for s in symbols if s.get('status') == 'active'])
        inactive_count = len([s for s in symbols if s.get('status') == 'inactive'])
        other_count = len([s for s in symbols if s.get('status') not in ['active', 'inactive']])
        
        print(f"Active symbols: {active_count}")
        print(f"Inactive symbols: {inactive_count}")
        print(f"Other status: {other_count}")
        
        # Get last updated
        last_updated = db.get_symbols_last_updated()
        print(f"Last updated: {last_updated}")
        
        # Check settings file symbols
        try:
            settings = Settings.load('config/settings.yaml')
            enabled_pairs = settings.bot.get('enabled_pairs', [])
            print(f"\n=== Settings File Status ===")
            print(f"Enabled pairs in settings.yaml: {len(enabled_pairs)}")
        except Exception as e:
            print(f"Error loading settings: {e}")
        
        # Show first 10 active symbols from database
        if active_count > 0:
            active_symbols = [s['symbol'] for s in symbols if s.get('status') == 'active']
            print(f"\n=== First 10 Active Symbols ===")
            for i, symbol in enumerate(active_symbols[:10]):
                print(f"{i+1}. {symbol}")
        
        # Check if we're using fallback
        if active_count < 50:
            print(f"\n⚠️  WARNING: Only {active_count} active symbols in database!")
            print("This is likely causing the AI to use the 192 fallback symbols from settings.yaml")
            print("instead of the full database list.")
        else:
            print(f"\n✅ Database has sufficient symbols ({active_count} active)")
            
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    check_database_status()