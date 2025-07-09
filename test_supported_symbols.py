#!/usr/bin/env python3
"""
Test script to check and populate supported symbols in the database.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import TradingDatabase

def test_supported_symbols():
    """Test the supported symbols functionality"""
    try:
        db = TradingDatabase()
        
        # Check current symbols
        print("🔍 Checking current supported symbols in database...")
        symbols = db.get_supported_symbols()
        print(f"Found {len(symbols)} symbols in database")
        
        if symbols:
            print("\n📊 Current symbols:")
            for symbol in symbols[:10]:  # Show first 10
                print(f"  {symbol['symbol']} - {symbol['status']}")
            if len(symbols) > 10:
                print(f"  ... and {len(symbols) - 10} more")
        else:
            print("\n❌ No symbols found in database")
            print("🔄 Adding some default symbols for testing...")
            
            # Add some default symbols
            default_symbols = [
                {'symbol': 'BTCUSDT', 'base_currency': 'BTC', 'quote_currency': 'USDT', 'status': 'active', 'min_order_qty': 0.001, 'qty_step': 0.001, 'min_leverage': 1, 'max_leverage': 100},
                {'symbol': 'ETHUSDT', 'base_currency': 'ETH', 'quote_currency': 'USDT', 'status': 'active', 'min_order_qty': 0.01, 'qty_step': 0.01, 'min_leverage': 1, 'max_leverage': 100},
                {'symbol': 'ADAUSDT', 'base_currency': 'ADA', 'quote_currency': 'USDT', 'status': 'active', 'min_order_qty': 1, 'qty_step': 1, 'min_leverage': 1, 'max_leverage': 50},
                {'symbol': 'SOLUSDT', 'base_currency': 'SOL', 'quote_currency': 'USDT', 'status': 'active', 'min_order_qty': 0.1, 'qty_step': 0.1, 'min_leverage': 1, 'max_leverage': 75},
                {'symbol': 'DOTUSDT', 'base_currency': 'DOT', 'quote_currency': 'USDT', 'status': 'active', 'min_order_qty': 0.1, 'qty_step': 0.1, 'min_leverage': 1, 'max_leverage': 50},
            ]
            
            db.refresh_supported_symbols(default_symbols)
            print(f"✅ Added {len(default_symbols)} default symbols")
        
        # Check last updated
        last_updated = db.get_symbols_last_updated()
        print(f"\n📅 Last updated: {last_updated}")
        
        # Test the AI worker integration
        print("\n🤖 Testing AI worker integration...")
        from ai_worker import AIWorker
        worker = AIWorker()
        training_symbols = worker.get_supported_symbols()
        print(f"AI worker found {len(training_symbols)} symbols for training:")
        for symbol in training_symbols[:10]:
            print(f"  {symbol}")
        if len(training_symbols) > 10:
            print(f"  ... and {len(training_symbols) - 10} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing supported symbols functionality...")
    success = test_supported_symbols()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)