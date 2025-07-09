#!/usr/bin/env python3
"""
Test symbols API endpoints directly
"""
import os
import sys
from datetime import datetime
import json

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import TradingDatabase

def test_symbols_api():
    """Test the symbols API functionality"""
    try:
        print("=" * 60)
        print("TESTING SYMBOLS API LOGIC")
        print("=" * 60)
        
        # Initialize database
        db = TradingDatabase()
        
        print("\n1. Testing get_supported_symbols():")
        symbols = db.get_supported_symbols()
        print(f"   ✓ Found {len(symbols)} symbols")
        if symbols:
            print(f"   ✓ First symbol: {symbols[0]['symbol']}")
            print(f"   ✓ Sample symbol data: {symbols[0]}")
        
        print("\n2. Testing get_symbols_last_updated():")
        last_updated = db.get_symbols_last_updated()
        print(f"   ✓ Last updated: {last_updated}")
        print(f"   ✓ Type: {type(last_updated)}")
        
        print("\n3. Testing API response construction:")
        # Handle last_updated properly - could be string or datetime
        last_updated_str = None
        if last_updated:
            if hasattr(last_updated, 'isoformat'):
                last_updated_str = last_updated.isoformat()
            else:
                last_updated_str = str(last_updated)
        
        api_response = {
            'success': True,
            'last_updated': last_updated_str,
            'count': len(symbols)
        }
        
        print(f"   ✓ API Response: {json.dumps(api_response, indent=2)}")
        
        print("\n4. Testing active symbols count:")
        active_symbols = [s for s in symbols if s.get('status') == 'active']
        inactive_symbols = [s for s in symbols if s.get('status') != 'active']
        
        print(f"   ✓ Active symbols: {len(active_symbols)}")
        print(f"   ✓ Inactive symbols: {len(inactive_symbols)}")
        
        print("\n5. Testing training symbols logic:")
        # This mimics the /api/training_symbols endpoint
        if active_symbols:
            active_symbol_names = [s['symbol'] for s in active_symbols]
            print(f"   ✓ Using {len(active_symbol_names)} active symbols from database")
            print(f"   ✓ Sample active symbols: {active_symbol_names[:10]}")
        else:
            print("   ⚠️  No active symbols found - would fall back to settings")
        
        print("\n" + "=" * 60)
        print("TESTING COMPLETE")
        print("=" * 60)
        
        if len(symbols) > 0 and last_updated:
            print("✅ All API endpoints should work correctly")
            print(f"✅ Count: {len(symbols)}")
            print(f"✅ Last Updated: {last_updated_str}")
        else:
            print("❌ Issues found:")
            if len(symbols) == 0:
                print("   - No symbols in database")
            if not last_updated:
                print("   - No last updated timestamp")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing symbols API: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_symbols_api()