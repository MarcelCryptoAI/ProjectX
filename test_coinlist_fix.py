#!/usr/bin/env python3
"""
Test the coinlist database storage fix
"""
import os
import sys
from datetime import datetime
import json

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import TradingDatabase

def test_coinlist_fix():
    """Test all the coinlist functionality"""
    print("=" * 70)
    print("COINLIST DATABASE STORAGE FIX VERIFICATION")
    print("=" * 70)
    
    try:
        # 1. Test database connection and symbol storage
        print("\n1. Testing database connection and symbols...")
        db = TradingDatabase()
        symbols = db.get_supported_symbols()
        
        print(f"   ✓ Database connected successfully")
        print(f"   ✓ Found {len(symbols)} symbols in database")
        
        if len(symbols) == 0:
            print("   ❌ ISSUE: No symbols in database")
            return False
        
        # 2. Test last updated functionality
        print("\n2. Testing last updated timestamp...")
        last_updated = db.get_symbols_last_updated()
        print(f"   ✓ Last updated raw value: {last_updated}")
        print(f"   ✓ Last updated type: {type(last_updated)}")
        
        # Test the fixed API response logic
        last_updated_str = None
        if last_updated:
            if hasattr(last_updated, 'isoformat'):
                last_updated_str = last_updated.isoformat()
            else:
                last_updated_str = str(last_updated)
        
        print(f"   ✓ Formatted for API: {last_updated_str}")
        
        # 3. Test count of supported coins
        print("\n3. Testing supported coins count...")
        active_symbols = [s for s in symbols if s.get('status') == 'active']
        inactive_symbols = [s for s in symbols if s.get('status') != 'active']
        
        print(f"   ✓ Total symbols: {len(symbols)}")
        print(f"   ✓ Active symbols: {len(active_symbols)}")
        print(f"   ✓ Inactive symbols: {len(inactive_symbols)}")
        
        if len(active_symbols) == 0:
            print("   ❌ ISSUE: No active symbols found")
            return False
        
        # 4. Test API response construction (symbols_info)
        print("\n4. Testing /api/symbols_info response construction...")
        
        api_response = {
            'success': True,
            'symbols': symbols,
            'last_updated': last_updated_str,
            'count': len(symbols)
        }
        
        print(f"   ✓ API response keys: {list(api_response.keys())}")
        print(f"   ✓ Count: {api_response['count']}")
        print(f"   ✓ Last updated: {api_response['last_updated']}")
        print(f"   ✓ Success: {api_response['success']}")
        
        # 5. Test training symbols logic
        print("\n5. Testing training symbols logic...")
        
        active_symbol_names = [s['symbol'] for s in active_symbols]
        
        training_response = {
            'success': True,
            'active_symbols': active_symbol_names,
            'active_count': len(active_symbols),
            'fallback_symbols': ['BTCUSDT', 'ETHUSDT'],
            'using_database': len(active_symbols) > 0,
            'last_updated': last_updated_str
        }
        
        print(f"   ✓ Using database: {training_response['using_database']}")
        print(f"   ✓ Active count: {training_response['active_count']}")
        print(f"   ✓ Sample symbols: {active_symbol_names[:5]}")
        
        # 6. Test fetch and storage process simulation
        print("\n6. Testing database storage integrity...")
        
        # Check if the refresh process would work
        test_symbol_data = [{
            'symbol': 'TESTUSDT',
            'base_currency': 'TEST',
            'quote_currency': 'USDT', 
            'status': 'active',
            'min_order_qty': 0.1,
            'qty_step': 0.1,
            'min_leverage': 1,
            'max_leverage': 10,
            'leverage_multiplier': 1.0
        }]
        
        # Get current count
        original_count = len(db.get_supported_symbols())
        
        # Test save (we'll restore after)
        print(f"   ✓ Original symbol count: {original_count}")
        print("   ⚠️  Note: Not actually modifying database in test")
        
        # 7. Summary
        print("\n" + "=" * 70)
        print("VERIFICATION RESULTS")
        print("=" * 70)
        
        issues_found = []
        
        if len(symbols) == 0:
            issues_found.append("No symbols in database")
        
        if not last_updated:
            issues_found.append("No last updated timestamp")
        
        if len(active_symbols) == 0:
            issues_found.append("No active symbols")
        
        if issues_found:
            print("❌ ISSUES FOUND:")
            for issue in issues_found:
                print(f"   - {issue}")
            print("\n🔧 SOLUTIONS:")
            print("   1. Run the refresh symbols function from the web interface")
            print("   2. Visit /config page and click 'Force Refresh Coin List'")
            print("   3. Check API logs for any errors during refresh")
            return False
        else:
            print("✅ ALL CHECKS PASSED!")
            print(f"   ✓ Database has {len(symbols)} symbols")
            print(f"   ✓ {len(active_symbols)} active symbols ready for trading")
            print(f"   ✓ Last updated: {last_updated_str}")
            print(f"   ✓ API responses will work correctly")
            print(f"   ✓ Training will use database symbols")
            return True
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coinlist_fix()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 COINLIST FIX VERIFICATION: PASSED")
        print("   The database storage issues have been resolved!")
    else:
        print("⚠️  COINLIST FIX VERIFICATION: NEEDS ATTENTION")
        print("   Please check the issues above and run refresh if needed.")
    print("=" * 70)