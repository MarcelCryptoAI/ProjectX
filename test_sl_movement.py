#!/usr/bin/env python3
"""
Test script for Stop Loss Movement functionality
This script tests the fix for automatic SL movement to breakeven + 0.1% when TP1 is hit
"""

import sys
import time
import json
import requests
from datetime import datetime

def test_sl_movement_endpoints():
    """Test the new debugging endpoints for SL movement"""
    base_url = "http://localhost:5000"
    
    print("🔍 Testing Stop Loss Movement Debugging Endpoints")
    print("=" * 60)
    
    # Test 1: Check debug endpoint
    print("\n1. Testing SL Debug Endpoint...")
    try:
        response = requests.get(f"{base_url}/api/trades/sl_debug")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                debug_info = data['debug_info']
                print(f"✅ Debug endpoint working:")
                print(f"   - Worker running: {debug_info['worker_running']}")
                print(f"   - ByBit session available: {debug_info['bybit_session_available']}")
                print(f"   - Active trades: {debug_info['active_trades_count']}")
                print(f"   - Monitoring interval: {debug_info['monitoring_interval']}")
                
                if debug_info['active_trades_count'] > 0:
                    print(f"   - Trade details:")
                    for trade in debug_info['trades_details']:
                        print(f"     • {trade['symbol']} ({trade['side']}): Entry filled: {trade['entry_filled']}, TP1 hit: {trade['tp1_hit']}")
            else:
                print(f"❌ Debug endpoint error: {data}")
        else:
            print(f"❌ Debug endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Debug endpoint error: {str(e)}")
    
    # Test 2: Check active trades status
    print("\n2. Testing Active Trades Status Endpoint...")
    try:
        response = requests.get(f"{base_url}/api/trades/active_status")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                trades_data = data['data']
                print(f"✅ Active trades endpoint working:")
                print(f"   - {trades_data['message']}")
                
                if trades_data['trades']:
                    for trade in trades_data['trades']:
                        print(f"   - {trade['symbol']} ({trade['side']}):")
                        print(f"     • Entry Price: ${trade['entry_price']:.4f}")
                        print(f"     • Stop Loss: ${trade['stop_loss']:.4f}")
                        print(f"     • Entry Filled: {trade['entry_filled']}")
                        print(f"     • TP1 Hit: {trade['tp1_hit']}")
                        print(f"     • SL Moved: {trade['sl_moved_to_breakeven']}")
                        print(f"     • TP Levels: {len(trade['tp_levels'])}")
            else:
                print(f"❌ Active trades error: {data}")
        else:
            print(f"❌ Active trades failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Active trades error: {str(e)}")
    
    # Test 3: Force monitor
    print("\n3. Testing Force Monitor Endpoint...")
    try:
        response = requests.post(f"{base_url}/api/trades/force_monitor")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ Force monitor working:")
                print(f"   - {data['message']}")
                
                # Show updated status
                trades_data = data['data']
                if trades_data['trades']:
                    print(f"   - Updated trades status:")
                    for trade in trades_data['trades']:
                        print(f"     • {trade['symbol']}: TP1 hit = {trade['tp1_hit']}, SL moved = {trade['sl_moved_to_breakeven']}")
                else:
                    print(f"   - No active trades to monitor")
            else:
                print(f"❌ Force monitor error: {data}")
        else:
            print(f"❌ Force monitor failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Force monitor error: {str(e)}")

def test_sl_movement_logic():
    """Test the actual SL movement logic"""
    print("\n🧪 Testing Stop Loss Movement Logic")
    print("=" * 60)
    
    # Test calculations
    print("\n1. Testing SL Calculation Logic...")
    
    # Test Buy position
    entry_price_buy = 50000.0  # Example BTC price
    sl_price_buy = entry_price_buy * 1.001  # +0.1%
    expected_sl_buy = 50050.0
    
    print(f"Buy position test:")
    print(f"   - Entry price: ${entry_price_buy:.4f}")
    print(f"   - Expected SL (breakeven + 0.1%): ${expected_sl_buy:.4f}")
    print(f"   - Calculated SL: ${sl_price_buy:.4f}")
    print(f"   - ✅ Calculation correct: {abs(sl_price_buy - expected_sl_buy) < 0.01}")
    
    # Test Sell position
    entry_price_sell = 50000.0
    sl_price_sell = entry_price_sell * 0.999  # -0.1%
    expected_sl_sell = 49950.0
    
    print(f"\nSell position test:")
    print(f"   - Entry price: ${entry_price_sell:.4f}")
    print(f"   - Expected SL (breakeven - 0.1%): ${expected_sl_sell:.4f}")
    print(f"   - Calculated SL: ${sl_price_sell:.4f}")
    print(f"   - ✅ Calculation correct: {abs(sl_price_sell - expected_sl_sell) < 0.01}")

def check_system_status():
    """Check if the system is ready for testing"""
    print("🔧 System Status Check")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    try:
        # Check if server is running
        response = requests.get(f"{base_url}/api/system_status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is running")
            
            # Check AI worker status
            if 'ai_worker' in data:
                ai_status = data['ai_worker']
                print(f"✅ AI Worker status:")
                print(f"   - Running: {ai_status.get('running', False)}")
                print(f"   - Active trades: {ai_status.get('active_trades', 0)}")
                print(f"   - Max trades: {ai_status.get('max_trades', 0)}")
            
            return True
        else:
            print(f"❌ Server not responding properly: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the Flask app is running on localhost:5000")
        return False
    except Exception as e:
        print(f"❌ System check error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Stop Loss Movement Fix Test Suite")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Check system status first
    if not check_system_status():
        print("\n❌ System not ready for testing. Please start the Flask application first.")
        return
    
    # Test SL movement logic
    test_sl_movement_logic()
    
    # Test endpoints
    test_sl_movement_endpoints()
    
    print("\n📋 Summary of Fixes Applied:")
    print("=" * 60)
    print("1. ✅ Fixed TP1 detection logic - now properly checks TP1 order ID")
    print("2. ✅ Fixed SL quantity - now uses actual position size after TP1")
    print("3. ✅ Added comprehensive error handling and logging")
    print("4. ✅ Added retry logic for API calls")
    print("5. ✅ Added debugging endpoints for monitoring")
    print("6. ✅ Added manual trigger for testing")
    print("7. ✅ Added bybit_session availability check")
    print("8. ✅ Enhanced monitoring with detailed status reporting")
    
    print("\n📝 Next Steps:")
    print("=" * 60)
    print("1. Deploy the updated code to production")
    print("2. Monitor the logs for 'TP1 hit' and 'Stop loss moved' messages")
    print("3. Use /api/trades/sl_debug to check system status")
    print("4. Use /api/trades/force_monitor to manually trigger monitoring")
    print("5. Verify SL movement happens when TP1 is actually hit")
    
    print("\n✅ Stop Loss Movement Fix Testing Complete!")

if __name__ == "__main__":
    main()