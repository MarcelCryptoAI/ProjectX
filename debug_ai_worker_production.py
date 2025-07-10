#!/usr/bin/env python3
"""
Debug AI Worker in Production
Check if the AI worker is running, monitoring trades, and detecting manual closures
"""
import os
import sys
import json
import requests
from datetime import datetime, timedelta

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_ai_worker_production():
    """Debug AI worker status in production"""
    print("=" * 80)
    print("AI WORKER PRODUCTION DEBUG")
    print("=" * 80)
    
    # Production URL
    production_url = "https://bybit-ai-bot-eu-d0c891a4972a.herokuapp.com"
    
    try:
        # 1. Check if AI worker is running
        print("\n1. CHECKING AI WORKER STATUS")
        print("-" * 40)
        
        try:
            response = requests.get(f"{production_url}/api/worker_status", timeout=10)
            if response.status_code == 200:
                worker_status = response.json()
                print(f"✅ AI Worker Status API accessible")
                print(f"   Worker Running: {worker_status.get('is_running', False)}")
                print(f"   Training in Progress: {worker_status.get('training_in_progress', False)}")
                print(f"   Signal Count: {worker_status.get('signal_count', 0)}")
                print(f"   Active Trades: {worker_status.get('active_trades', 0)}")
                print(f"   Max Trades: {worker_status.get('max_trades', 0)}")
                print(f"   Last Model Update: {worker_status.get('last_model_update', 'Never')}")
                
                if not worker_status.get('is_running', False):
                    print("❌ ISSUE: AI Worker is NOT running!")
                    print("   SOLUTION: Start the AI worker via /api/start_worker")
                    
            else:
                print(f"❌ Worker status API returned: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ Failed to check worker status: {e}")
        
        # 2. Check active trades monitoring
        print("\n2. CHECKING ACTIVE TRADES MONITORING")
        print("-" * 40)
        
        try:
            response = requests.get(f"{production_url}/api/active_trades_status", timeout=10)
            if response.status_code == 200:
                trades_status = response.json()
                print(f"✅ Active trades API accessible")
                print(f"   Message: {trades_status.get('message', 'N/A')}")
                trades = trades_status.get('trades', [])
                print(f"   Number of tracked trades: {len(trades)}")
                
                for i, trade in enumerate(trades[:3], 1):  # Show first 3 trades
                    print(f"   Trade {i}: {trade.get('symbol')} {trade.get('side')}")
                    print(f"     Entry Filled: {trade.get('entry_filled', False)}")
                    print(f"     TP1 Hit: {trade.get('tp1_hit', False)}")
                    print(f"     SL Moved: {trade.get('sl_moved_to_breakeven', False)}")
                    
            else:
                print(f"❌ Active trades API returned: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to check active trades: {e}")
        
        # 3. Check production analytics
        print("\n3. CHECKING PRODUCTION ANALYTICS")
        print("-" * 40)
        
        try:
            response = requests.get(f"{production_url}/api/production_analytics", timeout=15)
            if response.status_code == 200:
                analytics = response.json()
                print(f"✅ Production analytics API accessible")
                print(f"   Database Type: {analytics.get('database_type', 'Unknown')}")
                print(f"   Total Signals: {analytics.get('total_signals', 0)}")
                print(f"   Completed Signals: {analytics.get('completed_signals', 0)}")
                print(f"   Completed with P&L: {analytics.get('completed_with_pnl', 0)}")
                
                status_breakdown = analytics.get('status_breakdown', {})
                if status_breakdown:
                    print("   Status Breakdown:")
                    for status, count in status_breakdown.items():
                        print(f"     {status}: {count}")
                
                analytics_data = analytics.get('analytics_data', {})
                if analytics_data:
                    print("   Analytics Data:")
                    print(f"     Total Trades: {analytics_data.get('total_trades', 0)}")
                    print(f"     Win Rate: {analytics_data.get('win_rate', 0):.1f}%")
                    print(f"     Realized P&L All-time: {analytics_data.get('realized_pnl_all_time', 0):.4f} USDT")
                
                issues = analytics.get('issues', [])
                if issues:
                    print("   Issues Found:")
                    for issue in issues:
                        print(f"     ❌ {issue}")
                        
                recommendations = analytics.get('recommendations', [])
                if recommendations:
                    print("   Recommendations:")
                    for rec in recommendations:
                        print(f"     📝 {rec}")
                        
            else:
                print(f"❌ Production analytics API returned: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to check production analytics: {e}")
        
        # 4. Force trade monitoring
        print("\n4. FORCING TRADE MONITORING")
        print("-" * 40)
        
        try:
            response = requests.post(f"{production_url}/api/force_monitor_trades", timeout=15)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Force monitor executed successfully")
                print(f"   Message: {result.get('message', 'N/A')}")
            else:
                print(f"❌ Force monitor returned: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ Failed to force monitor trades: {e}")
        
        # 5. Check console logs
        print("\n5. CHECKING RECENT CONSOLE LOGS")
        print("-" * 40)
        
        try:
            response = requests.get(f"{production_url}/api/console_logs", timeout=10)
            if response.status_code == 200:
                logs = response.json()
                print(f"✅ Console logs API accessible")
                print(f"   Number of log entries: {len(logs)}")
                
                # Show last 10 log entries
                for log in logs[-10:]:
                    timestamp = log.get('timestamp', 'N/A')
                    level = log.get('level', 'INFO')
                    message = log.get('message', 'No message')
                    print(f"   [{timestamp}] {level}: {message}")
                    
            else:
                print(f"❌ Console logs API returned: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to check console logs: {e}")
        
        # 6. Summary and recommendations
        print("\n6. SUMMARY AND RECOMMENDATIONS")
        print("-" * 40)
        
        print("Based on the checks above:")
        print("\nIf AI Worker is NOT running:")
        print("  1. Start it via: POST /api/start_worker")
        print("  2. Check server logs for startup errors")
        
        print("\nIf completed signals show 0 P&L:")
        print("  1. Check monitor_trades_and_move_sl() function")
        print("  2. Verify position closure detection logic")
        print("  3. Test manual trade completion endpoint")
        
        print("\nIf monitoring isn't working:")
        print("  1. Check ByBit API connectivity")
        print("  2. Verify trade data structure")
        print("  3. Look for orphaned positions")
        
        print(f"\nProduction URL: {production_url}")
        print("Use the web interface to start/stop the AI worker and check real-time logs.")
        
    except Exception as e:
        print(f"❌ Major error during debugging: {e}")
        import traceback
        traceback.print_exc()

def test_manual_completion_endpoint():
    """Test creating a manual trade completion endpoint"""
    print("\n" + "=" * 80)
    print("TESTING MANUAL TRADE COMPLETION")
    print("=" * 80)
    
    production_url = "https://bybit-ai-bot-eu-d0c891a4972a.herokuapp.com"
    
    # Test data for manual completion
    test_data = {
        "signal_id": "test_manual_completion",
        "entry_price": 50000.0,
        "exit_price": 51000.0,
        "realized_pnl": 100.0
    }
    
    try:
        response = requests.post(
            f"{production_url}/api/manual_complete_trade",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Manual completion endpoint works")
            print(f"   Result: {result}")
        else:
            print(f"❌ Manual completion returned: {response.status_code}")
            print(f"   Response: {response.text}")
            print(f"   This endpoint might not exist yet - we should create it")
            
    except Exception as e:
        print(f"❌ Failed to test manual completion: {e}")
        print("   The endpoint likely doesn't exist yet")

if __name__ == "__main__":
    debug_ai_worker_production()
    test_manual_completion_endpoint()