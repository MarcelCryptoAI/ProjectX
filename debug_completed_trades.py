#!/usr/bin/env python3
"""
Debug why completed trades aren't showing up in production analytics
"""
import os
import sys
from datetime import datetime, timedelta

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import TradingDatabase

def debug_completed_trades():
    """Debug the completed trades issue"""
    print("=" * 80)
    print("COMPLETED TRADES DEBUG")
    print("=" * 80)
    
    # Initialize database
    db = TradingDatabase()
    print(f"✓ Database initialized")
    print(f"  Using PostgreSQL: {db.use_postgres}")
    
    # Check if we're in production
    if not db.use_postgres:
        print("⚠️  WARNING: Running on local SQLite database")
        print("   This is NOT the production database!")
        print("   Production uses PostgreSQL via DATABASE_URL environment variable")
        print("   Set DATABASE_URL to test production database")
        print()
    else:
        print("✓ Running on production PostgreSQL database")
        print()
    
    # 1. Check trading_signals table
    print("1. CHECKING TRADING SIGNALS TABLE")
    print("-" * 40)
    
    try:
        signals = db.get_trading_signals()
        print(f"✓ Total signals in database: {len(signals)}")
        
        if len(signals) == 0:
            print("❌ CRITICAL: No signals in database at all!")
            print("   This means:")
            print("   - The AI worker is not running")
            print("   - OR the AI worker is not generating signals")
            print("   - OR there's a database connection issue")
            return False
        
        # Analyze signal statuses
        status_counts = {}
        for signal in signals:
            status = signal.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("   Signal statuses:")
        for status, count in status_counts.items():
            print(f"     {status}: {count}")
        
        # Check completed signals
        completed_signals = [s for s in signals if s.get('status') == 'completed']
        completed_with_pnl = [s for s in completed_signals if s.get('realized_pnl') is not None]
        
        print(f"   Completed signals: {len(completed_signals)}")
        print(f"   Completed with P&L: {len(completed_with_pnl)}")
        
        if len(completed_signals) == 0:
            print("❌ ISSUE: No completed signals found!")
            print("   This means positions are not being detected as closed")
            print("   Possible causes:")
            print("   - AI worker monitor_trades_and_move_sl() not running")
            print("   - Position monitoring logic not working correctly")
            print("   - No positions are actually closing")
        
        elif len(completed_with_pnl) == 0:
            print("❌ ISSUE: Completed signals exist but no P&L data!")
            print("   This means positions are detected as closed but P&L calculation fails")
            print("   Possible causes:")
            print("   - P&L calculation logic has bugs")
            print("   - Execution history API calls failing")
            print("   - update_signal_with_pnl() not working")
        
        else:
            print("✅ Good: Found completed signals with P&L data")
            
            # Calculate totals
            total_pnl = sum(float(s.get('realized_pnl', 0)) for s in completed_with_pnl)
            winning_trades = len([s for s in completed_with_pnl if float(s.get('realized_pnl', 0)) > 0])
            losing_trades = len([s for s in completed_with_pnl if float(s.get('realized_pnl', 0)) < 0])
            
            print(f"   Total P&L: {total_pnl:.4f} USDT")
            print(f"   Winning trades: {winning_trades}")
            print(f"   Losing trades: {losing_trades}")
            
            # Check recent activity
            now = datetime.utcnow()
            recent_signals = []
            for signal in completed_with_pnl:
                if signal.get('exit_time'):
                    try:
                        exit_time_str = str(signal.get('exit_time'))
                        if exit_time_str:
                            # Handle different datetime formats
                            if 'T' in exit_time_str:
                                exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                            else:
                                exit_time = datetime.strptime(exit_time_str, '%Y-%m-%d %H:%M:%S')
                            
                            if exit_time.tzinfo is not None:
                                exit_time = exit_time.replace(tzinfo=None)
                            
                            if now - exit_time < timedelta(days=7):
                                recent_signals.append(signal)
                    except:
                        pass
            
            print(f"   Recent completed trades (last 7 days): {len(recent_signals)}")
        
    except Exception as e:
        print(f"❌ Error checking signals: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # 2. Test the analytics query logic
    print("2. TESTING ANALYTICS QUERY LOGIC")
    print("-" * 40)
    
    try:
        signals = db.get_trading_signals()
        
        # Replicate the exact logic from analytics_data endpoint
        completed_signals = [s for s in signals if s.get('status') == 'completed' and s.get('realized_pnl') is not None]
        
        print(f"✓ Analytics query finds {len(completed_signals)} completed trades")
        
        if len(completed_signals) > 0:
            # Calculate the same metrics as analytics
            total_trades = len(completed_signals)
            winning_trades = 0
            losing_trades = 0
            total_profit = 0
            total_loss = 0
            largest_win = 0
            largest_loss = 0
            
            # Calculate realized P&L for different periods
            realized_pnl_24h = 0
            realized_pnl_all_time = 0
            
            now = datetime.utcnow()
            yesterday = now - timedelta(days=1)
            
            for signal in completed_signals:
                pnl = float(signal.get('realized_pnl', 0))
                realized_pnl_all_time += pnl
                
                if pnl > 0:
                    winning_trades += 1
                    total_profit += pnl
                    largest_win = max(largest_win, pnl)
                elif pnl < 0:
                    losing_trades += 1
                    total_loss += abs(pnl)
                    largest_loss = min(largest_loss, pnl)
                
                # Check if trade was in last 24 hours
                if signal.get('exit_time'):
                    try:
                        exit_time_str = str(signal.get('exit_time'))
                        if 'T' in exit_time_str:
                            exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                        else:
                            exit_time = datetime.strptime(exit_time_str, '%Y-%m-%d %H:%M:%S')
                        
                        if exit_time.tzinfo is not None:
                            exit_time = exit_time.replace(tzinfo=None)
                        
                        if exit_time >= yesterday:
                            realized_pnl_24h += pnl
                    except:
                        pass
            
            print(f"   Analytics would show:")
            print(f"     Total trades: {total_trades}")
            print(f"     Winning trades: {winning_trades}")
            print(f"     Losing trades: {losing_trades}")
            print(f"     Total profit: {total_profit:.4f} USDT")
            print(f"     Total loss: {total_loss:.4f} USDT")
            print(f"     Largest win: {largest_win:.4f} USDT")
            print(f"     Largest loss: {largest_loss:.4f} USDT")
            print(f"     Realized P&L 24h: {realized_pnl_24h:.4f} USDT")
            print(f"     Realized P&L all-time: {realized_pnl_all_time:.4f} USDT")
            
            if total_trades > 0:
                win_rate = (winning_trades / total_trades) * 100
                print(f"     Win rate: {win_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Error testing analytics logic: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # 3. Show sample signals
    print("3. SAMPLE SIGNALS")
    print("-" * 40)
    
    try:
        signals = db.get_trading_signals()
        
        if signals:
            print("Most recent 5 signals:")
            for i, signal in enumerate(signals[:5]):
                print(f"   {i+1}. {signal['symbol']} - {signal['side']} - Status: {signal['status']}")
                print(f"      Signal ID: {signal['signal_id']}")
                print(f"      Created: {signal.get('created_at', 'N/A')}")
                print(f"      Entry Price: {signal.get('entry_price', 'N/A')}")
                print(f"      Exit Price: {signal.get('exit_price', 'N/A')}")
                print(f"      Realized P&L: {signal.get('realized_pnl', 'N/A')}")
                print(f"      Exit Time: {signal.get('exit_time', 'N/A')}")
                print()
        
    except Exception as e:
        print(f"❌ Error showing sample signals: {e}")
    
    # 4. Final recommendations
    print("4. RECOMMENDATIONS")
    print("-" * 40)
    
    try:
        signals = db.get_trading_signals()
        completed_with_pnl = [s for s in signals if s.get('status') == 'completed' and s.get('realized_pnl') is not None]
        
        if len(signals) == 0:
            print("❌ CRITICAL: Start the AI worker to generate signals")
            print("   Command: python ai_worker.py")
            
        elif len(completed_with_pnl) == 0:
            print("❌ CRITICAL: No completed trades with P&L")
            print("   Possible solutions:")
            print("   1. Check if AI worker is running and monitoring trades")
            print("   2. Verify that positions are actually closing")
            print("   3. Check if monitor_trades_and_move_sl() function is working")
            print("   4. Verify ByBit API access for position and execution data")
            print("   5. Check for errors in P&L calculation logic")
            
        else:
            print("✅ Database has completed trades - analytics should work")
            print("   If analytics still shows 0, check:")
            print("   1. Web app database connection")
            print("   2. Analytics endpoint error handling")
            print("   3. Database environment variables")
        
    except Exception as e:
        print(f"❌ Error in recommendations: {e}")
    
    return True

if __name__ == "__main__":
    debug_completed_trades()