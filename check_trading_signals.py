#!/usr/bin/env python3
"""
Check trading signals table and analyze completed trades
"""
import os
import sys
from datetime import datetime

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import TradingDatabase

def check_trading_signals():
    """Check the trading signals table and analyze completed trades"""
    try:
        print("=" * 60)
        print("TRADING SIGNALS TABLE ANALYSIS")
        print("=" * 60)
        
        # Initialize database
        db = TradingDatabase()
        print(f"✓ Database initialized")
        print(f"  Using PostgreSQL: {db.use_postgres}")
        
        # Get all trading signals
        signals = db.get_trading_signals()
        print(f"✓ Total signals in database: {len(signals)}")
        
        if not signals:
            print("⚠️  No trading signals found in database!")
            return False
        
        # Analyze signal statuses
        status_counts = {}
        completed_with_pnl = []
        completed_without_pnl = []
        
        for signal in signals:
            status = signal.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if status == 'completed':
                if signal.get('realized_pnl') is not None:
                    completed_with_pnl.append(signal)
                else:
                    completed_without_pnl.append(signal)
        
        print("\n" + "=" * 60)
        print("SIGNAL STATUS BREAKDOWN")
        print("=" * 60)
        
        for status, count in status_counts.items():
            print(f"  {status}: {count} signals")
        
        print(f"\n✓ Completed signals with P&L data: {len(completed_with_pnl)}")
        print(f"⚠️  Completed signals WITHOUT P&L data: {len(completed_without_pnl)}")
        
        # Show sample signals
        print("\n" + "=" * 60)
        print("RECENT SIGNALS (Last 10)")
        print("=" * 60)
        
        for i, signal in enumerate(signals[:10]):
            print(f"\n{i+1}. Signal: {signal['signal_id']}")
            print(f"   Symbol: {signal['symbol']}")
            print(f"   Side: {signal['side']}")
            print(f"   Status: {signal['status']}")
            print(f"   Entry Price: {signal.get('entry_price', 'N/A')}")
            print(f"   Exit Price: {signal.get('exit_price', 'N/A')}")
            print(f"   Realized P&L: {signal.get('realized_pnl', 'N/A')}")
            print(f"   Created: {signal.get('created_at', 'N/A')}")
            print(f"   Exit Time: {signal.get('exit_time', 'N/A')}")
        
        # Analyze completed trades with P&L
        if completed_with_pnl:
            print("\n" + "=" * 60)
            print("COMPLETED TRADES WITH P&L ANALYSIS")
            print("=" * 60)
            
            total_pnl = 0
            winning_trades = 0
            losing_trades = 0
            
            for signal in completed_with_pnl:
                pnl = float(signal.get('realized_pnl', 0))
                total_pnl += pnl
                
                if pnl > 0:
                    winning_trades += 1
                elif pnl < 0:
                    losing_trades += 1
            
            print(f"✓ Total completed trades with P&L: {len(completed_with_pnl)}")
            print(f"✓ Winning trades: {winning_trades}")
            print(f"✓ Losing trades: {losing_trades}")
            print(f"✓ Total realized P&L: {total_pnl:.4f} USDT")
            
            if len(completed_with_pnl) > 0:
                win_rate = (winning_trades / len(completed_with_pnl)) * 100
                print(f"✓ Win rate: {win_rate:.1f}%")
        
        # Check for missing P&L data
        if completed_without_pnl:
            print("\n" + "=" * 60)
            print("COMPLETED TRADES WITHOUT P&L DATA")
            print("=" * 60)
            
            print(f"⚠️  Found {len(completed_without_pnl)} completed trades without P&L data:")
            for signal in completed_without_pnl[:5]:  # Show first 5
                print(f"   - {signal['signal_id']} ({signal['symbol']}) - Status: {signal['status']}")
            
            if len(completed_without_pnl) > 5:
                print(f"   ... and {len(completed_without_pnl) - 5} more")
        
        # Direct database query to check table structure
        print("\n" + "=" * 60)
        print("DATABASE TABLE STRUCTURE")
        print("=" * 60)
        
        conn = db.get_connection()
        cursor = conn.cursor()
        
        try:
            if db.use_postgres:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable 
                    FROM information_schema.columns 
                    WHERE table_name = 'trading_signals' 
                    ORDER BY ordinal_position
                """)
                columns = cursor.fetchall()
                print("✓ PostgreSQL table structure:")
                for col in columns:
                    print(f"  - {col[0]} ({col[1]}) - Nullable: {col[2]}")
            else:
                cursor.execute("PRAGMA table_info(trading_signals)")
                columns = cursor.fetchall()
                print("✓ SQLite table structure:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]}) - Nullable: {col[3] == 0}")
        
        except Exception as e:
            print(f"❌ Error checking table structure: {e}")
        
        conn.close()
        
        # Recommendations
        print("\n" + "=" * 60)
        print("ANALYSIS & RECOMMENDATIONS")
        print("=" * 60)
        
        if len(completed_with_pnl) == 0:
            print("❌ CRITICAL ISSUE: No completed trades with P&L data found!")
            print("   This explains why analytics shows 0 trades.")
            print("   Possible causes:")
            print("   1. Trades are not being marked as 'completed' status")
            print("   2. P&L data is not being saved when trades complete")
            print("   3. The bot is not actually completing trades")
            print("   4. There's an issue with the trade completion logic")
        
        elif len(completed_with_pnl) < 10:
            print("⚠️  WARNING: Very few completed trades with P&L data")
            print("   This might indicate:")
            print("   1. The bot is running but not completing many trades")
            print("   2. Some trades are completing but not being recorded properly")
        
        else:
            print("✅ Good: Found completed trades with P&L data")
            print("   The analytics should show these trades")
        
        if completed_without_pnl:
            print(f"\n⚠️  ISSUE: {len(completed_without_pnl)} completed trades lack P&L data")
            print("   These trades completed but P&L wasn't recorded")
            print("   This could be due to:")
            print("   1. Bug in trade completion logic")
            print("   2. Missing update_signal_with_pnl() calls")
            print("   3. API issues when retrieving final trade data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking trading signals: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_trading_signals()