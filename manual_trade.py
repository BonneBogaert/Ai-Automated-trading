#!/usr/bin/env python3
"""
Manual Trading Script
Run this for a single trading session when you want to test or manually trigger trading
"""

from ai_trader import AITrader
from datetime import datetime

def main():
    """Run a single manual trading session"""
    print("🚀 Starting Manual Trading Session...")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Initialize trader
    trader = AITrader()
    
    # Send manual session notification
    trader.notifier.send_message("🎯 Manual Trading Session Started")
    
    # Run analysis and trading with conservative settings
    session_data = trader.analyze_and_trade(
        max_trades=2,      # Limit to 2 trades for manual sessions
        min_confidence=8   # Higher confidence threshold for manual sessions
    )
    
    if session_data:
        print(f"\n✅ Manual session completed!")
        print(f"📊 Stocks analyzed: {session_data.get('stocks_analyzed', 0)}")
        print(f"📈 Trades executed: {session_data.get('trades_executed', 0)}")
    else:
        print("\n⚠️ Manual session completed with no data")
    
    # Send completion notification
    trader.notifier.send_message("✅ Manual Trading Session Completed")
    
    print("🎉 Manual trading session finished!")

if __name__ == "__main__":
    main() 