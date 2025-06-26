#!/usr/bin/env python3
"""
Manual AI Trading Session
Run this script to manually execute an AI trading session with your portfolio.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run a manual AI trading session"""
    try:
        print("🤖 Starting Manual AI Trading Session...")
        print("=" * 60)
        
        # Import AITrader
        from ai_trader import AITrader
        
        # Initialize the trader
        trader = AITrader()
        
        # Get current portfolio status
        account = trader.get_account_info()
        if account:
            print(f"💰 Current Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
            print(f"💵 Available Cash: ${account.get('cash', 0):,.2f}")
        
        positions = trader.get_portfolio_positions()
        if positions:
            print(f"📈 Active Positions: {len(positions)}")
            for ticker, pos in positions.items():
                print(f"   {ticker}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
        else:
            print("📈 No active positions")
        
        print("\n" + "=" * 60)
        
        # Ask user for trading parameters
        print("🎯 Trading Session Configuration:")
        try:
            max_trades = int(input("Maximum trades to execute (default: 3): ") or "3")
            min_confidence = int(input("Minimum confidence score (1-10, default: 6): ") or "6")
        except ValueError:
            print("⚠️ Using default values due to invalid input")
            max_trades = 3
            min_confidence = 6
        
        print(f"\n🚀 Starting AI analysis with max_trades={max_trades}, min_confidence={min_confidence}")
        print("=" * 60)
        
        # Run the trading session
        session_data = trader.analyze_and_trade(
            max_trades=3,
            min_confidence=8
        )
        
        if session_data:
            print("\n" + "=" * 60)
            print("✅ MANUAL TRADING SESSION COMPLETED")
            print("=" * 60)
            print(f"📊 Stocks Analyzed: {session_data.get('stocks_analyzed', 0)}")
            print(f"🔄 Trades Executed: {session_data.get('trades_executed', 0)}")
            print(f"📈 Buy Orders: {session_data.get('buy_orders', 0)}")
            print(f"📉 Sell Orders: {session_data.get('sell_orders', 0)}")
            
            # Show top decisions
            top_decisions = session_data.get('top_decisions', [])
            if top_decisions:
                print(f"\n🏆 Top Trading Decisions:")
                for i, decision in enumerate(top_decisions[:3], 1):
                    emoji = "🟢" if decision['action'] == "BUY" else "🔴" if decision['action'] == "SELL" else "🟡"
                    print(f"   {i}. {emoji} {decision['ticker']}: {decision['action']} (Confidence: {decision['confidence']}/10)")
            
            # Ask if user wants to generate a report
            generate_report = input("\n📄 Generate PDF report? (y/n, default: y): ").lower() != 'n'
            if generate_report:
                print("📊 Generating PDF report...")
                pdf_buffer = trader.generate_position_report_pdf()
                if pdf_buffer:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"manual_trading_report_{timestamp}.pdf"

                    os.makedirs('reports', exist_ok=True)
                    filename = os.path.join('reports', filename)
                    with open(filename, "wb") as f:
                        f.write(pdf_buffer.getvalue())
                        print(f"✅ Daily portfolio report saved as: {filename}")
                        print(f"📁 File location: {os.path.abspath(filename)}")                    
                        print(f"✅ Trading report saved as: {filename}")
                else:
                    print("❌ Failed to generate trading report")
        else:
            print("❌ Trading session failed or returned no data")
            
    except KeyboardInterrupt:
        print("\n⚠️ Manual trading session interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error in manual trading session: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 