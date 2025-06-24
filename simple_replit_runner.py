#!/usr/bin/env python3
"""
Simple Replit runner for AI trading system
Just runs the AI trader periodically with basic error handling
"""

import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

def run_trading_session():
    """Run a single trading session"""
    try:
        from ai_trader import AITrader
        
        logging.info("🚀 Starting AI Trading Session")
        logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize trader
        trader = AITrader()
        
        # Run analysis and trading
        session_data = trader.analyze_and_trade(max_trades=2, min_confidence=6)
        
        if session_data:
            trades = session_data.get('trades_executed', 0)
            logging.info(f"✅ Session completed! Trades executed: {trades}")
        else:
            logging.info("⚠️ Session completed with no data")
            
    except Exception as e:
        logging.error(f"❌ Error in trading session: {e}")
        # Try to send error notification
        try:
            trader.notifier.send_message(f"❌ Trading session failed: {str(e)}")
        except:
            pass

def main():
    """Main function - runs trading sessions periodically"""
    print("🤖 Simple AI Trading Runner for Replit")
    print("=" * 50)
    
    # Check if we're on Replit
    if os.getenv('REPL_ID'):
        print("✅ Running on Replit")
    else:
        print("⚠️ Not running on Replit")
    
    # Run initial session
    run_trading_session()
    
    # Wait and run again (simple approach)
    while True:
        try:
            print(f"\n⏰ Waiting 4 hours until next session...")
            print(f"Next run at: {(datetime.now().replace(hour=datetime.now().hour + 4)).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Sleep for 4 hours (14400 seconds)
            time.sleep(14400)
            
            # Run another session
            run_trading_session()
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Waiting 1 hour before retrying...")
            time.sleep(3600)

if __name__ == "__main__":
    main() 