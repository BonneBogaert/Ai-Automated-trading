#!/usr/bin/env python3
"""
Replit-optimized scheduler for AI Trading System
Runs every hour to minimize API costs while maintaining effectiveness
"""

import time
import schedule
import logging
from datetime import datetime, timedelta
from ai_trader import AITrader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

class ReplitScheduler:
    """Scheduler optimized for Replit deployment"""
    
    def __init__(self):
        self.trader = AITrader()
        self.last_run = None
        self.run_count = 0
        self.max_runs_per_day = 4  # Limit to save API costs
        
    def run_trading_session(self):
        """Execute a single trading session"""
        try:
            current_time = datetime.now()
            
            # Check if we've exceeded daily run limit
            if self.last_run and (current_time - self.last_run).days == 0:
                if self.run_count >= self.max_runs_per_day:
                    logging.info("Daily run limit reached. Skipping session.")
                    return
            
            # Check if market is open (simplified check - you can enhance this)
            if not self.is_market_open():
                logging.info("Market is closed. Skipping session.")
                return
            
            logging.info("="*60)
            logging.info("🚀 STARTING SCHEDULED AI TRADING SESSION")
            logging.info(f"Session #{self.run_count + 1} of {self.max_runs_per_day} today")
            logging.info(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info("="*60)
            
            # Run the trading session
            session_data = self.trader.analyze_and_trade(max_trades=2, min_confidence=6)
            
            # Update tracking
            self.last_run = current_time
            self.run_count += 1
            
            logging.info("="*60)
            logging.info("✅ SCHEDULED TRADING SESSION COMPLETED")
            logging.info(f"Trades executed: {session_data.get('trades_executed', 0) if session_data else 0}")
            logging.info("="*60)
            
        except Exception as e:
            logging.error(f"❌ Error in scheduled trading session: {e}")
            # Send error notification
            try:
                self.trader.notifier.send_message(f"❌ Scheduled trading session failed: {str(e)}")
            except:
                pass
    
    def is_market_open(self):
        """Check if market is open (simplified version)"""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM EST (simplified)
        # You can enhance this with proper timezone handling
        current_hour = now.hour
        if 9 <= current_hour < 16:
            return True
        
        return False
    
    def reset_daily_counter(self):
        """Reset daily run counter at midnight"""
        self.run_count = 0
        logging.info("🔄 Daily run counter reset")
    
    def start_scheduler(self):
        """Start the scheduling system"""
        logging.info("🤖 Starting Replit AI Trading Scheduler...")
        
        # Schedule trading sessions
        schedule.every().day.at("10:00").do(self.run_trading_session)  # Morning session
        schedule.every().day.at("12:00").do(self.run_trading_session)  # Midday session
        schedule.every().day.at("14:00").do(self.run_trading_session)  # Afternoon session
        schedule.every().day.at("15:30").do(self.run_trading_session)  # Pre-close session
        
        # Reset daily counter at midnight
        schedule.every().day.at("00:00").do(self.reset_daily_counter)
        
        # Send startup notification
        try:
            self.trader.notifier.send_message("🤖 Replit AI Trading Scheduler started successfully!")
        except:
            logging.warning("Could not send startup notification")
        
        logging.info("📅 Scheduler started with the following schedule:")
        logging.info("   - 10:00 AM: Morning trading session")
        logging.info("   - 12:00 PM: Midday trading session")
        logging.info("   - 02:00 PM: Afternoon trading session")
        logging.info("   - 03:30 PM: Pre-close trading session")
        logging.info("   - 00:00 AM: Daily counter reset")
        
        # Main scheduling loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logging.info("🛑 Scheduler stopped by user")
                break
            except Exception as e:
                logging.error(f"❌ Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

def main():
    """Main function to run the scheduler"""
    scheduler = ReplitScheduler()
    scheduler.start_scheduler()

if __name__ == "__main__":
    main() 