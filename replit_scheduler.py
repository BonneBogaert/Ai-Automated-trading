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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('replit_scheduler.log'),
        logging.StreamHandler()
    ]
)

class ReplitTradingScheduler:
    """Optimized scheduler for Replit deployment"""
    
    def __init__(self):
        self.trader = AITrader()
        self.is_running = False
        self.session_count = 0
        self.max_sessions_per_day = 8  # Limit to 8 sessions per day to save API costs
        
    def should_run_session(self) -> bool:
        """Check if we should run a trading session based on various factors"""
        now = datetime.now()
        
        # Don't run on weekends
        if now.weekday() >= 5:
            print("📅 Weekend detected - skipping trading session")
            return False
        
        # Don't run outside market hours (9:30 AM - 4:00 PM ET)
        # Simplified check - you might want to use a proper market calendar
        hour = now.hour
        if hour < 9 or hour > 16:
            print(f"⏰ Outside market hours ({hour}:00) - skipping trading session")
            return False
        
        # Check session limit
        if self.session_count >= self.max_sessions_per_day:
            print(f"🛑 Daily session limit reached ({self.max_sessions_per_day}) - skipping")
            return False
        
        return True
    
    def run_trading_session(self):
        """Run a single trading session with comprehensive logging"""
        try:
            if not self.should_run_session():
                return
            
            print(f"\n{'='*60}")
            print(f"🚀 TRADING SESSION #{self.session_count + 1} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            # Send session start notification
            self.trader.notifier.send_message(f"🔄 Starting Trading Session #{self.session_count + 1}")
            
            # Run analysis and trading
            session_data = self.trader.analyze_and_trade(max_trades=2, min_confidence=8)  # Higher confidence threshold
            
            if session_data:
                self.session_count += 1
                print(f"✅ Session #{self.session_count} completed successfully!")
                
                # Send completion notification
                self.trader.notifier.send_message(f"✅ Trading Session #{self.session_count} completed!")
            else:
                print("⚠️ Session completed with no data")
                
        except Exception as e:
            error_msg = f"❌ Error in trading session: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            self.trader.notifier.send_message(error_msg)
    
    def daily_reset(self):
        """Reset daily counters and send daily summary"""
        try:
            print("🔄 Resetting daily counters...")
            self.session_count = 0
            
            # Get portfolio summary
            account = self.trader.get_account_info()
            positions = self.trader.get_portfolio_positions()
            
            if account:
                total_value = account.get('portfolio_value', 0)
                cash = account.get('cash', 0)
                active_positions = len(positions)
                
                # Calculate total P&L
                total_pl = sum(pos['unrealized_pl'] for pos in positions.values())
                
                daily_summary = f"""
📊 <b>Daily Trading Summary</b>

💰 <b>Portfolio Status:</b>
• Total Value: ${total_value:,.2f}
• Available Cash: ${cash:,.2f}
• Active Positions: {active_positions}
• Unrealized P&L: ${total_pl:,.2f}

📈 <b>Session Activity:</b>
• Sessions Run: {self.session_count}
• Max Sessions: {self.max_sessions_per_day}

📅 Date: {datetime.now().strftime('%Y-%m-%d')}
                """
                
                self.trader.notifier.send_message(daily_summary)
                print("✅ Daily reset completed")
            
        except Exception as e:
            print(f"❌ Error in daily reset: {e}")
    
    def setup_schedule(self):
        """Setup the optimized trading schedule"""
        print("📅 Setting up Replit-optimized trading schedule...")
        
        # Trading sessions every hour during market hours
        for hour in range(9, 17):  # 9 AM to 4 PM
            schedule.every().day.at(f"{hour:02d}:30").do(self.run_trading_session)
        
        # Daily reset at midnight
        schedule.every().day.at("00:00").do(self.daily_reset)
        
        # Market open notification
        schedule.every().monday.at("09:25").do(self.market_open_notification)
        schedule.every().tuesday.at("09:25").do(self.market_open_notification)
        schedule.every().wednesday.at("09:25").do(self.market_open_notification)
        schedule.every().thursday.at("09:25").do(self.market_open_notification)
        schedule.every().friday.at("09:25").do(self.market_open_notification)
        
        print("✅ Schedule setup completed!")
        print("📋 Scheduled sessions:")
        for job in schedule.jobs:
            print(f"   • {job.at_time} - {job.job_func.__name__}")
    
    def market_open_notification(self):
        """Send market open notification"""
        try:
            self.trader.notifier.send_message("🌅 Market is open - AI trader is active and monitoring for opportunities!")
        except Exception as e:
            print(f"❌ Error sending market open notification: {e}")
    
    def run(self):
        """Run the scheduler"""
        self.is_running = True
        self.setup_schedule()
        
        # Send startup notification
        startup_msg = f"""
🤖 <b>AI Trading Scheduler Started on Replit</b>

📊 <b>Configuration:</b>
• Sessions per day: {self.max_sessions_per_day}
• Market hours: 9:30 AM - 4:30 PM ET
• Confidence threshold: 8/10
• Max trades per session: 2

⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        self.trader.notifier.send_message(startup_msg)
        
        print("🚀 Replit Trading Scheduler started!")
        print("📋 Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\n🛑 Scheduler stopped by user")
            self.trader.notifier.send_message("🛑 AI Trading Scheduler stopped by user")
            self.is_running = False
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        print("🛑 Scheduler stopped")

def main():
    """Main function to run the Replit scheduler"""
    print("🚀 Starting Replit-optimized AI Trading Scheduler...")
    
    # Check if running on Replit
    if os.getenv('REPL_ID'):
        print("✅ Running on Replit - using optimized configuration")
    else:
        print("⚠️ Not running on Replit - some optimizations may not apply")
    
    scheduler = ReplitTradingScheduler()
    scheduler.run()

if __name__ == "__main__":
    main() 