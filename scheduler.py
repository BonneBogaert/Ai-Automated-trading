import schedule
import time
import logging
from datetime import datetime
from ai_trader import AITrader
from config import NOTIFICATION_CONFIG, SCREENING_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

class TradingScheduler:
    """Schedule and manage automated trading sessions"""
    
    def __init__(self):
        self.trader = AITrader()
        self.is_running = False
        
    def start_trading_session(self):
        """Run a complete trading session"""
        try:
            logging.info("Starting scheduled trading session...")
            self.trader.notifier.send_message("🔄 Starting scheduled trading session...")
            
            # Run analysis and trading
            self.trader.analyze_and_trade(max_trades=3, min_confidence=7)
            
            logging.info("Trading session completed successfully!")
            self.trader.notifier.send_message("✅ Trading session completed!")
            
        except Exception as e:
            error_msg = f"❌ Error in trading session: {str(e)}"
            logging.error(error_msg)
            if NOTIFICATION_CONFIG['send_error_alerts']:
                self.trader.notifier.send_message(error_msg)
    
    def daily_summary(self):
        """Send daily portfolio summary"""
        try:
            logging.info("Generating daily summary...")
            
            # Get current portfolio
            positions = self.trader.get_portfolio_positions()
            account = self.trader.get_account_info()
            
            if not account:
                return
            
            total_value = account.get('portfolio_value', 0)
            cash = account.get('cash', 0)
            
            # Calculate daily performance
            active_positions = len(positions)
            total_unrealized_pl = sum(pos['unrealized_pl'] for pos in positions.values())
            
            message = f"""
📊 <b>Daily Portfolio Summary</b>

Portfolio Value: ${total_value:,.2f}
Available Cash: ${cash:,.2f}
Active Positions: {active_positions}
Unrealized P&L: ${total_unrealized_pl:,.2f}

Date: {datetime.now().strftime('%Y-%m-%d')}
            """
            
            self.trader.notifier.send_message(message)
            logging.info("Daily summary sent successfully!")
            
        except Exception as e:
            logging.error(f"Error generating daily summary: {e}")
    
    def weekly_report(self):
        """Generate and send weekly report"""
        try:
            logging.info("Generating weekly report...")
            self.trader.generate_weekly_report()
            logging.info("Weekly report sent successfully!")
            
        except Exception as e:
            logging.error(f"Error generating weekly report: {e}")
    
    def market_open_check(self):
        """Check if market is open and send notification"""
        try:
            # Simple market hours check (US market: 9:30 AM - 4:00 PM ET)
            now = datetime.now()
            # This is a simplified check - you might want to use a proper market calendar
            if now.weekday() < 5:  # Monday to Friday
                self.trader.notifier.send_message("🌅 Market is open - AI trader is active!")
                
        except Exception as e:
            logging.error(f"Error in market open check: {e}")
    
    def setup_schedule(self):
        """Setup the trading schedule"""
        
        # Trading sessions (during market hours)
        schedule.every().monday.at("09:30").do(self.start_trading_session)
        schedule.every().tuesday.at("09:30").do(self.start_trading_session)
        schedule.every().wednesday.at("09:30").do(self.start_trading_session)
        schedule.every().thursday.at("09:30").do(self.start_trading_session)
        schedule.every().friday.at("09:30").do(self.start_trading_session)
        
        # Additional trading sessions during the day
        schedule.every().monday.at("12:00").do(self.start_trading_session)
        schedule.every().wednesday.at("12:00").do(self.start_trading_session)
        schedule.every().friday.at("12:00").do(self.start_trading_session)
        
        # Daily summary at end of trading day
        schedule.every().monday.at("16:30").do(self.daily_summary)
        schedule.every().tuesday.at("16:30").do(self.daily_summary)
        schedule.every().wednesday.at("16:30").do(self.daily_summary)
        schedule.every().thursday.at("16:30").do(self.daily_summary)
        schedule.every().friday.at("16:30").do(self.daily_summary)
        
        # Weekly report on Sunday evening
        schedule.every().sunday.at("20:00").do(self.weekly_report)
        
        # Market open notification
        schedule.every().monday.at("09:25").do(self.market_open_check)
        schedule.every().tuesday.at("09:25").do(self.market_open_check)
        schedule.every().wednesday.at("09:25").do(self.market_open_check)
        schedule.every().thursday.at("09:25").do(self.market_open_check)
        schedule.every().friday.at("09:25").do(self.market_open_check)
        
        logging.info("Trading schedule setup completed!")
    
    def run(self):
        """Run the scheduler"""
        self.is_running = True
        self.setup_schedule()
        
        # Send startup notification
        self.trader.notifier.send_message("🤖 AI Trading Scheduler started! Monitoring for trading opportunities...")
        
        logging.info("Scheduler started. Press Ctrl+C to stop.")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user.")
            self.trader.notifier.send_message("🛑 AI Trading Scheduler stopped.")
            self.is_running = False
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        logging.info("Scheduler stopped.")

def main():
    """Main function to run the scheduler"""
    scheduler = TradingScheduler()
    scheduler.run()

if __name__ == "__main__":
    main() 