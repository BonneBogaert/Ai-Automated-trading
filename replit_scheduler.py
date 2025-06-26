#!/usr/bin/env python3
"""
Replit-optimized scheduler for AI Trading System
Runs every hour to minimize API costs while maintaining effectiveness
"""
from keep_alive import keep_alive
keep_alive()

import time
import schedule
import logging
from datetime import datetime, timedelta, time as dtime
from ai_trader import AITrader
import os
from dotenv import load_dotenv
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    brussels_tz = ZoneInfo("Europe/Brussels")
except ImportError:
    from pytz import timezone
    brussels_tz = timezone("Europe/Brussels")

# Load environment variables
load_dotenv()

class ReplitScheduler:
    """Scheduler optimized for Replit deployment, now timezone-aware for Brussels"""
    
    def __init__(self):
        self.trader = AITrader()
    
    def run_trading_session(self):
        """Execute a single trading session and send a Telegram message"""
        try:
            current_time = datetime.now()
            if not self.is_market_open_brussels():
                logging.info("Market is closed. Skipping session.")
                return
            logging.info("="*60)
            logging.info("STARTING HOURLY AI TRADING SESSION")
            logging.info(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info("="*60)
            session_data = self.trader.analyze_and_trade(max_trades=2, min_confidence=6)
            trades = session_data.get('trades_executed', 0) if session_data else 0
            self.trader.notifier.send_message(f"Hourly trading session completed! Trades executed: {trades}")
            logging.info(f"HOURLY TRADING SESSION COMPLETED. Trades executed: {trades}")
            logging.info("="*60)
        except Exception as e:
            logging.error(f"Error in hourly trading session: {e}")
            try:
                self.trader.notifier.send_message(f"Hourly trading session failed: {str(e)}")
            except:
                pass

    def send_daily_report(self):
        """Send a comprehensive daily report via Telegram"""
        try:
            logging.info("Sending comprehensive daily portfolio summary...")
            
            # Get detailed position performance
            performance = self.trader.get_position_performance()
            account = self.trader.get_account_info()
            
            if not account:
                return
            
            if not performance:
                # No positions - send simple account summary
                total_value = account.get('portfolio_value', 0)
                cash = account.get('cash', 0)
                message = f"""
Daily Portfolio Summary

Portfolio Value: ${total_value:,.2f}
Available Cash: ${cash:,.2f}
Active Positions: 0

Date: {datetime.now().strftime('%Y-%m-%d')}
                """
                self.trader.notifier.send_message(message)
                return
            
            # Create comprehensive daily report
            summary = performance['portfolio_summary']
            
            # Determine overall performance emoji
            if summary['total_unrealized_pl'] > 0:
                overall_emoji = "📈"
            elif summary['total_unrealized_pl'] < 0:
                overall_emoji = "📉"
            else:
                overall_emoji = "➡️"
            
            message = f"""
{overall_emoji} Daily Portfolio Summary

💰 Portfolio Overview:
• Total Value: ${summary['total_value']:,.2f}
• Available Cash: ${summary['cash']:,.2f}
• Invested Amount: ${summary['invested_amount']:,.2f}
• Unrealized P&L: ${summary['total_unrealized_pl']:,.2f} ({summary['total_unrealized_pl_pct']:+.2f}%)
• Active Positions: {summary['positions_count']}

🏆 Performance Highlights:
"""
            
            # Add best and worst performers
            if performance['best_performer']:
                best = performance['positions'][performance['best_performer']]
                message += f"• Best: {performance['best_performer']} ({best['unrealized_pl_pct']:+.2f}%)\n"
            
            if performance['worst_performer']:
                worst = performance['positions'][performance['worst_performer']]
                message += f"• Worst: {performance['worst_performer']} ({worst['unrealized_pl_pct']:+.2f}%)\n"
            
            # Add top 3 positions by value
            positions_sorted_by_value = sorted(
                performance['positions'].items(), 
                key=lambda x: x[1]['market_value'], 
                reverse=True
            )
            
            message += f"\n📈 Top Positions by Value:\n"
            for i, (ticker, pos) in enumerate(positions_sorted_by_value[:3]):
                emoji = "🟢" if pos['unrealized_pl_pct'] >= 0 else "🔴"
                message += f"{i+1}. {emoji} {ticker}: ${pos['market_value']:,.0f} ({pos['unrealized_pl_pct']:+.2f}%)\n"
            
            message += f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d')}"
            
            self.trader.notifier.send_message(message)
            logging.info("Comprehensive daily summary sent successfully!")
            
        except Exception as e:
            logging.error(f"Error sending daily summary: {e}")

    def is_market_open_brussels(self):
        now = datetime.now(brussels_tz)
        if now.weekday() >= 5:
            return False
        market_open = dtime(9, 30)
        market_close = dtime(16, 30)
        return market_open <= now.time() <= market_close

    def start_scheduler(self):
        logging.info("Starting Brussels-timezone-aware AI Trading Scheduler...")
        self.trader.notifier.send_message("Replit AI Trading Scheduler (Brussels time) started successfully!")
        logging.info("Scheduler started: hourly trading sessions and daily report at 16:30 Brussels time.")

        # Immediate run on startup
        self.run_trading_session()

        last_hourly_run = None
        last_daily_report = None
        while True:
            now = datetime.now(brussels_tz)
            # Run hourly session at every half hour between 09:30 and 16:30
            if self.is_market_open_brussels() and now.minute == 30:
                # Only run once per hour
                if last_hourly_run != (now.date(), now.hour):
                    self.run_trading_session()
                    last_hourly_run = (now.date(), now.hour)
            # Run daily report at 16:30
            if now.hour == 16 and now.minute == 30:
                if last_daily_report != now.date():
                    self.send_daily_report()
                    self.send_daily_position_report()
                    last_daily_report = now.date()
            time.sleep(20)  # Check every 20 seconds

    def send_daily_position_report(self):
        """Send a daily PDF position report via Telegram at 16:30 Brussels time."""
        try:
            pdf_buffer = self.trader.generate_position_report_pdf()
            if pdf_buffer:
                filename = f"position_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                self.trader.notifier.send_document(
                    pdf_buffer.getvalue(),
                    filename,
                    f"📊 Daily Portfolio Position Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
        except Exception as e:
            logging.error(f"Error sending daily position report: {e}")

def main():
    """Main function to run the scheduler"""
    scheduler = ReplitScheduler()
    scheduler.start_scheduler()

if __name__ == "__main__":
    main() 