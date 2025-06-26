#!/usr/bin/env python3
"""
Hourly AI Trading Session for Render Cron Job
Runs every hour during market hours and generates daily reports.
"""

import os
import sys
from datetime import datetime, timedelta
import logging

def is_market_hours() -> bool:
    """Check if current time is during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)"""
    now = datetime.utcnow()
    
    # Convert UTC to ET (UTC-5 for EST, UTC-4 for EDT)
    # For simplicity, we'll use UTC-5 (EST) - you may want to adjust for daylight savings
    et_time = now - timedelta(hours=5)
    
    # Check if it's a weekday (Monday = 0, Sunday = 6)
    if et_time.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if it's during market hours (9:30 AM - 4:00 PM ET)
    market_start = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_start <= et_time <= market_end

def is_end_of_trading_day() -> bool:
    """Check if this is the last trading session of the day (around 4:00 PM ET)"""
    now = datetime.utcnow()
    et_time = now - timedelta(hours=5)
    
    # Check if it's between 3:30 PM and 4:30 PM ET (end of trading day)
    end_start = et_time.replace(hour=15, minute=30, second=0, microsecond=0)
    end_end = et_time.replace(hour=16, minute=30, second=0, microsecond=0)
    
    return end_start <= et_time <= end_end

def main():
    """Main function for hourly trading session"""
    try:
        logging.info("🚀 Starting Hourly AI Trading Session...")
        
        # Check if we're in market hours
        if not is_market_hours():
            logging.info("⏰ Outside market hours - skipping trading session")
            return
        
        # Import AITrader
        from ai_trader import AITrader
        
        # Initialize the trader
        trader = AITrader()
        
        # Get current time info
        now = datetime.utcnow()
        et_time = now - timedelta(hours=5)
        
        logging.info(f"📅 Trading Session: {et_time.strftime('%Y-%m-%d %H:%M:%S')} ET")
        
        # Get current portfolio status
        account = trader.get_account_info()
        if account:
            logging.info(f"💰 Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
            logging.info(f"💵 Available Cash: ${account.get('cash', 0):,.2f}")
        
        positions = trader.get_portfolio_positions()
        if positions:
            logging.info(f"📈 Active Positions: {len(positions)}")
            for ticker, pos in positions.items():
                logging.info(f"   {ticker}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
        else:
            logging.info("📈 No active positions")
        
        # Run the trading session with config values
        session_data = trader.analyze_and_trade(
            max_trades=3,
            min_confidence=8
        )
        
        if session_data:
            logging.info(f"✅ Trading session completed:")
            logging.info(f"   📊 Stocks Analyzed: {session_data.get('stocks_analyzed', 0)}")
            logging.info(f"   🔄 Trades Executed: {session_data.get('trades_executed', 0)}")
            logging.info(f"   📈 Buy Orders: {session_data.get('buy_orders', 0)}")
            logging.info(f"   📉 Sell Orders: {session_data.get('sell_orders', 0)}")
            
            # Show top decisions
            top_decisions = session_data.get('top_decisions', [])
            if top_decisions:
                logging.info("🏆 Top Trading Decisions:")
                for i, decision in enumerate(top_decisions[:3], 1):
                    emoji = "🟢" if decision['action'] == "BUY" else "🔴" if decision['action'] == "SELL" else "🟡"
                    logging.info(f"   {i}. {emoji} {decision['ticker']}: {decision['action']} (Confidence: {decision['confidence']}/10)")
        else:
            logging.warning("❌ Trading session failed or returned no data")
        
        # Check if this is the end of the trading day
        if is_end_of_trading_day():
            logging.info("📊 End of trading day - generating daily report...")
            
            try:
                # Generate and save daily report
                pdf_buffer = trader.generate_position_report_pdf()
                if pdf_buffer:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"reports/daily_report_{timestamp}.pdf"
                    
                    # Ensure reports directory exists
                    os.makedirs('reports', exist_ok=True)
                    
                    with open(filename, "wb") as f:
                        f.write(pdf_buffer.getvalue())
                    
                    logging.info(f"✅ Daily report saved as: {filename}")
                    
                    # Send daily summary via Telegram
                    performance = trader.get_position_performance()
                    if performance:
                        summary = performance['portfolio_summary']
                        message = f"""
📊 <b>End of Day Report - {et_time.strftime('%Y-%m-%d')}</b>

💰 <b>Portfolio Summary:</b>
• Total Value: ${summary['total_value']:,.2f}
• Cash: ${summary['cash']:,.2f}
• Invested: ${summary['invested_amount']:,.2f}
• Unrealized P&L: ${summary['total_unrealized_pl']:,.2f} ({summary['total_unrealized_pl_pct']:+.2f}%)
• Active Positions: {summary['positions_count']}

📈 <b>Today's Session:</b>
• Stocks Analyzed: {session_data.get('stocks_analyzed', 0) if session_data else 0}
• Trades Executed: {session_data.get('trades_executed', 0) if session_data else 0}
• Buy Orders: {session_data.get('buy_orders', 0) if session_data else 0}
• Sell Orders: {session_data.get('sell_orders', 0) if session_data else 0}

⏰ Generated: {et_time.strftime('%Y-%m-%d %H:%M:%S')} ET
                        """
                        trader.notifier.send_message(message)
                        
                        # Send PDF report
                        trader.notifier.send_document(
                            pdf_buffer.getvalue(),
                            f"daily_report_{timestamp}.pdf",
                            f"📊 Daily Trading Report - {et_time.strftime('%Y-%m-%d')}"
                        )
                    else:
                        logging.warning("❌ Could not generate position performance for daily report")
                else:
                    logging.error("❌ Failed to generate daily PDF report")
                    
            except Exception as e:
                logging.error(f"❌ Error generating daily report: {e}")
        
        logging.info("🎉 Hourly trading session completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Error in hourly trading session: {e}")
        # Try to send error notification
        try:
            from ai_trader import AITrader
            trader = AITrader()
            trader.notifier.send_message(f"❌ Hourly trading session error: {str(e)}")
        except:
            pass
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if exit_code else 0) 