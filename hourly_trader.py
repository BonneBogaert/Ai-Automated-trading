#!/usr/bin/env python3
"""
Hourly AI Trading Session for Render Cron Job
Runs every hour during market hours and generates daily reports.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
import logging
import gc
import pytz

# Suppress deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning, module="datetime")

# Test datetime functionality at import time
try:
    test_time = datetime.now(timezone.utc)
    print(f"✅ Datetime test successful: {test_time}")
except Exception as e:
    print(f"❌ Datetime test failed: {e}")
    sys.exit(1)

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except:
        return 0

def log_memory(message=""):
    """Log memory usage with a message"""
    try:
        memory = get_memory_usage()
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message} | Memory: {memory:.1f}MB"
        print(log_msg)
        logging.info(log_msg)
        
        if memory > 400:
            warning_msg = f"⚠️  WARNING: High memory usage ({memory:.1f}MB) - approaching Render limit (512MB)"
            print(warning_msg)
            logging.warning(warning_msg)
        
        return memory
    except Exception as e:
        print(f"❌ Error in log_memory: {e}")
        return 0

def get_stock_exchange(ticker: str) -> str:
    """Get the primary exchange for a given stock ticker"""
    # Map of stock tickers to their primary exchanges
    stock_exchanges = {
        # US Stocks (NYSE/NASDAQ)
        'AAPL': 'US', 'MSFT': 'US', 'GOOGL': 'US', 'AMZN': 'US', 'TSLA': 'US', 
        'NVDA': 'US', 'META': 'US', 'NFLX': 'US', 'ENPH': 'US', 'SEDG': 'US', 
        'RUN': 'US', 'SPWR': 'US', 'FSLR': 'US', 'NEE': 'US', 'BEP': 'US', 
        'CWEN': 'US', 'RIVN': 'US', 'LCID': 'US', 'NIO': 'US', 'XPEV': 'US', 
        'LI': 'US', 'PLUG': 'US', 'BLDP': 'US', 'FCEL': 'US', 'BEEM': 'US', 
        'MAXN': 'US', 'TDOC': 'US', 'CRWD': 'US', 'ZM': 'US', 'DOCU': 'US', 
        'TWLO': 'US', 'SQ': 'US', 'PYPL': 'US', 'COIN': 'US', 'HOOD': 'US', 
        'AFRM': 'US', 'NKE': 'US', 'SBUX': 'US', 'TGT': 'US', 'COST': 'US', 'HD': 'US',
        
        # European Stocks
        'ASML': 'LSE',      # ASML Holding (Netherlands, but trades on multiple exchanges)
        'SAP': 'LSE',       # SAP SE (Germany, but trades on multiple exchanges)
        'NESN.SW': 'SWX',   # Nestlé (Swiss Exchange)
        'NOVO-B.CO': 'CSE', # Novo Nordisk (Copenhagen Stock Exchange)
        'ROCHE.SW': 'SWX'   # Roche (Swiss Exchange)
    }
    
    return stock_exchanges.get(ticker, 'US')  # Default to US if unknown

def is_market_hours_for_stocks(stock_tickers: list = None) -> bool:
    """Check if markets are open for specific stocks"""
    try:
        # If no specific stocks provided, check all markets
        if not stock_tickers:
            return is_market_hours()
        
        # Get unique exchanges for the stocks
        exchanges = set()
        for ticker in stock_tickers:
            exchange = get_stock_exchange(ticker)
            exchanges.add(exchange)
        
        print(f"🔍 Checking markets for stocks: {stock_tickers}")
        print(f"📊 Relevant exchanges: {list(exchanges)}")
        
        # Check if any relevant exchange is open
        now_utc = datetime.now(timezone.utc)
        
        markets = {
            'US': {
                'timezone': pytz.timezone('US/Eastern'),
                'hours': (9, 30, 16, 0),
                'days': [0, 1, 2, 3, 4]
            },
            'LSE': {
                'timezone': pytz.timezone('Europe/London'),
                'hours': (8, 0, 16, 30),
                'days': [0, 1, 2, 3, 4]
            },
            'SWX': {
                'timezone': pytz.timezone('Europe/Zurich'),
                'hours': (9, 0, 17, 30),
                'days': [0, 1, 2, 3, 4]
            },
            'CSE': {
                'timezone': pytz.timezone('Europe/Copenhagen'),
                'hours': (9, 0, 17, 0),
                'days': [0, 1, 2, 3, 4]
            }
        }
        
        for exchange in exchanges:
            if exchange in markets:
                market_config = markets[exchange]
                try:
                    market_time = now_utc.astimezone(market_config['timezone'])
                    
                    if market_time.weekday() in market_config['days']:
                        start_hour, start_minute, end_hour, end_minute = market_config['hours']
                        market_start = market_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
                        market_end = market_time.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
                        
                        if market_start <= market_time <= market_end:
                            print(f"✅ Market OPEN: {exchange} - {market_time.strftime('%Y-%m-%d %H:%M:%S')} {market_config['timezone'].tzname(market_time)}")
                            return True
                            
                except Exception as e:
                    print(f"⚠️ Error checking {exchange} market: {e}")
                    continue
        
        print(f"❌ No relevant markets open for stocks: {stock_tickers}")
        return False
        
    except Exception as e:
        print(f"❌ Error in is_market_hours_for_stocks: {e}")
        return False

def is_market_hours() -> bool:
    """Check if current time is during market hours for any relevant exchange"""
    try:
        # Get current time in UTC
        now_utc = datetime.now(timezone.utc)
        
        # Define market hours for different exchanges
        markets = {
            'US': {
                'timezone': pytz.timezone('US/Eastern'),
                'hours': (9, 30, 16, 0),  # 9:30 AM - 4:00 PM ET
                'days': [0, 1, 2, 3, 4]   # Monday-Friday
            },
            'LSE': {  # London Stock Exchange
                'timezone': pytz.timezone('Europe/London'),
                'hours': (8, 0, 16, 30),  # 8:00 AM - 4:30 PM GMT/BST
                'days': [0, 1, 2, 3, 4]   # Monday-Friday
            },
            'SWX': {  # Swiss Exchange
                'timezone': pytz.timezone('Europe/Zurich'),
                'hours': (9, 0, 17, 30),  # 9:00 AM - 5:30 PM CET/CEST
                'days': [0, 1, 2, 3, 4]   # Monday-Friday
            },
            'CSE': {  # Copenhagen Stock Exchange
                'timezone': pytz.timezone('Europe/Copenhagen'),
                'hours': (9, 0, 17, 0),   # 9:00 AM - 5:00 PM CET/CEST
                'days': [0, 1, 2, 3, 4]   # Monday-Friday
            }
        }
        
        # Check each market
        for market_name, market_config in markets.items():
            try:
                # Convert to market timezone
                market_time = now_utc.astimezone(market_config['timezone'])
                
                # Check if it's a trading day
                if market_time.weekday() not in market_config['days']:
                    continue
                
                # Check if it's during market hours
                start_hour, start_minute, end_hour, end_minute = market_config['hours']
                market_start = market_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
                market_end = market_time.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
                
                if market_start <= market_time <= market_end:
                    print(f"✅ Market OPEN: {market_name} - {market_time.strftime('%Y-%m-%d %H:%M:%S')} {market_config['timezone'].tzname(market_time)}")
                    return True
                    
            except Exception as e:
                print(f"⚠️ Error checking {market_name} market: {e}")
                continue
        
        # If we get here, no markets are open
        print(f"❌ All markets CLOSED - Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Debug: Show current times in all markets
        print("🔍 Current times in all markets:")
        for market_name, market_config in markets.items():
            try:
                market_time = now_utc.astimezone(market_config['timezone'])
                print(f"  {market_name}: {market_time.strftime('%Y-%m-%d %H:%M:%S')} {market_config['timezone'].tzname(market_time)}")
            except:
                pass
        
        return False
        
    except Exception as e:
        print(f"❌ Error in is_market_hours: {e}")
        return False

def get_active_markets() -> list:
    """Get list of currently active markets"""
    try:
        now_utc = datetime.now(timezone.utc)
        active_markets = []
        
        markets = {
            'US': {
                'timezone': pytz.timezone('US/Eastern'),
                'hours': (9, 30, 16, 0),
                'days': [0, 1, 2, 3, 4]
            },
            'LSE': {
                'timezone': pytz.timezone('Europe/London'),
                'hours': (8, 0, 16, 30),
                'days': [0, 1, 2, 3, 4]
            },
            'SWX': {
                'timezone': pytz.timezone('Europe/Zurich'),
                'hours': (9, 0, 17, 30),
                'days': [0, 1, 2, 3, 4]
            },
            'CSE': {
                'timezone': pytz.timezone('Europe/Copenhagen'),
                'hours': (9, 0, 17, 0),
                'days': [0, 1, 2, 3, 4]
            }
        }
        
        for market_name, market_config in markets.items():
            try:
                market_time = now_utc.astimezone(market_config['timezone'])
                
                if market_time.weekday() in market_config['days']:
                    start_hour, start_minute, end_hour, end_minute = market_config['hours']
                    market_start = market_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
                    market_end = market_time.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
                    
                    if market_start <= market_time <= market_end:
                        active_markets.append(market_name)
                        
            except Exception as e:
                print(f"⚠️ Error checking {market_name} market: {e}")
                continue
        
        return active_markets
        
    except Exception as e:
        print(f"❌ Error getting active markets: {e}")
        return []

def is_end_of_trading_day() -> bool:
    """Check if this is the end of the trading day for any major market"""
    try:
        now_utc = datetime.now(timezone.utc)
        
        # Check end of day for major markets
        markets = {
            'US': {
                'timezone': pytz.timezone('US/Eastern'),
                'end_window': (15, 30, 16, 30)  # 3:30 PM - 4:30 PM ET
            },
            'LSE': {
                'timezone': pytz.timezone('Europe/London'),
                'end_window': (16, 0, 17, 0)    # 4:00 PM - 5:00 PM GMT/BST
            }
        }
        
        for market_name, market_config in markets.items():
            try:
                market_time = now_utc.astimezone(market_config['timezone'])
                start_hour, start_minute, end_hour, end_minute = market_config['end_window']
                
                end_start = market_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
                end_end = market_time.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
                
                if end_start <= market_time <= end_end:
                    print(f"📊 End of trading day detected for {market_name}")
                    return True
                    
            except Exception as e:
                print(f"⚠️ Error checking end of day for {market_name}: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"❌ Error in is_end_of_trading_day: {e}")
        return False

def main():
    """Main function for hourly trading session"""
    try:
        logging.info("🚀 Starting Hourly AI Trading Session...")
        initial_memory = log_memory("Session started")
        
        # Check if we're in market hours
        if not is_market_hours():
            logging.info("⏰ Outside market hours - skipping trading session")
            # Show which markets are active for debugging
            active_markets = get_active_markets()
            if active_markets:
                logging.info(f"🔍 Active markets: {', '.join(active_markets)}")
            return
        
        # Import AITrader
        log_memory("Before importing AITrader")
        try:
            from ai_trader import AITrader
            log_memory("After importing AITrader")
        except Exception as e:
            logging.error(f"❌ Error importing AITrader: {e}")
            return 1
        
        # Initialize the trader
        try:
            trader = AITrader()
            log_memory("After initializing AITrader")
        except Exception as e:
            logging.error(f"❌ Error initializing AITrader: {e}")
            return 1
        
        # Get current time info
        try:
            now_utc = datetime.now(timezone.utc)
            et_tz = pytz.timezone('US/Eastern')
            et_time = now_utc.astimezone(et_tz)
            
            logging.info(f"📅 Trading Session: {et_time.strftime('%Y-%m-%d %H:%M:%S')} ET")
        except Exception as e:
            logging.error(f"❌ Error getting current time: {e}")
            return 1
        
        # Get current portfolio status
        try:
            account = trader.get_account_info()
            log_memory("After getting account info")
            if account:
                logging.info(f"💰 Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
                logging.info(f"💵 Available Cash: ${account.get('cash', 0):,.2f}")
        except Exception as e:
            logging.error(f"❌ Error getting account info: {e}")
        
        try:
            positions = trader.get_portfolio_positions()
            log_memory("After getting positions")
            if positions:
                logging.info(f"📈 Active Positions: {len(positions)}")
                for ticker, pos in positions.items():
                    logging.info(f"   {ticker}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
            else:
                logging.info("📈 No active positions")
        except Exception as e:
            logging.error(f"❌ Error getting positions: {e}")
        
        # Run the trading session with config values
        try:
            log_memory("Before trading session")
            session_data = trader.analyze_and_trade(
                max_trades=3,
                min_confidence=8
            )
            log_memory("After trading session")
        except Exception as e:
            logging.error(f"❌ Error in trading session: {e}")
            session_data = None
        
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
                    logging.info(f"   {i}. {decision['ticker']}: {decision['action']} (Confidence: {decision['confidence']}/10)")
        else:
            logging.warning("❌ Trading session failed or returned no data")
        
        # Check if this is the end of the trading day
        if is_end_of_trading_day():
            logging.info("📊 End of trading day - generating daily report...")
            
            try:
                # Generate and save daily report
                log_memory("Before generating PDF report")
                pdf_buffer = trader.generate_position_report_pdf()
                log_memory("After generating PDF report")
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
                    log_memory("After getting position performance")
                    if performance:
                        summary = performance['portfolio_summary']
                        message = f"""
📊 <b>End of Day Report - {et_time.strftime('%Y-%m-%d')}</b>

💰 <b>Portfolio Summary:</b>
• Total Value: ${summary['total_value']:,.2f}
• Cash: ${summary['cash']:,.2f}
• Invested: ${summary['invested_amount']:,.2f}
• Unrealized P&L: ${summary['total_unrealized_pl']:,.2f} ({summary['total_unrealized_pl_pct']:+.2f}%)
• Active Positions: {len(positions)}

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
                        logging.warning("❌ Failed to generate daily PDF report")
                    
            except Exception as e:
                logging.error(f"❌ Error generating daily report: {e}")
        
        # Final cleanup
        gc.collect()
        final_memory = log_memory("After garbage collection")
        
        # Memory summary
        memory_diff = final_memory - initial_memory
        logging.info(f"📊 Memory Summary: Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB, Difference: {memory_diff:.1f}MB")
        
        if final_memory > 512:
            logging.error(f"❌ CRITICAL: Final memory ({final_memory:.1f}MB) exceeds Render limit!")
            return 1
        
        logging.info("✅ Hourly trading session completed successfully!")
        return 0
        
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
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Run the main function
    exit_code = main()
    sys.exit(exit_code) 