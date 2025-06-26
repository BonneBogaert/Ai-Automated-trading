# AI Trading System Configuration

# Trading Parameters
TRADING_CONFIG = {
    'max_trades_per_session': 10,      # Maximum number of trades per analysis session
    'min_confidence_score': 7,        # Minimum AI confidence score to execute trades
    'position_size_pct': 0.05,        # Percentage of portfolio per trade (2%)
    'max_position_size': 1000,        # Maximum dollar amount per position
    'min_stock_price': 5.0,           # Minimum stock price to consider
    'max_stock_price': 2000.0,        # Maximum stock price to consider
    'min_market_cap': 5e8,            # Minimum market cap ($500M)
    'min_volume_ratio': 0.5,          # Minimum volume ratio compared to average
}

# Risk Management
RISK_CONFIG = {
    'max_portfolio_risk': 0.05,       # Maximum 5% portfolio risk per trade
    'stop_loss_pct': 0.15,            # 10% stop loss
    'take_profit_pct': 0.30,          # 20% take profit
    'max_drawdown': 0.25,             # Maximum 15% drawdown before stopping
}

# AI Analysis Settings
AI_CONFIG = {
    'model': 'gpt-4',                 # OpenAI model to use
    'temperature': 0.5,               # AI response randomness (0-1)
    'max_tokens': 500,                # Maximum tokens for AI response
    'analysis_period': '6mo',         # Data period for analysis
    'news_lookback_days': 7,          # Days of news to analyze
}

# Screening Settings
SCREENING_CONFIG = {
    'max_stocks_to_analyze': 20,      # Maximum stocks to analyze per session
    'screening_interval_hours': 24,   # How often to screen for new stocks
    'rebalance_interval_days': 7,     # How often to rebalance portfolio
}

# Notification Settings
NOTIFICATION_CONFIG = {
    'send_trade_alerts': True,        # Send alerts for each trade
    'send_daily_summary': True,       # Send daily portfolio summary
    'send_weekly_report': True,       # Send weekly detailed report
    'send_error_alerts': True,        # Send alerts for errors
}

# Ethical Investment Settings
ETHICAL_CONFIG = {
    'exclude_military': True,
    'exclude_big_pharma': True,
    'exclude_fossil_fuels': True,
    'exclude_tobacco': True,
    'exclude_gambling': True,
    'prefer_renewable_energy': True,
    'prefer_clean_tech': True,
    'prefer_esg_companies': True,
}

# Stock Universe (can be customized)
STOCK_UNIVERSE = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    'renewable_energy': ['ENPH', 'SEDG', 'RUN', 'SPWR', 'FSLR', 'NEE', 'BEP', 'CWEN'],
    'ev_automotive': ['RIVN', 'LCID', 'NIO', 'XPEV', 'LI'],
    'clean_tech': ['PLUG', 'BLDP', 'FCEL', 'BEEM', 'MAXN'],
    'healthcare_tech': ['TDOC', 'CRWD', 'ZM', 'DOCU', 'TWLO'],
    'fintech': ['SQ', 'PYPL', 'COIN', 'HOOD', 'AFRM'],
    'consumer': ['NKE', 'SBUX', 'TGT', 'COST', 'HD'],
    'european': ['ASML', 'SAP', 'NESN.SW', 'NOVO-B.CO', 'ROCHE.SW']
}

# Technical Analysis Parameters
TECHNICAL_CONFIG = {
    'sma_periods': [20, 50, 200],     # Simple Moving Average periods
    'ema_periods': [12, 26],          # Exponential Moving Average periods
    'rsi_period': 14,                 # RSI calculation period
    'macd_fast': 12,                  # MACD fast period
    'macd_slow': 26,                  # MACD slow period
    'macd_signal': 9,                 # MACD signal period
    'bb_period': 20,                  # Bollinger Bands period
    'bb_std': 2,                      # Bollinger Bands standard deviation
    'volume_sma_period': 20,          # Volume SMA period
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'ai_trader.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
} 