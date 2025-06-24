# AI-Powered Trading System 🤖📈

An intelligent, automated trading system that uses AI to analyze stocks, make trading decisions, and manage your portfolio ethically. Perfect for investors who want to leverage AI for pattern recognition, forecasting, and automated trading while maintaining ethical investment principles.

## Features ✨

- **🤖 AI-Powered Analysis**: Uses OpenAI GPT-4 for intelligent stock analysis and decision making
- **📊 Technical Analysis**: Comprehensive technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **📰 News Sentiment Analysis**: Real-time news sentiment analysis for better decision making
- **🌱 Ethical Filtering**: Automatically excludes military, big pharma, fossil fuels, and other unethical sectors
- **📱 Telegram Notifications**: Real-time alerts for trades, daily summaries, and weekly reports
- **📄 PDF Reports**: Detailed PDF reports with charts and analysis sent via Telegram
- **⚡ Automated Trading**: Integrates with Alpaca for paper trading (can be configured for live trading)
- **📈 Portfolio Management**: Automatic position sizing and risk management
- **🕒 Optimized Scheduling**: Hourly sessions during market hours to minimize API costs
- **📊 Performance Tracking**: Detailed logging and performance analytics
- **☁️ Replit Ready**: Optimized for cloud deployment on Replit

## Ethical Investment Focus 🌍

This system is designed for ethical investors and automatically filters out:
- Military/Defense companies
- Big Pharma
- Fossil Fuel companies
- Tobacco companies
- Gambling companies
- Private prisons

While prioritizing:
- Renewable Energy
- Clean Technology
- ESG-focused companies
- Sustainable businesses

## Setup Instructions 🚀

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project directory:

```env
# OpenAI API (for AI analysis)
OPENAI_API_KEY=your_openai_api_key_here

# News API (for sentiment analysis)
NEWSAPI_KEY=your_newsapi_key_here

# Alpaca Trading API (paper trading)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# Telegram Bot (for notifications)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```

### 3. API Setup

#### OpenAI API
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an account and get your API key
3. Add funds to your account (GPT-4 usage costs money)

#### News API
1. Go to [NewsAPI](https://newsapi.org/)
2. Sign up for a free account
3. Get your API key

#### Alpaca Trading
1. Go to [Alpaca Markets](https://alpaca.markets/)
2. Create a paper trading account
3. Get your API key and secret

#### Telegram Bot
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Create a new bot with `/newbot`
3. Get your bot token
4. Start a chat with your bot and get your chat ID

### 4. Configuration

Edit `config.py` to customize:
- Trading parameters (position sizes, confidence thresholds)
- Risk management settings
- Stock universe (add/remove stocks)
- Notification preferences
- Ethical filtering rules

## Usage 📖

### Quick Start

Run a single trading session:
```bash
python ai_trader.py
```

### Manual Trading

Run a manual trading session (for testing):
```bash
python manual_trade.py
```

### Automated Trading

#### Local Deployment
Run the scheduler for automated trading:
```bash
python scheduler.py
```

#### Replit Deployment (Recommended)
For cloud deployment on Replit, use the optimized scheduler:
```bash
python replit_scheduler.py
```

The Replit scheduler will:
- Run trading sessions every hour during market hours (9:30 AM - 4:30 PM ET)
- Limit to 8 sessions per day to minimize API costs
- Send daily summaries at midnight
- Send market open notifications
- Generate detailed PDF reports with charts

### Manual Trading

You can also run specific functions:

```python
from ai_trader import AITrader

# Initialize the trader
trader = AITrader()

# Run analysis and trading
trader.analyze_and_trade(max_trades=3, min_confidence=7)

# Generate weekly report
trader.generate_weekly_report()
```

## Replit Deployment ☁️

### 1. Create a Replit Account
1. Go to [Replit](https://replit.com/)
2. Create a free account
3. Create a new Python repl

### 2. Upload Your Code
1. Copy all your Python files to the Replit workspace
2. Create the `.env` file with your API keys
3. Install dependencies by running `pip install -r requirements.txt`

### 3. Configure for Always-On
1. In your Replit, go to "Tools" → "Secrets"
2. Add your environment variables as secrets
3. Set up a "Always On" repl (requires Replit Pro)

### 4. Start the Scheduler
```bash
python replit_scheduler.py
```

### 5. Monitor Your Bot
- Check the console output in Replit
- Receive Telegram notifications
- Get PDF reports with detailed analysis

## API Usage Optimization 💰

The system is optimized to minimize API costs:

- **OpenAI**: Limited to 300 tokens per analysis, higher confidence thresholds
- **News API**: Limited to 3 articles per stock, 5 stocks per session
- **Yahoo Finance**: Free, no limits
- **Alpaca**: Free for paper trading
- **Telegram**: Free

Estimated monthly costs:
- OpenAI: $5-20 (depending on usage)
- News API: Free tier (1000 requests/day)
- Total: $5-20/month

## How It Works 🔧

### 1. Stock Screening
- Screens a predefined universe of stocks
- Filters out unethical companies
- Applies basic criteria (market cap, volume, price)

### 2. Technical Analysis
- Calculates multiple technical indicators
- Analyzes price patterns and momentum
- Evaluates volume and volatility

### 3. News Sentiment Analysis
- Fetches recent news for each stock
- Analyzes sentiment using TextBlob
- Incorporates news into AI decision making

### 4. AI Decision Making
- Uses GPT-4 to analyze all data
- Considers technical indicators, sentiment, and news
- Provides trading decision with confidence score

### 5. Risk Management
- Position sizing based on portfolio percentage
- Maximum position limits
- Confidence thresholds for trade execution

### 6. Execution & Monitoring
- Executes trades via Alpaca API
- Sends real-time notifications
- Generates PDF reports with charts
- Tracks performance and generates reports

## Telegram Notifications 📱

You'll receive:

1. **Session Start/End**: When trading sessions begin and complete
2. **Trade Alerts**: Real-time notifications for each trade with reasoning
3. **Session Summary**: Overview of what was analyzed and traded
4. **PDF Reports**: Detailed reports with charts and analysis
5. **Daily Summary**: Portfolio status and daily performance
6. **Error Alerts**: If something goes wrong

## PDF Reports 📄

Each trading session generates a detailed PDF report including:

- Session summary with metrics
- Trading decisions with confidence scores
- Technical analysis charts
- Price charts with moving averages
- Volume analysis
- Risk assessments
- AI reasoning for each decision

## Customization 🎛️

### Adding New Stocks

Edit the `STOCK_UNIVERSE` in `config.py`:

```python
STOCK_UNIVERSE = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', ...],
    'renewable_energy': ['ENPH', 'SEDG', ...],
    'your_sector': ['STOCK1', 'STOCK2', ...]
}
```

### Modifying Trading Parameters

Edit `TRADING_CONFIG` in `config.py`:

```python
TRADING_CONFIG = {
    'max_trades_per_session': 5,      # More/fewer trades
    'min_confidence_score': 7,        # Higher/lower confidence threshold
    'position_size_pct': 0.02,        # Position size as % of portfolio
    # ... other parameters
}
```

### Custom Ethical Filters

Modify the `EthicalFilter` class in `ai_trader.py`:

```python
UNETHICAL_KEYWORDS = [
    'military', 'defense', 'weapons',
    # Add your own keywords
]
```

### Adjusting Schedule

For Replit deployment, modify `replit_scheduler.py`:

```python
self.max_sessions_per_day = 8  # Change number of sessions per day
```

## Risk Disclaimer ⚠️

**This is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.**

- Start with paper trading
- Never invest more than you can afford to lose
- Monitor the system regularly
- Understand that AI can make mistakes
- Consider consulting with a financial advisor

## Performance Monitoring 📊

The system provides several ways to monitor performance:

1. **Telegram Notifications**: Real-time trade alerts and summaries
2. **PDF Reports**: Detailed analysis with charts
3. **Log Files**: Detailed logs in `ai_trader.log` and `replit_scheduler.log`
4. **Alpaca Dashboard**: View your paper trading account
5. **Console Output**: Real-time status updates

## Troubleshooting 🔧

### Common Issues

1. **API Rate Limits**: The system handles rate limits automatically, but you may need to adjust timing
2. **Market Hours**: Paper trading is available 24/7, but real market data has limitations
3. **Network Issues**: The system will retry failed requests automatically
4. **Replit Timeout**: Use "Always On" repls to prevent timeouts

### Debug Mode

Enable debug logging by modifying the logging level in the scripts:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Test Your Setup

Run the test script to verify everything works:

```bash
python test_system.py
```

## Contributing 🤝

Feel free to contribute improvements:
- Add new technical indicators
- Improve ethical filtering
- Enhance AI prompts
- Add new data sources
- Optimize performance
- Improve PDF report formatting

## License 📄

This project is for educational purposes. Use at your own risk.

## Support 💬

For questions or issues:
1. Check the logs for error messages
2. Verify your API keys are correct
3. Ensure all dependencies are installed
4. Test with paper trading first
5. Check the Telegram notifications for status updates

---

**Happy Trading! 🚀📈** 