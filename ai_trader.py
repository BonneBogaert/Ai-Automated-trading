import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from textblob import TextBlob
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import time
import json
from typing import List, Dict, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Setup logging with more detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_trader.log'),
        logging.StreamHandler()
    ]
)

class PDFReportGenerator:
    """Generate detailed PDF reports with charts and analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
            textColor=colors.darkgreen
        )
    
    def create_price_chart(self, ticker: str, data: pd.DataFrame) -> str:
        """Create price chart with technical indicators"""
        try:
            if data.empty:
                print(f"⚠️ No data available for {ticker} price chart")
                return None

            plt.style.use('default')
            sns.set_palette("husl")
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='#1f77b4')

            if len(data) >= 20:
                sma_20 = data['Close'].rolling(window=20).mean()
                ax.plot(data.index, sma_20, label='20-day SMA', alpha=0.7, color='orange')
            if len(data) >= 50:
                sma_50 = data['Close'].rolling(window=50).mean()
                ax.plot(data.index, sma_50, label='50-day SMA', alpha=0.7, color='red')

            if len(data) >= 20:
                sma_20 = data['Close'].rolling(window=20).mean()
                std_20 = data['Close'].rolling(window=20).std()
                upper_band = sma_20 + (std_20 * 2)
                lower_band = sma_20 - (std_20 * 2)

                # Use dropna to get only valid points
                valid = ~(upper_band.isna() | lower_band.isna())
                upper_valid = upper_band[valid]
                lower_valid = lower_band[valid]
                idx_valid = upper_valid.index

                if len(upper_valid) > 0:
                    ax.plot(idx_valid, upper_valid.values, '--', alpha=0.5, color='gray', label='Bollinger Upper')
                    ax.plot(idx_valid, lower_valid.values, '--', alpha=0.5, color='gray', label='Bollinger Lower')
                    ax.fill_between(idx_valid, upper_valid.values, lower_valid.values, alpha=0.1, color='gray')

            ax.set_title(f'{ticker} Price Chart with Technical Indicators', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            print(f"✅ Price chart created successfully for {ticker}")
            return buffer
        except Exception as e:
            print(f"❌ Error creating price chart for {ticker}: {e}")
            logging.error(f"Error creating price chart for {ticker}: {e}")
            return None
    
    def create_technical_indicators_chart(self, ticker: str, data: pd.DataFrame, indicators: Dict) -> str:
        """Create technical indicators chart"""
        try:
            if data.empty:
                print(f"⚠️ No data available for {ticker} technical chart")
                return None
            plt.style.use('default')
            sns.set_palette("husl")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='#1f77b4')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2.bar(data.index, data['Volume'], alpha=0.7, color='lightblue', label='Volume')
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            if len(data) >= 14:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_valid = rsi.dropna()
                if len(rsi_valid) > 0:
                    ax3.plot(rsi_valid.index, rsi_valid.values, label='RSI', linewidth=2, color='purple')
                    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
                    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
                    ax3.set_ylabel('RSI', fontsize=12)
                    ax3.set_ylim(0, 100)
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
            fig.suptitle(f'{ticker} Technical Indicators', fontsize=16, fontweight='bold')
            ax3.set_xlabel('Date', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            print(f"✅ Technical indicators chart created successfully for {ticker}")
            return buffer
        except Exception as e:
            print(f"❌ Error creating technical indicators chart for {ticker}: {e}")
            logging.error(f"Error creating technical indicators chart for {ticker}: {e}")
            return None
    
    def generate_trading_report(self, trading_session: Dict) -> str:
        """Generate comprehensive PDF trading report"""
        try:
            # Create PDF document
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []
            
            # Title
            story.append(Paragraph("AI Trading System Report", self.title_style))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Session Summary
            story.append(Paragraph("Session Summary", self.subtitle_style))
            summary_data = [
                ['Metric', 'Value'],
                ['Total Stocks Analyzed', str(trading_session.get('stocks_analyzed', 0))],
                ['Trades Executed', str(trading_session.get('trades_executed', 0))],
                ['Buy Orders', str(trading_session.get('buy_orders', 0))],
                ['Sell Orders', str(trading_session.get('sell_orders', 0))],
                ['Total Portfolio Value', f"${trading_session.get('portfolio_value', 0):,.2f}"],
                ['Available Cash', f"${trading_session.get('available_cash', 0):,.2f}"],
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Trading Decisions
            if trading_session.get('trading_decisions'):
                story.append(Paragraph("Trading Decisions", self.subtitle_style))
                
                for decision in trading_session['trading_decisions']:
                    story.append(Paragraph(f"<b>{decision['ticker']}</b> - {decision['action']}", self.styles['Heading3']))
                    story.append(Paragraph(f"Confidence: {decision['confidence']}/10", self.styles['Normal']))
                    story.append(Paragraph(f"Reasoning: {decision['reasoning']}", self.styles['Normal']))
                    story.append(Paragraph(f"Risk Level: {decision['risk']}", self.styles['Normal']))
                    story.append(Spacer(1, 10))
                    
                    # Add charts if available
                    if 'price_chart' in decision and decision['price_chart']:
                        story.append(Image(decision['price_chart'], width=6*inch, height=4*inch))
                        story.append(Spacer(1, 10))
                    
                    if 'technical_chart' in decision and decision['technical_chart']:
                        story.append(Image(decision['technical_chart'], width=6*inch, height=4*inch))
                        story.append(Spacer(1, 10))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            
            return buffer
            
        except Exception as e:
            logging.error(f"Error generating PDF report: {e}")
            return None

class EthicalFilter:
    """Filter out unethical companies based on keywords and sectors"""
    
    UNETHICAL_KEYWORDS = [
        'military', 'defense', 'weapons', 'arms', 'ammunition', 'missile',
        'pharmaceutical', 'big pharma', 'oil', 'gas', 'fossil fuel', 'coal',
        'tobacco', 'cigarette', 'casino', 'gambling', 'alcohol', 'beer', 'wine',
        'pornography', 'adult entertainment', 'private prison', 'detention'
    ]
    
    UNETHICAL_SECTORS = [
        'Aerospace & Defense', 'Oil & Gas', 'Tobacco', 'Gambling',
        'Pharmaceuticals', 'Private Prisons'
    ]
    
    @staticmethod
    def is_ethical(company_info: Dict) -> bool:
        """Check if a company is ethical based on its description and sector"""
        if not company_info:
            return False
            
        description = company_info.get('description', '').lower()
        sector = company_info.get('sector', '').lower()
        
        # Check for unethical keywords in description
        for keyword in EthicalFilter.UNETHICAL_KEYWORDS:
            if keyword in description:
                print(f"❌ Excluded {company_info.get('symbol', 'Unknown')}: Contains unethical keyword '{keyword}'")
                return False
                
        # Check for unethical sectors
        for unethical_sector in EthicalFilter.UNETHICAL_SECTORS:
            if unethical_sector.lower() in sector:
                print(f"❌ Excluded {company_info.get('symbol', 'Unknown')}: Unethical sector '{unethical_sector}'")
                return False
                
        print(f"✅ {company_info.get('symbol', 'Unknown')}: Passed ethical screening")
        return True

class StockAnalyzer:
    """Analyze stocks using technical and fundamental data"""
    
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        self.pdf_generator = PDFReportGenerator()
        
    def get_stock_data(self, ticker: str, period: str = "6mo") -> pd.DataFrame:
        """Get stock data from Yahoo Finance"""
        try:
            print(f"📊 Fetching data for {ticker}...")
            data = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
            if not data.empty:
                print(f"✅ Successfully fetched {len(data)} days of data for {ticker}")
            else:
                print(f"❌ No data found for {ticker}")
            return data
        except Exception as e:
            print(f"❌ Error getting data for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        if data.empty:
            return {}
            
        print("🔧 Calculating technical indicators...")
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
        indicators['sma_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
        indicators['ema_12'] = data['Close'].ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = data['Close'].ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        
        # Bollinger Bands - Fixed to extract scalar values
        sma_20 = data['Close'].rolling(window=20).mean()
        std_20 = data['Close'].rolling(window=20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        
        # Extract scalar values for Bollinger Bands
        indicators['bb_upper'] = bb_upper.iloc[-1]
        indicators['bb_lower'] = bb_lower.iloc[-1]
        indicators['bb_position'] = (data['Close'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Volume analysis
        indicators['volume_sma'] = data['Volume'].rolling(window=20).mean().iloc[-1]
        indicators['volume_ratio'] = data['Volume'].iloc[-1] / indicators['volume_sma']
        
        # Price momentum
        indicators['price_momentum'] = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1) * 100
        
        print(f"✅ Calculated {len(indicators)} technical indicators")
        return indicators
    
    def get_news_sentiment(self, ticker: str) -> Tuple[float, str]:
        """Get news sentiment for a ticker"""
        try:
            print(f"📰 Fetching news for {ticker}...")
            
            # Get company name for better news search
            stock_info = yf.Ticker(ticker).info
            company_name = stock_info.get('longName', ticker)
            
            # Search for news
            url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey={NEWSAPI_KEY}&language=en&sortBy=publishedAt&pageSize=5"
            response = requests.get(url)
            news_data = response.json()
            
            if news_data.get('status') != 'ok' or not news_data.get('articles'):
                print(f"⚠️ No recent news found for {ticker}")
                return 0.0, "No recent news found."
            
            # Analyze sentiment
            sentiments = []
            news_summary = []
            
            for article in news_data['articles'][:3]:  # Limit to 3 articles to save API calls
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = TextBlob(text).sentiment.polarity
                sentiments.append(sentiment)
                news_summary.append(f"{article.get('title', '')}: {article.get('description', '')[:100]}...")
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            news_text = " | ".join(news_summary[:2])  # Limit summary length
            
            print(f"✅ News sentiment for {ticker}: {avg_sentiment:.3f}")
            return avg_sentiment, news_text
            
        except Exception as e:
            print(f"❌ Error getting news for {ticker}: {e}")
            return 0.0, "Error fetching news."
    
    def analyze_stock_with_ai(self, ticker: str, indicators: Dict, sentiment: float, news: str) -> Dict:
        """Use AI to analyze stock and make trading decision"""
        try:
            print(f"🤖 Running AI analysis for {ticker}...")
            
            # Ensure all indicators are scalar values to prevent formatting errors
            def safe_float(value, default=0.0):
                """Safely convert value to float, handling pandas Series and NaN values"""
                # Handle None values first
                if value is None:
                    return default
                
                # Handle pandas Series
                if hasattr(value, 'iloc'):  # pandas Series
                    try:
                        if value.empty:
                            return default
                        last_value = value.iloc[-1]
                        if pd.isna(last_value):
                            return default
                        return float(last_value)
                    except (IndexError, TypeError):
                        return default
                
                # Handle scalar values
                try:
                    if pd.isna(value):
                        return default
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # Extract and convert all indicators to safe scalar values
            current_price = safe_float(indicators.get('current_price', 0))
            sma_20 = safe_float(indicators.get('sma_20', 0))
            sma_50 = safe_float(indicators.get('sma_50', 0))
            rsi = safe_float(indicators.get('rsi', 0))
            macd = safe_float(indicators.get('macd', 0))
            bb_position = safe_float(indicators.get('bb_position', 0))
            volume_ratio = safe_float(indicators.get('volume_ratio', 0))
            price_momentum = safe_float(indicators.get('price_momentum', 0))
            
            prompt = f"""
Analyze {ticker} for trading. Data:
Price: ${current_price:.2f}
SMA20: ${sma_20:.2f}
SMA50: ${sma_50:.2f}
RSI: {rsi:.2f}
MACD: {macd:.4f}
BB Pos: {bb_position:.2f}
Vol Ratio: {volume_ratio:.2f}
Momentum(5d): {price_momentum:.2f}%
News Sentiment: {sentiment:.3f}
News: {news}

Return JSON:
{{
  "decision": "BUY/SELL/HOLD",
  "confidence": 1-10,
  "reasoning": "...",
  "risk": "LOW/MEDIUM/HIGH",
  "expected_movement": "UP/DOWN/SIDEWAYS"
}}
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300  # Limit tokens to save costs
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                print(f"✅ AI decision for {ticker}: {result['decision']} (confidence: {result['confidence']}/10)")
                return result
            except json.JSONDecodeError:
                print(f"❌ Failed to parse AI response for {ticker}")
                return {
                    "decision": "HOLD",
                    "confidence": 5,
                    "reasoning": "AI analysis failed",
                    "risk": "MEDIUM",
                    "expected_movement": "SIDEWAYS"
                }
                
        except Exception as e:
            print(f"❌ Error in AI analysis for {ticker}: {e}")
            return {
                "decision": "HOLD",
                "confidence": 5,
                "reasoning": f"Analysis error: {str(e)}",
                "risk": "MEDIUM",
                "expected_movement": "SIDEWAYS"
            }

class StockScreener:
    """Screen for potential stocks to trade"""
    
    def __init__(self):
        self.ethical_filter = EthicalFilter()
    
    def get_stock_universe(self) -> List[str]:
        """Get a list of stocks to screen"""
        # Focused list for better performance and lower API costs
        stocks = [
            # Tech (your current holdings)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            # Renewable Energy (ethical focus)
            'ENPH', 'SEDG', 'RUN', 'SPWR', 'FSLR', 'NEE', 'BEP', 'CWEN',
            # Electric Vehicles
            'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
            # Clean Tech
            'PLUG', 'BLDP', 'FCEL', 'BEEM', 'MAXN',
            # Healthcare Tech (excluding big pharma)
            'TDOC', 'CRWD', 'ZM', 'DOCU', 'TWLO',
            # Financial Tech
            'SQ', 'PYPL', 'COIN', 'HOOD', 'AFRM',
            # Consumer
            'NKE', 'SBUX', 'TGT', 'COST', 'HD',
            # European stocks (since you're in EUR)
            'ASML', 'SAP', 'NESN.SW', 'NOVO-B.CO', 'ROCHE.SW'
        ]
        return stocks
    
    def screen_stocks(self, min_market_cap: float = 1e9) -> List[Dict]:
        """Screen stocks based on criteria"""
        stocks = self.get_stock_universe()
        screened_stocks = []
        
        print(f"🔍 Screening {len(stocks)} stocks...")
        
        for ticker in stocks:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Basic filters
                if (info.get('marketCap', 0) < min_market_cap or
                    info.get('regularMarketPrice', 0) < 5 or
                    info.get('regularMarketPrice', 0) > 1000):
                    print(f"⏭️ Skipped {ticker}: Failed basic filters")
                    continue
                
                # Ethical filter
                if not self.ethical_filter.is_ethical(info):
                    continue
                
                # Get recent price data
                data = stock.history(period="1mo")
                if data.empty:
                    print(f"⏭️ Skipped {ticker}: No price data")
                    continue
                
                # Calculate basic metrics
                current_price = info.get('regularMarketPrice', 0)
                volume = info.get('volume', 0)
                avg_volume = info.get('averageVolume', 0)
                
                if volume < avg_volume * 0.5:  # Skip low volume stocks
                    print(f"⏭️ Skipped {ticker}: Low volume")
                    continue
                
                screened_stocks.append({
                    'ticker': ticker,
                    'name': info.get('longName', ticker),
                    'sector': info.get('sector', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'current_price': current_price,
                    'volume': volume,
                    'avg_volume': avg_volume,
                    'pe_ratio': info.get('trailingPE', 0),
                    'beta': info.get('beta', 1.0)
                })
                
                print(f"✅ {ticker} passed screening")
                
            except Exception as e:
                print(f"❌ Error screening {ticker}: {e}")
                continue
        
        print(f"✅ Screening complete: {len(screened_stocks)} stocks passed")
        return screened_stocks

class TelegramNotifier:
    """Send notifications via Telegram"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_message(self, message: str):
        """Send a message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data)
            if response.status_code != 200:
                print(f"❌ Failed to send Telegram message: {response.text}")
            else:
                print("✅ Telegram message sent successfully")
        except Exception as e:
            print(f"❌ Error sending Telegram message: {e}")
    
    def send_document(self, document_data: bytes, filename: str, caption: str = ""):
        """Send a document (PDF) to Telegram"""
        try:
            url = f"{self.base_url}/sendDocument"
            files = {'document': (filename, document_data, 'application/pdf')}
            data = {
                'chat_id': self.chat_id,
                'caption': caption
            }
            response = requests.post(url, data=data, files=files)
            if response.status_code != 200:
                print(f"❌ Failed to send PDF: {response.text}")
            else:
                print("✅ PDF report sent successfully")
        except Exception as e:
            print(f"❌ Error sending PDF: {e}")
    
    def send_trade_alert(self, ticker: str, action: str, price: float, reasoning: str):
        """Send trade alert"""
        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
        message = f"""
{emoji} <b>Trade Alert: {ticker}</b>

Action: {action}
Price: ${price:.2f}

Reasoning: {reasoning}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        self.send_message(message)
    
    def send_session_summary(self, session_data: Dict):
        """Send session summary"""
        message = f"""
🤖 <b>AI Trading Session Summary</b>

📊 <b>Analysis Results:</b>
• Stocks Analyzed: {session_data.get('stocks_analyzed', 0)}
• Trades Executed: {session_data.get('trades_executed', 0)}
• Buy Orders: {session_data.get('buy_orders', 0)}
• Sell Orders: {session_data.get('sell_orders', 0)}

💰 <b>Portfolio Status:</b>
• Total Value: ${session_data.get('portfolio_value', 0):,.2f}
• Available Cash: ${session_data.get('available_cash', 0):,.2f}
• Active Positions: {session_data.get('active_positions', 0)}

📈 <b>Top Decisions:</b>
{self._format_top_decisions(session_data.get('top_decisions', []))}

⏰ Session Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        self.send_message(message)
    
    def _format_top_decisions(self, decisions: List) -> str:
        """Format top trading decisions for Telegram"""
        if not decisions:
            return "No significant decisions made"
        
        formatted = ""
        for i, decision in enumerate(decisions[:3], 1):
            emoji = "🟢" if decision['action'] == "BUY" else "🔴" if decision['action'] == "SELL" else "🟡"
            formatted += f"{i}. {emoji} {decision['ticker']}: {decision['action']} (Confidence: {decision['confidence']}/10)\n"
        
        return formatted

class AITrader:
    """Main AI trading system"""
    
    def __init__(self):
        print("🚀 Initializing AI Trading System...")
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.analyzer = StockAnalyzer(self.openai_client)
        self.screener = StockScreener()
        self.notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        
        # Initialize Alpaca with new SDK
        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        # Portfolio tracking
        self.portfolio = {}
        self.trade_history = []
        
        print("✅ AI Trading System initialized successfully!")
        
    def get_account_info(self) -> Dict:
        """Get account information from Alpaca"""
        try:
            print("📊 Fetching account information...")
            account = self.trading_client.get_account()
            info = {
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity)
            }
            print(f"✅ Account info: Portfolio value: ${info['portfolio_value']:,.2f}, Cash: ${info['cash']:,.2f}")
            return info
        except Exception as e:
            print(f"❌ Error getting account info: {e}")
            return {}
    
    def get_portfolio_positions(self) -> Dict:
        """Get current portfolio positions"""
        try:
            print("📈 Fetching portfolio positions...")
            positions = self.trading_client.get_all_positions()
            portfolio = {}
            
            for position in positions:
                portfolio[position.symbol] = {
                    'quantity': int(position.qty),
                    'avg_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl)
                }
            
            print(f"✅ Found {len(portfolio)} active positions")
            return portfolio
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return {}
    
    def place_trade(self, ticker: str, action: str, quantity: int = 1, reason: str = ""):
        """Place a trade on Alpaca"""
        try:
            print(f"📝 Placing {action} order for {quantity} shares of {ticker}...")
            
            # Create order request
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            
            # Submit order
            order = self.trading_client.submit_order(order_data)
            
            if order:
                self.notifier.send_trade_alert(ticker, action, 0, reason)
                
                # Record trade
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': action,
                    'quantity': quantity,
                    'reason': reason
                })
                
                print(f"✅ Trade placed successfully: {action} {quantity} {ticker}")
                return True
            else:
                print(f"❌ Failed to place trade for {ticker}")
                return False
            
        except Exception as e:
            print(f"❌ Trade failed for {ticker}: {e}")
            return False
    
    def analyze_and_trade(self, max_trades: int = 3, min_confidence: int = 5):
        """Main analysis and trading function"""
        print("\n" + "="*60)
        print("🤖 STARTING AI ANALYSIS AND TRADING SESSION")
        print("="*60)
        
        session_data = {
            'stocks_analyzed': 0,
            'trades_executed': 0,
            'buy_orders': 0,
            'sell_orders': 0,
            'trading_decisions': [],
            'top_decisions': []
        }
        
        # Get account info
        account = self.get_account_info()
        if not account:
            print("❌ Could not get account information - stopping session")
            return
        
        session_data['portfolio_value'] = account.get('portfolio_value', 0)
        session_data['available_cash'] = account.get('cash', 0)
        
        # Screen for stocks
        screened_stocks = self.screener.screen_stocks()
        session_data['stocks_analyzed'] = len(screened_stocks)
        
        print(f"\n🔍 Screening Results:")
        print(f"  Total stocks screened: {len(screened_stocks)}")
        
        if not screened_stocks:
            print("❌ No stocks passed screening - stopping session")
            return
        
        # Analyze each stock (limit to save API costs)
        analysis_results = []
        stocks_to_analyze = screened_stocks[:10]  # Limit to top 10 for efficiency
        
        print(f"\n🔍 Analyzing {len(stocks_to_analyze)} stocks...")
        
        for stock in stocks_to_analyze:
            ticker = stock['ticker']
            
            try:
                print(f"\n📊 Analyzing {ticker}...")
                
                # Get technical data
                data = self.analyzer.get_stock_data(ticker)
                if data.empty:
                    continue
                
                indicators = self.analyzer.calculate_technical_indicators(data)
                indicators['current_price'] = stock['current_price']
                
                # Get news sentiment
                sentiment, news = self.analyzer.get_news_sentiment(ticker)
                
                # AI analysis
                ai_result = self.analyzer.analyze_stock_with_ai(ticker, indicators, sentiment, news)
                
                # Create charts for PDF report
                price_chart = self.analyzer.pdf_generator.create_price_chart(ticker, data)
                technical_chart = self.analyzer.pdf_generator.create_technical_indicators_chart(ticker, data, indicators)
                
                analysis_results.append({
                    'ticker': ticker,
                    'stock_info': stock,
                    'indicators': indicators,
                    'sentiment': sentiment,
                    'news': news,
                    'ai_analysis': ai_result,
                    'price_chart': price_chart,
                    'technical_chart': technical_chart
                })
                
                session_data['stocks_analyzed'] += 1
                
            except Exception as e:
                print(f"❌ Error analyzing {ticker}: {e}")
                continue
        
        # Sort by confidence and make trading decisions
        analysis_results.sort(key=lambda x: x['ai_analysis']['confidence'], reverse=True)
        
        print(f"\n📋 Trading decisions sorted by confidence:")
        for i, result in enumerate(analysis_results[:5]):
            analysis = result['ai_analysis']
            print(f"{i+1}. {result['ticker']}: {analysis['decision']} (Confidence: {analysis['confidence']}/10)")
        
        # Debug: Show all decisions and confidence levels
        print(f"\n🔍 DEBUG: All AI decisions (min_confidence={min_confidence}):")
        for result in analysis_results:
            analysis = result['ai_analysis']
            confidence = analysis['confidence']
            decision = analysis['decision']
            status = "✅ PASS" if confidence >= min_confidence else "❌ LOW CONFIDENCE"
            print(f"  {result['ticker']}: {decision} (Confidence: {confidence}/10) - {status}")
        
        # Execute trades
        trades_made = 0
        current_positions = self.get_portfolio_positions()
        session_data['active_positions'] = len(current_positions)
        
        print(f"\n💰 Current positions: {list(current_positions.keys()) if current_positions else 'None'}")
        print(f"💵 Portfolio value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"💵 Available cash: ${account.get('cash', 0):,.2f}")
        
        for result in analysis_results:
            if trades_made >= max_trades:
                print(f"⏹️ Reached max trades limit ({max_trades})")
                break
                
            ticker = result['ticker']
            analysis = result['ai_analysis']
            
            if analysis['confidence'] < min_confidence:
                print(f"⏭️ Skipping {ticker}: Confidence too low ({analysis['confidence']}/10)")
                continue
            
            decision = analysis['decision']
            reasoning = analysis['reasoning']
            
            # Add to top decisions for summary
            session_data['top_decisions'].append({
                'ticker': ticker,
                'action': decision,
                'confidence': analysis['confidence'],
                'reasoning': reasoning
            })
            
            print(f"\n🎯 Processing {ticker}: {decision} (Confidence: {analysis['confidence']}/10)")
            
            if decision == "BUY" and ticker not in current_positions:
                # Calculate position size (2% of portfolio per trade)
                portfolio_value = account['portfolio_value']
                position_size = portfolio_value * 0.02
                price = result['stock_info']['current_price']
                quantity = max(1, int(position_size / price))
                
                print(f"  💰 Position size: ${position_size:,.2f} ({quantity} shares @ ${price:.2f})")
                
                if self.place_trade(ticker, "BUY", quantity, reasoning):
                    trades_made += 1
                    session_data['trades_executed'] += 1
                    session_data['buy_orders'] += 1
                    print(f"  ✅ BUY trade executed for {ticker}")
                else:
                    print(f"  ❌ BUY trade failed for {ticker}")
                    
            elif decision == "SELL" and ticker in current_positions:
                quantity = current_positions[ticker]['quantity']
                print(f"  📉 Selling {quantity} shares of {ticker}")
                
                if self.place_trade(ticker, "SELL", quantity, reasoning):
                    trades_made += 1
                    session_data['trades_executed'] += 1
                    session_data['sell_orders'] += 1
                    print(f"  ✅ SELL trade executed for {ticker}")
                else:
                    print(f"  ❌ SELL trade failed for {ticker}")
            else:
                if decision == "BUY":
                    print(f"  ⏭️ Skipping BUY: {ticker} already in portfolio")
                elif decision == "SELL":
                    print(f"  ⏭️ Skipping SELL: {ticker} not in portfolio")
                else:
                    print(f"  ⏭️ Skipping HOLD: {ticker}")
        
        # Store trading decisions for PDF report
        session_data['trading_decisions'] = [
            {
                'ticker': result['ticker'],
                'action': result['ai_analysis']['decision'],
                'confidence': result['ai_analysis']['confidence'],
                'reasoning': result['ai_analysis']['reasoning'],
                'risk': result['ai_analysis']['risk'],
                'price_chart': result.get('price_chart'),
                'technical_chart': result.get('technical_chart')
            }
            for result in analysis_results[:5]  # Top 5 decisions
        ]
        
        print(f"\n✅ Trading session completed: {trades_made} trades executed")
        
        # Send summary and generate PDF
        self.notifier.send_session_summary(session_data)
        
        # Generate and send PDF report
        pdf_buffer = self.analyzer.pdf_generator.generate_trading_report(session_data)
        if pdf_buffer:
            filename = f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            self.notifier.send_document(
                pdf_buffer.getvalue(),
                filename,
                f"📊 Detailed AI Trading Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
        
        print("="*60)
        print("✅ AI TRADING SESSION COMPLETED")
        print("="*60)
        
        return session_data

def main():
    """Main function to run the AI trader"""
    print("🚀 Starting AI Trading System...")
    trader = AITrader()
    
    # Send startup notification
    trader.notifier.send_message("🤖 AI Trader started successfully!")
    
    # Run analysis and trading with lower confidence threshold
    session_data = trader.analyze_and_trade(max_trades=3, min_confidence=5)
    
    print("🎉 AI Trading System completed successfully!")

if __name__ == "__main__":
    main() 