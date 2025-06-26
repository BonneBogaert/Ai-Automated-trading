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
import json
from typing import List, Dict, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import matplotlib.font_manager as fm
import random

# Import all configuration dictionaries from config.py
from config import (
    TRADING_CONFIG, RISK_CONFIG, AI_CONFIG, SCREENING_CONFIG,
    NOTIFICATION_CONFIG, ETHICAL_CONFIG, STOCK_UNIVERSE, TECHNICAL_CONFIG, LOGGING_CONFIG
)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Configure fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 10

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
        """Create simple price chart"""
        try:
            if data.empty:
                print(f"⚠️ No data available for {ticker} price chart")
                return None

            # Simple approach - just plot the data
            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data['Close'], label='Close Price', linewidth=2)
            plt.title(f'{ticker} Price Chart')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
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
        """Create comprehensive technical indicators chart with enhanced visuals"""
        try:
            if data.empty:
                return None

            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                ticker_name = data.columns.get_level_values(1)[0]
                close_data = data[('Close', ticker_name)]
                volume_data = data[('Volume', ticker_name)]
            else:
                close_data = data['Close']
                volume_data = data['Volume']

            # Set style for better visuals
            plt.style.use('seaborn-v0_8')
            
            # Create comprehensive technical analysis chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle(f'{ticker} Technical Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
            
            # Color scheme
            colors = {
                'price': '#1f77b4',
                'sma20': '#ff7f0e',
                'sma50': '#d62728',
                'volume': '#2ca02c',
                'rsi': '#9467bd',
                'macd': '#8c564b',
                'signal': '#e377c2',
                'histogram': '#7f7f7f',
                'overbought': '#ff4444',
                'oversold': '#44ff44',
                'neutral': '#ffaa00'
            }
            
            # Price chart with moving averages and decision clues
            ax1.plot(data.index, close_data, label='Close Price', linewidth=2.5, color=colors['price'], alpha=0.9)
            
            # Add moving averages if available
            if 'sma_20' in indicators and 'sma_50' in indicators:
                sma_20_series = close_data.rolling(window=20).mean()
                sma_50_series = close_data.rolling(window=50).mean()
                
                ax1.plot(data.index, sma_20_series, label='SMA 20', linewidth=2, color=colors['sma20'], alpha=0.8)
                ax1.plot(data.index, sma_50_series, label='SMA 50', linewidth=2, color=colors['sma50'], alpha=0.8)
                
                # Add decision clues based on moving average crossovers
                current_price = close_data.iloc[-1]
                sma_20_current = sma_20_series.iloc[-1]
                sma_50_current = sma_50_series.iloc[-1]
                
                # Add markers for key levels
                ax1.axhline(y=sma_20_current, color=colors['sma20'], linestyle='--', alpha=0.6, label=f'SMA 20: ${sma_20_current:.2f}')
                ax1.axhline(y=sma_50_current, color=colors['sma50'], linestyle='--', alpha=0.6, label=f'SMA 50: ${sma_50_current:.2f}')
                
                # Add decision annotation
                if current_price > sma_20_current > sma_50_current:
                    decision_text = "G BULLISH: Price > SMA20 > SMA50"
                    decision_color = colors['neutral']
                elif current_price < sma_20_current < sma_50_current:
                    decision_text = "R BEARISH: Price < SMA20 < SMA50"
                    decision_color = colors['overbought']
                else:
                    decision_text = "Y MIXED: Mixed signals"
                    decision_color = colors['neutral']
                
                # Try to use emoji, fallback to text if font doesn't support it
                ax1.text(0.02, 0.98, decision_text, transform=ax1.transAxes, fontsize=12, fontname='DejaVu Sans',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=decision_color, alpha=0.7),
                        verticalalignment='top')
            
            # Add Bollinger Bands if available
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                bb_upper_series = close_data.rolling(window=20).mean() + (close_data.rolling(window=20).std() * 2)
                bb_lower_series = close_data.rolling(window=20).mean() - (close_data.rolling(window=20).std() * 2)
                
                ax1.fill_between(data.index, bb_upper_series, bb_lower_series, alpha=0.1, color='gray', label='Bollinger Bands')
                ax1.plot(data.index, bb_upper_series, color='gray', linestyle=':', alpha=0.7, linewidth=1)
                ax1.plot(data.index, bb_lower_series, color='gray', linestyle=':', alpha=0.7, linewidth=1)
            
            ax1.set_title(f'{ticker} Price Chart with Technical Indicators', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Add current price annotation
            ax1.annotate(f'${current_price:.2f}', xy=(data.index[-1], current_price), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            # Volume chart with enhanced styling
            volume_colors = ['green' if close_data.iloc[i] >= close_data.iloc[i-1] else 'red' 
                           for i in range(1, len(close_data))]
            volume_colors.insert(0, 'green')  # First bar
            
            ax2.bar(data.index, volume_data, alpha=0.7, color=volume_colors, label='Volume')
            
            # Add volume moving average
            volume_sma = volume_data.rolling(window=20).mean()
            ax2.plot(data.index, volume_sma, color='blue', linewidth=2, label='Volume SMA (20)', alpha=0.8)
            
            ax2.set_title(f'{ticker} Volume Analysis', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.legend(loc='upper left', fontsize=10)
            ax2.grid(True, alpha=0.3)

            # RSI chart with enhanced styling
            if 'rsi' in indicators:
                delta = close_data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_series = 100 - (100 / (1 + rs))
                
                ax3.plot(data.index, rsi_series, label='RSI (14)', linewidth=2.5, color=colors['rsi'])
                
                # Add overbought/oversold zones with fill
                ax3.fill_between(data.index, 70, 100, alpha=0.2, color=colors['overbought'], label='Overbought Zone')
                ax3.fill_between(data.index, 0, 30, alpha=0.2, color=colors['oversold'], label='Oversold Zone')
                
                ax3.axhline(y=70, color=colors['overbought'], linestyle='--', alpha=0.8, linewidth=2, label='Overbought (70)')
                ax3.axhline(y=30, color=colors['oversold'], linestyle='--', alpha=0.8, linewidth=2, label='Oversold (30)')
                ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.5, linewidth=1, label='Neutral (50)')
                
                # Add RSI decision annotation
                current_rsi = rsi_series.iloc[-1]
                if current_rsi > 70:
                    rsi_decision = "R OVERBOUGHT: Consider selling"
                    rsi_color = colors['overbought']
                elif current_rsi < 30:
                    rsi_decision = "G OVERSOLD: Consider buying"
                    rsi_color = colors['oversold']
                else:
                    rsi_decision = "Y NEUTRAL: No extreme signals"
                    rsi_color = colors['neutral']
                
                ax3.text(0.02, 0.98, f"{rsi_decision}\nRSI: {current_rsi:.1f}", 
                        transform=ax3.transAxes, fontsize=11, fontname='DejaVu Sans',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=rsi_color, alpha=0.7),
                        verticalalignment='top')
                
                ax3.set_title(f'{ticker} RSI (14) Momentum Indicator', fontsize=14, fontweight='bold')
                ax3.set_ylabel('RSI', fontsize=12)
                ax3.set_ylim(0, 100)
                ax3.legend(loc='upper left', fontsize=10)
                ax3.grid(True, alpha=0.3)

            # MACD chart with enhanced styling
            if 'macd' in indicators:
                ema_12_series = close_data.ewm(span=12).mean()
                ema_26_series = close_data.ewm(span=26).mean()
                macd_series = ema_12_series - ema_26_series
                signal_series = macd_series.ewm(span=9).mean()
                histogram = macd_series - signal_series
                
                # Color histogram based on MACD vs Signal
                histogram_colors = ['green' if h >= 0 else 'red' for h in histogram]
                
                ax4.plot(data.index, macd_series, label='MACD', linewidth=2.5, color=colors['macd'])
                ax4.plot(data.index, signal_series, label='Signal', linewidth=2.5, color=colors['signal'])
                ax4.bar(data.index, histogram, alpha=0.6, color=histogram_colors, label='Histogram', width=0.8)
                
                # Add zero line
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                
                # Add MACD decision annotation
                current_macd = macd_series.iloc[-1]
                current_signal = signal_series.iloc[-1]
                
                if current_macd > current_signal and current_macd > 0:
                    macd_decision = "G BULLISH: MACD > Signal > 0"
                    macd_color = colors['oversold']
                elif current_macd < current_signal and current_macd < 0:
                    macd_decision = "R BEARISH: MACD < Signal < 0"
                    macd_color = colors['overbought']
                elif current_macd > current_signal:
                    macd_decision = "Y WEAK: MACD > Signal"
                    macd_color = colors['neutral']
                else:
                    macd_decision = "Y WEAK: MACD < Signal"
                    macd_color = colors['neutral']
                
                ax4.text(0.02, 0.98, f"{macd_decision}\nMACD: {current_macd:.4f}", 
                        transform=ax4.transAxes, fontsize=11, fontname='DejaVu Sans',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=macd_color, alpha=0.7),
                        verticalalignment='top')
                
                ax4.set_title(f'{ticker} MACD Trend Indicator', fontsize=14, fontweight='bold')
                ax4.set_ylabel('MACD', fontsize=12)
                ax4.legend(loc='upper left', fontsize=10)
                ax4.grid(True, alpha=0.3)

            # Overall styling
            plt.tight_layout()
            
            # Add timestamp
            fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=10, ha='right', va='bottom', alpha=0.7)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except Exception as e:
            logging.error(f"Error creating technical indicators chart for {ticker}: {e}")
            return None
    
    def generate_trading_report(self, trading_session: Dict, analyzer=None) -> str:
        """Generate comprehensive PDF trading report with vertical stack layout per stock"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []
            story.append(Paragraph("AI Trading System Report", self.title_style))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
            story.append(Spacer(1, 20))
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
            # For each trading decision, add a section with header and vertical stack of graphs
            if trading_session.get('trading_decisions'):
                story.append(Paragraph("Trading Decisions & Technical Analysis", self.subtitle_style))
                for decision in trading_session['trading_decisions']:
                    ticker = decision['ticker']
                    story.append(Paragraph(f"{ticker} - {decision['action']} (Confidence: {decision['confidence']}/10)", self.styles['Heading3']))
                    story.append(Paragraph(f"Risk Level: {decision['risk']}", self.styles['Normal']))
                    story.append(Paragraph(f"Reasoning: {decision['reasoning']}", self.styles['Normal']))
                    story.append(Spacer(1, 10))
                    # Generate and add the full analysis figure
                    if analyzer:
                        data = analyzer.get_stock_data(ticker, period="6mo")
                        indicators = analyzer.calculate_technical_indicators(data)
                        # For RSI/MACD, add full series to indicators
                        if not data.empty:
                            # RSI
                            delta = data['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi_series = 100 - (100 / (1 + rs))
                            indicators['rsi_series'] = rsi_series
                            # MACD
                            ema_12_series = data['Close'].ewm(span=12).mean()
                            ema_26_series = data['Close'].ewm(span=26).mean()
                            macd_series = ema_12_series - ema_26_series
                            signal_series = macd_series.ewm(span=9).mean()
                            histogram = macd_series - signal_series
                            indicators['macd_series'] = macd_series
                            indicators['signal_series'] = signal_series
                            indicators['histogram'] = histogram
                            # Create the figure
                            fig_buffer, n_graphs = self.create_full_stock_analysis_figure(ticker, data, indicators)
                            if fig_buffer:
                                # Calculate appropriate image size for PDF page - reduced further
                                max_width = 5.5 * inch  # Reduced from 6.5*inch
                                max_height = 2.0 * inch * n_graphs  # Reduced from 2.5*inch
                                story.append(Image(fig_buffer, width=max_width, height=max_height))
                                story.append(Spacer(1, 20))
            doc.build(story)
            buffer.seek(0)
            return buffer
        except Exception as e:
            logging.error(f"Error generating PDF report: {e}")
            return None
    
    def create_full_stock_analysis_figure(self, ticker, data, indicators):
        """Create a vertical stack of price, volume, RSI, MACD for a stock"""
        try:
            n_graphs = 2  # Always price and volume
            if 'rsi' in indicators:
                n_graphs += 1
            if 'macd' in indicators:
                n_graphs += 1
            fig, axes = plt.subplots(n_graphs, 1, figsize=(10, 3*n_graphs), sharex=True)
            if n_graphs == 1:
                axes = [axes]
            fig.suptitle(f"{ticker} - Technical Analysis", fontsize=16, fontweight='bold')
            idx = 0
            # 1. Price
            try:
                # Ensure 'Close' and 'Volume' are Series, not DataFrames
                close_series = data['Close']
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                volume_series = data['Volume']
                if isinstance(volume_series, pd.DataFrame):
                    volume_series = volume_series.iloc[:, 0]

                axes[idx].plot(data.index, close_series, label='Close Price')
                axes[idx].set_ylabel('Price')
                axes[idx].legend()
                axes[idx].set_title('Price Chart')
            except Exception as e:
                logging.error(f"Error plotting price for {ticker}: {e}")
            idx += 1
            # 2. Volume
            try:
                axes[idx].bar(data.index, volume_series, label='Volume')
                axes[idx].set_ylabel('Volume')
                axes[idx].legend()
                axes[idx].set_title('Volume Chart')
            except Exception as e:
                logging.error(f"Error plotting volume for {ticker}: {e}")
            idx += 1
            # 3. RSI
            if 'rsi' in indicators:
                rsi_series = indicators.get('rsi_series')
                # Ensure rsi_series is a Series, not a DataFrame
                if isinstance(rsi_series, pd.DataFrame):
                    rsi_series = rsi_series.iloc[:, 0]
                try:
                    if rsi_series is not None and hasattr(rsi_series, '__len__') and len(rsi_series) == len(data.index):
                        axes[idx].plot(data.index, rsi_series, label='RSI')
                        axes[idx].set_ylabel('RSI')
                        axes[idx].legend()
                        axes[idx].set_title('RSI Chart')
                except Exception as e:
                    logging.error(f"Error plotting RSI for {ticker}: {e}")
                idx += 1
            # 4. MACD
            if 'macd' in indicators:
                macd_series = indicators.get('macd_series')
                signal_series = indicators.get('signal_series')
                histogram = indicators.get('histogram')
                # Ensure all are Series, not DataFrames
                if isinstance(macd_series, pd.DataFrame):
                    macd_series = macd_series.iloc[:, 0]
                if isinstance(signal_series, pd.DataFrame):
                    signal_series = signal_series.iloc[:, 0]
                if isinstance(histogram, pd.DataFrame):
                    histogram = histogram.iloc[:, 0]
                try:
                    if (macd_series is not None and signal_series is not None and histogram is not None and
                        hasattr(macd_series, '__len__') and hasattr(signal_series, '__len__') and hasattr(histogram, '__len__') and
                        len(macd_series) == len(data.index) and len(signal_series) == len(data.index) and len(histogram) == len(data.index)):
                        axes[idx].plot(data.index, macd_series, label='MACD')
                        axes[idx].plot(data.index, signal_series, label='Signal')
                        axes[idx].bar(data.index, histogram, label='Histogram')
                        axes[idx].set_ylabel('MACD')
                        axes[idx].legend()
                        axes[idx].set_title('MACD Chart')
                except Exception as e:
                    logging.error(f"Error plotting MACD for {ticker}: {e}")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer, n_graphs
        except Exception as e:
            logging.error(f"Error creating full stock analysis figure for {ticker}: {e}")
            return None, 0
    
    def create_decision_matrix_chart(self, decisions: List[Dict], analyzer=None) -> str:
        """Create a matrix layout of technical charts for multiple decisions"""
        try:
            if not decisions:
                return None
            
            # Determine matrix dimensions
            n_decisions = len(decisions)
            if n_decisions <= 2:
                cols = 2
                rows = 1
            elif n_decisions <= 4:
                cols = 2
                rows = 2
            else:
                cols = 3
                rows = (n_decisions + 2) // 3  # Ceiling division
            
            # Create subplot grid
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
            fig.suptitle('Technical Analysis Matrix', fontsize=16, fontweight='bold', y=0.98)
            
            # Ensure axes is always a 2D array
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Color scheme for consistency
            colors = {
                'price': '#1f77b4',
                'sma20': '#ff7f0e',
                'sma50': '#d62728',
                'volume': '#2ca02c',
                'rsi': '#9467bd',
                'macd': '#8c564b',
                'signal': '#e377c2',
                'overbought': '#ff4444',
                'oversold': '#44ff44',
                'neutral': '#ffaa00'
            }
            
            for idx, decision in enumerate(decisions):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col]
                
                ticker = decision['ticker']
                
                # Get stock data for this ticker
                try:
                    if analyzer:
                        stock_data = analyzer.get_stock_data(ticker, period="1mo")
                    else:
                        # Fallback: try to get data directly
                        stock_data = yf.download(ticker, period="1mo", interval="1d", auto_adjust=True, progress=False)
                    
                    if stock_data.empty:
                        ax.text(0.5, 0.5, f'No data for {ticker}', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{ticker} - {decision["action"]}', fontsize=10, fontweight='bold')
                        continue
                    
                    # Handle MultiIndex if present
                    if isinstance(stock_data.columns, pd.MultiIndex):
                        ticker_name = stock_data.columns.get_level_values(1)[0]
                        close_data = stock_data[('Close', ticker_name)]
                        volume_data = stock_data[('Volume', ticker_name)]
                    else:
                        close_data = stock_data['Close']
                        volume_data = stock_data['Volume']
                    
                    # Create mini technical chart
                    # Price and moving averages
                    ax.plot(stock_data.index, close_data, label='Price', linewidth=1.5, color=colors['price'], alpha=0.8)
                    
                    # Add moving averages
                    sma_20 = close_data.rolling(window=20).mean()
                    sma_50 = close_data.rolling(window=50).mean()
                    ax.plot(stock_data.index, sma_20, label='SMA20', linewidth=1, color=colors['sma20'], alpha=0.7)
                    ax.plot(stock_data.index, sma_50, label='SMA50', linewidth=1, color=colors['sma50'], alpha=0.7)
                    
                    # Add decision annotation
                    current_price = close_data.iloc[-1]
                    decision_letter = "G" if decision['action'] == "BUY" else "R" if decision['action'] == "SELL" else "Y"
                    
                    # Color based on decision
                    if decision['action'] == "BUY":
                        bg_color = colors['oversold']
                    elif decision['action'] == "SELL":
                        bg_color = colors['overbought']
                    else:
                        bg_color = colors['neutral']
                    
                    # Try to use emoji, fallback to text if font doesn't support it
                    ax.text(0.02, 0.98, f"{decision_letter} {decision['action']}\nConf: {decision['confidence']}/10", 
                           transform=ax.transAxes, fontsize=8, fontname='DejaVu Sans',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color, alpha=0.7),
                           verticalalignment='top')
                    
                    # Add current price annotation
                    ax.annotate(f'${current_price:.2f}', xy=(stock_data.index[-1], current_price), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8,
                              bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=0.5))
                    
                    ax.set_title(f'{ticker} - {decision["action"]} ({decision["confidence"]}/10)', 
                               fontsize=10, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    
                    # Rotate x-axis labels for better readability
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {ticker}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{ticker} - Error', fontsize=10, fontweight='bold')
            
            # Hide empty subplots
            for idx in range(n_decisions, rows * cols):
                row = idx // cols
                col = idx % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # Save to buffer
            chart_buffer = io.BytesIO()
            plt.savefig(chart_buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            chart_buffer.seek(0)
            plt.close()
            
            return chart_buffer
            
        except Exception as e:
            logging.error(f"Error creating decision matrix chart: {e}")
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
                
        #print(f"✅ {company_info.get('symbol', 'Unknown')}: Passed ethical screening")
        return True

class StockAnalyzer:
    """Analyze stocks using technical and fundamental data"""
    
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        self.pdf_generator = PDFReportGenerator()
        
    def get_stock_data(self, ticker: str, period: str = None) -> pd.DataFrame:
        """Get stock data from Yahoo Finance"""
        try:
            if period is None:
                period = AI_CONFIG['analysis_period']
            print(f"📊 Fetching data for {ticker}...")
            data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
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
            
        indicators = {}
        
        def safe_scalar(value, default=0.0):
            """Safely extract scalar value from pandas Series/DataFrame"""
            try:
                if hasattr(value, 'iloc'):
                    # Handle Series with MultiIndex or complex structure
                    if hasattr(value, 'values'):
                        return float(value.values[-1])
                    else:
                        return float(value.iloc[-1])
                elif isinstance(value, (np.ndarray, list)):
                    return float(value[-1])
                else:
                    return float(value)
            except Exception as e:
                return default
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            ticker_name = data.columns.get_level_values(1)[0]
            close_data = data[('Close', ticker_name)]
            volume_data = data[('Volume', ticker_name)]
        else:
            close_data = data['Close']
            volume_data = data['Volume']
        
        # Moving averages
        sma_20_series = close_data.rolling(window=20).mean()
        sma_50_series = close_data.rolling(window=50).mean()
        ema_12_series = close_data.ewm(span=12).mean()
        ema_26_series = close_data.ewm(span=26).mean()
        
        indicators['sma_20'] = safe_scalar(sma_20_series)
        indicators['sma_50'] = safe_scalar(sma_50_series)
        indicators['ema_12'] = safe_scalar(ema_12_series)
        indicators['ema_26'] = safe_scalar(ema_26_series)
        
        # RSI
        delta = close_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        indicators['rsi'] = safe_scalar(rsi_series)
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        
        # Bollinger Bands
        sma_20_series = close_data.rolling(window=20).mean()
        std_20_series = close_data.rolling(window=20).std()
        bb_upper_series = sma_20_series + (std_20_series * 2)
        bb_lower_series = sma_20_series - (std_20_series * 2)
        
        indicators['bb_upper'] = safe_scalar(bb_upper_series)
        indicators['bb_lower'] = safe_scalar(bb_lower_series)
        indicators['bb_position'] = (safe_scalar(close_data) - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Volume analysis
        volume_sma_series = volume_data.rolling(window=20).mean()
        indicators['volume_sma'] = safe_scalar(volume_sma_series)
        indicators['volume_ratio'] = safe_scalar(volume_data) / indicators['volume_sma']
        
        # Price momentum
        current_price = safe_scalar(close_data)
        price_5_days_ago = safe_scalar(close_data.iloc[-5] if len(close_data) >= 5 else close_data.iloc[0])
        indicators['price_momentum'] = (current_price / price_5_days_ago - 1) * 100
        
        return indicators
    
    def get_news_sentiment(self, ticker: str) -> Tuple[float, str]:
        """Get news sentiment for a ticker"""
        try:
            print(f"📰 Fetching news for {ticker}...")
            try:
                stock_info = yf.Ticker(ticker).info
                company_name = stock_info.get('longName', ticker) if stock_info else ticker
            except Exception as e:
                print(f"⚠️ Could not get company info for {ticker}: {e}")
                company_name = ticker

            # Check if NEWSAPI_KEY is available
            if not NEWSAPI_KEY:
                print(f"⚠️ No NewsAPI key configured for {ticker}")
                return 0.0, "No news API configured."

            # Try with ticker first
            url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWSAPI_KEY}&language=en&sortBy=publishedAt&pageSize=5"
            response = requests.get(url)
            news_data = response.json()
            #print(f"[DEBUG] NewsAPI response for {ticker}: {news_data}")

            # If no news, try with company_name
            if news_data.get('status') != 'ok' or not news_data.get('articles'):
                print(f"⚠️ No news found for ticker {ticker}, trying company name: {company_name}")
                url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey={NEWSAPI_KEY}&language=en&sortBy=publishedAt&pageSize=5"
                response = requests.get(url)
                news_data = response.json()
                #print(f"[DEBUG] NewsAPI response for {company_name}: {news_data}")

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
                model=AI_CONFIG['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=AI_CONFIG['temperature'],
                max_tokens=AI_CONFIG['max_tokens']
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
        # Use STOCK_UNIVERSE from config
        stocks = []
        for sector_list in STOCK_UNIVERSE.values():
            stocks.extend(sector_list)
        return stocks
    
    def screen_stocks(self, min_market_cap: float = None) -> List[Dict]:
        """Screen stocks based on criteria"""
        stocks = self.get_stock_universe()
        screened_stocks = []
        print(f"🔍 Screening {len(stocks)} stocks...")
        if min_market_cap is None:
            min_market_cap = TRADING_CONFIG['min_market_cap']
        for ticker in stocks:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', 0)
                current_price = info.get('regularMarketPrice', 0)
                volume = info.get('volume', 0)
                avg_volume = info.get('averageVolume', 0)
                if market_cap < min_market_cap:
                    print(f"⏭️ Skipped {ticker}: Market cap too low (${market_cap:,.0f} < ${min_market_cap:,.0f})")
                    continue
                if current_price < TRADING_CONFIG['min_stock_price']:
                    print(f"⏭️ Skipped {ticker}: Price too low (${current_price:.2f} < ${TRADING_CONFIG['min_stock_price']:.2f})")
                    continue
                if current_price > TRADING_CONFIG['max_stock_price']:
                    print(f"⏭️ Skipped {ticker}: Price too high (${current_price:.2f} > ${TRADING_CONFIG['max_stock_price']:.2f})")
                    continue
                # Ethical filter
                if not self.ethical_filter.is_ethical(info):
                    continue
                data = stock.history(period="1mo")
                if data.empty:
                    print(f"⏭️ Skipped {ticker}: No price data available")
                    continue
                if avg_volume > 0 and volume < avg_volume * TRADING_CONFIG['min_volume_ratio']:
                    print(f"⏭️ Skipped {ticker}: Low volume (current: {volume:,.0f}, avg: {avg_volume:,.0f}, ratio: {volume/avg_volume:.2f})")
                    continue
                screened_stocks.append({
                    'ticker': ticker,
                    'name': info.get('longName', ticker),
                    'sector': info.get('sector', 'Unknown'),
                    'market_cap': market_cap,
                    'current_price': current_price,
                    'volume': volume,
                    'avg_volume': avg_volume,
                    'pe_ratio': info.get('trailingPE', 0),
                    'beta': info.get('beta', 1.0)
                })
            except Exception as e:
                print(f"❌ Error screening {ticker}: {e}")
                continue
        print(f"✅ Screening complete: {len(screened_stocks)} stocks passed")
        return screened_stocks

class TelegramNotifier:
    """Send notifications via Telegram"""
    
    def __init__(self, token: str, chat_id: str, trader=None):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.trader = trader
    
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
        # Get detailed position performance for invested amount and unrealized P&L
        performance = None
        invested_amount = None
        unrealized_pl = None
        unrealized_pl_pct = None
        try:
            performance = self.trader.get_position_performance()
            if performance and 'portfolio_summary' in performance:
                summary = performance['portfolio_summary']
                invested_amount = summary.get('invested_amount', 0)
                unrealized_pl = summary.get('total_unrealized_pl', 0)
                unrealized_pl_pct = summary.get('total_unrealized_pl_pct', 0)
        except Exception as e:
            print(f"[DEBUG] Could not fetch position performance for session summary: {e}")

        message = (
            f"🤖 <b>AI Trading Session Summary</b>\n\n"
            f"📊 <b>Analysis Results:</b>\n"
            f"• Stocks Analyzed: {session_data.get('stocks_analyzed', 0)}\n"
            f"• Trades Executed: {session_data.get('trades_executed', 0)}\n"
            f"• Buy Orders: {session_data.get('buy_orders', 0)}\n"
            f"• Sell Orders: {session_data.get('sell_orders', 0)}\n\n"
            f"💰 <b>Portfolio Status:</b>\n"
            f"• Total Value: ${session_data.get('portfolio_value', 0):,.2f}\n"
            f"• Available Cash: ${session_data.get('available_cash', 0):,.2f}"
        )
        if invested_amount is not None:
            message += f"\n• Invested Amount: ${invested_amount:,.2f}"
        if unrealized_pl is not None:
            message += f"\n• Unrealized P&L: ${unrealized_pl:,.2f} ({unrealized_pl_pct:+.2f}%)"
        message += f"\n• Active Positions: {session_data.get('active_positions', 0)}\n\n"
        message += (
            f"📈 <b>Top Decisions:</b>\n"
            f"{self._format_top_decisions(session_data.get('top_decisions', []))}\n\n"
            f"⏰ Session Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
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
        self.notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, trader=self)
        
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
    
    def analyze_and_trade(self, max_trades: int = None, min_confidence: int = None):
        """Main analysis and trading function"""
        print("\n" + "="*60)
        print("🤖 STARTING AI ANALYSIS AND TRADING SESSION")
        print("="*60)
        if max_trades is None:
            max_trades = TRADING_CONFIG['max_trades_per_session']
        if min_confidence is None:
            min_confidence = TRADING_CONFIG['min_confidence_score']
        
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
        
        # Get current portfolio positions
        current_positions = self.get_portfolio_positions()
        portfolio_tickers = set(current_positions.keys())
        portfolio_stocks = []
        other_stocks = []
        stocks_to_analyze = []
        if len(current_positions) < 8:
            # Screen for stocks only if we have less than 8 positions
            screened_stocks = self.screener.screen_stocks()
            session_data['stocks_analyzed'] = len(screened_stocks)
            portfolio_stocks = [s for s in screened_stocks if s['ticker'] in portfolio_tickers]
            other_stocks = [s for s in screened_stocks if s['ticker'] not in portfolio_tickers]
            num_to_pick = min(8 - len(portfolio_stocks), len(other_stocks))
            random_others = random.sample(other_stocks, num_to_pick) if num_to_pick > 0 else []
            stocks_to_analyze = portfolio_stocks + random_others
            print(f"  Stocks to analyze this session: {[s['ticker'] for s in stocks_to_analyze]}")
        else:
            # Only analyze current portfolio stocks
            stocks_to_analyze = []
            for ticker, pos in current_positions.items():
                stocks_to_analyze.append({
                    'ticker': ticker,
                    'current_price': pos['current_price'],
                    # Add any other fields needed for downstream code
                })
            session_data['stocks_analyzed'] = len(stocks_to_analyze)
            print(f"  (Portfolio >=8) Only analyzing current positions: {[s['ticker'] for s in stocks_to_analyze]}")
        
        # Analyze each stock (limit to save API costs)
        analysis_results = []
        
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
                position_size = min(portfolio_value * TRADING_CONFIG['position_size_pct'], TRADING_CONFIG['max_position_size'])
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
        pdf_buffer = self.analyzer.pdf_generator.generate_trading_report(session_data, self.analyzer)
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

    def get_position_performance(self) -> Dict:
        """Get detailed performance analysis of current positions"""
        try:
            positions = self.get_portfolio_positions()
            account = self.get_account_info()
            
            if not positions or not account:
                return {}
            
            position_analysis = {
                'total_positions': len(positions),
                'total_market_value': 0,
                'total_unrealized_pl': 0,
                'total_unrealized_pl_pct': 0,
                'positions': {},
                'best_performer': None,
                'worst_performer': None,
                'portfolio_summary': {}
            }
            
            best_pl_pct = -float('inf')
            worst_pl_pct = float('inf')
            
            for ticker, position in positions.items():
                # Calculate position metrics
                quantity = position['quantity']
                avg_price = position['avg_price']
                current_price = position['current_price']
                market_value = position['market_value']
                unrealized_pl = position['unrealized_pl']
                
                # Calculate percentage return
                unrealized_pl_pct = ((current_price - avg_price) / avg_price) * 100
                
                # Calculate position size as percentage of portfolio
                portfolio_value = account.get('portfolio_value', 0)
                position_size_pct = (market_value / portfolio_value) * 100 if portfolio_value > 0 else 0
                
                # Get historical data for performance analysis
                try:
                    stock_data = self.analyzer.get_stock_data(ticker, period="1mo")
                    if not stock_data.empty:
                        # Handle MultiIndex if present
                        if isinstance(stock_data.columns, pd.MultiIndex):
                            ticker_name = stock_data.columns.get_level_values(1)[0]
                            close_data = stock_data[('Close', ticker_name)]
                        else:
                            close_data = stock_data['Close']
                        
                        # Calculate performance metrics
                        price_1w_ago = close_data.iloc[-7] if len(close_data) >= 7 else close_data.iloc[0]
                        price_1m_ago = close_data.iloc[0]
                        
                        performance_1w = ((current_price - price_1w_ago) / price_1w_ago) * 100
                        performance_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
                        
                        # Calculate volatility (standard deviation of returns)
                        returns = close_data.pct_change().dropna()
                        volatility = returns.std() * 100
                        
                    else:
                        performance_1w = 0
                        performance_1m = 0
                        volatility = 0
                        
                except Exception as e:
                    performance_1w = 0
                    performance_1m = 0
                    volatility = 0
                
                position_info = {
                    'ticker': ticker,
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'market_value': market_value,
                    'unrealized_pl': unrealized_pl,
                    'unrealized_pl_pct': unrealized_pl_pct,
                    'position_size_pct': position_size_pct,
                    'performance_1w': performance_1w,
                    'performance_1m': performance_1m,
                    'volatility': volatility,
                    'days_held': 0  # Could be calculated if we track entry dates
                }
                
                position_analysis['positions'][ticker] = position_info
                position_analysis['total_market_value'] += market_value
                position_analysis['total_unrealized_pl'] += unrealized_pl
                
                # Track best and worst performers
                if unrealized_pl_pct > best_pl_pct:
                    best_pl_pct = unrealized_pl_pct
                    position_analysis['best_performer'] = ticker
                
                if unrealized_pl_pct < worst_pl_pct:
                    worst_pl_pct = unrealized_pl_pct
                    position_analysis['worst_performer'] = ticker
            
            # Calculate total portfolio metrics
            if position_analysis['total_market_value'] > 0:
                position_analysis['total_unrealized_pl_pct'] = (
                    position_analysis['total_unrealized_pl'] / 
                    (position_analysis['total_market_value'] - position_analysis['total_unrealized_pl'])
                ) * 100
            
            # Portfolio summary
            position_analysis['portfolio_summary'] = {
                'total_value': account.get('portfolio_value', 0),
                'cash': account.get('cash', 0),
                'invested_amount': position_analysis['total_market_value'],
                'total_unrealized_pl': position_analysis['total_unrealized_pl'],
                'total_unrealized_pl_pct': position_analysis['total_unrealized_pl_pct'],
                'positions_count': position_analysis['total_positions']
            }
            
            return position_analysis
            
        except Exception as e:
            logging.error(f"Error getting position performance: {e}")
            return {}
    
    def generate_position_report(self) -> str:
        """Generate a detailed position performance report"""
        try:
            performance = self.get_position_performance()
            
            if not performance:
                return "No positions to report."
            
            # Create detailed report
            report_lines = []
            report_lines.append("📊 **POSITION PERFORMANCE REPORT**")
            report_lines.append("=" * 50)
            
            # Portfolio Summary
            summary = performance['portfolio_summary']
            report_lines.append(f"💰 **Portfolio Summary:**")
            report_lines.append(f"Total Portfolio Value: ${summary['total_value']:,.2f}")
            report_lines.append(f"Cash Available: ${summary['cash']:,.2f}")
            report_lines.append(f"Invested Amount: ${summary['invested_amount']:,.2f}")
            report_lines.append(f"Total Unrealized P&L: ${summary['total_unrealized_pl']:,.2f} ({summary['total_unrealized_pl_pct']:+.2f}%)")
            report_lines.append(f"Active Positions: {summary['positions_count']}")
            report_lines.append("")
            
            # Best and Worst Performers
            if performance['best_performer']:
                best = performance['positions'][performance['best_performer']]
                report_lines.append(f"🏆 **Best Performer:** {performance['best_performer']}")
                report_lines.append(f"   Unrealized P&L: ${best['unrealized_pl']:,.2f} ({best['unrealized_pl_pct']:+.2f}%)")
                report_lines.append(f"   1-Week Performance: {best['performance_1w']:+.2f}%")
                report_lines.append(f"   1-Month Performance: {best['performance_1m']:+.2f}%")
                report_lines.append("")
            
            if performance['worst_performer']:
                worst = performance['positions'][performance['worst_performer']]
                report_lines.append(f"📉 **Worst Performer:** {performance['worst_performer']}")
                report_lines.append(f"   Unrealized P&L: ${worst['unrealized_pl']:,.2f} ({worst['unrealized_pl_pct']:+.2f}%)")
                report_lines.append(f"   1-Week Performance: {worst['performance_1w']:+.2f}%")
                report_lines.append(f"   1-Month Performance: {worst['performance_1m']:+.2f}%")
                report_lines.append("")
            
            # Individual Position Details
            report_lines.append("📈 **Position Details:**")
            report_lines.append("-" * 50)
            
            for ticker, pos in performance['positions'].items():
                emoji = "🟢" if pos['unrealized_pl'] >= 0 else "🔴"
                report_lines.append(f"{emoji} **{ticker}**")
                report_lines.append(f"   Quantity: {pos['quantity']} shares")
                report_lines.append(f"   Avg Price: ${pos['avg_price']:.2f} | Current: ${pos['current_price']:.2f}")
                report_lines.append(f"   Market Value: ${pos['market_value']:,.2f} ({pos['position_size_pct']:.1f}% of portfolio)")
                report_lines.append(f"   Unrealized P&L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_pl_pct']:+.2f}%)")
                report_lines.append(f"   1-Week: {pos['performance_1w']:+.2f}% | 1-Month: {pos['performance_1m']:+.2f}% | Volatility: {pos['volatility']:.1f}%")
                report_lines.append("")
            
            report_lines.append(f"📅 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logging.error(f"Error generating position report: {e}")
            return f"Error generating position report: {str(e)}"
    
    def send_position_update(self, force_update: bool = False):
        """Send position performance update via Telegram"""
        try:
            # Only send if we have positions or if forced
            performance = self.get_position_performance()
            
            if not performance and not force_update:
                return
            
            if not performance:
                self.notifier.send_message("📊 **Position Update:** No active positions.")
                return
            
            # Create concise update message
            summary = performance['portfolio_summary']
            
            # Determine overall performance emoji
            if summary['total_unrealized_pl'] > 0:
                overall_emoji = "📈"
            elif summary['total_unrealized_pl'] < 0:
                overall_emoji = "📉"
            else:
                overall_emoji = "➡️"
            
            message = f"""
{overall_emoji} **Position Update**

💰 **Portfolio Summary:**
• Total Value: ${summary['total_value']:,.2f}
• Cash: ${summary['cash']:,.2f}
• Unrealized P&L: ${summary['total_unrealized_pl']:,.2f} ({summary['total_unrealized_pl_pct']:+.2f}%)
• Active Positions: {summary['positions_count']}

🏆 **Top Performers:**
"""
            
            # Add top 3 performers
            positions_sorted = sorted(
                performance['positions'].items(), 
                key=lambda x: x[1]['unrealized_pl_pct'], 
                reverse=True
            )
            
            for i, (ticker, pos) in enumerate(positions_sorted[:3]):
                emoji = "🟢" if pos['unrealized_pl_pct'] >= 0 else "🔴"
                message += f"{i+1}. {emoji} {ticker}: {pos['unrealized_pl_pct']:+.2f}% (${pos['unrealized_pl']:,.2f})\n"
            
            message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            self.notifier.send_message(message)
            
        except Exception as e:
            logging.error(f"Error sending position update: {e}")
            self.notifier.send_message(f"❌ Error sending position update: {str(e)}")

    def request_position_update(self, request_type: str = "summary"):
        """Handle on-demand position update requests"""
        try:
            if request_type.lower() in ["summary", "brief", "quick"]:
                # Send concise position update
                self.send_position_update(force_update=True)
                
            elif request_type.lower() in ["detailed", "full", "complete"]:
                # Send detailed position report
                report = self.generate_position_report()
                if report and report != "No positions to report.":
                    # Split long reports if needed
                    if len(report) > 4000:
                        # Send in parts
                        parts = [report[i:i+4000] for i in range(0, len(report), 4000)]
                        for i, part in enumerate(parts):
                            self.notifier.send_message(f"📊 Position Report (Part {i+1}/{len(parts)}):\n\n{part}")
                    else:
                        self.notifier.send_message(f"📊 **Detailed Position Report:**\n\n{report}")
                else:
                    self.notifier.send_message("📊 **Position Report:** No active positions to report.")
                    
            elif request_type.lower() in ["performance", "perf"]:
                # Send performance-focused update
                performance = self.get_position_performance()
                if not performance:
                    self.notifier.send_message("📊 **Performance Update:** No active positions.")
                    return
                
                summary = performance['portfolio_summary']
                message = f"""
📊 <b>Performance Update</b>

💰 <b>Portfolio Performance:</b>
• Total Return: {summary['total_unrealized_pl_pct']:+.2f}%
• Unrealized P&L: ${summary['total_unrealized_pl']:,.2f}
• Invested: ${summary['invested_amount']:,.2f}

🏆 <b>Top Performers (1-Month):</b>
"""
                
                # Sort by 1-month performance
                positions_sorted = sorted(
                    performance['positions'].items(), 
                    key=lambda x: x[1]['performance_1m'], 
                    reverse=True
                )
                
                for i, (ticker, pos) in enumerate(positions_sorted[:3]):
                    emoji = "🟢" if pos['performance_1m'] >= 0 else "🔴"
                    message += f"{i+1}. {emoji} {ticker}: {pos['performance_1m']:+.2f}% (1M)\n"
                
                message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                self.notifier.send_message(message)
                
            else:
                # Default to summary
                self.send_position_update(force_update=True)
                
        except Exception as e:
            logging.error(f"Error handling position update request: {e}")
            self.notifier.send_message(f"❌ Error generating position update: {str(e)}")
    
    def get_position_summary(self) -> str:
        """Get a quick position summary for internal use"""
        try:
            performance = self.get_position_performance()
            if not performance:
                return "No active positions"
            
            summary = performance['portfolio_summary']
            return f"${summary['total_unrealized_pl']:,.2f} ({summary['total_unrealized_pl_pct']:+.2f}%) - {summary['positions_count']} positions"
            
        except Exception as e:
            logging.error(f"Error getting position summary: {e}")
            return "Error getting position summary"

    def create_portfolio_performance_chart(self):
        """Create a chart showing total unrealized P&L performance over time (USD only)."""
        try:
            positions = self.get_portfolio_positions()
            if not positions:
                return None
            # Get historical data for all positions (last month)
            portfolio_data = {}
            for ticker in positions.keys():
                try:
                    data = yf.download(ticker, period="1mo", interval="1d", auto_adjust=True, progress=False)
                    if not data.empty:
                        portfolio_data[ticker] = data
                except Exception as e:
                    logging.error(f"Error fetching data for {ticker}: {e}")
                    continue
            if not portfolio_data:
                return None
            account = self.get_account_info()
            cash_usd = account.get('cash', 0)
            dates = list(portfolio_data.values())[0].index
            portfolio_values_usd = []
            invested_usd = []
            for date in dates:
                total_value = 0
                total_invested = 0
                for ticker, position in positions.items():
                    if ticker in portfolio_data:
                        data = portfolio_data[ticker]
                        if date in data.index:
                            quantity = position['quantity']
                            avg_price = position['avg_price']
                            close_price = data.loc[date, 'Close']
                            if isinstance(close_price, pd.Series):
                                close_price = float(close_price.iloc[0])
                            else:
                                close_price = float(close_price)
                            total_value += quantity * close_price
                            total_invested += quantity * avg_price
                total_value += cash_usd
                portfolio_values_usd.append(total_value)
                invested_usd.append(total_invested)
            pnl_usd = [v - i for v, i in zip(portfolio_values_usd, invested_usd)]
            import matplotlib.pyplot as plt
            import numpy as np
            dates = np.array(dates)
            pnl_usd = np.array(pnl_usd)
            portfolio_values_usd = np.array(portfolio_values_usd)
            invested_usd = np.array(invested_usd)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, pnl_usd, label='Unrealized P&L (USD)', color='blue', linewidth=2)
            ax.set_ylabel('Unrealized P&L (USD)', fontsize=12)
            ax.set_title('Portfolio Unrealized P&L in USD (Last 30 Days)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            return buffer
        except Exception as e:
            logging.error(f"Error creating USD portfolio performance chart: {e}")
            return None

    def create_per_stock_performance_charts(self):
        """Create a dict of {ticker: buffer} for each active position, showing value in USD over last month."""
        try:
            positions = self.get_portfolio_positions()
            if not positions:
                return {}
            charts = {}
            for ticker, position in positions.items():
                try:
                    data = yf.download(ticker, period="1mo", interval="1d", auto_adjust=True, progress=False)
                    if data.empty:
                        continue
                    values_usd = []
                    dates = data.index
                    for date in dates:
                        close_price = data.loc[date, 'Close']
                        if isinstance(close_price, pd.Series):
                            close_price = float(close_price.iloc[0])
                        else:
                            close_price = float(close_price)
                        values_usd.append(position['quantity'] * close_price)
                    import matplotlib.pyplot as plt
                    import numpy as np
                    dates = np.array(dates)
                    values_usd = np.array(values_usd)
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(dates, values_usd, label=f'{ticker} Value (USD)', color='green', linewidth=2)
                    ax.set_ylabel('Value (USD)', fontsize=10)
                    ax.set_title(f'{ticker} Position Value in USD (Last 30 Days)', fontsize=11)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                    buffer.seek(0)
                    plt.close()
                    charts[ticker] = buffer
                except Exception as e:
                    logging.error(f"Error creating per-stock USD chart for {ticker}: {e}")
                    continue
            return charts
        except Exception as e:
            logging.error(f"Error creating per-stock USD charts: {e}")
            return {}

    def generate_position_report_pdf(self):
        """Generate a PDF with the current portfolio performance summary and details, using USD charts only."""
        try:
            performance = self.get_position_performance()
            if not performance:
                return None
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            import io
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=20)
            subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
            story = []
            story.append(Paragraph("Portfolio Performance Report", title_style))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 12))
            # Portfolio Performance Chart (USD)
            chart_buffer = self.create_portfolio_performance_chart()
            if chart_buffer:
                story.append(Paragraph("Portfolio Unrealized P&L in USD (Last 30 Days)", subtitle_style))
                story.append(Image(chart_buffer, width=6*inch, height=3*inch))
                story.append(Spacer(1, 12))
            # Portfolio Summary
            summary = performance['portfolio_summary']
            summary_data = [
                ["Metric", "Value"],
                ["Total Portfolio Value", f"${summary['total_value']:,.2f}"],
                ["Cash Available", f"${summary['cash']:,.2f}"],
                ["Invested Amount", f"${summary['invested_amount']:,.2f}"],
                ["Total Unrealized P&L", f"${summary['total_unrealized_pl']:,.2f} ({summary['total_unrealized_pl_pct']:+.2f}%)"],
                ["Active Positions", f"{summary['positions_count']}"]
            ]
            table = Table(summary_data, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 16))
            # Best and Worst Performers
            if performance['best_performer']:
                best = performance['positions'][performance['best_performer']]
                story.append(Paragraph(f"Best Performer: {performance['best_performer']}", subtitle_style))
                story.append(Paragraph(f"Unrealized P&L: ${best['unrealized_pl']:,.2f} ({best['unrealized_pl_pct']:+.2f}%)", styles['Normal']))
                story.append(Paragraph(f"1-Week: {best['performance_1w']:+.2f}% | 1-Month: {best['performance_1m']:+.2f}% | Volatility: {best['volatility']:.1f}%", styles['Normal']))
                story.append(Spacer(1, 8))
            if performance['worst_performer']:
                worst = performance['positions'][performance['worst_performer']]
                story.append(Paragraph(f"Worst Performer: {performance['worst_performer']}", subtitle_style))
                story.append(Paragraph(f"Unrealized P&L: ${worst['unrealized_pl']:,.2f} ({worst['unrealized_pl_pct']:+.2f}%)", styles['Normal']))
                story.append(Paragraph(f"1-Week: {worst['performance_1w']:+.2f}% | 1-Month: {worst['performance_1m']:+.2f}% | Volatility: {worst['volatility']:.1f}%", styles['Normal']))
                story.append(Spacer(1, 8))
            # Individual Position Details with Total Invested vs Current Worth
            story.append(Paragraph("Position Details:", subtitle_style))
            pos_data = [["Ticker", "Qty", "Avg Price", "Current Price", "Total Invested", "Current Worth", "Unrealized P&L", "% of Portfolio", "1W %", "1M %", "Volatility"]]
            for ticker, pos in performance['positions'].items():
                total_invested = pos['quantity'] * pos['avg_price']
                current_worth = pos['quantity'] * pos['current_price']
                pos_data.append([
                    ticker,
                    str(pos['quantity']),
                    f"${pos['avg_price']:.2f}",
                    f"${pos['current_price']:.2f}",
                    f"${total_invested:,.2f}",
                    f"${current_worth:,.2f}",
                    f"${pos['unrealized_pl']:,.2f} ({pos['unrealized_pl_pct']:+.2f}%)",
                    f"{pos['position_size_pct']:.1f}%",
                    f"{pos['performance_1w']:+.2f}%",
                    f"{pos['performance_1m']:+.2f}%",
                    f"{pos['volatility']:.1f}%"
                ])
            pos_table = Table(pos_data, hAlign='LEFT', repeatRows=1)
            pos_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(pos_table)
            story.append(Spacer(1, 12))
            # Per-stock USD charts
            charts = self.create_per_stock_performance_charts()
            if charts:
                story.append(Paragraph("Per-Stock Position Value in USD (Last 30 Days):", subtitle_style))
                for ticker, buf in charts.items():
                    story.append(Paragraph(f"{ticker}", styles['Heading3']))
                    story.append(Image(buf, width=5*inch, height=2*inch))
                    story.append(Spacer(1, 8))
            doc.build(story)
            buffer.seek(0)
            return buffer
        except Exception as e:
            logging.error(f"Error generating position PDF report: {e}")
            return None

def main():
    """Main function to run the AI trader"""
    print("🚀 Starting AI Trading System...")
    trader = AITrader()
    
    # Send startup notification
    trader.notifier.send_message("🤖 AI Trader started successfully!")
    
    # Run analysis and trading with lower confidence threshold
    session_data = trader.analyze_and_trade()  # Now uses config defaults
    
    print("🎉 AI Trading System completed successfully!")

if __name__ == "__main__":
    main() 
