#!/usr/bin/env python3
"""
Test script for the AI Trading System
Run this to verify all components are working correctly
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test if all environment variables are set"""
    print("🔍 Testing environment variables...")
    
    required_vars = [
        'OPENAI_API_KEY',
        'NEWSAPI_KEY', 
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please check your .env file")
        return False
    else:
        print("✅ All environment variables are set")
        return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n📦 Testing package imports...")
    
    try:
        import requests
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from openai import OpenAI
        from textblob import TextBlob
        import alpaca_trade_api as tradeapi
        import schedule
        print("✅ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def test_openai():
    """Test OpenAI API connection"""
    print("\n🤖 Testing OpenAI API...")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        
        if response.choices[0].message.content:
            print("✅ OpenAI API connection successful")
            return True
        else:
            print("❌ OpenAI API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False

def test_news_api():
    """Test News API connection"""
    print("\n📰 Testing News API...")
    
    try:
        import requests
        api_key = os.getenv("NEWSAPI_KEY")
        url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok':
                print("✅ News API connection successful")
                return True
            else:
                print(f"❌ News API error: {data.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ News API HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ News API error: {e}")
        return False

def test_alpaca():
    """Test Alpaca API connection"""
    print("\n📈 Testing Alpaca API...")
    
    try:
        import alpaca_trade_api as tradeapi
        
        alpaca = tradeapi.REST(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_SECRET_KEY"),
            base_url='https://paper-api.alpaca.markets'
        )
        
        account = alpaca.get_account()
        if account:
            print(f"✅ Alpaca API connection successful")
            print(f"   Account: {account.id}")
            print(f"   Status: {account.status}")
            return True
        else:
            print("❌ Alpaca API returned no account data")
            return False
            
    except Exception as e:
        print(f"❌ Alpaca API error: {e}")
        return False

def test_telegram():
    """Test Telegram bot connection"""
    print("\n📱 Testing Telegram Bot...")
    
    try:
        import requests
        
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        url = f"https://api.telegram.org/bot{token}/getMe"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                bot_info = data.get('result', {})
                print(f"✅ Telegram Bot connection successful")
                print(f"   Bot: @{bot_info.get('username', 'Unknown')}")
                print(f"   Chat ID: {chat_id}")
                return True
            else:
                print(f"❌ Telegram Bot error: {data.get('description', 'Unknown error')}")
                return False
        else:
            print(f"❌ Telegram Bot HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Telegram Bot error: {e}")
        return False

def test_yfinance():
    """Test Yahoo Finance data fetching"""
    print("\n📊 Testing Yahoo Finance...")
    
    try:
        import yfinance as yf
        
        # Test with a simple stock
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info and 'regularMarketPrice' in info:
            print(f"✅ Yahoo Finance connection successful")
            print(f"   Test stock: AAPL - ${info.get('regularMarketPrice', 0):.2f}")
            return True
        else:
            print("❌ Yahoo Finance returned no data")
            return False
            
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
        return False

def test_ai_trader():
    """Test AI Trader initialization"""
    print("\n🤖 Testing AI Trader...")
    
    try:
        from ai_trader import AITrader
        
        trader = AITrader()
        print("✅ AI Trader initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ AI Trader error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Trading System - System Test")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_imports,
        test_openai,
        test_news_api,
        test_alpaca,
        test_telegram,
        test_yfinance,
        test_ai_trader
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your AI Trading System is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python ai_trader.py' for a single trading session")
        print("2. Run 'python scheduler.py' for automated trading")
    else:
        print("⚠️  Some tests failed. Please fix the issues before using the system.")
        sys.exit(1)

if __name__ == "__main__":
    main() 