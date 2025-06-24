#!/usr/bin/env python3
"""
Test script to verify Replit setup before running the full scheduler
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import requests
        print("✅ requests")
    except ImportError as e:
        print(f"❌ requests: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✅ yfinance")
    except ImportError as e:
        print(f"❌ yfinance: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        from openai import OpenAI
        print("✅ openai")
    except ImportError as e:
        print(f"❌ openai: {e}")
        return False
    
    try:
        from textblob import TextBlob
        print("✅ textblob")
    except ImportError as e:
        print(f"❌ textblob: {e}")
        return False
    
    try:
        from alpaca.trading.client import TradingClient
        print("✅ alpaca-py")
    except ImportError as e:
        print(f"❌ alpaca-py: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
    except ImportError as e:
        print(f"❌ matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ seaborn")
    except ImportError as e:
        print(f"❌ seaborn: {e}")
        return False
    
    try:
        import schedule
        print("✅ schedule")
    except ImportError as e:
        print(f"❌ schedule: {e}")
        return False
    
    return True

def test_environment():
    """Test if environment variables are set"""
    print("\n🔍 Testing environment variables...")
    
    required_vars = [
        'OPENAI_API_KEY',
        'ALPACA_API_KEY', 
        'ALPACA_SECRET_KEY',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    optional_vars = ['NEWSAPI_KEY']
    
    all_good = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * min(len(value), 8)}...")
        else:
            print(f"❌ {var}: Not set")
            all_good = False
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * min(len(value), 8)}... (optional)")
        else:
            print(f"⚠️ {var}: Not set (optional)")
    
    return all_good

def test_ai_trader():
    """Test if AI trader can be imported and initialized"""
    print("\n🔍 Testing AI trader...")
    
    try:
        from ai_trader import AITrader
        print("✅ AI trader imported successfully")
        
        # Try to initialize (this will test API connections)
        print("🔄 Initializing AI trader...")
        trader = AITrader()
        print("✅ AI trader initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ AI trader test failed: {e}")
        return False

def test_scheduler():
    """Test if scheduler can be imported"""
    print("\n🔍 Testing scheduler...")
    
    try:
        from replit_scheduler import ReplitScheduler
        print("✅ Scheduler imported successfully")
        return True
    except Exception as e:
        print(f"❌ Scheduler test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Replit Setup Test")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("AI Trader", test_ai_trader),
        ("Scheduler", test_scheduler)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Replit setup is ready.")
        print("You can now run: python replit_scheduler.py")
    else:
        print("⚠️ Some tests failed. Please fix the issues before running the scheduler.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 