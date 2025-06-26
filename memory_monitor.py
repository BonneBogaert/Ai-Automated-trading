#!/usr/bin/env python3
"""
Memory Monitoring Script for AI Trading System
Use this to monitor memory usage during trading sessions.
"""

import psutil
import os
import time
import gc
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except:
        return 0

def get_system_memory():
    """Get system memory usage"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / 1024 / 1024,  # MB
            'available': memory.available / 1024 / 1024,  # MB
            'used': memory.used / 1024 / 1024,  # MB
            'percent': memory.percent
        }
    except:
        return {'total': 0, 'available': 0, 'used': 0, 'percent': 0}

def log_memory_usage(message=""):
    """Log current memory usage"""
    process_memory = get_memory_usage()
    system_memory = get_system_memory()
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {message} | Process: {process_memory:.1f}MB | System: {system_memory['used']:.1f}MB/{system_memory['total']:.1f}MB ({system_memory['percent']:.1f}%)"
    
    print(log_msg)
    
    # Check if we're approaching Render's 512MB limit
    if process_memory > 400:
        print(f"⚠️  WARNING: High memory usage ({process_memory:.1f}MB) - approaching Render limit (512MB)")
    
    return process_memory

def monitor_memory_during_trading():
    """Monitor memory during a trading session"""
    print("🔍 Memory Monitoring for Trading Session")
    print("=" * 60)
    
    # Initial memory check
    initial_memory = log_memory_usage("Initial memory")
    
    # Import and initialize trader
    log_memory_usage("Before importing AITrader")
    from ai_trader import AITrader
    log_memory_usage("After importing AITrader")
    
    trader = AITrader()
    log_memory_usage("After initializing AITrader")
    
    # Get account info
    account = trader.get_account_info()
    log_memory_usage("After getting account info")
    
    # Get positions
    positions = trader.get_portfolio_positions()
    log_memory_usage("After getting positions")
    
    # Run trading session
    session_data = trader.analyze_and_trade(
        max_trades=3,
        min_confidence=8
    )
    log_memory_usage("After trading session")
    
    # Generate report
    if session_data:
        pdf_buffer = trader.generate_position_report_pdf()
        log_memory_usage("After generating PDF report")
    
    # Final cleanup
    gc.collect()
    final_memory = log_memory_usage("After garbage collection")
    
    print("=" * 60)
    print(f"📊 Memory Summary:")
    print(f"   Initial: {initial_memory:.1f}MB")
    print(f"   Final: {final_memory:.1f}MB")
    print(f"   Difference: {final_memory - initial_memory:.1f}MB")
    print(f"   Render Limit: 512MB")
    
    if final_memory > 512:
        print(f"❌ CRITICAL: Final memory ({final_memory:.1f}MB) exceeds Render limit!")
    elif final_memory > 400:
        print(f"⚠️  WARNING: High memory usage ({final_memory:.1f}MB)")
    else:
        print(f"✅ Memory usage OK ({final_memory:.1f}MB)")

def quick_memory_check():
    """Quick memory check"""
    process_memory = get_memory_usage()
    system_memory = get_system_memory()
    
    print(f"📊 Quick Memory Check - {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Process Memory: {process_memory:.1f}MB")
    print(f"   System Memory: {system_memory['used']:.1f}MB/{system_memory['total']:.1f}MB ({system_memory['percent']:.1f}%)")
    
    if process_memory > 512:
        print(f"❌ CRITICAL: Process memory ({process_memory:.1f}MB) exceeds Render limit (512MB)")
    elif process_memory > 400:
        print(f"⚠️  WARNING: High memory usage ({process_memory:.1f}MB)")
    else:
        print(f"✅ Memory usage OK")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            monitor_memory_during_trading()
        else:
            print("Usage: python memory_monitor.py [monitor]")
            print("Example: python memory_monitor.py monitor")
    else:
        quick_memory_check() 