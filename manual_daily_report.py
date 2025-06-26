#!/usr/bin/env python3
"""
Manual Daily Portfolio Report Generator
Run this script to generate a PDF report of your current portfolio performance.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Generate a manual daily portfolio report"""
    try:
        print("📊 Generating Manual Daily Portfolio Report...")
        
        # Import AITrader
        from ai_trader import AITrader
        
        # Initialize the trader
        trader = AITrader()
        
        # Generate the PDF report
        pdf_buffer = trader.generate_position_report_pdf()
        
        if pdf_buffer:
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"manual_daily_report_{timestamp}.pdf"
            
            #save report in folder
            os.makedirs('reports', exist_ok=True)
            filename = os.path.join('reports', filename)
            with open(filename, "wb") as f:
                f.write(pdf_buffer.getvalue())
                print(f"✅ Daily portfolio report saved as: {filename}")
                print(f"📁 File location: {os.path.abspath(filename)}")
            
            # Also print a quick summary to console
            performance = trader.get_position_performance()
            if performance:
                summary = performance['portfolio_summary']
                print(f"\n📈 Quick Summary:")
                print(f"   Total Portfolio Value: ${summary['total_value']:,.2f}")
                print(f"   Cash Available: ${summary['cash']:,.2f}")
                print(f"   Invested Amount: ${summary['invested_amount']:,.2f}")
                print(f"   Unrealized P&L: ${summary['total_unrealized_pl']:,.2f} ({summary['total_unrealized_pl_pct']:+.2f}%)")
                print(f"   Active Positions: {summary['positions_count']}")
            else:
                print("📊 No active positions found.")
                
        else:
            print("❌ Failed to generate portfolio report (no positions or error).")
            
    except Exception as e:
        print(f"❌ Error generating manual daily report: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 