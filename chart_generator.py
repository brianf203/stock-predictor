import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

class ChartGenerator:
    def __init__(self):
        self.style = 'dark_background'
        plt.style.use(self.style)
    
    def generate_chart(self, ticker: str, period: str = "6mo") -> BytesIO:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            raise ValueError(f"No data available for {ticker}")
        
        info = stock.info
        company_name = info.get('longName', ticker) if info else ticker
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(4, 1, hspace=0.3, height_ratios=[3, 1, 1, 1])
        
        ax_price = fig.add_subplot(gs[0, 0])
        ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_price)
        ax_rsi = fig.add_subplot(gs[2, 0], sharex=ax_price)
        ax_macd = fig.add_subplot(gs[3, 0], sharex=ax_price)
        
        closes = hist['Close']
        highs = hist['High']
        lows = hist['Low']
        opens = hist['Open']
        volumes = hist['Volume']
        
        current_price = closes.iloc[-1]
        price_change = closes.iloc[-1] - closes.iloc[-20] if len(closes) >= 20 else 0
        price_change_pct = (price_change / closes.iloc[-20]) * 100 if len(closes) >= 20 else 0
        
        color = '#2ecc71' if price_change >= 0 else '#e74c3c'
        
        ax_price.plot(closes.index, closes.values, color=color, linewidth=2, label='Close Price', zorder=3)
        ax_price.fill_between(closes.index, closes.values, alpha=0.3, color=color, zorder=1)
        
        ema_20 = EMAIndicator(close=closes, window=20).ema_indicator()
        ema_50 = EMAIndicator(close=closes, window=50).ema_indicator()
        
        if not ema_20.empty:
            ax_price.plot(ema_20.index, ema_20.values, color='#f39c12', linewidth=1.5, label='EMA(20)', alpha=0.8, zorder=2)
        if not ema_50.empty:
            ax_price.plot(ema_50.index, ema_50.values, color='#9b59b6', linewidth=1.5, label='EMA(50)', alpha=0.8, zorder=2)
        
        bb = BollingerBands(close=closes, window=20)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()
        
        if not bb_upper.empty:
            ax_price.fill_between(bb_upper.index, bb_upper.values, bb_lower.values, 
                                 alpha=0.15, color='#3498db', label='Bollinger Bands', zorder=0)
            ax_price.plot(bb_mid.index, bb_mid.values, color='#3498db', linewidth=1, alpha=0.5, linestyle='--', zorder=1)
        
        ax_price.set_title(f'{company_name} ({ticker}) - ${current_price:.2f} ({price_change_pct:+.2f}%)', 
                          fontsize=16, fontweight='bold', color='white', pad=20)
        ax_price.set_ylabel('Price (USD)', fontsize=12, color='white')
        ax_price.legend(loc='upper left', fontsize=9, framealpha=0.3)
        ax_price.grid(True, alpha=0.3, linestyle='--')
        ax_price.set_facecolor('#1e1e1e')
        
        for spine in ax_price.spines.values():
            spine.set_edgecolor('#444444')
        
        ax_volume.bar(volumes.index, volumes.values, color=color, alpha=0.6, width=0.8)
        ax_volume.set_ylabel('Volume', fontsize=10, color='white')
        ax_volume.set_facecolor('#1e1e1e')
        ax_volume.grid(True, alpha=0.2, linestyle='--', axis='y')
        
        for spine in ax_volume.spines.values():
            spine.set_edgecolor('#444444')
        
        rsi = RSIIndicator(close=closes, window=14).rsi()
        if not rsi.empty:
            ax_rsi.plot(rsi.index, rsi.values, color='#e67e22', linewidth=1.5, label='RSI(14)')
            ax_rsi.axhline(y=70, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)
            ax_rsi.axhline(y=30, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1)
            ax_rsi.fill_between(rsi.index, 70, 100, alpha=0.2, color='#e74c3c', label='Overbought')
            ax_rsi.fill_between(rsi.index, 0, 30, alpha=0.2, color='#2ecc71', label='Oversold')
            ax_rsi.set_ylabel('RSI', fontsize=10, color='white')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.legend(loc='upper left', fontsize=8, framealpha=0.3)
            ax_rsi.set_facecolor('#1e1e1e')
            ax_rsi.grid(True, alpha=0.2, linestyle='--')
        
        for spine in ax_rsi.spines.values():
            spine.set_edgecolor('#444444')
        
        from ta.trend import MACD
        macd = MACD(close=closes)
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        histogram = macd.macd_diff()
        
        if not macd_line.empty:
            ax_macd.plot(macd_line.index, macd_line.values, color='#3498db', linewidth=1.5, label='MACD')
            ax_macd.plot(signal_line.index, signal_line.values, color='#e74c3c', linewidth=1.5, label='Signal')
            colors = ['#2ecc71' if h >= 0 else '#e74c3c' for h in histogram.values]
            ax_macd.bar(histogram.index, histogram.values, color=colors, alpha=0.6, width=0.8, label='Histogram')
            ax_macd.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=0.5)
            ax_macd.set_ylabel('MACD', fontsize=10, color='white')
            ax_macd.set_xlabel('Date', fontsize=12, color='white')
            ax_macd.legend(loc='upper left', fontsize=8, framealpha=0.3)
            ax_macd.set_facecolor('#1e1e1e')
            ax_macd.grid(True, alpha=0.2, linestyle='--')
        
        for spine in ax_macd.spines.values():
            spine.set_edgecolor('#444444')
        
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8, color='white')
        
        fig.patch.set_facecolor('#1e1e1e')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e', edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        
        return buf



