# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Create .env File

Create a `.env` file with your Discord bot token:

```
DISCORD_BOT_TOKEN=your_token_here
```

## 3. Run the Bot

```bash
python bot.py
```

## 4. Test Commands

In your Discord server:

- `$sp help` - See all commands
- `$sp predict AAPL` - Get predictions for Apple stock

## Bot Features

✅ **Short-term predictions** (1-5 days)
- Technical indicators (RSI, MACD, EMA, Bollinger Bands)
- Price action and momentum analysis
- Volume trends

✅ **Long-term predictions** (6-12 months)
- Fundamental analysis (earnings, revenue growth)
- DCF valuation model
- Financial health metrics

✅ **Smart features**
- Confidence scores for each prediction
- Key factors explaining predictions
- Warnings for risky stocks
- Beautiful embed messages

## Example Output

When you run `$sp predict AAPL`, you'll get:

- Current stock price
- Short-term prediction with confidence
- Long-term prediction with confidence
- Key factors influencing each prediction
- Any warnings about the stock

## Notes

- The bot uses **yfinance** (free, no API key needed)
- Predictions are estimates, not financial advice
- First request may take a few seconds to fetch data
- Works with any valid stock ticker symbol



