# Setup Guide

## Prerequisites

- Python 3.8 or higher
- Discord Bot Token
- (Optional) API keys for enhanced features

## Installation Steps

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```
DISCORD_BOT_TOKEN=your_discord_bot_token_here
ALPHA_VANTAGE_API_KEY=optional_alpha_vantage_key
FRED_API_KEY=optional_fred_api_key
NEWS_API_KEY=optional_news_api_key
```

5. **Get your Discord Bot Token**

- Go to https://discord.com/developers/applications
- Create a new application or select an existing one
- Go to the "Bot" section
- Click "Reset Token" or "Copy" to get your bot token
- Enable "Message Content Intent" in the Bot settings
- Copy the token to your `.env` file

6. **Invite the bot to your server**

- In Discord Developer Portal, go to OAuth2 > URL Generator
- Select scopes: `bot` and `applications.commands`
- Select bot permissions: `Send Messages`, `Embed Links`, `Read Message History`
- Copy the generated URL and open it in your browser
- Select your server and authorize

7. **Run the bot**
```bash
python bot.py
```

## Architecture Overview

### Short-Term Predictions (1-5 days)
- **Technical Indicators**: RSI, MACD, EMA, Bollinger Bands, VWAP
- **Price Action**: Recent trend analysis, momentum scoring
- **Volume Analysis**: Volume trends and patterns
- **Volatility Assessment**: Recent price volatility

### Long-Term Predictions (6-12 months)
- **Fundamental Analysis**: Earnings growth, revenue growth, P/E ratios, PEG ratios
- **DCF Model**: Discounted Cash Flow valuation
- **Historical Trends**: Annual return patterns
- **Financial Health**: Debt-to-equity, cash flow analysis

## Data Sources

- **yfinance**: Primary data source (free, no API key required)
  - Stock prices, historical data
  - Company fundamentals
  - Financial statements

- **Alpha Vantage** (optional): Additional technical indicators
  - Free tier: 500 calls/day
  - Get API key at: https://www.alphavantage.co/support/#api-key

- **FRED API** (optional): Economic indicators
  - Free tier available
  - Get API key at: https://fred.stlouisfed.org/docs/api/api_key.html

- **NewsAPI** (optional): News sentiment analysis
  - Free tier: 100 requests/day
  - Get API key at: https://newsapi.org/

## Current Features

✅ Basic Discord bot with `$sp` prefix
✅ Help command (`$sp help`)
✅ Predict command (`$sp predict <ticker>`)
✅ Short-term prediction engine (technical analysis)
✅ Long-term prediction engine (fundamental analysis)
✅ Beautiful embed messages
✅ Error handling

## Future Enhancements

- News sentiment analysis integration
- Economic indicator integration
- Machine learning model training
- Historical accuracy tracking
- Portfolio analysis features
- Alert system for price targets



