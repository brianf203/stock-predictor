# Stock Predictor Discord Bot

A sophisticated Discord bot for predicting stock prices with both short-term and long-term forecasts.

## Features

- **Short-term predictions** (1-5 days): Technical indicators, price action, market sentiment
- **Long-term predictions** (6 months - 1 year): DCF models, fundamental analysis, economic factors

## Commands

- `$sp help` - Display list of commands
- `$sp predict <ticker>` - Get predictions for a stock ticker

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables (see `.env.example`)
3. Run the bot: `python bot.py`

## Data Sources

- **yfinance**: Stock prices, financials, technical indicators
- **Alpha Vantage**: Additional technical indicators and time series data
- **FRED API**: Economic indicators (GDP, inflation, interest rates)
- **NewsAPI**: News sentiment analysis



