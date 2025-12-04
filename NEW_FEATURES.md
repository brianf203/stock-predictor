# New Features: Charts & News

## 📊 Stock Charts

The bot now generates professional stock charts similar to Finviz when you use the `$sp predict` command.

### Chart Features:
- **Price Chart**: 6-month price history with EMA(20) and EMA(50) indicators
- **Bollinger Bands**: Volatility bands showing overbought/oversold zones
- **Volume Bars**: Trading volume visualization
- **RSI Indicator**: Relative Strength Index with overbought/oversold zones
- **MACD Indicator**: Moving Average Convergence Divergence with signal line and histogram
- **Dark Theme**: Professional dark background for better readability
- **Real-time Data**: Shows current price and percentage change

### Chart Layout:
1. **Top Panel**: Price chart with EMAs and Bollinger Bands
2. **Second Panel**: Volume bars
3. **Third Panel**: RSI indicator (0-100 scale)
4. **Bottom Panel**: MACD with signal line and histogram

## 📰 News Articles with Sentiment Analysis

### In Predict Command:
When you run `$sp predict <ticker>`, you'll now get:
- **Top 3 Most Relevant News Articles**
- **Sentiment Analysis**: Each article is labeled as:
  - 🟢 **Bullish**: Positive news that may drive price up
  - 🔴 **Bearish**: Negative news that may drive price down
  - ⚪ **Neutral**: News with no clear directional bias
- **Article Details**: Title, date, publisher, and link

### New News Command:
Use `$sp news <ticker>` to get:
- **Up to 10 Recent News Articles**
- **Organized by Sentiment**: 
  - Bullish news section
  - Bearish news section
  - Neutral news (if no bullish/bearish available)
- **Sentiment Summary**: Count of bullish, bearish, and neutral articles

### Sentiment Analysis Method:
The bot uses a hybrid approach:
1. **Keyword Analysis**: Identifies bullish/bearish keywords in headlines
2. **TextBlob Sentiment**: Natural language processing for polarity
3. **Combined Scoring**: Weighted combination of both methods
4. **Smart Classification**: Articles are classified based on sentiment strength

### Example Usage:
```
$sp predict AAPL
```
Returns: Predictions + Chart + Top 3 News Articles

```
$sp news TSLA
```
Returns: Recent news articles organized by sentiment

## Installation

Make sure to install the new dependencies:

```bash
pip install -r requirements.txt
```

New dependencies:
- `matplotlib>=3.7.0` - For chart generation
- `textblob>=0.17.1` - For sentiment analysis

## Technical Details

### Chart Generation:
- Uses matplotlib with dark theme
- Generates charts as PNG images
- Sent as Discord file attachments
- Optimized for Discord's image display

### News Fetching:
- Uses yfinance's built-in news API
- Fetches from multiple financial news sources
- Analyzes sentiment in real-time
- Sorts by relevance and sentiment strength

### Performance:
- Charts and news are fetched in parallel for speed
- Non-blocking async operations
- Error handling for missing data
- Graceful degradation if services unavailable

## Notes

- Charts may take a few seconds to generate
- News articles are sourced from Yahoo Finance
- Sentiment analysis is automated and may not always be 100% accurate
- Some stocks may have limited news coverage
- Chart generation requires sufficient historical data (minimum 50 days)



