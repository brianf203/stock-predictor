# Architecture & Prediction Methodology

## System Overview

The Stock Predictor bot uses a dual-model approach:
1. **Short-term predictions** (1-5 days): Technical analysis focused
2. **Long-term predictions** (6-12 months): Fundamental analysis focused

## Short-Term Prediction Model

### Time Horizon
- **1-5 days**: Captures intraday and short-term momentum patterns

### Data Sources
- Historical price data (2 years)
- Volume data
- Real-time price information

### Technical Indicators Used

1. **RSI (Relative Strength Index)**
   - Period: 14 days
   - Signals: Oversold (<30) = bullish, Overbought (>70) = bearish
   - Weight: 15% of momentum score

2. **MACD (Moving Average Convergence Divergence)**
   - Signals: MACD > Signal = bullish, MACD < Signal = bearish
   - Weight: 10% of momentum score

3. **EMA (Exponential Moving Averages)**
   - EMA(20): Short-term trend
   - EMA(50): Medium-term trend
   - Price above EMA = bullish, below = bearish
   - Weight: 10% each (20% total)

4. **Bollinger Bands**
   - Period: 20 days, 2 standard deviations
   - Near lower band = potential bounce, near upper band = potential reversal
   - Weight: 10% of momentum score

5. **VWAP (Volume Weighted Average Price)**
   - Used for context, not direct scoring
   - Helps identify institutional buying/selling patterns

6. **Volume Analysis**
   - Increasing volume = stronger trend confirmation
   - Decreasing volume = weakening trend
   - Weight: 5% of momentum score

### Prediction Algorithm

```
momentum_score = sum of all indicator signals (-1 to +1 range)
trend_factor = recent 5-day price change * 0.3
volatility_factor = 1 - (recent_volatility * 0.5)

predicted_change = (momentum_score + trend_factor) * volatility_factor * 0.02
predicted_price = current_price * (1 + predicted_change)
```

### Confidence Calculation
- Base confidence: 70%
- Adjusted by momentum strength: ±20%
- Range: 40-85%

## Long-Term Prediction Model

### Time Horizon
- **6-12 months**: Captures fundamental trends and economic cycles

### Data Sources
- 2+ years of historical data
- Company financials (earnings, revenue, cash flow)
- Valuation metrics (P/E, PEG ratios)
- Balance sheet data

### Fundamental Factors

1. **Earnings Growth**
   - Quarterly earnings growth rate
   - Strong growth (>15%) = +20% score
   - Declining (<-10%) = -20% score

2. **Revenue Growth**
   - Annual revenue growth rate
   - Growth (>10%) = +15% score
   - Decline (<-5%) = -15% score

3. **P/E Ratio (Price-to-Earnings)**
   - Reasonable (0-25) = +10% score
   - High (>40) = -10% score
   - Indicates valuation relative to earnings

4. **PEG Ratio (Price/Earnings to Growth)**
   - Good (<1.5) = +10% score
   - High (>2.0) = -10% score
   - Accounts for growth in valuation

5. **DCF Model (Discounted Cash Flow)**
   - Projects future cash flows
   - Discounts to present value
   - Terminal value calculation
   - Weight: 30% of fundamental score

### DCF Calculation Details

```
Free Cash Flow per Share = Total FCF / Shares Outstanding
Growth Rate = Earnings Growth or Historical Return (capped at 25%)
Discount Rate = 10% (WACC approximation)
Terminal Growth = 3% (long-term GDP growth)

For each year (1-5):
  Future FCF = FCF_per_share * (1 + growth_rate)^year
  Present Value = Future FCF / (1 + discount_rate)^year

Terminal Value = (Final FCF * 1.03) / (0.10 - 0.03)
PV Terminal = Terminal Value / (1 + discount_rate)^5

Intrinsic Value = Sum of PV FCF + PV Terminal
```

### Prediction Algorithm

```
fundamental_score = sum of all fundamental signals (-1 to +1 range)
trend_factor = annual_return * 0.4
volatility_adjustment = 1 - (volatility * 0.3)

predicted_change = (fundamental_score + trend_factor) * volatility_adjustment
predicted_price = current_price * (1 + predicted_change)
```

### Confidence Calculation
- Base confidence: 60%
- Adjusted by fundamental strength: ±15%
- Range: 35-80%

## Risk Factors & Warnings

The system generates warnings for:
- Limited historical data (<100 days)
- Low trading volume (<100k daily)
- High P/E ratios (>50)
- High debt-to-equity (>200%)

## Data Quality Checks

1. **Ticker Validation**: Verifies ticker exists and has data
2. **Price Validation**: Multiple fallback methods for current price
3. **Data Sufficiency**: Checks minimum data requirements
4. **Error Handling**: Graceful degradation with informative messages

## Performance Considerations

- **Caching**: Consider implementing cache for frequently requested tickers
- **Rate Limiting**: yfinance has rate limits; implement delays if needed
- **Async Operations**: Prediction engine uses async for non-blocking operations
- **Error Recovery**: Multiple fallback methods for data retrieval

## Future Enhancements

### Machine Learning Integration
- LSTM networks for time series prediction
- Random Forest for feature importance
- XGBoost for ensemble predictions

### Additional Data Sources
- News sentiment analysis (NewsAPI)
- Economic indicators (FRED API)
- Social media sentiment
- Options flow data

### Advanced Models
- Attention-based CNN-LSTM hybrid
- Transformer models for long-term trends
- Ensemble methods combining multiple models

## Accuracy Notes

- **Short-term**: More volatile, lower confidence (40-85%)
- **Long-term**: More stable, moderate confidence (35-80%)
- **Market Conditions**: Predictions are more accurate in trending markets
- **Disclaimer**: All predictions are estimates, not financial advice



