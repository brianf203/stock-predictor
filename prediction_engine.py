import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.others import DailyReturnIndicator
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

class PredictionEngine:
    def __init__(self):
        self.short_term_days = 5
        self.long_term_months = 12
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def _fetch_stock_data(self, ticker: str):
        stock = yf.Ticker(ticker)
        info = stock.info
        hist_data = stock.history(period="2y")
        return stock, info, hist_data
        
    def _get_exchange_prefix(self, info: dict):
        exchange = info.get('exchange', '')
        exchange_map = {
            'NMS': 'NASDAQ',
            'NGM': 'NASDAQ',
            'NCM': 'NASDAQ',
            'NYQ': 'NYSE',
            'ASE': 'NYSE',
            'PCX': 'NYSE',
            'TOR': 'TSX',
            'VAN': 'TSXV',
            'LON': 'LSE',
            'GER': 'XETR',
            'FRA': 'XFRA',
            'AMS': 'AMS',
            'BRU': 'BRU',
            'LIS': 'LIS',
            'MIL': 'MIL',
            'PAR': 'PAR',
            'VIE': 'VIE',
            'STO': 'STO',
            'HEL': 'HEL',
            'CPH': 'CPH',
            'OSL': 'OSL',
            'ASX': 'ASX',
            'NZE': 'NZE',
            'TSE': 'TSE',
            'HKG': 'HKEX',
            'SHE': 'SZSE',
            'SHG': 'SSE',
            'KRX': 'KRX',
            'SES': 'SGX',
            'BOM': 'BSE',
            'NSE': 'NSE',
        }
        return exchange_map.get(exchange, exchange) if exchange else ''
    
    async def get_predictions(self, ticker: str):
        loop = asyncio.get_event_loop()
        stock, info, hist_data = await loop.run_in_executor(
            self.executor, self._fetch_stock_data, ticker
        )
        
        if not info or 'currentPrice' not in info:
            raise ValueError(f"Invalid ticker or unable to fetch data for {ticker}")
        
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        if current_price == 0:
            def get_current_price():
                hist = stock.history(period="1d")
                if not hist.empty:
                    return hist['Close'].iloc[-1]
                return None
            current_price = await loop.run_in_executor(self.executor, get_current_price)
            if current_price is None:
                raise ValueError(f"Unable to determine current price for {ticker}")
        
        if hist_data.empty:
            raise ValueError(f"Insufficient historical data for {ticker}")
        
        exchange_prefix = self._get_exchange_prefix(info)
        full_ticker = f"{exchange_prefix}:{ticker}" if exchange_prefix else ticker
        
        short_term = self._predict_short_term_advanced(hist_data, current_price, info)
        long_term = self._predict_long_term_advanced(hist_data, current_price, info, stock)
        
        return {
            'current_price': current_price,
            'ticker': full_ticker,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'short_term': short_term,
            'long_term': long_term,
            'warnings': self._generate_warnings(info, hist_data)
        }
    
    def _predict_short_term_advanced(self, hist_data: pd.DataFrame, current_price: float, info: dict):
        if len(hist_data) < 100:
            return {
                'predicted_price': current_price,
                'change_percent': 0.0,
                'confidence': 30.0,
                'key_factors': 'Insufficient data for accurate prediction'
            }
        
        closes = hist_data['Close'].values
        highs = hist_data['High'].values
        lows = hist_data['Low'].values
        opens = hist_data['Open'].values
        volumes = hist_data['Volume'].values
        
        df = hist_data.copy()
        
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi()
        df['RSI_14'] = rsi.rsi()
        df['RSI_7'] = RSIIndicator(close=df['Close'], window=7).rsi()
        
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        ema_9 = EMAIndicator(close=df['Close'], window=9)
        df['EMA_9'] = ema_9.ema_indicator()
        ema_20 = EMAIndicator(close=df['Close'], window=20)
        df['EMA_20'] = ema_20.ema_indicator()
        ema_50 = EMAIndicator(close=df['Close'], window=50)
        df['EMA_50'] = ema_50.ema_indicator()
        ema_200 = EMAIndicator(close=df['Close'], window=200)
        df['EMA_200'] = ema_200.ema_indicator()
        
        sma_20 = SMAIndicator(close=df['Close'], window=20)
        df['SMA_20'] = sma_20.sma_indicator()
        sma_50 = SMAIndicator(close=df['Close'], window=50)
        df['SMA_50'] = sma_50.sma_indicator()
        
        bb = BollingerBands(close=df['Close'], window=20)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
        
        kc = KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
        df['KC_upper'] = kc.keltner_channel_hband()
        df['KC_lower'] = kc.keltner_channel_lband()
        df['KC_mid'] = kc.keltner_channel_mband()
        
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ATR'] = atr.average_true_range()
        
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        williams = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Williams_R'] = williams.williams_r()
        
        roc = ROCIndicator(close=df['Close'], window=10)
        df['ROC'] = roc.roc()
        
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['ADX_Pos'] = adx.adx_pos()
        df['ADX_Neg'] = adx.adx_neg()
        
        obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
        df['OBV'] = obv.on_balance_volume()
        
        acc_dist = AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
        df['ADI'] = acc_dist.acc_dist_index()
        
        vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
        df['VWAP'] = vwap.volume_weighted_average_price()
        
        try:
            ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'])
            df['Ichimoku_A'] = ichimoku.ichimoku_a()
            df['Ichimoku_B'] = ichimoku.ichimoku_b()
            df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
            df['Ichimoku_Conv'] = ichimoku.ichimoku_conversion_line()
        except Exception:
            df['Ichimoku_A'] = np.nan
            df['Ichimoku_B'] = np.nan
            df['Ichimoku_Base'] = np.nan
            df['Ichimoku_Conv'] = np.nan
        
        daily_return = DailyReturnIndicator(close=df['Close'])
        df['Daily_Return'] = daily_return.daily_return()
        
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Price_Volume'] = df['Close'] * df['Volume']
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        support, resistance = self._find_support_resistance(df)
        market_regime = self._detect_market_regime(df)
        volatility_model = self._model_volatility(df)
        
        predictions = []
        factors = []
        confidence_scores = []
        
        model1_pred, model1_conf, model1_factors = self._momentum_model(df, latest, prev, current_price)
        predictions.append(model1_pred)
        confidence_scores.append(model1_conf)
        factors.extend(model1_factors)
        
        model2_pred, model2_conf, model2_factors = self._mean_reversion_model(df, latest, support, resistance, current_price)
        predictions.append(model2_pred)
        confidence_scores.append(model2_conf)
        factors.extend(model2_factors)
        
        model3_pred, model3_conf, model3_factors = self._trend_following_model(df, latest, current_price, market_regime)
        predictions.append(model3_pred)
        confidence_scores.append(model3_conf)
        factors.extend(model3_factors)
        
        model4_pred, model4_conf, model4_factors = self._volume_price_model(df, latest, current_price)
        predictions.append(model4_pred)
        confidence_scores.append(model4_conf)
        factors.extend(model4_factors)
        
        model5_pred, model5_conf, model5_factors = self._volatility_model(df, latest, volatility_model, current_price)
        predictions.append(model5_pred)
        confidence_scores.append(model5_conf)
        factors.extend(model5_factors)
        
        ensemble_pred = np.mean(predictions)
        ensemble_std = np.std(predictions)
        agreement = 1 - (ensemble_std / current_price) if current_price > 0 else 0.5
        
        final_confidence = min(92.0, max(45.0, np.mean(confidence_scores) * (0.7 + agreement * 0.3)))
        
        predicted_price = ensemble_pred
        predicted_change = (predicted_price - current_price) / current_price
        
        unique_factors = list(dict.fromkeys(factors))[:5]
        
        return {
            'predicted_price': predicted_price,
            'change_percent': predicted_change * 100,
            'confidence': final_confidence,
            'key_factors': ', '.join(unique_factors) if unique_factors else 'Multiple technical signals'
        }
    
    def _momentum_model(self, df: pd.DataFrame, latest: pd.Series, prev: pd.Series, current_price: float):
        score = 0
        factors = []
        confidence = 70.0
        
        if not pd.isna(latest['RSI_14']):
            if latest['RSI_14'] < 25:
                score += 0.20
                factors.append("Strongly oversold (RSI<25)")
                confidence += 5
            elif latest['RSI_14'] < 35:
                score += 0.10
                factors.append("Oversold")
            elif latest['RSI_14'] > 75:
                score -= 0.20
                factors.append("Strongly overbought (RSI>75)")
                confidence += 5
            elif latest['RSI_14'] > 65:
                score -= 0.10
                factors.append("Overbought")
        
        if not pd.isna(latest['MACD_hist']):
            if latest['MACD_hist'] > 0 and prev['MACD_hist'] <= 0:
                score += 0.15
                factors.append("MACD bullish crossover")
                confidence += 3
            elif latest['MACD_hist'] < 0 and prev['MACD_hist'] >= 0:
                score -= 0.15
                factors.append("MACD bearish crossover")
                confidence += 3
            elif latest['MACD_hist'] > prev['MACD_hist']:
                score += 0.08
            else:
                score -= 0.08
        
        if not pd.isna(latest['Stoch_K']) and not pd.isna(latest['Stoch_D']):
            if latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20:
                score += 0.12
                factors.append("Stochastic oversold")
            elif latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:
                score -= 0.12
                factors.append("Stochastic overbought")
        
        if not pd.isna(latest['Williams_R']):
            if latest['Williams_R'] < -80:
                score += 0.10
                factors.append("Williams %R oversold")
            elif latest['Williams_R'] > -20:
                score -= 0.10
                factors.append("Williams %R overbought")
        
        if not pd.isna(latest['ROC']):
            if latest['ROC'] > 5:
                score += 0.08
            elif latest['ROC'] < -5:
                score -= 0.08
        
        predicted_change = score * 0.025
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _mean_reversion_model(self, df: pd.DataFrame, latest: pd.Series, support: float, resistance: float, current_price: float):
        score = 0
        factors = []
        confidence = 65.0
        
        bb_position = (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']) if not pd.isna(latest['BB_upper']) and latest['BB_upper'] != latest['BB_lower'] else 0.5
        
        if bb_position < 0.1:
            score += 0.25
            factors.append("Extreme lower Bollinger Band")
            confidence += 8
        elif bb_position < 0.2:
            score += 0.15
            factors.append("Near lower Bollinger Band")
        elif bb_position > 0.9:
            score -= 0.25
            factors.append("Extreme upper Bollinger Band")
            confidence += 8
        elif bb_position > 0.8:
            score -= 0.15
            factors.append("Near upper Bollinger Band")
        
        if support > 0 and current_price <= support * 1.02:
            score += 0.20
            factors.append(f"Near support ${support:.2f}")
            confidence += 5
        elif resistance > 0 and current_price >= resistance * 0.98:
            score -= 0.20
            factors.append(f"Near resistance ${resistance:.2f}")
            confidence += 5
        
        kc_position = (latest['Close'] - latest['KC_lower']) / (latest['KC_upper'] - latest['KC_lower']) if not pd.isna(latest['KC_upper']) and latest['KC_upper'] != latest['KC_lower'] else 0.5
        
        if kc_position < 0.2:
            score += 0.12
        elif kc_position > 0.8:
            score -= 0.12
        
        if not pd.isna(latest['EMA_20']):
            distance_from_ema = (current_price - latest['EMA_20']) / latest['EMA_20']
            if abs(distance_from_ema) > 0.05:
                score += np.sign(-distance_from_ema) * 0.10
                factors.append("Price deviation from EMA(20)")
        
        predicted_change = score * 0.02
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _trend_following_model(self, df: pd.DataFrame, latest: pd.Series, current_price: float, market_regime: str):
        score = 0
        factors = []
        confidence = 75.0
        
        if not pd.isna(latest['EMA_9']) and not pd.isna(latest['EMA_20']):
            if latest['EMA_9'] > latest['EMA_20']:
                score += 0.12
                factors.append("EMA(9) > EMA(20) - bullish")
            else:
                score -= 0.12
                factors.append("EMA(9) < EMA(20) - bearish")
        
        if not pd.isna(latest['EMA_20']) and not pd.isna(latest['EMA_50']):
            if latest['EMA_20'] > latest['EMA_50']:
                score += 0.15
                factors.append("EMA(20) > EMA(50) - uptrend")
                confidence += 3
            else:
                score -= 0.15
                factors.append("EMA(20) < EMA(50) - downtrend")
                confidence += 3
        
        if not pd.isna(latest['EMA_50']) and not pd.isna(latest['EMA_200']):
            if latest['EMA_50'] > latest['EMA_200']:
                score += 0.18
                factors.append("Golden cross (EMA50>EMA200)")
                confidence += 5
            else:
                score -= 0.18
                factors.append("Death cross (EMA50<EMA200)")
                confidence += 5
        
        if not pd.isna(latest['ADX']):
            if latest['ADX'] > 25:
                if latest['ADX_Pos'] > latest['ADX_Neg']:
                    score += 0.15
                    factors.append("Strong uptrend (ADX>25)")
                    confidence += 4
                else:
                    score -= 0.15
                    factors.append("Strong downtrend (ADX>25)")
                    confidence += 4
            elif latest['ADX'] < 20:
                score *= 0.7
                factors.append("Weak trend (ADX<20)")
        
        if market_regime == 'bull':
            score += 0.10
            factors.append("Bull market regime")
        elif market_regime == 'bear':
            score -= 0.10
            factors.append("Bear market regime")
        
        if 'Ichimoku_Base' in latest and 'Ichimoku_Conv' in latest:
            if not pd.isna(latest['Ichimoku_Base']) and not pd.isna(latest['Ichimoku_Conv']):
                if latest['Ichimoku_Conv'] > latest['Ichimoku_Base']:
                    score += 0.10
                    factors.append("Ichimoku bullish")
                else:
                    score -= 0.10
                    factors.append("Ichimoku bearish")
        
        predicted_change = score * 0.03
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _volume_price_model(self, df: pd.DataFrame, latest: pd.Series, current_price: float):
        score = 0
        factors = []
        confidence = 68.0
        
        if not pd.isna(latest['OBV']):
            obv_trend = df['OBV'].iloc[-5:].pct_change().mean() if len(df) >= 5 else 0
            price_trend = df['Close'].iloc[-5:].pct_change().mean() if len(df) >= 5 else 0
            
            if obv_trend > 0 and price_trend > 0:
                score += 0.15
                factors.append("Volume-price confirmation (bullish)")
                confidence += 4
            elif obv_trend < 0 and price_trend < 0:
                score -= 0.15
                factors.append("Volume-price confirmation (bearish)")
                confidence += 4
            elif obv_trend > 0 and price_trend < 0:
                score += 0.10
                factors.append("Volume divergence (bullish)")
            elif obv_trend < 0 and price_trend > 0:
                score -= 0.10
                factors.append("Volume divergence (bearish)")
        
        if not pd.isna(latest['ADI']):
            adi_trend = df['ADI'].iloc[-5:].pct_change().mean() if len(df) >= 5 else 0
            if adi_trend > 0.02:
                score += 0.12
                factors.append("Accumulation pattern")
            elif adi_trend < -0.02:
                score -= 0.12
                factors.append("Distribution pattern")
        
        if not pd.isna(latest['VWAP']):
            vwap_distance = (current_price - latest['VWAP']) / latest['VWAP']
            if vwap_distance > 0.02:
                score -= 0.10
                factors.append("Above VWAP - potential pullback")
            elif vwap_distance < -0.02:
                score += 0.10
                factors.append("Below VWAP - potential bounce")
        
        volume_ma = df['Volume'].iloc[-20:].mean() if len(df) >= 20 else df['Volume'].iloc[-1]
        recent_volume = df['Volume'].iloc[-5:].mean() if len(df) >= 5 else df['Volume'].iloc[-1]
        
        if recent_volume > volume_ma * 1.5:
            if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                score += 0.12
                factors.append("High volume breakout")
                confidence += 3
            else:
                score -= 0.12
                factors.append("High volume breakdown")
                confidence += 3
        
        predicted_change = score * 0.022
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _volatility_model(self, df: pd.DataFrame, latest: pd.Series, volatility_params: dict, current_price: float):
        score = 0
        factors = []
        confidence = 72.0
        
        if not pd.isna(latest['ATR']):
            atr_pct = latest['ATR'] / current_price if current_price > 0 else 0
            if atr_pct > 0.04:
                factors.append("High volatility (ATR>4%)")
                score *= 0.85
            elif atr_pct < 0.01:
                factors.append("Low volatility (ATR<1%)")
                confidence += 3
        
        if 'realized_vol' in volatility_params:
            rv = volatility_params['realized_vol']
            if rv > 0.4:
                factors.append("Extreme volatility")
                score *= 0.80
            elif rv < 0.15:
                confidence += 2
        
        if 'volatility_regime' in volatility_params:
            if volatility_params['volatility_regime'] == 'high':
                score *= 0.90
                factors.append("High volatility regime")
            elif volatility_params['volatility_regime'] == 'low':
                confidence += 3
                factors.append("Low volatility regime")
        
        bb_width = latest['BB_width'] if not pd.isna(latest['BB_width']) else 0.05
        if bb_width > 0.1:
            factors.append("Wide Bollinger Bands - high volatility")
            score *= 0.88
        elif bb_width < 0.03:
            factors.append("Narrow Bollinger Bands - low volatility")
            confidence += 2
        
        predicted_change = score * 0.018 if score != 0 else 0
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20):
        closes = df['Close'].values
        if len(closes) < window * 2:
            return 0, 0
        
        recent_closes = closes[-window*2:]
        highs = df['High'].values[-window*2:]
        lows = df['Low'].values[-window*2:]
        
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(recent_closes) - window):
            local_min = np.min(lows[i-window:i+window])
            local_max = np.max(highs[i-window:i+window])
            
            if lows[i] <= local_min * 1.01:
                support_levels.append(lows[i])
            if highs[i] >= local_max * 0.99:
                resistance_levels.append(highs[i])
        
        support = np.median(support_levels) if support_levels else np.min(lows[-window:])
        resistance = np.median(resistance_levels) if resistance_levels else np.max(highs[-window:])
        
        return support, resistance
    
    def _detect_market_regime(self, df: pd.DataFrame):
        if len(df) < 200:
            return 'neutral'
        
        closes = df['Close'].values
        ema_50 = df['EMA_50'].values if 'EMA_50' in df.columns else None
        ema_200 = df['EMA_200'].values if 'EMA_200' in df.columns else None
        
        if ema_50 is not None and ema_200 is not None and not pd.isna(ema_50[-1]) and not pd.isna(ema_200[-1]):
            if ema_50[-1] > ema_200[-1] and closes[-1] > ema_50[-1]:
                return 'bull'
            elif ema_50[-1] < ema_200[-1] and closes[-1] < ema_50[-1]:
                return 'bear'
        
        recent_trend = (closes[-1] - closes[-50]) / closes[-50] if len(closes) >= 50 else 0
        if recent_trend > 0.15:
            return 'bull'
        elif recent_trend < -0.15:
            return 'bear'
        
        return 'neutral'
    
    def _model_volatility(self, df: pd.DataFrame, window: int = 20):
        returns = df['Close'].pct_change().dropna()
        if len(returns) < window:
            return {'realized_vol': 0.2, 'volatility_regime': 'neutral'}
        
        recent_returns = returns.iloc[-window:]
        realized_vol = recent_returns.std() * np.sqrt(252)
        
        if realized_vol > 0.35:
            regime = 'high'
        elif realized_vol < 0.15:
            regime = 'low'
        else:
            regime = 'normal'
        
        return {
            'realized_vol': realized_vol,
            'volatility_regime': regime
        }
    
    def _predict_long_term_advanced(self, hist_data: pd.DataFrame, current_price: float, info: dict, stock):
        if len(hist_data) < 252:
            return {
                'predicted_price': current_price,
                'change_percent': 0.0,
                'confidence': 30.0,
                'key_factors': 'Insufficient historical data for long-term prediction'
            }
        
        closes = hist_data['Close'].values
        
        earnings_growth = info.get('earningsQuarterlyGrowth', 0) or 0
        earnings_growth_annual = info.get('earningsGrowth', 0) or 0
        
        if earnings_growth and abs(earnings_growth) > 1.0:
            earnings_growth = earnings_growth / 100.0
        
        if earnings_growth_annual and abs(earnings_growth_annual) > 1.0:
            earnings_growth_annual = earnings_growth_annual / 100.0
        
        if earnings_growth_annual and abs(earnings_growth_annual) < abs(earnings_growth):
            earnings_growth = earnings_growth_annual
        
        if earnings_growth and abs(earnings_growth) > 0.5:
            earnings_growth = earnings_growth * 0.25
        
        revenue_growth = info.get('revenueGrowth', 0) or 0
        if revenue_growth and abs(revenue_growth) > 1.0:
            revenue_growth = revenue_growth / 100.0
        
        pe_ratio = info.get('trailingPE', 0) or info.get('forwardPE', 0) or 0
        peg_ratio = info.get('pegRatio', 0) or 0
        roe = info.get('returnOnEquity', 0) or 0
        if roe and abs(roe) > 2.0:
            roe = roe / 100.0
        elif roe and abs(roe) > 1.0 and abs(roe) <= 2.0:
            pass
        elif roe and abs(roe) <= 1.0:
            pass
        
        roa = info.get('returnOnAssets', 0) or 0
        if roa and roa > 1.0:
            roa = roa / 100.0
        
        profit_margin = info.get('profitMargins', 0) or 0
        if profit_margin and profit_margin > 1.0:
            profit_margin = profit_margin / 100.0
        
        operating_margin = info.get('operatingMargins', 0) or 0
        if operating_margin and operating_margin > 1.0:
            operating_margin = operating_margin / 100.0
        debt_to_equity = info.get('debtToEquity', 0) or 0
        current_ratio = info.get('currentRatio', 0) or 0
        quick_ratio = info.get('quickRatio', 0) or 0
        beta = info.get('beta', 1.0) or 1.0
        dividend_yield = info.get('dividendYield', 0) or 0
        payout_ratio = info.get('payoutRatio', 0) or 0
        market_cap = info.get('marketCap', 0) or 0
        sector = info.get('sector', '') or ''
        industry = info.get('industry', '') or ''
        book_value = info.get('bookValue', 0) or 0
        price_to_book = info.get('priceToBook', 0) or 0
        
        annual_return = (closes[-1] / closes[-252]) ** (1/1) - 1 if len(closes) >= 252 else 0
        
        historical_pe = []
        if len(closes) >= 252 and pe_ratio > 0:
            try:
                for i in range(max(0, len(closes)-252), len(closes), 20):
                    if i < len(closes):
                        historical_price = closes[i]
                        historical_pe_estimate = historical_price / (historical_price / pe_ratio) if historical_price > 0 else pe_ratio
                        historical_pe.append(historical_pe_estimate)
            except:
                pass
        if len(historical_pe) < 3:
            historical_pe = [pe_ratio * 0.8, pe_ratio, pe_ratio * 1.2]
        pe_mean = np.mean(historical_pe) if historical_pe else pe_ratio
        pe_std = np.std(historical_pe) if len(historical_pe) > 1 and np.std(historical_pe) > 0 else pe_ratio * 0.15
        
        if len(closes) >= 252:
            volatility = closes[-252:].std() / closes[-252:].mean()
            sharpe_ratio = (annual_return - 0.02) / (closes[-252:].std() * np.sqrt(252)) if closes[-252:].std() > 0 else 0
        else:
            volatility = closes.std() / closes.mean()
            sharpe_ratio = 0
        
        predictions = []
        factors = []
        confidence_scores = []
        
        model1_pred, model1_conf, model1_factors = self._fundamental_model(
            earnings_growth, revenue_growth, pe_ratio, peg_ratio, roe, roa, 
            profit_margin, operating_margin, current_price, market_cap
        )
        predictions.append(model1_pred)
        confidence_scores.append(model1_conf)
        factors.extend(model1_factors)
        
        model2_pred, model2_conf, model2_factors = self._dcf_model_advanced(info, annual_return, current_price)
        if model2_pred:
            predictions.append(model2_pred)
            confidence_scores.append(model2_conf)
            factors.extend(model2_factors)
        
        model3_pred, model3_conf, model3_factors = self._valuation_model(
            pe_ratio, peg_ratio, roe, roa, profit_margin, beta, current_price, pe_mean, pe_std
        )
        predictions.append(model3_pred)
        confidence_scores.append(model3_conf)
        factors.extend(model3_factors)
        
        model4_pred, model4_conf, model4_factors = self._financial_health_model(
            debt_to_equity, current_ratio, quick_ratio, profit_margin, 
            operating_margin, current_price
        )
        predictions.append(model4_pred)
        confidence_scores.append(model4_conf)
        factors.extend(model4_factors)
        
        model5_pred, model5_conf, model5_factors = self._trend_extrapolation_model(
            closes, annual_return, sharpe_ratio, volatility, current_price
        )
        predictions.append(model5_pred)
        confidence_scores.append(model5_conf)
        factors.extend(model5_factors)
        
        if dividend_yield > 0:
            model6_pred, model6_conf, model6_factors = self._dividend_discount_model(
                dividend_yield, payout_ratio, earnings_growth, current_price
            )
            predictions.append(model6_pred)
            confidence_scores.append(model6_conf)
            factors.extend(model6_factors)
        
        model7_pred, model7_conf, model7_factors = self._pe_mean_reversion_model(
            pe_ratio, pe_mean, pe_std, current_price, earnings_growth
        )
        predictions.append(model7_pred)
        confidence_scores.append(model7_conf)
        factors.extend(model7_factors)
        
        shares_outstanding = info.get('sharesOutstanding', 0) or 0
        model8_pred, model8_conf, model8_factors = self._market_cap_valuation_model(
            market_cap, price_to_book, book_value, roe, current_price, shares_outstanding
        )
        if model8_pred:
            predictions.append(model8_pred)
            confidence_scores.append(model8_conf)
            factors.extend(model8_factors)
        
        if not predictions:
            return {
                'predicted_price': current_price,
                'change_percent': 0.0,
                'confidence': 30.0,
                'key_factors': 'Insufficient data for prediction'
            }
        
        predictions_array = np.array(predictions)
        changes = (predictions_array - current_price) / current_price
        
        outlier_threshold = 2.0
        median_change = np.median(changes)
        mad = np.median(np.abs(changes - median_change))
        
        if mad > 0:
            z_scores = np.abs((changes - median_change) / (mad + 1e-10))
            valid_mask = z_scores < outlier_threshold
        else:
            valid_mask = np.ones(len(predictions), dtype=bool)
        
        if np.sum(valid_mask) < 2:
            valid_mask = np.ones(len(predictions), dtype=bool)
        
        filtered_predictions = predictions_array[valid_mask]
        filtered_confidences = np.array(confidence_scores)[valid_mask]
        
        if len(filtered_predictions) > 0:
            fundamental_preds = [predictions[i] for i in range(len(predictions)) if valid_mask[i] and i < 3]
            other_preds = [predictions[i] for i in range(len(predictions)) if valid_mask[i] and i >= 3]
            
            if len(fundamental_preds) > 0 and len(other_preds) > 0:
                fundamental_weight = 0.55
                other_weight = 0.45
                fundamental_avg = np.mean(fundamental_preds)
                other_avg = np.mean(other_preds)
                weighted_pred = fundamental_avg * fundamental_weight + other_avg * other_weight
            else:
                ensemble_pred = np.median(filtered_predictions)
                ensemble_mean = np.mean(filtered_predictions)
                weighted_pred = ensemble_pred * 0.6 + ensemble_mean * 0.4
        else:
            weighted_pred = np.mean(predictions)
        
        ensemble_std = np.std(filtered_predictions) if len(filtered_predictions) > 1 else np.std(predictions)
        
        agreement = 1 - min(1.0, ensemble_std / current_price) if current_price > 0 else 0.5
        
        final_confidence = min(88.0, max(40.0, np.mean(filtered_confidences) * (0.75 + agreement * 0.25)))
        
        predicted_price = weighted_pred
        predicted_change = (predicted_price - current_price) / current_price
        
        unique_factors = list(dict.fromkeys(factors))[:5]
        
        return {
            'predicted_price': predicted_price,
            'change_percent': predicted_change * 100,
            'confidence': final_confidence,
            'key_factors': ', '.join(unique_factors) if unique_factors else 'Multiple fundamental signals'
        }
    
    def _fundamental_model(self, earnings_growth, revenue_growth, pe_ratio, peg_ratio, 
                          roe, roa, profit_margin, operating_margin, current_price, market_cap=0):
        score = 0
        factors = []
        confidence = 70.0
        
        if earnings_growth > 0.20:
            score += 0.25
            factors.append(f"Exceptional earnings growth ({earnings_growth*100:.1f}%)")
            confidence += 5
        elif earnings_growth > 0.15:
            score += 0.18
            factors.append(f"Strong earnings growth ({earnings_growth*100:.1f}%)")
        elif earnings_growth > 0.10:
            score += 0.12
            factors.append(f"Moderate earnings growth ({earnings_growth*100:.1f}%)")
        elif earnings_growth < -0.15:
            score -= 0.25
            factors.append(f"Severe earnings decline ({earnings_growth*100:.1f}%)")
            confidence += 5
        elif earnings_growth < -0.05:
            score -= 0.15
            factors.append(f"Earnings decline ({earnings_growth*100:.1f}%)")
        
        if revenue_growth > 0.15:
            score += 0.20
            factors.append(f"Strong revenue growth ({revenue_growth*100:.1f}%)")
            confidence += 3
        elif revenue_growth > 0.08:
            score += 0.12
            factors.append(f"Revenue growth ({revenue_growth*100:.1f}%)")
        elif revenue_growth < -0.10:
            score -= 0.20
            factors.append(f"Revenue decline ({revenue_growth*100:.1f}%)")
            confidence += 3
        elif revenue_growth < 0:
            score -= 0.10
            factors.append(f"Revenue contraction ({revenue_growth*100:.1f}%)")
        
        if 0 < pe_ratio < 15:
            score += 0.15
            factors.append(f"Undervalued P/E ({pe_ratio:.1f})")
            confidence += 4
        elif 15 <= pe_ratio < 25:
            score += 0.08
            factors.append(f"Reasonable P/E ({pe_ratio:.1f})")
        elif pe_ratio > 50:
            score -= 0.20
            factors.append(f"Overvalued P/E ({pe_ratio:.1f})")
            confidence += 4
        elif pe_ratio > 35:
            score -= 0.12
            factors.append(f"High P/E ({pe_ratio:.1f})")
        
        if 0 < peg_ratio < 1.0:
            score += 0.18
            factors.append(f"Excellent PEG ({peg_ratio:.2f})")
            confidence += 4
        elif 1.0 <= peg_ratio < 1.5:
            score += 0.10
            factors.append(f"Good PEG ({peg_ratio:.2f})")
        elif peg_ratio > 2.5:
            score -= 0.18
            factors.append(f"Poor PEG ({peg_ratio:.2f})")
            confidence += 3
        
        roe_display = roe * 100 if roe <= 1.0 else roe
        if roe > 1.5 or (roe > 0.20 and roe <= 1.0):
            score += 0.20
            factors.append(f"Exceptional ROE ({roe_display:.1f}%)")
            confidence += 5
        elif roe > 1.0 or (roe > 0.15 and roe <= 1.0):
            score += 0.15
            factors.append(f"High ROE ({roe_display:.1f}%)")
            confidence += 3
        elif roe > 0.10:
            score += 0.10
            factors.append(f"Good ROE ({roe_display:.1f}%)")
        elif roe < 0.05:
            score -= 0.12
            factors.append(f"Low ROE ({roe_display:.1f}%)")
        
        if roa > 0.10:
            score += 0.12
            factors.append(f"Strong ROA ({roa*100:.1f}%)")
        elif roa < 0.03:
            score -= 0.10
            factors.append(f"Weak ROA ({roa*100:.1f}%)")
        
        if profit_margin > 0.20:
            score += 0.15
            factors.append(f"Excellent margins ({profit_margin*100:.1f}%)")
            confidence += 3
        elif profit_margin > 0.10:
            score += 0.10
            factors.append(f"Good margins ({profit_margin*100:.1f}%)")
        elif profit_margin < 0.05:
            score -= 0.12
            factors.append(f"Thin margins ({profit_margin*100:.1f}%)")
        
        if operating_margin > 0.25:
            score += 0.12
            factors.append(f"Strong operating margin ({operating_margin*100:.1f}%)")
        elif operating_margin < 0.10:
            score -= 0.10
            factors.append(f"Weak operating margin ({operating_margin*100:.1f}%)")
        
        market_cap_adjustment = 1.0
        if market_cap > 0:
            if market_cap > 200_000_000_000:
                market_cap_adjustment = 0.85
            elif market_cap < 2_000_000_000:
                market_cap_adjustment = 1.15
        
        score = max(-1.0, min(1.0, score))
        if abs(score) > 0.6:
            base_multiplier = 0.20
        elif abs(score) > 0.4:
            base_multiplier = 0.16
        elif abs(score) > 0.2:
            base_multiplier = 0.14
        else:
            base_multiplier = 0.12
        predicted_change = score * base_multiplier * market_cap_adjustment
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _dcf_model_advanced(self, info: dict, annual_return: float, current_price: float):
        try:
            free_cash_flow = info.get('freeCashflow', 0) or 0
            operating_cash_flow = info.get('operatingCashflow', 0) or 0
            total_cash = info.get('totalCash', 0) or 0
            total_debt = info.get('totalDebt', 0) or 0
            shares_outstanding = info.get('sharesOutstanding', 0) or 0
            
            if shares_outstanding <= 0:
                return None, 0, []
            
            if free_cash_flow <= 0:
                if operating_cash_flow > 0:
                    free_cash_flow = operating_cash_flow * 0.7
                else:
                    return None, 0, []
            
            fcf_per_share = free_cash_flow / shares_outstanding
            
            earnings_growth_quarterly = info.get('earningsQuarterlyGrowth', 0) or 0
            earnings_growth_annual = info.get('earningsGrowth', 0) or 0
            
            if earnings_growth_quarterly and abs(earnings_growth_quarterly) > 1.0:
                earnings_growth_quarterly = earnings_growth_quarterly / 100.0
            
            if earnings_growth_annual and abs(earnings_growth_annual) > 1.0:
                earnings_growth_annual = earnings_growth_annual / 100.0
            
            if earnings_growth_annual:
                earnings_growth = earnings_growth_annual
            elif earnings_growth_quarterly:
                earnings_growth = earnings_growth_quarterly * 0.25
            else:
                earnings_growth = 0
            
            if abs(earnings_growth) > 0.50:
                earnings_growth = np.sign(earnings_growth) * 0.30
            
            revenue_growth = info.get('revenueGrowth', 0) or 0
            if revenue_growth and abs(revenue_growth) > 1.0:
                revenue_growth = revenue_growth / 100.0
            
            growth_rate = max(earnings_growth, revenue_growth, annual_return * 0.6, 0.02)
            growth_rate = max(-0.10, min(0.15, growth_rate))
            
            roe = info.get('returnOnEquity', 0.10) or 0.10
            if roe and abs(roe) > 2.0:
                roe = roe / 100.0
            elif roe and abs(roe) > 1.0 and abs(roe) <= 2.0:
                roe = roe / 100.0
            beta = info.get('beta', 1.0) or 1.0
            risk_free_rate = 0.04
            market_risk_premium = 0.06
            cost_of_equity = risk_free_rate + beta * market_risk_premium
            
            debt_to_equity = info.get('debtToEquity', 0) or 0
            if debt_to_equity > 0:
                cost_of_debt = 0.05
                tax_rate = 0.21
                wacc = (cost_of_equity * (1 / (1 + debt_to_equity)) + 
                       cost_of_debt * (debt_to_equity / (1 + debt_to_equity)) * (1 - tax_rate))
            else:
                wacc = cost_of_equity
            
            discount_rate = max(0.08, min(0.15, wacc))
            
            terminal_growth = min(0.04, max(0.02, growth_rate * 0.5))
            projection_years = 10
            
            pv_fcf = 0
            for year in range(1, projection_years + 1):
                year_growth = growth_rate * (1 - (year / projection_years) * 0.5)
                year_growth = max(terminal_growth, year_growth)
                future_fcf = fcf_per_share * ((1 + year_growth) ** year)
                pv_fcf += future_fcf / ((1 + discount_rate) ** year)
            
            terminal_fcf = fcf_per_share * ((1 + growth_rate) ** projection_years) * (1 + terminal_growth)
            
            if discount_rate <= terminal_growth:
                discount_rate = terminal_growth + 0.01
            
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)
            
            net_cash_per_share = (total_cash - total_debt) / shares_outstanding if shares_outstanding > 0 else 0
            
            intrinsic_value = pv_fcf + pv_terminal + net_cash_per_share
            
            if intrinsic_value > current_price * 3.0:
                intrinsic_value = current_price * 2.0
            elif intrinsic_value < current_price * 0.3:
                intrinsic_value = current_price * 0.5
            
            factors = [f"DCF: ${intrinsic_value:.2f}"]
            if intrinsic_value > current_price * 1.2:
                factors.append("Significantly undervalued")
            elif intrinsic_value > current_price * 1.1:
                factors.append("Undervalued")
            elif intrinsic_value < current_price * 0.8:
                factors.append("Overvalued")
            
            confidence = 75.0 if free_cash_flow > 0 and shares_outstanding > 0 else 55.0
            
            return intrinsic_value, confidence, factors
        except Exception as e:
            return None, 0, []
    
    def _valuation_model(self, pe_ratio, peg_ratio, roe, roa, profit_margin, beta, current_price, pe_mean=0, pe_std=0):
        score = 0
        factors = []
        confidence = 68.0
        
        fair_pe = roe * 10 if roe > 0 else 15
        if pe_ratio > 0:
            pe_ratio_vs_fair = fair_pe / pe_ratio
            if pe_ratio_vs_fair > 1.3:
                score += 0.20
                factors.append(f"Undervalued vs ROE (P/E {pe_ratio:.1f} vs fair {fair_pe:.1f})")
                confidence += 5
            elif pe_ratio_vs_fair < 0.7:
                score -= 0.20
                factors.append(f"Overvalued vs ROE")
                confidence += 5
        
        if pe_mean > 0 and pe_std > 0:
            pe_z_score = (pe_ratio - pe_mean) / pe_std if pe_std > 0 else 0
            if pe_z_score > 1.5:
                score -= 0.15
                factors.append(f"P/E significantly above mean ({pe_z_score:.1f}σ)")
                confidence += 3
            elif pe_z_score < -1.5:
                score += 0.15
                factors.append(f"P/E significantly below mean ({pe_z_score:.1f}σ)")
                confidence += 3
        
        if beta < 0.8:
            score += 0.08
            factors.append(f"Low beta ({beta:.2f}) - defensive")
        elif beta > 1.3:
            score -= 0.05
            factors.append(f"High beta ({beta:.2f}) - volatile")
        
        if profit_margin > 0.15 and roe > 0.15:
            score += 0.12
            factors.append("High quality metrics")
            confidence += 3
        
        score = max(-1.0, min(1.0, score))
        predicted_change = score * 0.12
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _pe_mean_reversion_model(self, pe_ratio, pe_mean, pe_std, current_price, earnings_growth):
        if pe_mean <= 0 or pe_std <= 0 or abs(pe_mean - pe_ratio) / pe_std < 0.5:
            return current_price, 50.0, []
        
        score = 0
        factors = []
        confidence = 65.0
        
        pe_z_score = (pe_ratio - pe_mean) / pe_std
        
        if abs(pe_z_score) > 2.5:
            reversion_strength = 0.25
            confidence += 8
        elif abs(pe_z_score) > 2.0:
            reversion_strength = 0.18
            confidence += 6
        elif abs(pe_z_score) > 1.5:
            reversion_strength = 0.12
            confidence += 4
        elif abs(pe_z_score) > 1.0:
            reversion_strength = 0.08
            confidence += 2
        else:
            return current_price, 50.0, []
        
        if pe_z_score > 0:
            score = -reversion_strength
            factors.append(f"P/E {pe_z_score:.1f}σ above mean")
        else:
            score = reversion_strength
            factors.append(f"P/E {abs(pe_z_score):.1f}σ below mean")
        
        if earnings_growth > 0.15:
            growth_adjustment = 1.0 - (earnings_growth * 0.4)
            score *= max(0.3, growth_adjustment)
            factors.append("Strong growth mitigates reversion")
            confidence -= 3
        elif earnings_growth > 0.10:
            growth_adjustment = 1.0 - (earnings_growth * 0.3)
            score *= max(0.5, growth_adjustment)
            confidence -= 2
        elif earnings_growth > 0.05:
            growth_adjustment = 1.0 - (earnings_growth * 0.2)
            score *= max(0.7, growth_adjustment)
        
        predicted_change = score
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _market_cap_valuation_model(self, market_cap, price_to_book, book_value, roe, current_price, shares_outstanding=0):
        if market_cap <= 0 or shares_outstanding <= 0:
            return None, 0, []
        
        score = 0
        factors = []
        confidence = 60.0
        
        if price_to_book > 0:
            if price_to_book < 1.0:
                score += 0.20
                factors.append(f"Undervalued P/B ({price_to_book:.2f})")
                confidence += 5
            elif price_to_book < 2.0:
                score += 0.08
                factors.append(f"Reasonable P/B ({price_to_book:.2f})")
            elif price_to_book > 5.0:
                score -= 0.15
                factors.append(f"High P/B ({price_to_book:.2f})")
                confidence += 3
        
        if roe > 0.15 and price_to_book > 0:
            justified_pb = roe * 1.5
            pb_ratio = price_to_book / justified_pb if justified_pb > 0 else 1.0
            if pb_ratio < 0.7:
                score += 0.12
                factors.append("P/B below justified level")
            elif pb_ratio > 1.5:
                score -= 0.12
                factors.append("P/B above justified level")
        
        market_cap_category = 'mega' if market_cap > 200_000_000_000 else 'large' if market_cap > 10_000_000_000 else 'mid' if market_cap > 2_000_000_000 else 'small'
        
        if market_cap_category == 'small':
            score *= 1.2
            factors.append("Small cap - higher volatility expected")
        elif market_cap_category == 'mega':
            score *= 0.8
            factors.append("Mega cap - lower volatility expected")
        
        score = max(-1.0, min(1.0, score))
        predicted_change = score * 0.10
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _financial_health_model(self, debt_to_equity, current_ratio, quick_ratio, 
                                profit_margin, operating_margin, current_price):
        score = 0
        factors = []
        confidence = 65.0
        
        if debt_to_equity < 30:
            score += 0.15
            factors.append(f"Low debt (D/E {debt_to_equity:.1f}%)")
            confidence += 4
        elif debt_to_equity > 100:
            score -= 0.20
            factors.append(f"High debt (D/E {debt_to_equity:.1f}%)")
            confidence += 4
        elif debt_to_equity > 70:
            score -= 0.12
            factors.append(f"Elevated debt (D/E {debt_to_equity:.1f}%)")
        
        if current_ratio > 2.0:
            score += 0.10
            factors.append(f"Strong liquidity (CR {current_ratio:.2f})")
        elif current_ratio < 1.0:
            score -= 0.15
            factors.append(f"Weak liquidity (CR {current_ratio:.2f})")
            confidence += 3
        
        if quick_ratio > 1.5:
            score += 0.08
            factors.append(f"Good quick ratio ({quick_ratio:.2f})")
        elif quick_ratio < 0.8:
            score -= 0.12
            factors.append(f"Poor quick ratio ({quick_ratio:.2f})")
        
        if profit_margin > 0.15 and operating_margin > 0.20:
            score += 0.12
            factors.append("Strong profitability")
            confidence += 2
        
        score = max(-1.0, min(1.0, score))
        predicted_change = score * 0.08
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _trend_extrapolation_model(self, closes: np.ndarray, annual_return: float, 
                                  sharpe_ratio: float, volatility: float, current_price: float):
        if len(closes) < 252:
            return current_price, 50.0, []
        
        factors = []
        confidence = 70.0
        
        recent_trend = (closes[-1] - closes[-63]) / closes[-63] if len(closes) >= 63 else annual_return
        medium_trend = annual_return
        
        momentum_factor = recent_trend * 0.6 + medium_trend * 0.4
        
        if sharpe_ratio > 1.5:
            factors.append(f"Excellent risk-adjusted return (Sharpe {sharpe_ratio:.2f})")
            confidence += 5
            momentum_factor *= 1.1
        elif sharpe_ratio > 1.0:
            factors.append(f"Good risk-adjusted return (Sharpe {sharpe_ratio:.2f})")
            confidence += 2
        elif sharpe_ratio < 0.5:
            factors.append(f"Poor risk-adjusted return (Sharpe {sharpe_ratio:.2f})")
            momentum_factor *= 0.9
        
        if volatility > 0.5:
            momentum_factor *= 0.85
            factors.append("High volatility - adjusted")
        elif volatility < 0.2:
            confidence += 2
            factors.append("Low volatility")
        
        if annual_return > 0.20:
            factors.append(f"Strong annual return ({annual_return*100:.1f}%)")
            confidence += 3
        elif annual_return < -0.10:
            factors.append(f"Negative annual return ({annual_return*100:.1f}%)")
            confidence += 3
        
        predicted_change = momentum_factor * 0.3
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _dividend_discount_model(self, dividend_yield: float, payout_ratio: float, 
                                 earnings_growth: float, current_price: float):
        if dividend_yield <= 0:
            return current_price, 50.0, []
        
        factors = []
        confidence = 65.0
        
        current_dividend = current_price * dividend_yield
        
        growth_rate = max(0.02, min(0.06, earnings_growth * 0.5))
        required_return = 0.08
        
        if growth_rate >= required_return:
            growth_rate = required_return * 0.95
        
        terminal_value = current_dividend * (1 + growth_rate) / (required_return - growth_rate)
        
        if payout_ratio > 0.8:
            factors.append(f"High payout ratio ({payout_ratio*100:.1f}%) - sustainability risk")
            terminal_value *= 0.9
            confidence -= 5
        elif payout_ratio < 0.3:
            factors.append(f"Low payout ratio ({payout_ratio*100:.1f}%) - growth potential")
            confidence += 3
        
        factors.append(f"Dividend yield {dividend_yield*100:.2f}%")
        
        predicted_price = terminal_value
        predicted_change = (predicted_price - current_price) / current_price
        
        return predicted_price, confidence, factors
    
    def _generate_warnings(self, info: dict, hist_data: pd.DataFrame):
        warnings = []
        
        if len(hist_data) < 100:
            warnings.append("Limited historical data available")
        
        if info.get('regularMarketVolume', 0) < 100000:
            warnings.append("Low trading volume - predictions may be less reliable")
        
        if info.get('trailingPE', 0) and info.get('trailingPE', 0) > 50:
            warnings.append("High P/E ratio detected")
        
        if info.get('debtToEquity', 0) and info.get('debtToEquity', 0) > 200:
            warnings.append("High debt-to-equity ratio")
        
        if info.get('currentRatio', 0) and info.get('currentRatio', 0) < 1.0:
            warnings.append("Current ratio below 1.0 - liquidity concerns")
        
        if info.get('profitMargins', 0) and info.get('profitMargins', 0) < 0:
            warnings.append("Negative profit margins")
        
        return warnings
