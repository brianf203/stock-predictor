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
        
        market_cap = info.get('marketCap', 0) or 0
        if market_cap == 0:
            shares_outstanding = info.get('sharesOutstanding', 0) or 0
            current_price_val = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0) or current_price
            if shares_outstanding > 0 and current_price_val > 0:
                market_cap = shares_outstanding * current_price_val
        
        market_cap_params = self._get_market_cap_parameters(market_cap) if market_cap > 0 else None
        
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
        volatility_model = self._model_volatility(df, market_cap_params=market_cap_params)
        
        predictions = []
        factors = []
        confidence_scores = []
        
        model1_pred, model1_conf, model1_factors = self._momentum_model(df, latest, prev, current_price, market_cap_params)
        predictions.append(model1_pred)
        confidence_scores.append(model1_conf)
        factors.extend(model1_factors)
        
        model2_pred, model2_conf, model2_factors = self._mean_reversion_model(df, latest, support, resistance, current_price, market_cap_params)
        predictions.append(model2_pred)
        confidence_scores.append(model2_conf)
        factors.extend(model2_factors)
        
        model3_pred, model3_conf, model3_factors = self._trend_following_model(df, latest, current_price, market_regime, market_cap_params)
        predictions.append(model3_pred)
        confidence_scores.append(model3_conf)
        factors.extend(model3_factors)
        
        model4_pred, model4_conf, model4_factors = self._volume_price_model(df, latest, current_price, market_cap_params)
        predictions.append(model4_pred)
        confidence_scores.append(model4_conf)
        factors.extend(model4_factors)
        
        model5_pred, model5_conf, model5_factors = self._volatility_model(df, latest, volatility_model, current_price, market_cap_params)
        predictions.append(model5_pred)
        confidence_scores.append(model5_conf)
        factors.extend(model5_factors)
        
        predictions_array = np.array(predictions)
        confidence_array = np.array(confidence_scores)
        
        confidence_weights = confidence_array / confidence_array.sum() if confidence_array.sum() > 0 else np.ones(len(predictions)) / len(predictions)
        
        changes = (predictions_array - current_price) / current_price
        median_change = np.median(changes)
        mad = np.median(np.abs(changes - median_change))
        
        if mad > 0:
            z_scores = np.abs((changes - median_change) / (mad + 1e-10))
            outlier_mask = z_scores < 2.5
        else:
            outlier_mask = np.ones(len(predictions), dtype=bool)
        
        if np.sum(outlier_mask) < 2:
            outlier_mask = np.ones(len(predictions), dtype=bool)
        
        filtered_predictions = predictions_array[outlier_mask]
        filtered_confidences = confidence_array[outlier_mask]
        filtered_weights = confidence_weights[outlier_mask]
        filtered_weights = filtered_weights / filtered_weights.sum() if filtered_weights.sum() > 0 else np.ones(len(filtered_predictions)) / len(filtered_predictions)
        
        ensemble_pred = np.average(filtered_predictions, weights=filtered_weights)
        ensemble_median = np.median(filtered_predictions)
        
        ensemble_pred = ensemble_pred * 0.6 + ensemble_median * 0.4
        
        ensemble_std = np.std(filtered_predictions)
        agreement = 1 - min(1.0, ensemble_std / current_price) if current_price > 0 else 0.5
        
        market_regime = self._detect_market_regime(df)
        regime_weight_adjustment = self._get_regime_weights(market_regime, filtered_predictions, current_price)
        
        if regime_weight_adjustment is not None:
            adjusted_weights = filtered_weights * regime_weight_adjustment
            adjusted_weights = adjusted_weights / adjusted_weights.sum() if adjusted_weights.sum() > 0 else filtered_weights
            ensemble_pred = np.average(filtered_predictions, weights=adjusted_weights) * 0.7 + ensemble_median * 0.3
        else:
            ensemble_pred = ensemble_pred * 0.6 + ensemble_median * 0.4
        
        monte_carlo_result = self._monte_carlo_simulation(
            filtered_predictions, filtered_confidences, current_price, volatility_model
        )
        
        if monte_carlo_result:
            ensemble_pred = ensemble_pred * 0.75 + monte_carlo_result['mean'] * 0.25
            ensemble_std = monte_carlo_result['std']
        
        avg_confidence = np.average(filtered_confidences, weights=filtered_weights)
        
        prediction_interval_width = ensemble_std / current_price if current_price > 0 else 0.1
        interval_adjustment = 1 - min(0.3, prediction_interval_width * 2)
        
        market_cap_confidence_adjustment = market_cap_params.get('confidence_multiplier', 1.0) if market_cap_params else 1.0
        
        final_confidence = min(92.0, max(45.0, avg_confidence * (0.60 + agreement * 0.25 + interval_adjustment * 0.15) * market_cap_confidence_adjustment))
        
        predicted_price = ensemble_pred
        predicted_change = (predicted_price - current_price) / current_price
        
        bearish_keywords = ['oversold', 'lower', 'support', 'breakdown', 'bearish', 'death', 'downtrend', 'decline', 'distribution', 'below']
        bullish_keywords = ['overbought', 'upper', 'resistance', 'breakout', 'bullish', 'golden', 'uptrend', 'growth', 'accumulation', 'above', 'crossover']
        
        bearish_factors = [f for f in factors if any(kw.lower() in f.lower() for kw in bearish_keywords)]
        bullish_factors = [f for f in factors if any(kw.lower() in f.lower() for kw in bullish_keywords)]
        neutral_factors = [f for f in factors if f not in bearish_factors and f not in bullish_factors]
        
        unique_factors_all = list(dict.fromkeys(factors))
        unique_bearish = list(dict.fromkeys(bearish_factors))
        unique_bullish = list(dict.fromkeys(bullish_factors))
        unique_neutral = list(dict.fromkeys(neutral_factors))
        
        if predicted_change < -0.01:
            bearish_count = min(3, len(unique_bearish))
            bullish_count = min(2, len(unique_bullish))
            selected_factors = unique_bearish[:bearish_count] + unique_bullish[:bullish_count]
            if len(selected_factors) < 3 and unique_neutral:
                selected_factors.extend(unique_neutral[:3-len(selected_factors)])
        elif predicted_change > 0.01:
            bullish_count = min(3, len(unique_bullish))
            bearish_count = min(2, len(unique_bearish))
            selected_factors = unique_bullish[:bullish_count] + unique_bearish[:bearish_count]
            if len(selected_factors) < 3 and unique_neutral:
                selected_factors.extend(unique_neutral[:3-len(selected_factors)])
        else:
            selected_factors = unique_factors_all[:5]
        
        if not selected_factors:
            selected_factors = unique_factors_all[:5] if unique_factors_all else ['Multiple technical signals']
        
        return {
            'predicted_price': predicted_price,
            'change_percent': predicted_change * 100,
            'confidence': final_confidence,
            'key_factors': ', '.join(selected_factors[:5]) if selected_factors else 'Multiple technical signals'
        }
    
    def _momentum_model(self, df: pd.DataFrame, latest: pd.Series, prev: pd.Series, current_price: float, market_cap_params: dict = None):
        score = 0
        factors = []
        confidence = 70.0
        
        rsi_14 = latest['RSI_14'] if not pd.isna(latest['RSI_14']) else None
        rsi_7 = latest['RSI_7'] if not pd.isna(latest['RSI_7']) else None
        
        if rsi_14 is not None:
            if rsi_14 < 25:
                score += 0.22
                factors.append("Strongly oversold (RSI<25)")
                confidence += 6
            elif rsi_14 < 35:
                score += 0.12
                factors.append("Oversold")
                confidence += 2
            elif rsi_14 > 75:
                score -= 0.22
                factors.append("Strongly overbought (RSI>75)")
                confidence += 6
            elif rsi_14 > 65:
                score -= 0.12
                factors.append("Overbought")
                confidence += 2
            
            if rsi_7 is not None:
                if rsi_14 < rsi_7 and rsi_14 < 40:
                    score += 0.08
                    factors.append("RSI divergence (bullish)")
                elif rsi_14 > rsi_7 and rsi_14 > 60:
                    score -= 0.08
                    factors.append("RSI divergence (bearish)")
        
        macd_hist = latest['MACD_hist'] if not pd.isna(latest['MACD_hist']) else None
        prev_macd_hist = prev['MACD_hist'] if not pd.isna(prev['MACD_hist']) else None
        
        if macd_hist is not None and prev_macd_hist is not None:
            if macd_hist > 0 and prev_macd_hist <= 0:
                score += 0.18
                factors.append("MACD bullish crossover")
                confidence += 4
            elif macd_hist < 0 and prev_macd_hist >= 0:
                score -= 0.18
                factors.append("MACD bearish crossover")
                confidence += 4
            elif macd_hist > prev_macd_hist * 1.2:
                score += 0.10
                factors.append("MACD momentum accelerating")
            elif macd_hist < prev_macd_hist * 0.8:
                score -= 0.10
                factors.append("MACD momentum decelerating")
        
        macd = latest['MACD'] if not pd.isna(latest['MACD']) else None
        macd_signal = latest['MACD_signal'] if not pd.isna(latest['MACD_signal']) else None
        if macd is not None and macd_signal is not None:
            if macd > macd_signal * 1.05:
                score += 0.05
            elif macd < macd_signal * 0.95:
                score -= 0.05
        
        stoch_k = latest['Stoch_K'] if not pd.isna(latest['Stoch_K']) else None
        stoch_d = latest['Stoch_D'] if not pd.isna(latest['Stoch_D']) else None
        
        if stoch_k is not None and stoch_d is not None:
            if stoch_k < 20 and stoch_d < 20:
                score += 0.14
                factors.append("Stochastic oversold")
                confidence += 2
            elif stoch_k > 80 and stoch_d > 80:
                score -= 0.14
                factors.append("Stochastic overbought")
                confidence += 2
            elif stoch_k > stoch_d and stoch_k < 50:
                score += 0.06
            elif stoch_k < stoch_d and stoch_k > 50:
                score -= 0.06
        
        williams_r = latest['Williams_R'] if not pd.isna(latest['Williams_R']) else None
        if williams_r is not None:
            if williams_r < -80:
                score += 0.12
                factors.append("Williams %R oversold")
            elif williams_r > -20:
                score -= 0.12
                factors.append("Williams %R overbought")
        
        roc = latest['ROC'] if not pd.isna(latest['ROC']) else None
        if roc is not None:
            if roc > 5:
                score += 0.10
                factors.append("Strong momentum (ROC>5%)")
            elif roc < -5:
                score -= 0.10
                factors.append("Weak momentum (ROC<-5%)")
        
        price_momentum_5 = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) if len(df) >= 6 else 0
        price_momentum_10 = (df['Close'].iloc[-1] / df['Close'].iloc[-11] - 1) if len(df) >= 11 else 0
        
        if price_momentum_5 > 0.03 and price_momentum_10 > 0.05:
            score += 0.12
            factors.append("Multi-timeframe momentum")
            confidence += 3
        elif price_momentum_5 < -0.03 and price_momentum_10 < -0.05:
            score -= 0.12
            factors.append("Multi-timeframe weakness")
            confidence += 3
        
        predicted_change = score * 0.022
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _mean_reversion_model(self, df: pd.DataFrame, latest: pd.Series, support: float, resistance: float, current_price: float, market_cap_params: dict = None):
        score = 0
        factors = []
        confidence = 65.0
        
        bb_position = (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']) if not pd.isna(latest['BB_upper']) and latest['BB_upper'] != latest['BB_lower'] else 0.5
        
        bb_width = latest['BB_width'] if not pd.isna(latest['BB_width']) else 0.05
        
        if bb_position < 0.1:
            reversion_strength = 0.28
            factors.append("Extreme lower Bollinger Band")
            confidence += 10
        elif bb_position < 0.2:
            reversion_strength = 0.18
            factors.append("Near lower Bollinger Band")
            confidence += 5
        elif bb_position > 0.9:
            reversion_strength = -0.28
            factors.append("Extreme upper Bollinger Band")
            confidence += 10
        elif bb_position > 0.8:
            reversion_strength = -0.18
            factors.append("Near upper Bollinger Band")
            confidence += 5
        else:
            reversion_strength = 0
        
        if bb_width > 0.08:
            reversion_strength *= 0.7
        
        score += reversion_strength
        
        if support > 0:
            distance_to_support = (current_price - support) / support
            if -0.03 <= distance_to_support <= 0.02:
                support_strength = min(0.22, abs(distance_to_support) * 10)
                score += support_strength
                factors.append(f"Near support ${support:.2f}")
                confidence += 6
            elif distance_to_support < -0.03:
                score += 0.15
                factors.append(f"Below support ${support:.2f}")
        
        if resistance > 0:
            distance_to_resistance = (resistance - current_price) / current_price
            if -0.02 <= distance_to_resistance <= 0.03:
                resistance_strength = min(0.22, abs(distance_to_resistance) * 10)
                score -= resistance_strength
                factors.append(f"Near resistance ${resistance:.2f}")
                confidence += 6
            elif distance_to_resistance < -0.02:
                score -= 0.15
                factors.append(f"Above resistance ${resistance:.2f}")
        
        kc_position = (latest['Close'] - latest['KC_lower']) / (latest['KC_upper'] - latest['KC_lower']) if not pd.isna(latest['KC_upper']) and latest['KC_upper'] != latest['KC_lower'] else 0.5
        
        if kc_position < 0.2:
            score += 0.10
            factors.append("Keltner Channel oversold")
        elif kc_position > 0.8:
            score -= 0.10
            factors.append("Keltner Channel overbought")
        
        if not pd.isna(latest['EMA_20']):
            distance_from_ema = (current_price - latest['EMA_20']) / latest['EMA_20']
            if abs(distance_from_ema) > 0.05:
                mean_reversion_signal = np.sign(-distance_from_ema) * min(0.12, abs(distance_from_ema) * 2)
                score += mean_reversion_signal
                factors.append("Price deviation from EMA(20)")
        
        if not pd.isna(latest['SMA_50']):
            distance_from_sma = (current_price - latest['SMA_50']) / latest['SMA_50']
            if abs(distance_from_sma) > 0.08:
                mean_reversion_signal = np.sign(-distance_from_sma) * min(0.10, abs(distance_from_sma) * 1.5)
                score += mean_reversion_signal
                factors.append("Price deviation from SMA(50)")
        
        base_change = score * 0.018
        
        if market_cap_params:
            volatility_mult = market_cap_params.get('volatility_multiplier', 1.0)
            base_change *= volatility_mult
            if market_cap_params.get('category') in ['mega', 'large']:
                base_change *= 0.9
            elif market_cap_params.get('category') in ['small', 'micro']:
                base_change *= 1.1
        
        predicted_change = base_change
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _trend_following_model(self, df: pd.DataFrame, latest: pd.Series, current_price: float, market_regime: str, market_cap_params: dict = None):
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
        
        base_change = score * 0.03
        
        if market_cap_params:
            market_correlation = market_cap_params.get('market_correlation', 0.70)
            if market_regime == 'bull' or market_regime == 'bear':
                base_change *= (0.6 + market_correlation * 0.4)
            else:
                base_change *= (0.8 + market_correlation * 0.2)
        
        predicted_change = base_change
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _volume_price_model(self, df: pd.DataFrame, latest: pd.Series, current_price: float, market_cap_params: dict = None):
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
        
        base_change = score * 0.022
        
        if market_cap_params:
            liquidity_factor = market_cap_params.get('liquidity_factor', 1.0)
            base_change *= liquidity_factor
        
        predicted_change = base_change
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _volatility_model(self, df: pd.DataFrame, latest: pd.Series, volatility_params: dict, current_price: float, market_cap_params: dict = None):
        score = 0
        factors = []
        confidence = 72.0
        
        recent_returns = df['Close'].pct_change().iloc[-5:].dropna()
        if len(recent_returns) > 0:
            recent_vol = recent_returns.std() * np.sqrt(252)
            if recent_vol > 0.5:
                score -= 0.15
                factors.append("Extreme recent volatility")
                confidence -= 5
            elif recent_vol < 0.10:
                confidence += 3
                factors.append("Low recent volatility")
        
        if not pd.isna(latest['ATR']):
            atr_pct = latest['ATR'] / current_price if current_price > 0 else 0
            if atr_pct > 0.04:
                factors.append("High volatility (ATR>4%)")
                score -= 0.10
                confidence -= 3
            elif atr_pct < 0.01:
                factors.append("Low volatility (ATR<1%)")
                confidence += 3
        
        if 'realized_vol' in volatility_params and 'garch_vol' in volatility_params:
            rv = volatility_params['realized_vol']
            garch_vol = volatility_params['garch_vol']
            
            if garch_vol > rv * 1.2:
                factors.append("Volatility clustering (GARCH>realized)")
                score -= 0.08
            elif garch_vol < rv * 0.8:
                factors.append("Volatility mean reversion")
                confidence += 2
            
            if rv > 0.4:
                factors.append("Extreme volatility")
                score -= 0.12
                confidence -= 4
            elif rv < 0.15:
                confidence += 2
        
        if 'volatility_cluster' in volatility_params:
            cluster = volatility_params['volatility_cluster']
            if cluster > 1.5:
                factors.append("High volatility cluster")
                score -= 0.10
                confidence -= 3
            elif cluster < 0.7:
                factors.append("Low volatility cluster")
                confidence += 2
        
        if 'volatility_regime' in volatility_params:
            if volatility_params['volatility_regime'] == 'high':
                score -= 0.08
                factors.append("High volatility regime")
                confidence -= 2
            elif volatility_params['volatility_regime'] == 'low':
                confidence += 3
                factors.append("Low volatility regime")
        
        bb_width = latest['BB_width'] if not pd.isna(latest['BB_width']) else 0.05
        if bb_width > 0.1:
            factors.append("Wide Bollinger Bands")
            score -= 0.05
        elif bb_width < 0.03:
            factors.append("Narrow Bollinger Bands")
            confidence += 2
        
        trend = df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1 if len(df) >= 5 else 0
        volatility_adjusted_trend = trend / (volatility_params.get('realized_vol', 0.2) + 0.1)
        
        if volatility_adjusted_trend > 0.1:
            score += 0.10
            factors.append("Strong risk-adjusted momentum")
        elif volatility_adjusted_trend < -0.1:
            score -= 0.10
            factors.append("Weak risk-adjusted momentum")
        
        predicted_change = score * 0.015
        predicted_price = current_price * (1 + predicted_change)
        
        return predicted_price, confidence, factors
    
    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20):
        closes = df['Close'].values
        if len(closes) < window * 2:
            return 0, 0
        
        recent_closes = closes[-window*3:]
        highs = df['High'].values[-window*3:]
        lows = df['Low'].values[-window*3:]
        
        support_levels = []
        resistance_levels = []
        support_strength = []
        resistance_strength = []
        
        pivot_window = max(5, window // 4)
        
        for i in range(pivot_window, len(recent_closes) - pivot_window):
            local_min = np.min(lows[i-pivot_window:i+pivot_window])
            local_max = np.max(highs[i-pivot_window:i+pivot_window])
            
            if lows[i] <= local_min * 1.005:
                support_levels.append(lows[i])
                touches = np.sum((lows[max(0, i-window):min(len(lows), i+window)] <= lows[i] * 1.02) & 
                               (lows[max(0, i-window):min(len(lows), i+window)] >= lows[i] * 0.98))
                support_strength.append(touches)
            
            if highs[i] >= local_max * 0.995:
                resistance_levels.append(highs[i])
                touches = np.sum((highs[max(0, i-window):min(len(highs), i+window)] >= highs[i] * 0.98) & 
                               (highs[max(0, i-window):min(len(highs), i+window)] <= highs[i] * 1.02))
                resistance_strength.append(touches)
        
        if support_levels and support_strength:
            weighted_support = np.average(support_levels, weights=support_strength)
            support = weighted_support
        else:
            support = np.min(lows[-window:])
        
        if resistance_levels and resistance_strength:
            weighted_resistance = np.average(resistance_levels, weights=resistance_strength)
            resistance = weighted_resistance
        else:
            resistance = np.max(highs[-window:])
        
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
    
    def _get_market_cap_parameters(self, market_cap: float):
        if market_cap >= 200_000_000_000:
            return {
                'category': 'mega',
                'volatility_multiplier': 0.75,
                'beta_adjustment': 0.85,
                'market_correlation': 0.85,
                'confidence_multiplier': 1.10,
                'prediction_stability': 1.15,
                'liquidity_factor': 1.20
            }
        elif market_cap >= 10_000_000_000:
            return {
                'category': 'large',
                'volatility_multiplier': 0.85,
                'beta_adjustment': 0.92,
                'market_correlation': 0.80,
                'confidence_multiplier': 1.05,
                'prediction_stability': 1.08,
                'liquidity_factor': 1.10
            }
        elif market_cap >= 2_000_000_000:
            return {
                'category': 'mid',
                'volatility_multiplier': 1.0,
                'beta_adjustment': 1.0,
                'market_correlation': 0.70,
                'confidence_multiplier': 1.0,
                'prediction_stability': 1.0,
                'liquidity_factor': 1.0
            }
        elif market_cap >= 300_000_000:
            return {
                'category': 'small',
                'volatility_multiplier': 1.25,
                'beta_adjustment': 1.15,
                'market_correlation': 0.55,
                'confidence_multiplier': 0.90,
                'prediction_stability': 0.85,
                'liquidity_factor': 0.80
            }
        else:
            return {
                'category': 'micro',
                'volatility_multiplier': 1.50,
                'beta_adjustment': 1.30,
                'market_correlation': 0.40,
                'confidence_multiplier': 0.80,
                'prediction_stability': 0.70,
                'liquidity_factor': 0.65
            }
    
    def _model_volatility(self, df: pd.DataFrame, window: int = 20, market_cap_params: dict = None):
        returns = df['Close'].pct_change().dropna()
        if len(returns) < window:
            return {'realized_vol': 0.2, 'volatility_regime': 'neutral', 'volatility_cluster': 1.0, 'garch_vol': 0.2, 'long_term_vol': 0.2}
        
        recent_returns = returns.iloc[-window:]
        realized_vol = recent_returns.std() * np.sqrt(252)
        
        if realized_vol > 0.35:
            regime = 'high'
        elif realized_vol < 0.15:
            regime = 'low'
        else:
            regime = 'normal'
        
        long_term_returns = returns.iloc[-min(252, len(returns)):]
        long_term_vol = long_term_returns.std() * np.sqrt(252)
        
        volatility_cluster = realized_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        squared_returns = recent_returns ** 2
        if len(squared_returns) >= 5:
            alpha = 0.1
            beta = 0.85
            omega = long_term_vol ** 2 * (1 - alpha - beta) if long_term_vol > 0 else 0.04
            
            garch_var = omega
            for ret_sq in squared_returns.iloc[-5:]:
                garch_var = omega + alpha * ret_sq + beta * garch_var
            
            garch_vol = np.sqrt(garch_var * 252) if garch_var > 0 else realized_vol
        else:
            garch_vol = realized_vol
        
        return {
            'realized_vol': realized_vol,
            'volatility_regime': regime,
            'volatility_cluster': volatility_cluster,
            'garch_vol': garch_vol,
            'long_term_vol': long_term_vol
        }
    
    def _monte_carlo_simulation(self, predictions: np.ndarray, confidences: np.ndarray, 
                                current_price: float, volatility_params: dict, n_simulations: int = 1000):
        if len(predictions) < 2:
            return None
        
        try:
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            realized_vol = volatility_params.get('realized_vol', 0.2)
            garch_vol = volatility_params.get('garch_vol', realized_vol)
            
            vol_to_use = (realized_vol + garch_vol) / 2
            
            confidence_weight = np.mean(confidences) / 100.0
            
            simulations = []
            for _ in range(n_simulations):
                random_shock = np.random.normal(0, vol_to_use / np.sqrt(252) * np.sqrt(5))
                weighted_pred = np.random.choice(predictions, p=confidences/confidences.sum())
                simulated_price = weighted_pred * (1 + random_shock * confidence_weight)
                simulations.append(simulated_price)
            
            simulations = np.array(simulations)
            simulations = simulations[(simulations > current_price * 0.5) & (simulations < current_price * 2.0)]
            
            if len(simulations) > 100:
                return {
                    'mean': np.mean(simulations),
                    'std': np.std(simulations),
                    'percentile_5': np.percentile(simulations, 5),
                    'percentile_95': np.percentile(simulations, 95)
                }
        except Exception:
            pass
        
        return None
    
    def _monte_carlo_simulation_long_term(self, predictions: np.ndarray, confidences: np.ndarray,
                                         current_price: float, annual_return: float, volatility: float,
                                         n_simulations: int = 500):
        if len(predictions) < 2:
            return None
        
        try:
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            vol_to_use = max(volatility, 0.15)
            confidence_weight = np.mean(confidences) / 100.0
            
            simulations = []
            for _ in range(n_simulations):
                months_ahead = 9
                random_shock = np.random.normal(annual_return / 12, vol_to_use / np.sqrt(12))
                weighted_pred = np.random.choice(predictions, p=confidences/confidences.sum())
                
                monthly_returns = []
                for _ in range(months_ahead):
                    monthly_return = random_shock + np.random.normal(0, vol_to_use / np.sqrt(12) * 0.5)
                    monthly_returns.append(monthly_return)
                
                cumulative_return = np.prod([1 + r for r in monthly_returns])
                simulated_price = weighted_pred * cumulative_return * confidence_weight + weighted_pred * (1 - confidence_weight)
                simulations.append(simulated_price)
            
            simulations = np.array(simulations)
            simulations = simulations[(simulations > current_price * 0.3) & (simulations < current_price * 3.0)]
            
            if len(simulations) > 50:
                return {
                    'mean': np.mean(simulations),
                    'std': np.std(simulations),
                    'percentile_5': np.percentile(simulations, 5),
                    'percentile_95': np.percentile(simulations, 95)
                }
        except Exception:
            pass
        
        return None
    
    def _get_regime_weights(self, market_regime: str, predictions: np.ndarray, current_price: float):
        if len(predictions) < 3:
            return None
        
        changes = (predictions - current_price) / current_price
        
        if market_regime == 'bull':
            bullish_models = (changes > 0).astype(float)
            weights = bullish_models * 1.3 + (1 - bullish_models) * 0.7
        elif market_regime == 'bear':
            bearish_models = (changes < 0).astype(float)
            weights = bearish_models * 1.3 + (1 - bearish_models) * 0.7
        else:
            return None
        
        return weights / weights.sum() if weights.sum() > 0 else None
    
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
        roe_original = roe
        if roe and abs(roe) > 2.0:
            roe = roe / 100.0
        elif roe and abs(roe) > 1.0 and abs(roe) <= 2.0:
            roe = roe / 100.0
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
        if market_cap == 0:
            shares_outstanding = info.get('sharesOutstanding', 0) or 0
            if shares_outstanding > 0 and current_price > 0:
                market_cap = shares_outstanding * current_price
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
        
        annual_return = (closes[-1] / closes[-252]) ** (1/1) - 1 if len(closes) >= 252 else 0
        volatility = closes[-252:].std() / closes[-252:].mean() if len(closes) >= 252 else 0.2
        
        long_term_monte_carlo = self._monte_carlo_simulation_long_term(
            filtered_predictions, filtered_confidences, current_price, annual_return, volatility
        )
        
        if long_term_monte_carlo:
            weighted_pred = weighted_pred * 0.8 + long_term_monte_carlo['mean'] * 0.2
            ensemble_std = long_term_monte_carlo['std']
        
        prediction_interval_width = ensemble_std / current_price if current_price > 0 else 0.15
        interval_adjustment = 1 - min(0.25, prediction_interval_width * 1.5)
        
        market_cap_confidence_adjustment = market_cap_params.get('confidence_multiplier', 1.0) if market_cap_params else 1.0
        
        final_confidence = min(88.0, max(40.0, np.mean(filtered_confidences) * (0.70 + agreement * 0.20 + interval_adjustment * 0.10) * market_cap_confidence_adjustment))
        
        predicted_price = weighted_pred
        predicted_change = (predicted_price - current_price) / current_price
        
        unique_factors = list(dict.fromkeys(factors))
        
        bearish_keywords = ['decline', 'declining', 'High P/E', 'Overvalued', 'Poor', 'Low', 'Weak', 'Thin', 'High debt', 'Low liquidity', 'Poor liquidity', 'Negative', 'contraction', 'decrease', 'deteriorating']
        bullish_keywords = ['growth', 'Strong', 'Exceptional', 'Excellent', 'Good', 'Undervalued', 'Reasonable', 'High ROE', 'Strong ROA', 'High margins', 'Good margins', 'positive', 'improving', 'increase']
        
        bearish_factors = [f for f in unique_factors if any(kw.lower() in f.lower() for kw in bearish_keywords)]
        bullish_factors = [f for f in unique_factors if any(kw.lower() in f.lower() for kw in bullish_keywords)]
        neutral_factors = [f for f in unique_factors if f not in bearish_factors and f not in bullish_factors]
        
        if predicted_change < -0.01:
            bearish_count = min(3, len(bearish_factors))
            bullish_count = min(2, len(bullish_factors))
            selected_factors = bearish_factors[:bearish_count] + bullish_factors[:bullish_count]
            if len(selected_factors) < 3 and neutral_factors:
                selected_factors.extend(neutral_factors[:3-len(selected_factors)])
        elif predicted_change > 0.01:
            bullish_count = min(3, len(bullish_factors))
            bearish_count = min(2, len(bearish_factors))
            selected_factors = bullish_factors[:bullish_count] + bearish_factors[:bearish_count]
            if len(selected_factors) < 3 and neutral_factors:
                selected_factors.extend(neutral_factors[:3-len(selected_factors)])
        else:
            selected_factors = unique_factors[:5]
        
        if not selected_factors:
            selected_factors = unique_factors[:5] if unique_factors else ['Multiple fundamental signals']
        
        return {
            'predicted_price': predicted_price,
            'change_percent': predicted_change * 100,
            'confidence': final_confidence,
            'key_factors': ', '.join(selected_factors[:5]) if selected_factors else 'Multiple fundamental signals'
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
        
        roe_display = roe * 100
        
        if roe > 0.20:
            score += 0.20
            factors.append(f"Exceptional ROE ({roe_display:.1f}%)")
            confidence += 5
        elif roe > 0.15:
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
        
        market_cap_params = self._get_market_cap_parameters(market_cap) if market_cap > 0 else None
        
        market_cap_adjustment = 1.0
        if market_cap_params:
            volatility_mult = market_cap_params.get('volatility_multiplier', 1.0)
            beta_adj = market_cap_params.get('beta_adjustment', 1.0)
            market_cap_adjustment = (volatility_mult + beta_adj) / 2
        
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
        
        if market_cap_params:
            confidence *= market_cap_params.get('confidence_multiplier', 1.0)
        
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
            
            base_growth = max(earnings_growth, revenue_growth, annual_return * 0.5, 0.02)
            
            roe_factor = roe if roe > 0 else 0.10
            sustainable_growth = min(roe_factor * 0.7, 0.15)
            
            growth_rate = min(base_growth, sustainable_growth)
            growth_rate = max(-0.10, min(0.20, growth_rate))
            
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
            
            discount_rate = max(0.08, min(0.18, wacc))
            
            terminal_growth = min(0.04, max(0.015, growth_rate * 0.4))
            if terminal_growth >= discount_rate:
                terminal_growth = discount_rate - 0.01
            
            projection_years = 10
            
            pv_fcf = 0
            current_fcf = fcf_per_share
            
            for year in range(1, projection_years + 1):
                decay_factor = 1 - (year / projection_years) * 0.6
                year_growth = growth_rate * decay_factor
                year_growth = max(terminal_growth, min(growth_rate, year_growth))
                
                if year == 1:
                    future_fcf = current_fcf * (1 + year_growth)
                else:
                    future_fcf = current_fcf * ((1 + year_growth) ** year)
                
                pv_fcf += future_fcf / ((1 + discount_rate) ** year)
                current_fcf = future_fcf
            
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
