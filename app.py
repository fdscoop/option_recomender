from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.stats import norm
from datetime import datetime
import logging
import re
import os
from typing import Dict, List, Any
import json

# For ARIMA forecasting
from statsmodels.tsa.arima.model import ARIMA

# For LSTM forecasting
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# For scaling data before LSTM training
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Setup Logging and Flask
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

###############################################################################
# PART 1: Options & Market Analysis Components
###############################################################################

class OptionsGreeksCalculator:
    """
    Computes option Greeks (delta, gamma, theta, vega, etc.) 
    given the underlying price, strike, expiry, implied volatility, and option type.
    """
    def __init__(self, risk_free_rate: float = None):
        self.risk_free_rate = risk_free_rate or float(os.getenv('RISK_FREE_RATE', 0.07))

    def calculate_greeks(self, spot: float, strike: float, expiry: str, iv: float, opt_type: str) -> Dict[str, float]:
        try:
            if spot <= 0 or strike <= 0 or iv <= 0:
                return {}
            t = self._time_to_expiry(expiry)
            if t <= 0:
                return self._expiry_greeks(spot, strike, opt_type)
            sqrt_t = np.sqrt(t)
            d1 = (np.log(spot / strike) + (self.risk_free_rate + 0.5 * iv ** 2) * t) / (iv * sqrt_t)
            d2 = d1 - iv * sqrt_t
            if opt_type.upper() == 'CE':
                delta = norm.cdf(d1)
                theta = (-(spot * norm.pdf(d1) * iv) / (2 * sqrt_t) -
                         self.risk_free_rate * strike * np.exp(-self.risk_free_rate * t) * norm.cdf(d2))
            else:
                delta = -norm.cdf(-d1)
                theta = (-(spot * norm.pdf(d1) * iv) / (2 * sqrt_t) +
                         self.risk_free_rate * strike * np.exp(-self.risk_free_rate * t) * norm.cdf(-d2))
            return {
                'delta': delta,
                'gamma': norm.pdf(d1) / (spot * iv * sqrt_t),
                'theta': theta / 365,
                'vega': spot * sqrt_t * norm.pdf(d1) / 100,
                'iv_impact': iv / 20
            }
        except Exception as e:
            logger.error(f"Greeks calculation error: {e}")
            return {}

    def _time_to_expiry(self, expiry: str) -> float:
        try:
            expiry_date = self._parse_exchange_expiry(expiry)
            now = datetime.now().astimezone()
            t = (expiry_date - now).total_seconds() / (365 * 24 * 3600)
            return max(t, 0)
        except Exception as e:
            logger.error(f"Time to expiry calculation error: {e}")
            return 0.0

    def _parse_exchange_expiry(self, expiry_str: str) -> datetime:
        try:
            clean_str = re.sub(r'[^a-zA-Z0-9]', '', expiry_str).upper()
            for fmt in ['%d%b%Y', '%Y%m%d']:
                try:
                    parsed = datetime.strptime(clean_str, fmt)
                    return parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
                except ValueError:
                    continue
            raise ValueError(f"Unsupported expiry format: {expiry_str}")
        except Exception as e:
            logger.error(f"Expiry parsing error for {expiry_str}: {e}")
            raise

    def _expiry_greeks(self, spot: float, strike: float, opt_type: str) -> Dict[str, float]:
        intrinsic = max(spot - strike, 0) if opt_type.upper() == 'CE' else max(strike - spot, 0)
        return {
            'delta': 1.0 if intrinsic > 0 else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'iv_impact': 0.0
        }

class IndexOptionsAnalyzer:
    """
    Extracts historical index, VIX, futures, and options data from the payload.
    It reorganizes the options data into exactly three calls and three puts,
    validating (and filling in) the expiry if missing.
    """
    def __init__(self):
        self.greeks_calculator = OptionsGreeksCalculator()

    def analyze_options(self, payload: Dict) -> Dict:
        try:
            analysis_data = payload.get('analysis', {})
            required_keys = ['current_market', 'historical_data']
            if not all(k in analysis_data for k in required_keys):
                return {'error': f"Missing required keys: {required_keys}"}
            current_market = analysis_data['current_market']
            historical_data = analysis_data['historical_data']
            market_required = ['index', 'vix', 'futures', 'options']
            if not all(k in current_market for k in market_required):
                return {'error': f"Current market missing keys: {market_required}"}
            index_data = current_market.get('index', {})
            current_price = index_data.get('ltp', 0)
            if current_price == 0:
                current_price = index_data.get('close', 0)
            vix = current_market.get('vix', {}).get('ltp', 0) / 100
            futures_data = current_market.get('futures', {})

            logger.info(f"Processing options with current price: {current_price}, VIX: {vix}")

            reorganized_options = self.reorganize_options_data(current_market, current_price)
            processed_calls = [self._process_contract(opt, current_price, vix, futures_data) 
                               for opt in reorganized_options.get("calls", [])]
            processed_puts = [self._process_contract(opt, current_price, vix, futures_data) 
                              for opt in reorganized_options.get("puts", [])]
            options_chain = {"calls": processed_calls, "puts": processed_puts}

            # Extract historical index prices from the payload.
            historical_index_prices = [
                entry["price_data"]["close"]
                for entry in historical_data.get("index", [])
                if "price_data" in entry and "close" in entry["price_data"]
            ]
            # Build the analysis result.
            result = {
                'current_price': current_price,
                'vix': vix * 100,
                'options_chain': options_chain,
                'market_conditions': self._analyze_market_conditions(historical_data, vix),
                'strategy_ratings': self._calculate_strategy_ratings(options_chain, vix),
                # Keep historical data internally for forecasting—but it will be removed from the final output.
                'historical_index_prices': historical_index_prices
            }
            return result
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def reorganize_options_data(self, current_market: Dict, spot: float) -> Dict[str, List[Dict]]:
        options_data = current_market.get('options', {})
        flattened = self._flatten_options(options_data)
        calls = flattened.get("calls", [])
        puts = flattened.get("puts", [])
        calls_sorted = sorted(calls, key=lambda opt: abs(opt.get('strikePrice', float('inf')) - spot))
        puts_sorted = sorted(puts, key=lambda opt: abs(opt.get('strikePrice', float('inf')) - spot))
        return {"calls": calls_sorted[:3], "puts": puts_sorted[:3]}

    def _flatten_options(self, options_data: Dict) -> Dict[str, List[Dict]]:
        flattened = {"calls": [], "puts": []}
        if "byExpiry" in options_data:
            for expiry, expiry_data in options_data["byExpiry"].items():
                calls = expiry_data.get("calls", {})
                for strike_key, contract in calls.items():
                    if not contract.get("expiry"):
                        contract["expiry"] = expiry
                    flattened["calls"].append(contract)
                puts = expiry_data.get("puts", {})
                for strike_key, contract in puts.items():
                    if not contract.get("expiry"):
                        contract["expiry"] = expiry
                    flattened["puts"].append(contract)
        else:
            flattened["calls"] = options_data.get("calls", [])
            flattened["puts"] = options_data.get("puts", [])
        return flattened

    def _process_contract(self, contract: Dict, spot: float, iv: float, futures: Dict) -> Dict:
        required_fields = ['ltp', 'strikePrice', 'expiry', 'optionType']
        if any(f not in contract for f in required_fields):
            trading_symbol = contract.get('tradingSymbol', '')
            if not trading_symbol:
                raise ValueError("Missing tradingSymbol to parse contract details")
            parsed_data = self._parse_trading_symbol(trading_symbol)
            contract = {**contract, **parsed_data}
        expiry_date = self.greeks_calculator._parse_exchange_expiry(str(contract['expiry']))
        greeks = self.greeks_calculator.calculate_greeks(
            spot=spot,
            strike=contract['strikePrice'],
            expiry=expiry_date.strftime('%d%b%Y').upper(),
            iv=iv,
            opt_type='CE' if contract['optionType'].upper() == 'CALL' else 'PE'
        )
        return {
            'symbol': contract.get('tradingSymbol', ''),
            'strike': contract['strikePrice'],
            'premium': contract['ltp'],
            'expiry': expiry_date.strftime('%d-%b-%Y').upper(),
            'type': contract['optionType'].upper(),
            'greeks': greeks,
            'liquidity_score': self._calculate_liquidity(contract, futures),
            'depth': self._process_depth(contract.get('depth', {})),
            'timeframe_suitability': {
                'scalping': greeks.get('gamma', 0) * 2,
                'intraday': greeks.get('delta', 0) ** 2,
                'swing': greeks.get('vega', 0) * 0.7
            }
        }

    def _parse_trading_symbol(self, symbol: str) -> Dict[str, Any]:
        pattern = r'^(?P<root>[A-Z]+)(?P<day>\d{2})(?P<mon>[A-Z]{3})(?P<year>\d{2,4})(?P<strike>\d+)(?P<opt_code>CE|PE)$'
        m = re.match(pattern, symbol)
        if not m:
            raise ValueError(f"Unable to parse trading symbol: {symbol}")
        day = m.group('day')
        mon = m.group('mon')
        year = m.group('year')
        if len(year) == 2:
            year = '20' + year
        expiry = f"{day}{mon}{year}"
        strike = float(m.group('strike'))
        opt_type = 'CALL' if m.group('opt_code') == 'CE' else 'PUT'
        return {
            'strikePrice': strike,
            'expiry': expiry,
            'optionType': opt_type
        }

    def _calculate_liquidity(self, option: Dict, futures: Dict) -> float:
        try:
            option_oi = option.get('opnInterest', 0)
            futures_oi = futures.get('opnInterest', 1)
            option_vol = option.get('tradeVolume', 0)
            futures_vol = futures.get('tradeVolume', 1)
            oi_ratio = option_oi / futures_oi if futures_oi > 0 else 0
            vol_ratio = option_vol / futures_vol if futures_vol > 0 else 0
            return min(0.4 * oi_ratio + 0.6 * vol_ratio, 1.0)
        except Exception:
            return 0.5

    def _process_depth(self, depth: Dict) -> Dict:
        try:
            bids = sorted([b.get('price', 0) for b in depth.get('buy', [])[:5]], reverse=True)
            asks = sorted([a.get('price', 0) for a in depth.get('sell', [])[:5]])
            return {
                'best_bid': bids[0] if bids else 0,
                'best_ask': asks[0] if asks else 0,
                'spread': (asks[0] - bids[0]) if bids and asks else 0
            }
        except Exception:
            return {'best_bid': 0, 'best_ask': 0, 'spread': 0}

    def _analyze_market_conditions(self, historical_data: Dict, vix: float) -> Dict:
        try:
            index_history = historical_data.get('index', [])
            if not index_history:
                return {'trend': 'neutral', 'volatility': 'low', 'vix': vix * 100}
            closes = [entry['price_data']['close'] for entry in index_history 
                      if 'price_data' in entry and 'close' in entry['price_data']]
            if len(closes) < 20:
                return {'trend': 'neutral', 'volatility': 'low', 'vix': vix * 100}
            ma5 = np.mean(closes[-5:])
            ma20 = np.mean(closes[-20:])
            daily_returns = np.diff(closes) / closes[:-1]
            volatility = 'high' if np.std(daily_returns) > 0.015 or vix > 0.15 else 'low'
            return {
                'trend': 'bullish' if ma5 > ma20 else 'bearish',
                'volatility': volatility,
                'vix': vix * 100,
                'last_close': closes[-1] if closes else 0
            }
        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return {'trend': 'neutral', 'volatility': 'low', 'vix': vix * 100}

    def _calculate_strategy_ratings(self, options_chain: Dict, vix: float) -> Dict:
        try:
            scores = {'scalping': 0.01, 'intraday': 0.01, 'swing': 0.01}
            for opt_type in ['calls', 'puts']:
                for opt in options_chain.get(opt_type, []):
                    scores['scalping'] += opt.get('greeks', {}).get('gamma', 0) * 2
                    scores['intraday'] += opt.get('greeks', {}).get('delta', 0) ** 2
                    scores['swing'] += opt.get('greeks', {}).get('vega', 0) * 0.7
            total = sum(scores.values()) or 1.0
            vix_factor = max(min(vix / 0.20, 2.0), 0.5)
            return {
                'scalping': max(round(scores['scalping'] / total * (2.0 - vix_factor), 2), 0),
                'intraday': max(round(scores['intraday'] / total * vix_factor, 2), 0),
                'swing': max(round(scores['swing'] / total * (1 + vix_factor), 2), 0)
            }
        except Exception as e:
            logger.error(f"Strategy ratings error: {str(e)}")
            return {'scalping': 0.34, 'intraday': 0.33, 'swing': 0.33}

###############################################################################
# PART 2: Forecasting Engine (ARIMA and LSTM with Scaling)
###############################################################################

class ForecastingEngine:
    def forecast_arima(self, index_prices: List[float]) -> float:
        try:
            model = ARIMA(index_prices, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]
            return float(forecast)
        except Exception as e:
            logger.error(f"ARIMA forecasting error: {e}")
            return float(index_prices[-1])

    def forecast_lstm(self, index_prices: List[float], lookback: int = 10) -> float:
        try:
            prices = np.array(index_prices).reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(prices)
            X, y = [], []
            for i in range(len(scaled_prices) - lookback):
                X.append(scaled_prices[i:i + lookback])
                y.append(scaled_prices[i + lookback])
            X = np.array(X)
            y = np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            model = Sequential()
            model.add(Input(shape=(lookback, 1)))
            model.add(LSTM(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=20, verbose=0)
            last_sequence = np.array(scaled_prices[-lookback:]).reshape((1, lookback, 1))
            forecast_scaled = model.predict(last_sequence, verbose=0)[0, 0]
            forecast = scaler.inverse_transform(np.array([[forecast_scaled]]))[0, 0]
            return float(forecast)
        except Exception as e:
            logger.error(f"LSTM forecasting error: {e}")
            return float(index_prices[-1])

    def forecast_market(self, index_prices: List[float]) -> Dict[str, Any]:
        forecast_arima = self.forecast_arima(index_prices)
        forecast_lstm = self.forecast_lstm(index_prices)
        forecast = (forecast_arima + forecast_lstm) / 2
        current_price = index_prices[-1]
        direction = "bullish" if forecast > current_price else "bearish"
        return {
            "forecast": float(forecast),
            "current_price": float(current_price),
            "direction": direction,
            "forecast_arima": float(forecast_arima),
            "forecast_lstm": float(forecast_lstm)
        }

###############################################################################
# PART 3: Trading Strategy Engine
###############################################################################

class TradingStrategyEngine:
    def generate_strategies(self, analysis: Dict) -> Dict:
        try:
            # Remove and downsample historical data to reduce memory footprint.
            historical_prices = analysis.pop("historical_index_prices", [])
            if len(historical_prices) > 1000:
                n = len(historical_prices) // 1000
                historical_prices = historical_prices[::n]
            forecast_info = {}
            if historical_prices and len(historical_prices) >= 10:
                forecasting_engine = ForecastingEngine()
                forecast_info = forecasting_engine.forecast_market(historical_prices)
            return {
                'scalping': self._base_strategy(analysis, 'scalping', '10-15 minutes'),
                'intraday': self._base_strategy(analysis, 'intraday', '1-4 hours'),
                'swing': self._base_strategy(analysis, 'swing', '1-3 days'),
                'risk_management': self._risk_parameters(analysis),
                'market_conditions': analysis.get('market_conditions', {}),
                'forecast': forecast_info
            }
        except Exception as e:
            logger.error(f"Strategy generation error: {str(e)}")
            return {'error': 'Strategy generation failed'}

    def _base_strategy(self, analysis: Dict, timeframe: str, holding: str) -> Dict:
        options = self._filter_options(analysis, timeframe)
        return {
            'strategy': timeframe.capitalize(),
            'holding_time': holding,
            'recommended_options': options,
            'execution': {
                'entry_type': 'LIMIT' if timeframe == 'scalping' else 'LIMIT with STOP',
                'exit_type': 'TRAILING STOP' if timeframe == 'intraday' else 'TARGET',
                'slippage': '0.05%' if timeframe == 'scalping' else '0.1%',
                'position_size': self._position_size(analysis, timeframe)
            }
        }

    def _filter_options(self, analysis: Dict, timeframe: str) -> List[Dict]:
        selected = []
        for opt_type in ['calls', 'puts']:
            options = analysis.get('options_chain', {}).get(opt_type, [])
            if options:
                sorted_options = sorted(options,
                                        key=lambda x: x.get('timeframe_suitability', {}).get(timeframe, 0),
                                        reverse=True)[:1]
                selected.extend(sorted_options)
        return selected

    def _position_size(self, analysis: Dict, timeframe: str) -> str:
        base = {'scalping': 5, 'intraday': 3, 'swing': 2}.get(timeframe, 2)
        vix = analysis.get('vix', 15)
        return f"{base * (1 + vix / 20):.1f} lots"

    def _risk_parameters(self, analysis: Dict) -> Dict:
        vix = analysis.get('vix', 15)
        return {
            'max_loss': f"{min(5 + vix / 2, 10):.1f}%",
            'stop_loss': '0.5%' if vix < 18 else '1%',
            'hedging': 'Required' if vix > 20 else 'Recommended'
        }

###############################################################################
# PART 4: Flask Endpoints
###############################################################################

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    try:
        payload = request.get_json()
        logger.info(f"Received webhook payload: {json.dumps(payload, indent=2)}")
        if not payload or 'success' not in payload or 'analysis' not in payload:
            return jsonify({
                "success": False,
                "error": "Invalid payload format - missing required root keys"
            }), 400
        analysis_data = payload['analysis']
        analyzer = IndexOptionsAnalyzer()
        analysis_result = analyzer.analyze_options({'analysis': analysis_data})
        if 'error' in analysis_result:
            return jsonify({
                "success": False,
                "error": analysis_result['error']
            }), 400
        strategy_engine = TradingStrategyEngine()
        strategies = strategy_engine.generate_strategies(analysis_result)
        # Do not include historical_index_prices in the output.
        return jsonify({
            "success": True,
            "analysis": analysis_result,
            "strategies": strategies
        })
    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route("/")
def home():
    return "Hello, Flask is running!"

###############################################################################
# PART 5: Main (Heroku Deployment)
###############################################################################

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
