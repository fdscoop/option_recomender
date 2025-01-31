from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import logging
import re
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class OptionsGreeksCalculator:
    def __init__(self, risk_free_rate: float = 0.07):
        self.risk_free_rate = risk_free_rate

    def calculate_greeks(self, spot: float, strike: float, 
                       expiry: str, iv: float, opt_type: str) -> Dict[str, float]:
        try:
            if spot <= 0 or strike <= 0 or iv <= 0:
                return {}
                
            t = self._time_to_expiry(expiry)
            if t <= 0:
                return self._expiry_greeks(spot, strike, opt_type)

            sqrt_t = np.sqrt(t)
            d1 = (np.log(spot/strike) + (self.risk_free_rate + 0.5*iv**2)*t) / (iv*sqrt_t)
            d2 = d1 - iv*sqrt_t

            if opt_type == 'CE':
                delta = norm.cdf(d1)
                theta = (-(spot*norm.pdf(d1)*iv)/(2*sqrt_t) 
                        - self.risk_free_rate*strike*np.exp(-self.risk_free_rate*t)*norm.cdf(d2))
            else:
                delta = -norm.cdf(-d1)
                theta = (-(spot*norm.pdf(d1)*iv)/(2*sqrt_t) 
                        + self.risk_free_rate*strike*np.exp(-self.risk_free_rate*t)*norm.cdf(-d2))

            return {
                'delta': delta,
                'gamma': norm.pdf(d1)/(spot*iv*sqrt_t),
                'theta': theta/365,
                'vega': spot*sqrt_t*norm.pdf(d1)/100,
                'iv_impact': iv/20
            }
        except Exception as e:
            logger.error(f"Greeks calculation error: {e}")
            return {}

    def _time_to_expiry(self, expiry: str) -> float:
        try:
            expiry_date = datetime.strptime(expiry, '%d%b%Y')
            return max((expiry_date - datetime.now()).total_seconds()/(365*24*3600), 1/365)
        except:
            return 1/365

    def _expiry_greeks(self, spot: float, strike: float, opt_type: str) -> Dict[str, float]:
        try:
            intrinsic = max(spot - strike, 0) if opt_type == 'CE' else max(strike - spot, 0)
            return {
                'delta': 1.0 if intrinsic > 0 else 0.0,
                'gamma': 0.0,
                'theta': -intrinsic,
                'vega': 0.0,
                'iv_impact': 0.0
            }
        except:
            return {}

class IndexOptionsAnalyzer:
    def __init__(self):
        self.greeks_calculator = OptionsGreeksCalculator()
        self.symbol_pattern = re.compile(
            r'^(NIFTY|BANKNIFTY|FINNIFTY|MIDCPNIFTY)(\d{2}[A-Z]{3}\d{2})(\d{5})(CE|PE)$', 
            re.IGNORECASE
        )

    def analyze_options(self, payload: Dict) -> Dict:
        try:
            analysis_data = payload.get('analysis', {})
            
            required_keys = ['current_market', 'historical_data']
            if not all(k in analysis_data for k in required_keys):
                return {'error': f"Missing required keys: {required_keys}"}

            current_market = analysis_data['current_market']
            historical_data = analysis_data.get('historical_data', {})
            
            market_required = ['index', 'options']
            if not all(k in current_market for k in market_required):
                return {'error': f"Missing market data: {market_required}"}

            index_data = current_market.get('index', {})
            current_price = index_data.get('ltp', 0)
            vix = current_market.get('vix', {}).get('ltp', 0)
            futures_data = current_market.get('futures', {})
            options_data = current_market.get('options', {})

            logger.info(f"Processing options with current price: {current_price}, VIX: {vix}")

            options_chain = {
                'calls': self._process_options(
                    options_data.get('calls', []), 
                    current_price,
                    vix,
                    futures_data
                ),
                'puts': self._process_options(
                    options_data.get('puts', []),
                    current_price,
                    vix,
                    futures_data
                )
            }

            return {
                'current_price': current_price,
                'vix': vix,
                'options_chain': options_chain,
                'market_conditions': self._analyze_market_conditions(historical_data, vix),
                'strategy_ratings': self._calculate_strategy_ratings(options_chain, vix)
            }
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def _process_options(self, options: List[Dict], spot: float, 
                       vix: float, futures: Dict) -> List[Dict]:
        processed = []
        for opt in options:
            try:
                symbol = opt.get('tradingSymbol', '')
                match = self.symbol_pattern.match(symbol)
                if not match:
                    logger.warning(f"Skipping option with invalid symbol: {symbol}")
                    continue
                
                expiry_str = match.group(2)
                strike = float(match.group(3))
                opt_type = match.group(4).upper()

                try:
                    expiry_date = datetime.strptime(expiry_str, '%d%b%y')
                except ValueError:
                    expiry_date = datetime.strptime(expiry_str, '%d%b%Y')
                    
                formatted_expiry = expiry_date.strftime('%d%b%Y').upper()

                greeks = self.greeks_calculator.calculate_greeks(
                    spot=spot,
                    strike=strike,
                    expiry=formatted_expiry,
                    iv=vix/100 if vix > 0 else 0.2,
                    opt_type=opt_type
                )

                processed.append({
                    'strike': strike,
                    'premium': opt.get('ltp', 0),
                    'expiry': formatted_expiry,
                    'type': opt_type,
                    'greeks': greeks or {},
                    'liquidity_score': self._calculate_liquidity(opt, futures),
                    'depth': self._process_depth(opt.get('depth', {})),
                    'timeframe_suitability': {
                        'scalping': greeks.get('gamma', 0) * 2,
                        'intraday': greeks.get('delta', 0) ** 2,
                        'swing': greeks.get('vega', 0) * 0.7
                    }
                })
            except Exception as e:
                logger.warning(f"Skipping option {symbol}: {str(e)}")
        
        # Filter for ATM options (±5%) and select top 3 nearest
        filtered = [p for p in processed if 0.95 < (p['strike'] / spot) < 1.05]
        return sorted(filtered, key=lambda x: abs(x['strike'] - spot))[:3]

    def _calculate_liquidity(self, option: Dict, futures: Dict) -> float:
        try:
            option_oi = option.get('opnInterest', 0)
            futures_oi = futures.get('opnInterest', 1)
            option_vol = option.get('tradeVolume', 0)
            futures_vol = futures.get('tradeVolume', 1)
            
            oi_ratio = option_oi / futures_oi if futures_oi > 0 else 0
            vol_ratio = option_vol / futures_vol if futures_vol > 0 else 0
            
            return min(0.4 * oi_ratio + 0.6 * vol_ratio, 1.0)
        except:
            return 0.5

    def _process_depth(self, depth: Dict) -> Dict:
        try:
            bids = sorted([b.get('price',0) for b in depth.get('buy',[])[:5]], reverse=True)
            asks = sorted([a.get('price',0) for a in depth.get('sell',[])[:5]])
            
            avg_bid = sum(bids[:3])/3 if bids else 0
            avg_ask = sum(asks[:3])/3 if asks else 0
            
            return {
                'best_bid': bids[0] if bids else 0,
                'best_ask': asks[0] if asks else 0,
                'avg_spread': avg_ask - avg_bid
            }
        except:
            return {'best_bid': 0, 'best_ask': 0, 'avg_spread': 0}

    def _analyze_market_conditions(self, historical_data: Dict, vix: float) -> Dict:
        try:
            index_history = historical_data.get('index', [])
            if not index_history:
                return {'trend': 'neutral', 'volatility': 'low', 'vix': vix}

            closes = []
            for entry in index_history:
                if 'price_data' in entry and 'close' in entry['price_data']:
                    closes.append(entry['price_data']['close'])
                
            if len(closes) < 20:
                return {'trend': 'neutral', 'volatility': 'low', 'vix': vix}

            ma5 = np.mean(closes[-5:])
            ma20 = np.mean(closes[-20:])
            daily_returns = np.diff(closes) / closes[:-1]
            
            volatility = 'high' if np.std(daily_returns) > 0.015 else 'low'
            if vix > 15:  # Override volatility based on VIX
                volatility = 'high'
            
            return {
                'trend': 'bullish' if ma5 > ma20 else 'bearish',
                'volatility': volatility,
                'vix': vix,
                'last_close': closes[-1] if closes else 0
            }
        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return {'trend': 'neutral', 'volatility': 'low', 'vix': vix}

    def _calculate_strategy_ratings(self, options_chain: Dict, vix: float) -> Dict:
        try:
            scores = {'scalping': 0.01, 'intraday': 0.01, 'swing': 0.01}
            for opt_type in ['calls', 'puts']:
                for opt in options_chain.get(opt_type, []):
                    scores['scalping'] += opt.get('greeks', {}).get('gamma', 0) * 2
                    scores['intraday'] += opt.get('greeks', {}).get('delta', 0) ** 2
                    scores['swing'] += opt.get('greeks', {}).get('vega', 0) * 0.7

            total = sum(scores.values()) or 1.0
            vix_factor = max(min(vix / 20, 2.0), 0.5)
            
            return {
                'scalping': round(scores['scalping']/total * (1 - vix_factor), 2),
                'intraday': round(scores['intraday']/total * vix_factor, 2),
                'swing': round(scores['swing']/total * (1 + vix_factor), 2)
            }
        except Exception as e:
            logger.error(f"Strategy ratings error: {str(e)}")
            return {'scalping': 0.34, 'intraday': 0.33, 'swing': 0.33}

class TradingStrategyEngine:
    def generate_strategies(self, analysis: Dict) -> Dict:
        try:
            return {
                'scalping': self._base_strategy(analysis, 'scalping', '10-15 minutes'),
                'intraday': self._base_strategy(analysis, 'intraday', '1-4 hours'),
                'swing': self._base_strategy(analysis, 'swing', '1-3 days'),
                'risk_management': self._risk_parameters(analysis),
                'market_conditions': analysis.get('market_conditions', {})
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
                    reverse=True
                )[:1]
                selected.extend(sorted_options)
        return selected

    def _position_size(self, analysis: Dict, timeframe: str) -> str:
        base = {'scalping': 5, 'intraday': 3, 'swing': 2}.get(timeframe, 2)
        vix = analysis.get('vix', 15)
        return f"{base * (1 + vix/20):.1f} lots"

    def _risk_parameters(self, analysis: Dict) -> Dict:
        vix = analysis.get('vix', 15)
        return {
            'max_loss': f"{min(5 + vix/2, 10):.1f}%",
            'stop_loss': '0.5%' if vix < 18 else '1%',
            'hedging': 'Required' if vix > 20 else 'Recommended'
        }

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        payload = request.get_json()
        
        if not payload or not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON format"}), 400
            
        if 'analysis' not in payload:
            return jsonify({"error": "Missing 'analysis' key"}), 400
            
        analysis_data = payload['analysis']
        required_keys = ['current_market', 'historical_data']
        if not all(k in analysis_data for k in required_keys):
            return jsonify({"error": f"Missing analysis keys: {required_keys}"}), 400

        analyzer = IndexOptionsAnalyzer()
        analysis_result = analyzer.analyze_options(payload)
        
        if 'error' in analysis_result:
            return jsonify(analysis_result), 400

        strategy_engine = TradingStrategyEngine()
        return jsonify(strategy_engine.generate_strategies(analysis_result))
        
    except Exception as e:
        logger.error(f"API processing error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def home():
    return "Options Analysis API - POST /analyze with market data"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)