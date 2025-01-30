from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class OptionsGreeksCalculator:
    """Calculate Black-Scholes Greeks for index options with Nifty-specific adjustments"""
    def __init__(self, risk_free_rate: float = 0.07):
        self.risk_free_rate = risk_free_rate
        self.strike_step = 50  # Nifty strike interval

    def calculate_greeks(self, spot: float, strike: float, 
                       expiry: str, iv: float, opt_type: str) -> Dict[str, float]:
        try:
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
                'theta': theta/365,  # Daily theta
                'vega': spot*sqrt_t*norm.pdf(d1)/100,
                'iv_impact': iv/20  # Custom metric for IV sensitivity
            }
        except Exception as e:
            logger.error(f"Greeks calculation error: {e}")
            return {}

    def _time_to_expiry(self, expiry: str) -> float:
        expiry_date = datetime.strptime(expiry, '%d%b%Y')
        return max((expiry_date - datetime.now()).total_seconds()/(365*24*3600), 1/365)

    def _expiry_greeks(self, spot: float, strike: float, opt_type: str) -> Dict[str, float]:
        """Handle expiration day Greeks"""
        intrinsic = max(spot - strike, 0) if opt_type == 'CE' else max(strike - spot, 0)
        return {
            'delta': 1.0 if intrinsic > 0 else 0.0,
            'gamma': 0.0,
            'theta': -intrinsic,  # Full loss of intrinsic value
            'vega': 0.0,
            'iv_impact': 0.0
        }

class IndexOptionsAnalyzer:
    def __init__(self):
        self.greeks_calculator = OptionsGreeksCalculator()

    def analyze_options(self, payload: Dict) -> Dict:
        try:
            # Access nested analysis data
            analysis_data = payload.get('analysis', {})
            
            # Validate payload structure
            required_keys = ['current_market', 'options', 'historical_data']
            if not all(key in analysis_data for key in required_keys):
                return {'error': 'Invalid payload structure - missing required top-level keys'}

            current_market = analysis_data['current_market']
            options_data = analysis_data['options']
            historical_data = analysis_data['historical_data']

            # Safely extract values with defaults
            current_price = current_market.get('index', {}).get('ltp', 0)
            vix = current_market.get('vix', {}).get('ltp', 0)
            futures_data = current_market.get('futures', {})

            # Process options with validation
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
                'market_conditions': self._analyze_market_conditions(historical_data),
                'strategy_ratings': self._calculate_strategy_ratings(options_chain, vix)
            }

        except KeyError as e:
            logger.error(f"Missing key in payload: {e}")
            return {'error': f'Missing required field: {e}'}
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {'error': str(e)}

    def _process_options(self, options: List[Dict], spot: float, 
                       vix: float, futures: Dict) -> List[Dict]:
        processed = []
        for opt in options:
            try:
                greeks = self.greeks_calculator.calculate_greeks(
                    spot=spot,
                    strike=opt.get('strike', 0),
                    expiry=opt.get('expiry', ''),
                    iv=vix/100,
                    opt_type=opt.get('optionType', 'CE')
                )

                processed.append({
                    'strike': opt.get('strike', 0),
                    'premium': opt.get('ltp', 0),
                    'expiry': opt.get('expiry', ''),
                    'type': opt.get('optionType', 'CE').upper(),
                    'greeks': greeks,
                    'liquidity_score': self._calculate_liquidity(
                        opt.get('openInterest', 0),
                        futures.get('openInterest', 0)
                    ),
                    'depth': self._process_depth(opt.get('depth', {})),
                    'timeframe_suitability': {
                        'scalping': greeks.get('gamma', 0) * 2,
                        'intraday': greeks.get('delta', 0) ** 2,
                        'swing': greeks.get('vega', 0) * 0.7
                    }
                })
            except Exception as e:
                logger.warning(f"Skipping option {opt.get('symbol', '')}: {e}")
        return sorted(processed, key=lambda x: abs(x['strike'] - spot))[:3]

    def _calculate_liquidity(self, option_oi: int, futures_oi: int) -> float:
        """Calculate liquidity score based on open interest"""
        if futures_oi == 0:
            return 0.5  # Default score if futures data is missing
        return min(option_oi / (futures_oi / 1000), 1.0)

    def _process_depth(self, depth: Dict) -> Dict:
        return {
            'best_bid': depth.get('buy', [{}])[0].get('price', 0),
            'best_ask': depth.get('sell', [{}])[0].get('price', 0),
            'spread': abs(
                depth.get('sell', [{}])[0].get('price', 0) - 
                depth.get('buy', [{}])[0].get('price', 0)
            )
        }

    def _analyze_market_conditions(self, historical_data: Dict) -> Dict:
        """Analyze historical data for market trends"""
        try:
            index_data = historical_data.get('index', [])
            if not index_data:
                return {'trend': 'neutral', 'volatility': 'low'}

            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(index_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Calculate moving averages
            df['MA5'] = df['price_data'].apply(lambda x: x['close']).rolling(5).mean()
            df['MA20'] = df['price_data'].apply(lambda x: x['close']).rolling(20).mean()

            # Determine trend
            last_row = df.iloc[-1]
            trend = 'bullish' if last_row['MA5'] > last_row['MA20'] else 'bearish'

            # Calculate volatility
            daily_returns = df['price_data'].apply(lambda x: x['close']).pct_change().dropna()
            volatility = 'high' if daily_returns.std() > 0.02 else 'low'

            return {
                'trend': trend,
                'volatility': volatility,
                'last_close': last_row['price_data']['close']
            }
        except Exception as e:
            logger.error(f"Market conditions analysis error: {e}")
            return {'trend': 'neutral', 'volatility': 'low'}

    def _calculate_strategy_ratings(self, options_chain: Dict, vix: float) -> Dict:
        """Calculate strategy ratings based on options chain and VIX"""
        try:
            scores = {
                'scalping': 0,
                'intraday': 0,
                'swing': 0
            }

            for opt_type in ['calls', 'puts']:
                for opt in options_chain[opt_type]:
                    scores['scalping'] += opt['greeks'].get('gamma', 0) * 2
                    scores['intraday'] += opt['greeks'].get('delta', 0) ** 2
                    scores['swing'] += opt['greeks'].get('vega', 0) * 0.7

            # Normalize scores
            total = sum(scores.values())
            if total > 0:
                for key in scores:
                    scores[key] /= total

            # Adjust for VIX
            vix_factor = vix / 20
            scores['scalping'] *= (1 - vix_factor)
            scores['intraday'] *= vix_factor
            scores['swing'] *= (1 + vix_factor)

            return scores
        except Exception as e:
            logger.error(f"Strategy ratings calculation error: {e}")
            return {'scalping': 0, 'intraday': 0, 'swing': 0}

class TradingStrategyEngine:
    """Generate executable trading strategies"""
    def generate_strategies(self, analysis: Dict) -> Dict:
        return {
            'scalping': self._scalping_strategy(analysis),
            'intraday': self._intraday_strategy(analysis),
            'swing': self._swing_strategy(analysis),
            'risk_management': self._risk_parameters(analysis),
            'market_conditions': analysis['market_conditions']
        }

    def _scalping_strategy(self, analysis: Dict) -> Dict:
        """High-frequency trading strategy (10-15 minute holds)"""
        best_options = self._filter_options(analysis, 'scalping')
        return {
            'strategy': 'Scalping',
            'holding_time': '10-15 minutes',
            'options': best_options,
            'execution': {
                'entry_type': 'LIMIT',
                'exit_type': 'LIMIT',
                'max_slippage': '0.05%',
                'order_size': f"{analysis['strategy_ratings']['scalping']['recommended_lots']} lots"
            }
        }

    def _intraday_strategy(self, analysis: Dict) -> Dict:
        """Intraday trading strategy (1-4 hour holds)"""
        best_options = self._filter_options(analysis, 'intraday')
        return {
            'strategy': 'Intraday',
            'holding_time': '1-4 hours',
            'options': best_options,
            'execution': {
                'entry_type': 'LIMIT with STOP-LOSS',
                'exit_type': 'TRAILING STOP',
                'max_slippage': '0.1%',
                'order_size': f"{analysis['strategy_ratings']['intraday']['recommended_lots']} lots"
            }
        }

    def _swing_strategy(self, analysis: Dict) -> Dict:
        """Swing trading strategy (1-3 day holds)"""
        best_options = self._filter_options(analysis, 'swing')
        return {
            'strategy': 'Swing',
            'holding_time': '1-3 days',
            'options': best_options,
            'execution': {
                'entry_type': 'LIMIT with OCO',
                'exit_type': 'TARGET & STOP-LOSS',
                'max_slippage': '0.2%',
                'order_size': f"{analysis['strategy_ratings']['swing']['recommended_lots']} lots"
            }
        }

    def _filter_options(self, analysis: Dict, timeframe: str) -> List[Dict]:
        """Select top options for given timeframe"""
        selected = []
        for opt_type in ['calls', 'puts']:
            sorted_options = sorted(
                analysis['options_chain'][opt_type],
                key=lambda x: -x['timeframe_suitability'][timeframe]
            )
            selected.append({
                'type': opt_type[:-1].upper(),
                'strike': sorted_options[0]['strike'],
                'expiry': sorted_options[0]['expiry'],
                'premium_range': sorted_options[0]['trading_zones'][timeframe]
            })
        return selected

    def _risk_parameters(self, analysis: Dict) -> Dict:
        """Volatility-adjusted risk parameters"""
        vix = analysis['vix']
        return {
            'max_capital_per_trade': f"{min(10 + (vix-15), 15)}%",
            'daily_loss_limit': f"{3 + (vix/5)}%",
            'position_hedging': 'Required for swing trades',
            'margin_utilization': 'Max 60% during high volatility'
        }

# Add root route
@app.route('/')
def home():
    return """
    <h1>Stocxer AI F&O Analysis API</h1>
    <p>Send POST requests to /analyze with your market data payload</p>
    <p>Example curl command:</p>
    <pre>
    curl -X POST -H "Content-Type: application/json" \\
         -d '{"analysis": {...}}' \\
         https://<url>/analyze
    </pre>
    """

# Add health check route
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "version": "1.0.0"})

# Analysis endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No payload provided"}), 400
            
        analyzer = IndexOptionsAnalyzer()
        analysis = analyzer.analyze_options(payload)
        strategy_engine = TradingStrategyEngine()
        return jsonify(strategy_engine.generate_strategies(analysis))
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)