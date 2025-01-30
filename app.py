from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
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
    """Analyze Nifty options chain with zero volume handling"""
    def __init__(self):
        self.greeks_calculator = OptionsGreeksCalculator()
        self.nifty_lot_size = 75  # Standard Nifty lot size

    def analyze_options(self, payload: Dict) -> Dict:
        try:
            current_price = payload['current_market']['index']['ltp']
            vix = payload['current_market']['vix']['ltp']
            futures_data = payload['current_market']['futures']
            
            # Process options chain
            options_chain = {
                'calls': self._process_options(payload['options']['calls'], 'CE', current_price, vix, futures_data),
                'puts': self._process_options(payload['options']['puts'], 'PE', current_price, vix, futures_data)
            }

            return {
                'current_price': current_price,
                'vix': vix,
                'options_chain': options_chain,
                'strategy_ratings': self._calculate_strategy_ratings(options_chain, vix),
                'market_conditions': self._analyze_market_conditions(payload)
            }
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {}

    def _process_options(self, options: List[Dict], opt_type: str,
                       spot: float, vix: float, futures: Dict) -> List[Dict]:
        analyzed = []
        for opt in options:
            try:
                greeks = self.greeks_calculator.calculate_greeks(
                    spot, opt['strikePrice'], opt['expiry'], vix/100, opt_type
                )
                analyzed.append({
                    'strike': opt['strikePrice'],
                    'premium': opt['ltp'],
                    'expiry': opt['expiry'],
                    'oi': opt['openInterest'],
                    'greeks': greeks,
                    'liquidity_score': self._calculate_liquidity(opt, futures),
                    'trading_zones': self._calculate_trading_zones(opt, greeks, vix),
                    'timeframe_suitability': self._timeframe_suitability(greeks, vix)
                })
            except Exception as e:
                logger.warning(f"Skipping option {opt.get('symbolToken')}: {e}")
        return sorted(analyzed, key=lambda x: abs(x['strike'] - spot))[:3]  # Nearest 3 strikes

    def _calculate_liquidity(self, option: Dict, futures: Dict) -> float:
        """Calculate liquidity score based on OI and futures liquidity"""
        futures_oi = futures.get('openInterest', 0)
        option_oi = option.get('openInterest', 0)
        if futures_oi > 0:
            return min(option_oi / (futures_oi / 1000), 1.0)
        return 0.5  # Default score when futures data missing

    def _calculate_trading_zones(self, option: Dict, greeks: Dict, vix: float) -> Dict:
        """Calculate entry/exit points based on volatility and Greeks"""
        iv_factor = vix / 15  # Normalize around 15 VIX
        premium = option['ltp']
        return {
            'scalping': {
                'entry': premium * (1 - 0.002*iv_factor),
                'target': premium * (1 + 0.003*iv_factor),
                'stoploss': premium * (1 - 0.005*iv_factor)
            },
            'intraday': {
                'entry': premium * (1 - 0.005*iv_factor),
                'target': premium * (1 + 0.007*iv_factor),
                'stoploss': premium * (1 - 0.01*iv_factor)
            },
            'swing': {
                'entry': premium * (1 - 0.01*iv_factor),
                'target': premium * (1 + 0.015*iv_factor),
                'stoploss': premium * (1 - 0.03*iv_factor)
            }
        }

    def _timeframe_suitability(self, greeks: Dict, vix: float) -> Dict:
        """Score option suitability for different trading timeframes"""
        return {
            'scalping': greeks['gamma'] * 2 - abs(greeks['theta']) * 0.5,
            'intraday': greeks['delta']**2 + greeks['vega'] * (vix/20),
            'swing': greeks['vega'] * 0.7 - greeks['theta'] * 0.3
        }

    def _calculate_strategy_ratings(self, options: Dict, vix: float) -> Dict:
        """Generate strategy recommendations based on market conditions"""
        return {
            'scalping': self._score_strategy(options, 'scalping', vix),
            'intraday': self._score_strategy(options, 'intraday', vix),
            'swing': self._score_strategy(options, 'swing', vix)
        }

    def _score_strategy(self, options: Dict, timeframe: str, vix: float) -> Dict:
        scores = []
        for opt_type in ['calls', 'puts']:
            scores.extend([opt['timeframe_suitability'][timeframe] for opt in options[opt_type]])
        return {
            'confidence': min(max(np.mean(scores)*2, 0), 1),
            'recommended_lots': int((2 if vix < 18 else 1) * (1 if timeframe == 'scalping' else 0.5))
        }

    def _analyze_market_conditions(self, payload: Dict) -> Dict:
        """Determine overall market trend and sentiment"""
        hist_data = pd.DataFrame(payload['historical_data']['index'])
        hist_data['date'] = pd.to_datetime(hist_data['timestamp'])
        hist_data.set_index('date', inplace=True)
        
        return {
            'trend': self._detect_trend(hist_data),
            'volatility_regime': 'High' if payload['current_market']['vix']['ltp'] > 18 else 'Normal',
            'option_skew': self._calculate_skew(payload['options'])
        }

    def _detect_trend(self, data: pd.DataFrame) -> str:
        """Simple moving average trend detection"""
        if len(data) < 5: return 'Neutral'
        data['MA5'] = data['close'].rolling(5).mean()
        data['MA20'] = data['close'].rolling(20).mean()
        last = data.iloc[-1]
        return 'Bullish' if last['MA5'] > last['MA20'] else 'Bearish'

    def _calculate_skew(self, options: Dict) -> float:
        """Calculate put/call skew"""
        call_oi = sum(c['openInterest'] for c in options['calls'])
        put_oi = sum(p['openInterest'] for p in options['puts'])
        return put_oi / call_oi if call_oi > 0 else 0

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
         -d '{"your": "payload"}' \\
         https://<url>/analyze
    </pre>
    """

# Add health check route
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "version": "1.0.0"})

# Your existing analysis routes
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