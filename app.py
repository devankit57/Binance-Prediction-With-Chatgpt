import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
import json
import pytz
from pymongo import MongoClient
from bson.objectid import ObjectId
from openai import OpenAI

class MongoDBTradingUpdater:
    def __init__(self, api_key: str = "", 
                 api_secret: str = "",
                 openai_api_key: str = "",
                 mongo_uri: str = "mongodb://localhost:27017/",
                 db_name: str = "mttrader",
                 collection_name: str = "prediction"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://fapi.binance.com"
        self.symbol = "BTCUSDT"
        
        # MongoDB setup
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.collection = self.db[collection_name]
        
        # OpenAI setup
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Target document ID
        self.target_doc_id = ObjectId("68c4fe6a7855b7f504c54213")
        
        # Setup timezones
        self.utc = pytz.UTC
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Store historical data for analysis
        self.funding_history = []
        self.oi_history = []
        
        # Microstructure tracking
        self.trade_history = deque(maxlen=500)
        self.cvd_history = []
        self.delta_history = deque(maxlen=100)
        
        # Trading analysis system prompt
        self.system_prompt = """You are a trading analysis bot.
Your role:
Take the provided trading data package and analyze it strictly using the fixed multi-step system.
Return ONLY a JSON object. No text, no explanation outside JSON.
Input Data Package Should Include:
Price & spread
Candlesticks (OHLC + volume + buy/sell % per candle)
Orderbook snapshot (walls, bid/ask dominance)
Funding rate and trend
Open Interest
VWAP & price distance from VWAP
Multi-timeframe structure (15m, 30m, 1h, 4h)
Technical indicators (RSI, MACD, ATR, Bollinger position, momentum)
Liquidation clusters and dominant side
Correlations with other assets
Magnet levels & breakout probability
Aggressive orderflow / delta / CVD
Volume surge and volume imbalance
Key insights / signal confluence
JSON schema:
{
  "direction": "LONG" | "SHORT" | "HOLD",
  "entry": <number>,
  "tp1": <number or null>,
  "tp2": <number or null>,
  "tp3": <number or null>,
  "sl": <number or null>,
  "dynamic_sl": {
    "after_tp1": <number or null>,
    "after_tp2": <number or null>
  },
  "net_profit_est": <number>,
  "net_loss_est": <number>,
  "confidence": <0–1>,
  "reasoning": [
    "<bullet point>",
    "<bullet point>",
    "<bullet point>"
  ]
}
Rules:
Margin / Leverage & P/L estimates
Always use $100 margin × 5x leverage.
TP/SL distances are absolute points.
SL moves only after TP1 or TP2 is hit:
BUY: SL moves up 10 points after TP1/TP2
SELL: SL moves down 10 points after TP1/TP2
TP / SL logic
TP1: ±165 points from entry (65% of position size)
TP2: ±195 points from entry (75% of position size)
TP3: ±260 points from entry (full exit)
SL adjusts dynamically after TP1/TP2. TP3 is final exit.
Hard constraint on SL vs TP1:
- For LONG: (TP1 - SL) must be at least 250 points. If your initial SL is closer, push SL down so that TP1 - SL = 250.
- For SHORT: (SL - TP1) must be at least 250 points. If your initial SL is closer, push SL up so that SL - TP1 = 250.
Direction / HOLD
If signal is unclear or mixed, output "HOLD" with null TP/SL and zero net profit/loss.
Confidence score
0.6–0.75 = medium, 0.75+ = strong, <0.6 = weak
Reasoning
Must be concise, 3–5 bullet points summarizing key factors (structure, momentum, orderflow, liquidation, VWAP, correlations).
Output
Strictly valid JSON. No comments, no text outside JSON.
TP/SL Example (LONG):
Entry = 120,607.3
TP1 = 120,772.3 → SL moves to 120,762.3
TP2 = 120,802.3 → SL moves to 120,792.3
TP3 = 120,867.3 → final exit
TP/SL Example (SHORT):
Entry = 120,607.3
TP1 = 120,442.3 → SL moves to 120,452.3
TP2 = 120,412.3 → SL moves to 120,422.3
TP3 = 120,347.3 → final exit
Now wait for input data."""
        
    def get_headers(self):
        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        return headers
    
    def get_ist_time(self, utc_timestamp=None):
        """Get current IST time or convert UTC timestamp to IST"""
        if utc_timestamp:
            if isinstance(utc_timestamp, int):
                utc_dt = datetime.fromtimestamp(utc_timestamp / 1000, tz=self.utc)
            else:
                utc_dt = utc_timestamp.replace(tzinfo=self.utc) if utc_timestamp.tzinfo is None else utc_timestamp
            return utc_dt.astimezone(self.ist)
        else:
            return datetime.now(self.ist)
    
    def create_ist_timestamp(self):
        """Create a timestamp with IST timezone for MongoDB"""
        return self.get_ist_time()
    
    def format_ist_for_mongodb(self, dt=None):
        """Format IST datetime for MongoDB storage"""
        if dt is None:
            dt = self.get_ist_time()
        return dt
    
    def serialize_datetime(self, obj):
        """Custom JSON serializer for datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def safe_float_format(self, value, default=0.0):
        """Safely format float values, handling None"""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def validate_analysis(self, analysis: Dict) -> bool:
        """Validate that analysis contains all required fields"""
        if not analysis or not isinstance(analysis, dict):
            print("⚠️  Warning: Analysis is not a valid dictionary")
            return False
        
        required_fields = ['direction', 'entry', 'confidence']
        for field in required_fields:
            if field not in analysis:
                print(f"⚠️  Warning: Missing required field: {field}")
                return False
        
        # Validate direction is valid
        valid_directions = ['LONG', 'SHORT', 'HOLD']
        if analysis['direction'] not in valid_directions:
            print(f"⚠️  Warning: Invalid direction: {analysis['direction']}")
            return False
        
        return True
    
    def wait_for_next_15min_candle(self):
        now_ist = self.get_ist_time()
        current_minute = now_ist.minute
        current_second = now_ist.second
        
        # If we're exactly at the start of a 15-minute candle, return immediately
        if current_second == 0 and current_minute % 15 == 0:
            return now_ist
        
        minutes_to_wait = 15 - (current_minute % 15)
        wait_seconds = (minutes_to_wait * 60) - current_second
        
        print(f"⏰ Waiting {wait_seconds} seconds for next 15-minute candle...")
        time.sleep(wait_seconds)
        return self.get_ist_time()
    
    def get_higher_timeframe_structure(self) -> Dict:
        structure_data = {}
        timeframes = ['15m', '30m', '1h', '4h']
        
        for timeframe in timeframes:
            limit = 50
            url = f"{self.base_url}/fapi/v1/klines"
            params = {'symbol': self.symbol, 'interval': timeframe, 'limit': limit}
            
            try:
                response = requests.get(url, params=params, headers=self.get_headers())
                response.raise_for_status()
                data = response.json()
                
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                
                ema_20 = df['close'].ewm(span=20).mean()
                ema_50 = df['close'].ewm(span=50).mean()
                current_price = df['close'].iloc[-1]
                
                if current_price > ema_20.iloc[-1] > ema_50.iloc[-1]:
                    trend = "BULLISH"
                elif current_price < ema_20.iloc[-1] < ema_50.iloc[-1]:
                    trend = "BEARISH"
                else:
                    trend = "SIDEWAYS"
                
                momentum_pct = ((current_price - df['close'].iloc[-10]) / df['close'].iloc[-10]) * 100 if len(df) >= 10 else 0
                
                structure_data[timeframe] = {
                    'trend': trend,
                    'ema_20': float(ema_20.iloc[-1]),
                    'ema_50': float(ema_50.iloc[-1]),
                    'current_price': float(current_price),
                    'momentum_10_candles_pct': float(momentum_pct),
                    'price_above_ema20': bool(current_price > ema_20.iloc[-1]),
                    'price_above_ema50': bool(current_price > ema_50.iloc[-1])
                }
                
            except Exception as e:
                print(f"❌ Error fetching {timeframe} data: {e}")
                structure_data[timeframe] = {
                    'trend': 'ERROR',
                    'error': str(e)
                }
        
        trends = [structure_data[tf]['trend'] for tf in timeframes if structure_data[tf]['trend'] != 'ERROR']
        
        bullish_count = trends.count('BULLISH')
        bearish_count = trends.count('BEARISH')
        sideways_count = trends.count('SIDEWAYS')
        
        if bullish_count >= 3:
            overall_confluence = "STRONG_BULLISH"
        elif bearish_count >= 3:
            overall_confluence = "STRONG_BEARISH"
        elif bullish_count > bearish_count:
            overall_confluence = "MODERATE_BULLISH"
        elif bearish_count > bullish_count:
            overall_confluence = "MODERATE_BEARISH"
        else:
            overall_confluence = "MIXED_SIGNALS"
        
        max_agreement = max(bullish_count, bearish_count, sideways_count)
        confluence_strength = (max_agreement / len(trends)) * 100 if trends else 0
        
        structure_data['confluence'] = {
            'overall_bias': overall_confluence,
            'strength_percentage': float(confluence_strength),
            'bullish_timeframes': bullish_count,
            'bearish_timeframes': bearish_count,
            'sideways_timeframes': sideways_count,
            'trend_agreement': max_agreement >= 3
        }
        
        return structure_data
    
    def get_precise_15min_ohlcv_enhanced(self, interval: str = '15m', limit: int = 50) -> Dict:
        url = f"{self.base_url}/fapi/v1/klines"
        params = {'symbol': self.symbol, 'interval': interval, 'limit': limit + 10}
        
        try:
            response = requests.get(url, params=params, headers=self.get_headers())
            response.raise_for_status()
            data = response.json()
            
            candles = []
            for candle in data:
                utc_timestamp = pd.to_datetime(int(candle[0]), unit='ms', utc=True)
                ist_timestamp = utc_timestamp.tz_convert(self.ist)
                
                if ist_timestamp.second == 0 and ist_timestamp.minute % 15 == 0:
                    candles.append({
                        'timestamp': ist_timestamp.isoformat(),
                        'time': ist_timestamp.strftime('%H:%M:%S'),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5]),
                        'taker_buy_volume': float(candle[9])
                    })
            
            candles = candles[-limit:] if len(candles) >= limit else candles
            df = pd.DataFrame(candles)
            
            if df.empty:
                return {
                    'raw_data': [],
                    'dataframe': pd.DataFrame(),
                    'vwap_analysis': {},
                    'summary': {'current_price': 0, 'price_change_24h': 0, 'price_change_pct': 0}
                }
            
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            buy_pressure = (df['taker_buy_volume'] / df['volume'] * 100).tail(5).tolist()
            volume_imbalance = ((df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])) / df['volume'] * 100).tail(5).tolist()
            
            vwap_analysis = {
                'vwap_last_5': [float(v) for v in vwap.tail(5).tolist()],
                'price_distance_from_vwap': [float((close - vwap_val) / vwap_val * 100) for close, vwap_val in zip(df['close'].tail(5), vwap.tail(5))],
                'buy_pressure_last_5': [float(bp) for bp in buy_pressure],
                'volume_imbalance_last_5': [float(vi) for vi in volume_imbalance]
            }
            
            return {
                'raw_data': candles,
                'dataframe': df,
                'vwap_analysis': vwap_analysis,
                'summary': {
                    'current_price': float(df.iloc[-1]['close']) if len(candles) > 0 else 0,
                    'price_change_24h': float(df.iloc[-1]['close'] - df.iloc[0]['open']) if len(candles) > 0 else 0,
                    'price_change_pct': float(((df.iloc[-1]['close'] - df.iloc[0]['open']) / df.iloc[0]['open']) * 100) if len(candles) > 0 else 0
                }
            }
            
        except Exception as e:
            print(f"❌ Error fetching OHLCV data: {e}")
            return {
                'raw_data': [],
                'dataframe': pd.DataFrame(),
                'vwap_analysis': {},
                'summary': {'current_price': 0, 'price_change_24h': 0, 'price_change_pct': 0},
                'error': str(e)
            }
    
    def get_order_book(self) -> Dict:
        url = f"{self.base_url}/fapi/v1/depth"
        params = {'symbol': self.symbol, 'limit': 20}
        
        try:
            response = requests.get(url, params=params, headers=self.get_headers())
            response.raise_for_status()
            data = response.json()
            
            bids = [[float(bid[0]), float(bid[1])] for bid in data['bids']]
            asks = [[float(ask[0]), float(ask[1])] for ask in data['asks']]
            
            best_bid = bids[0] if bids else [0, 0]
            best_ask = asks[0] if asks else [0, 0]
            spread = best_ask[0] - best_bid[0] if bids and asks else 0
            
            bid_liquidity = sum([bid[1] for bid in bids])
            ask_liquidity = sum([ask[1] for ask in asks])
            total_liquidity = bid_liquidity + ask_liquidity
            
            return {
                'best_bid': best_bid[0],
                'best_ask': best_ask[0],
                'spread': spread,
                'bid_dominance': bid_liquidity / total_liquidity if total_liquidity > 0 else 0.5
            }
            
        except Exception as e:
            print(f"❌ Error fetching order book: {e}")
            return {
                'best_bid': 0,
                'best_ask': 0,
                'spread': 0,
                'bid_dominance': 0.5,
                'error': str(e)
            }
    
    def get_funding_rate(self) -> Dict:
        url = f"{self.base_url}/fapi/v1/fundingRate"
        params = {'symbol': self.symbol, 'limit': 10}
        
        try:
            response = requests.get(url, params=params, headers=self.get_headers())
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return {'current_rate': 0, 'trend': 'UNKNOWN'}
            
            rates = [float(rate['fundingRate']) * 100 for rate in data[:5]]
            current_rate = rates[0]
            
            trend = "STABLE"
            if len(rates) >= 3:
                if rates[0] > rates[2]:
                    trend = "INCREASING"
                elif rates[0] < rates[2]:
                    trend = "DECREASING"
            
            return {
                'current_rate': current_rate,
                'trend': trend,
                'last_5_rates': rates
            }
            
        except Exception as e:
            print(f"❌ Error fetching funding rate: {e}")
            return {
                'current_rate': 0,
                'trend': 'ERROR',
                'error': str(e)
            }
    
    def get_open_interest(self) -> Dict:
        url = f"{self.base_url}/fapi/v1/openInterest"
        params = {'symbol': self.symbol}
        
        try:
            response = requests.get(url, params=params, headers=self.get_headers())
            response.raise_for_status()
            data = response.json()
            
            if 'openInterest' not in data:
                return {'open_interest': 0}
            
            return {
                'open_interest': float(data['openInterest'])
            }
            
        except Exception as e:
            print(f"❌ Error fetching open interest: {e}")
            return {
                'open_interest': 0,
                'error': str(e)
            }
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        if len(df) == 0:
            return {}
        
        indicators = {}
        
        try:
            # RSI
            def calculate_rsi(prices, period=14):
                if len(prices) < period + 1:
                    return []
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs)))
                return [float(r) for r in rsi.tail(5).tolist() if not pd.isna(r)]
            
            indicators['rsi_14'] = calculate_rsi(df['close'])
            
            # MACD
            if len(df) >= 26:
                ema_12 = df['close'].ewm(span=12).mean()
                ema_26 = df['close'].ewm(span=26).mean()
                macd_line = ema_12 - ema_26
                indicators['macd'] = [float(m) for m in macd_line.tail(5).tolist()]
            
            # ATR
            if len(df) >= 14:
                high = df['high']
                low = df['low']
                close_prev = df['close'].shift(1)
                tr1 = high - low
                tr2 = abs(high - close_prev)
                tr3 = abs(low - close_prev)
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean()
                indicators['atr'] = [float(a) for a in atr.tail(5).tolist() if not pd.isna(a)]
            
            return indicators
            
        except Exception as e:
            print(f"❌ Error calculating technical indicators: {e}")
            return {'error': str(e)}
    
    def get_complete_trading_data(self) -> Dict:
        try:
            candle_time = self.wait_for_next_15min_candle()
            print(f"✅ Candle time: {candle_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            print("📊 Fetching higher timeframe structure...")
            htf_structure = self.get_higher_timeframe_structure()
            
            print("🕯️ Fetching OHLCV data...")
            ohlcv_data = self.get_precise_15min_ohlcv_enhanced('15m', 50)
            
            print("📈 Fetching order book...")
            order_book = self.get_order_book()
            
            print("💰 Fetching funding rate...")
            funding_rate = self.get_funding_rate()
            
            print("📊 Fetching open interest...")
            open_interest = self.get_open_interest()
            
            print("📉 Calculating technical indicators...")
            indicators = self.calculate_technical_indicators(ohlcv_data['dataframe'])
            
            # Build comprehensive data package with string timestamps for JSON
            data_package = {
                'timestamp': candle_time.isoformat(),
                'timestamp_str': candle_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'symbol': self.symbol,
                'current_price': ohlcv_data['summary']['current_price'],
                'spread': order_book['spread'],
                'candlesticks': {
                    'recent_5': [
                        {
                            'time': c['time'],
                            'open': c['open'],
                            'high': c['high'],
                            'low': c['low'],
                            'close': c['close'],
                            'volume': c['volume'],
                            'buy_sell_pct': (c['taker_buy_volume'] / c['volume'] * 100) if c['volume'] > 0 else 50
                        }
                        for c in ohlcv_data['raw_data'][-5:]
                    ]
                },
                'orderbook': {
                    'best_bid': order_book['best_bid'],
                    'best_ask': order_book['best_ask'],
                    'bid_dominance': order_book['bid_dominance']
                },
                'funding': {
                    'current_rate': funding_rate['current_rate'],
                    'trend': funding_rate['trend']
                },
                'open_interest': open_interest['open_interest'],
                'vwap': {
                    'current_vwap': ohlcv_data['vwap_analysis']['vwap_last_5'][-1] if ohlcv_data['vwap_analysis'].get('vwap_last_5') else 0,
                    'price_distance_pct': ohlcv_data['vwap_analysis']['price_distance_from_vwap'][-1] if ohlcv_data['vwap_analysis'].get('price_distance_from_vwap') else 0
                },
                'multi_timeframe': htf_structure,
                'technical_indicators': indicators,
                'volume_analysis': {
                    'volume_imbalance': ohlcv_data['vwap_analysis']['volume_imbalance_last_5'][-1] if ohlcv_data['vwap_analysis'].get('volume_imbalance_last_5') else 0
                }
            }
            
            print(f"✅ Data collection complete. Current price: ${data_package['current_price']:.2f}")
            return data_package
            
        except Exception as e:
            print(f"❌ Error in get_complete_trading_data: {e}")
            return {'error': str(e)}
    
    def get_gpt_analysis(self, data_package: Dict) -> Dict:
        """Send data to ChatGPT for analysis"""
        try:
            # Convert data package to JSON string with custom serializer
            data_str = json.dumps(data_package, indent=2, default=self.serialize_datetime)
            
            print("🤖 Sending data to ChatGPT for analysis...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Here is the trading data:\n\n{data_str}"}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Extract JSON from response
            response_text = response.choices[0].message.content
            
            # Parse JSON
            analysis = json.loads(response_text)
            print("✅ GPT analysis received successfully")
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing GPT response as JSON: {e}")
            print(f"Raw response: {response.choices[0].message.content if 'response' in locals() else 'N/A'}")
            return {'error': f'JSON parsing error: {str(e)}'}
        except Exception as e:
            print(f"❌ Error in GPT analysis: {e}")
            return {'error': str(e)}
    
    def update_mongodb_prediction(self, analysis: Dict, current_price: float):
        """Update the specific document in MongoDB with IST timestamps"""
        try:
            if not analysis or 'error' in analysis:
                print(f"❌ Error in analysis: {analysis.get('error', 'Unknown error') if analysis else 'No analysis data'}")
                return False
            
            # Validate analysis
            if not self.validate_analysis(analysis):
                print("❌ Invalid analysis data, skipping update")
                return False
            
            # Get current time in IST (timezone-aware)
            ist_now = self.format_ist_for_mongodb()
            
            # Read base values
            direction = analysis.get('direction', 'HOLD')
            entry = self.safe_float_format(analysis.get('entry'))
            tp1 = self.safe_float_format(analysis.get('tp1'))
            tp2 = self.safe_float_format(analysis.get('tp2'))
            tp3 = self.safe_float_format(analysis.get('tp3'))
            sl = self.safe_float_format(analysis.get('sl'))
            
            # ──────────────────────────────────────────────
            # HARD RULE: SL must be at least 250 points away from TP1
            # For LONG:  TP1 - SL >= 250  → if violated, set SL = TP1 - 250
            # For SHORT: SL  - TP1 >= 250 → if violated, set SL = TP1 + 250
            # ──────────────────────────────────────────────
            MIN_GAP = 250.0
            if tp1 != 0 and sl != 0 and direction in ("LONG", "SHORT"):
                original_sl = sl
                if direction == "LONG":
                    if (tp1 - sl) < MIN_GAP:
                        sl = tp1 - MIN_GAP
                elif direction == "SHORT":
                    if (sl - tp1) < MIN_GAP:
                        sl = tp1 + MIN_GAP

                if sl != original_sl:
                    print(
                        f"🔧 Adjusted SL from {original_sl:.2f} to {sl:.2f} "
                        f"to enforce ≥ {MIN_GAP} points distance from TP1 "
                        f"for {direction} setup."
                    )
            # ──────────────────────────────────────────────
            
            # Handle dynamic_sl safely
            dynamic_sl = analysis.get('dynamic_sl', {})
            if dynamic_sl:
                dynamic_sl = {
                    'after_tp1': self.safe_float_format(dynamic_sl.get('after_tp1')),
                    'after_tp2': self.safe_float_format(dynamic_sl.get('after_tp2'))
                }
            
            # Prepare update data with IST timestamps
            update_data = {
                'direction': direction,
                'entry': entry,
                'tp': tp1,  # Main TP field
                'sl': sl,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'dynamic_sl': dynamic_sl,
                'net_profit_est': self.safe_float_format(analysis.get('net_profit_est')),
                'net_loss_est': self.safe_float_format(analysis.get('net_loss_est')),
                'confidence': self.safe_float_format(analysis.get('confidence'), 0.0),
                'reasoning': analysis.get('reasoning', []),
                'updatedAt': ist_now,
                'updatedAtIST': ist_now.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'lastAnalysisTime': ist_now,
                'currentPrice': current_price
            }
            
            result = self.collection.update_one(
                {'_id': self.target_doc_id},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                print(f"✅ Successfully updated document at {ist_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"📊 Direction: {direction}")
                print(f"💰 Entry: ${entry:.2f}")
                print(f"🎯 TP1/TP2/TP3: ${tp1:.2f}/${tp2:.2f}/${tp3:.2f}")
                print(f"🛡️  SL: ${sl:.2f}")
                confidence_val = self.safe_float_format(analysis.get('confidence'), 0.0)
                print(f"📈 Confidence: {confidence_val:.2f}")
                return True
            else:
                print(f"ℹ️  Document found but no changes made at {ist_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                return False
                
        except Exception as e:
            print(f"❌ Error updating MongoDB: {e}")
            # Print the analysis for debugging with safe formatting
            print(f"📋 Analysis data for debugging:")
            print(f"   Direction: {analysis.get('direction', 'N/A')}")
            print(f"   Entry: {analysis.get('entry', 'N/A')}")
            print(f"   TP1: {analysis.get('tp1', 'N/A')}")
            print(f"   TP2: {analysis.get('tp2', 'N/A')}")
            print(f"   TP3: {analysis.get('tp3', 'N/A')}")
            print(f"   SL: {analysis.get('sl', 'N/A')}")
            print(f"   Confidence: {analysis.get('confidence', 'N/A')}")
            return False
    
    def run_continuous_updates(self, interval_minutes: int = 15):
        """Run continuous updates to MongoDB with IST timestamps"""
        print(f"🕒 Starting continuous MongoDB updates every {interval_minutes} minutes")
        print(f"🗄️  Target database: {self.db.name}.{self.collection.name}")
        print(f"📄 Target document ID: {self.target_doc_id}")
        print(f"🌏 Timezone: Indian Standard Time (IST)")
        print("-" * 80)
        
        try:
            while True:
                print(f"\n{'='*80}")
                current_ist = self.get_ist_time()
                print(f"🔄 Fetching data at {current_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"{'='*80}")
                
                # Get market data
                data = self.get_complete_trading_data()
                
                if data and 'error' not in data:
                    # Get GPT analysis
                    analysis = self.get_gpt_analysis(data)
                    
                    if 'error' not in analysis:
                        # Update MongoDB
                        self.update_mongodb_prediction(analysis, data['current_price'])
                    else:
                        print(f"❌ GPT analysis error: {analysis.get('error', 'Unknown error')}")
                else:
                    print(f"❌ Failed to fetch data: {data.get('error', 'Unknown error')}")
                
                print(f"\n⏳ Waiting for next {interval_minutes}-minute cycle...")
                time.sleep(interval_minutes * 60)  # Wait for the specified interval
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Updates stopped by user")
            self.mongo_client.close()
        except Exception as e:
            print(f"\n\n💥 Fatal error: {e}")
            self.mongo_client.close()


# Usage
if __name__ == "__main__":
    # Initialize with your API keys
    updater = MongoDBTradingUpdater(
        openai_api_key="YOUR_OPENAI_API_KEY_HERE",
        mongo_uri="mongodb+srv://netmanconnect:eDxdS7AkkimNGJdi@cluster0.exzvao3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        db_name="mttrader",
        collection_name="prediction"
    )
    
    # Run continuous updates
    updater.run_continuous_updates(interval_minutes=15)
