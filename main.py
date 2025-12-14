#!/usr/bin/env python3
"""
Bybit VWAP Strategy Monitor
Monitors top 20 OI USDT perpetual futures and sends Telegram signals
"""

import os
import time
import asyncio
import threading
import http.server
import socketserver
from datetime import datetime, timezone
from collections import defaultdict
import pytz
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import ccxt
import requests
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Load environment variables
load_dotenv()

# Configuration
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Monitor Settings
TIMEFRAMES = os.getenv('TIMEFRAMES', '3m,5m,15m').split(',')
TOP_OI_COUNT = int(os.getenv('TOP_OI_COUNT', '20'))
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))

# VWAP Strategy Parameters
STOP_POINTS = float(os.getenv('STOP_POINTS', '20.0'))
BAND_ENTRY_MULT = float(os.getenv('BAND_ENTRY_MULT', '2.0'))

# Exit Mode: "VWAP", "Deviation Band", or "None"
EXIT_MODE_LONG = os.getenv('EXIT_MODE_LONG', 'VWAP')
EXIT_MODE_SHORT = os.getenv('EXIT_MODE_SHORT', 'VWAP')

# Target Deviation (used when Exit Mode is "Deviation Band")
TARGET_LONG_DEVIATION = float(os.getenv('TARGET_LONG_DEVIATION', '2.0'))
TARGET_SHORT_DEVIATION = float(os.getenv('TARGET_SHORT_DEVIATION', '2.0'))

# Safety Exit
ENABLE_SAFETY_EXIT = os.getenv('ENABLE_SAFETY_EXIT', 'true').lower() == 'true'
NUM_OPPOSING_BARS = int(os.getenv('NUM_OPPOSING_BARS', '3'))

# Trade Direction
ALLOW_LONGS = os.getenv('ALLOW_LONGS', 'true').lower() == 'true'
ALLOW_SHORTS = os.getenv('ALLOW_SHORTS', 'true').lower() == 'true'

# Signal Strength & Volatility Filter
MIN_STRENGTH = float(os.getenv('MIN_STRENGTH', '0.7'))
MIN_VOL_RATIO = float(os.getenv('MIN_VOL_RATIO', '0.25'))

# No Trade Window Around 09:00 KST
NO_TRADE_AROUND_0900 = os.getenv('NO_TRADE_AROUND_0900', 'true').lower() == 'true'

# VWAP Session Reset Timezone (must match TradingView)
SESSION_TIMEZONE = os.getenv('SESSION_TIMEZONE', 'Asia/Seoul')

# Debug Mode
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Chart Generation
SEND_CHART = os.getenv('SEND_CHART', 'false').lower() == 'true'

# Chart Viewer Server
CHART_SERVER_PORT = int(os.getenv('CHART_SERVER_PORT', '8080'))

# Telegram notification
def send_telegram(message):
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code != 200:
            print(f"❌ Telegram error: {response.text}")
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")


def send_telegram_photo(photo_path, caption=""):
    """Send photo to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    try:
        with open(photo_path, 'rb') as photo:
            response = requests.post(
                url,
                data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'},
                files={'photo': photo},
                timeout=30
            )
            if response.status_code != 200:
                print(f"❌ Telegram photo error: {response.text}")
    except Exception as e:
        print(f"❌ Telegram photo send failed: {e}")


def generate_chart(df, symbol, timeframe, signal=None, save_path=None):
    """Generate candlestick chart with VWAP and bands"""
    # Use last 150 bars for wider time scale
    df = df.tail(150).copy()
    df = df.reset_index(drop=True)

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    fig, ax = plt.subplots(figsize=(14, 8))

    # Colors
    up_color = '#26a69a'  # Green
    down_color = '#ef5350'  # Red
    vwap_color = '#2196f3'  # Blue
    band_color = '#ff9800'  # Orange

    # Plot candlesticks
    width = 0.6
    for i in range(len(df)):
        row = df.iloc[i]
        color = up_color if row['close'] >= row['open'] else down_color

        # Wick
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)

        # Body
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open'])
        if body_height == 0:
            body_height = 0.001  # Doji
        rect = Rectangle((i - width/2, body_bottom), width, body_height,
                         facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # Plot VWAP
    ax.plot(range(len(df)), df['vwap'], color=vwap_color, linewidth=2, label='VWAP')

    # Plot bands
    ax.plot(range(len(df)), df['upper_band'], color=band_color, linewidth=1.5,
            linestyle='--', label=f'Upper Band ({BAND_ENTRY_MULT}σ)')
    ax.plot(range(len(df)), df['lower_band'], color=band_color, linewidth=1.5,
            linestyle='--', label=f'Lower Band ({BAND_ENTRY_MULT}σ)')

    # Fill between bands
    ax.fill_between(range(len(df)), df['upper_band'], df['lower_band'],
                   alpha=0.1, color=band_color)

    # Mark signal on last bar if present
    if signal and signal.get('type'):
        last_idx = len(df) - 1
        last_row = df.iloc[-1]
        if signal['type'] == 'LONG':
            ax.scatter(last_idx, last_row['low'] * 0.998, marker='^',
                      s=200, color='lime', edgecolors='black', zorder=5, label='LONG Signal')
        elif signal['type'] == 'SHORT':
            ax.scatter(last_idx, last_row['high'] * 1.002, marker='v',
                      s=200, color='red', edgecolors='black', zorder=5, label='SHORT Signal')

    # X-axis labels (show every 20th bar)
    x_labels = []
    x_ticks = []
    for i in range(0, len(df), 20):
        x_ticks.append(i)
        x_labels.append(df.iloc[i]['datetime'].strftime('%H:%M'))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)

    # Labels and title
    clean_symbol = symbol.replace(':USDT', '').replace('/USDT', '')
    ax.set_title(f'{clean_symbol} - {timeframe} | VWAP Strategy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add current values as text
    last = df.iloc[-1]
    info_text = (f"Close: {last['close']:.4f}\n"
                f"VWAP: {last['vwap']:.4f}\n"
                f"Upper: {last['upper_band']:.4f}\n"
                f"Lower: {last['lower_band']:.4f}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save chart
    if save_path is None:
        os.makedirs('charts', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        signal_type = signal.get('type', '').lower() if signal else ''
        save_path = f"charts/{clean_symbol}_{timeframe}_{signal_type}_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Update charts index for HTML viewer
    update_charts_index()

    print(f"📈 Chart saved: {save_path}")
    return save_path


def update_charts_index():
    """Update the charts/index.json file for the HTML viewer"""
    import json
    import glob

    charts_dir = 'charts'
    if not os.path.exists(charts_dir):
        return

    # Get all PNG files in charts directory
    chart_files = glob.glob(os.path.join(charts_dir, '*.png'))
    chart_files = [os.path.basename(f) for f in chart_files]

    # Sort by modification time (newest first)
    chart_files.sort(key=lambda f: os.path.getmtime(os.path.join(charts_dir, f)), reverse=True)

    # Write index
    index_path = os.path.join(charts_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump({'charts': chart_files, 'updated': datetime.now().isoformat()}, f)


def start_chart_server():
    """Start HTTP server for chart viewer in background thread"""
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress request logs
            pass

    try:
        with socketserver.TCPServer(("", CHART_SERVER_PORT), QuietHandler) as httpd:
            print(f"📊 Chart viewer: http://localhost:{CHART_SERVER_PORT}/chart_viewer.html")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"⚠️ Port {CHART_SERVER_PORT} already in use, chart server not started")
        else:
            print(f"⚠️ Chart server error: {e}")


class VWAPStrategy:
    """VWAP Mean Reversion Strategy Implementation"""

    def __init__(self):
        pass

    def calculate_vwap(self, df, symbol, timeframe):
        """Calculate VWAP and standard deviation bands - matching Pine Script logic"""
        df = df.copy()

        # Convert timestamp to datetime (UTC)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Convert to session timezone (must match TradingView chart timezone)
        # VWAP resets at 00:00 in this timezone
        try:
            tz = pytz.timezone(SESSION_TIMEZONE)
            df['datetime_local'] = df['datetime'].dt.tz_convert(tz)
            df['date'] = df['datetime_local'].dt.date
        except:
            # Fallback to UTC if timezone invalid
            df['date'] = df['datetime'].dt.date

        # Calculate hlc3 (typical price)
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3

        # Identify session starts (new day in local timezone)
        df['new_session'] = df['date'] != df['date'].shift(1)

        # Create session groups
        df['session_id'] = df['new_session'].cumsum()

        # Calculate cumulative sums within each session
        df['src_vol'] = df['hlc3'] * df['volume']
        df['src_sq_vol'] = (df['hlc3'] ** 2) * df['volume']

        df['cum_src_vol'] = df.groupby('session_id')['src_vol'].cumsum()
        df['cum_vol'] = df.groupby('session_id')['volume'].cumsum()
        df['cum_src_sq_vol'] = df.groupby('session_id')['src_sq_vol'].cumsum()

        # Calculate VWAP
        df['vwap'] = df['cum_src_vol'] / df['cum_vol']

        # Calculate standard deviation
        df['variance'] = (df['cum_src_sq_vol'] / df['cum_vol']) - (df['vwap'] ** 2)
        df['stdev'] = np.sqrt(df['variance'].clip(lower=0))

        # Calculate bands
        df['upper_band'] = df['vwap'] + df['stdev'] * BAND_ENTRY_MULT
        df['lower_band'] = df['vwap'] - df['stdev'] * BAND_ENTRY_MULT

        return df[['vwap', 'stdev', 'upper_band', 'lower_band']].reset_index(drop=True)
    
    def check_signal(self, df, symbol, timeframe):
        """Check for entry signals"""
        if len(df) < 15:  # Need enough data for ATR
            return None

        # Check for 09:00 KST no-trade window (±30 minutes)
        if NO_TRADE_AROUND_0900:
            from datetime import timezone, timedelta
            kst = timezone(timedelta(hours=9))
            now_kst = datetime.now(kst)
            # Check if within ±30 min of 09:00
            minutes_from_0900 = (now_kst.hour * 60 + now_kst.minute) - (9 * 60)
            if abs(minutes_from_0900) <= 30:
                return None

        # Reset index to avoid concat issues
        df = df.reset_index(drop=True)

        # Calculate VWAP
        vwap_df = self.calculate_vwap(df.copy(), symbol, timeframe)
        df = pd.concat([df, vwap_df], axis=1)

        # Calculate ATR for volatility filter
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # Latest bar
        current = df.iloc[-1]

        # Volatility filter
        if current['stdev'] > 0 and current['atr'] > 0:
            vol_ratio = current['stdev'] / current['atr']
            if vol_ratio < MIN_VOL_RATIO:
                return None
        else:
            return None

        # Signal strength
        bar_range = current['high'] - current['low']
        if bar_range <= 0:
            return None

        bull_strength = (current['close'] - current['low']) / bar_range
        bear_strength = (current['high'] - current['close']) / bar_range

        # Debug info for comparison with TradingView
        debug_info = {
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'vwap': current['vwap'],
            'stdev': current['stdev'],
            'upper_band': current['upper_band'],
            'lower_band': current['lower_band'],
            'atr': current['atr'],
            'vol_ratio': vol_ratio,
            'bull_strength': bull_strength,
            'bear_strength': bear_strength,
        }

        # Check H1/H2 pattern (Long signal)
        # Pine: open < entryLower and close > entryLower
        is_h1h2 = (
            ALLOW_LONGS and
            current['open'] < current['lower_band'] and
            current['close'] > current['lower_band'] and
            bull_strength >= MIN_STRENGTH
        )

        # Check L1/L2 pattern (Short signal)
        # Pine: open > entryUpper and close < entryUpper
        is_l1l2 = (
            ALLOW_SHORTS and
            current['open'] > current['upper_band'] and
            current['close'] < current['upper_band'] and
            bear_strength >= MIN_STRENGTH
        )

        # Debug: Show why signal conditions fail
        debug_info['long_conditions'] = {
            'open < lower_band': current['open'] < current['lower_band'],
            'close > lower_band': current['close'] > current['lower_band'],
            'bull_strength >= min': bull_strength >= MIN_STRENGTH,
        }
        debug_info['short_conditions'] = {
            'open > upper_band': current['open'] > current['upper_band'],
            'close < upper_band': current['close'] < current['upper_band'],
            'bear_strength >= min': bear_strength >= MIN_STRENGTH,
        }

        if is_h1h2:
            stop_loss = current['low'] - STOP_POINTS
            # Target based on exit mode
            if EXIT_MODE_LONG == 'VWAP':
                target = current['vwap']
            elif EXIT_MODE_LONG == 'Deviation Band':
                target = current['vwap'] + current['stdev'] * TARGET_LONG_DEVIATION
            else:
                target = None
            return {
                'type': 'LONG',
                'price': current['close'],
                'stop_loss': stop_loss,
                'target': target,
                'exit_mode': EXIT_MODE_LONG,
                'vwap': current['vwap'],
                'strength': bull_strength,
                'vol_ratio': vol_ratio,
                'debug': debug_info
            }

        if is_l1l2:
            stop_loss = current['high'] + STOP_POINTS
            # Target based on exit mode
            if EXIT_MODE_SHORT == 'VWAP':
                target = current['vwap']
            elif EXIT_MODE_SHORT == 'Deviation Band':
                target = current['vwap'] - current['stdev'] * TARGET_SHORT_DEVIATION
            else:
                target = None
            return {
                'type': 'SHORT',
                'price': current['close'],
                'stop_loss': stop_loss,
                'target': target,
                'exit_mode': EXIT_MODE_SHORT,
                'vwap': current['vwap'],
                'strength': bear_strength,
                'vol_ratio': vol_ratio,
                'debug': debug_info
            }

        # Return debug info even when no signal (for DEBUG_MODE)
        return {'type': None, 'debug': debug_info}


class BybitMonitor:
    """Bybit market monitor for top OI symbols"""

    def __init__(self):
        # No API keys needed - only using public endpoints (tickers, OHLCV)
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}  # USDT perpetual
        })
        self.strategy = VWAPStrategy()
        self.last_signals = {}  # Track last signal time to avoid spam
        self.live_data = {'symbols': [], 'signals': {}, 'signal_history': [], 'timeframe': TIMEFRAMES[0]}

    def generate_live_chart(self, df, symbol, timeframe, signal=None):
        """Generate live chart for a symbol"""
        os.makedirs('charts', exist_ok=True)
        clean_symbol = symbol.replace(':USDT', '').replace('/USDT', '')
        save_path = f"charts/live_{clean_symbol}.png"

        try:
            # Calculate VWAP
            df_chart = df.reset_index(drop=True)
            vwap_df = self.strategy.calculate_vwap(df_chart.copy(), symbol, timeframe)
            df_chart = pd.concat([df_chart, vwap_df], axis=1)

            # Generate chart
            generate_chart(df_chart, symbol, timeframe, signal, save_path)
            return True
        except Exception as e:
            print(f"⚠️ Chart error for {symbol}: {e}")
            return False

    def update_live_data(self, symbols_data):
        """Update live_data.json for the HTML viewer"""
        import json

        self.live_data['symbols'] = symbols_data
        self.live_data['updated'] = datetime.now().isoformat()

        os.makedirs('charts', exist_ok=True)
        with open('charts/live_data.json', 'w') as f:
            json.dump(self.live_data, f)

    def add_signal_to_history(self, symbol, timeframe, signal_type):
        """Add signal to history"""
        self.live_data['signal_history'].insert(0, {
            'symbol': symbol,
            'timeframe': timeframe,
            'type': signal_type,
            'time': datetime.now().isoformat()
        })
        # Keep only last 50 signals
        self.live_data['signal_history'] = self.live_data['signal_history'][:50]
    
    async def get_top_oi_symbols(self):
        """Get top 20 symbols by open interest"""
        try:
            tickers = self.exchange.fetch_tickers()
            
            # Symbols to exclude
            exclude_symbols = ['1000PEPE']

            # Filter USDT perpetual futures
            usdt_symbols = []
            for symbol, ticker in tickers.items():
                if ':USDT' in symbol and ticker.get('info', {}).get('openInterest'):
                    # Skip excluded symbols
                    if any(excl in symbol for excl in exclude_symbols):
                        continue
                    oi = float(ticker['info']['openInterest'])
                    usdt_symbols.append({
                        'symbol': symbol,
                        'oi': oi,
                        'oi_value': oi * ticker['last'] if ticker['last'] else 0
                    })
            
            # Sort by OI value and get top 20
            usdt_symbols.sort(key=lambda x: x['oi_value'], reverse=True)
            top_symbols = [s['symbol'] for s in usdt_symbols[:TOP_OI_COUNT]]
            
            print(f"📊 Top {TOP_OI_COUNT} OI symbols:")
            for i, s in enumerate(usdt_symbols[:TOP_OI_COUNT], 1):
                print(f"  {i}. {s['symbol']}: ${s['oi_value']:,.0f}")
            
            return top_symbols
            
        except Exception as e:
            print(f"❌ Error getting OI data: {e}")
            return []
    
    def fetch_ohlcv(self, symbol, timeframe, limit=1000):
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            print(f"❌ Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def format_signal_message(self, symbol, timeframe, signal):
        """Format signal for Telegram"""
        signal_type = signal['type']
        emoji = "🟢" if signal_type == "LONG" else "🔴"

        # Format target line based on exit mode
        exit_mode = signal.get('exit_mode', 'VWAP')
        if signal['target'] is not None:
            target_line = f"<b>Target ({exit_mode}):</b> ${signal['target']:.4f}"
            rr = abs(signal['price'] - signal['target']) / abs(signal['price'] - signal['stop_loss'])
            rr_line = f"<i>Risk/Reward:</i> {rr:.2f}"
        else:
            target_line = f"<b>Target:</b> None (Exit Mode: {exit_mode})"
            rr_line = ""

        message = f"""
{emoji} <b>{signal_type} SIGNAL</b> {emoji}

<b>Symbol:</b> {symbol.replace(':USDT', '')}
<b>Timeframe:</b> {timeframe}
<b>Entry:</b> ${signal['price']:.4f}
<b>Stop Loss:</b> ${signal['stop_loss']:.4f}
{target_line}

<b>Signal Strength:</b> {signal['strength']:.2%}
<b>Vol Ratio:</b> {signal['vol_ratio']:.2f}
{rr_line}

<i>Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
"""
        return message
    
    async def check_symbol(self, symbol):
        """Check a symbol across all timeframes"""
        for timeframe in TIMEFRAMES:
            try:
                # Check if we recently sent signal for this symbol/timeframe
                key = f"{symbol}_{timeframe}"
                last_signal_time = self.last_signals.get(key, 0)

                # Don't spam - wait at least 5 minutes between signals
                if time.time() - last_signal_time < 300:
                    continue

                # Fetch data
                df = self.fetch_ohlcv(symbol, timeframe)
                if df is None or len(df) < 15:
                    continue

                # Check for signal
                result = self.strategy.check_signal(df, symbol, timeframe)

                if result is None:
                    continue

                # Debug output
                if DEBUG_MODE and 'debug' in result:
                    d = result['debug']
                    print(f"\n📊 {symbol} ({timeframe}):")
                    print(f"   O: {d['open']:.4f} | H: {d['high']:.4f} | L: {d['low']:.4f} | C: {d['close']:.4f}")
                    print(f"   VWAP: {d['vwap']:.4f} | StDev: {d['stdev']:.4f}")
                    print(f"   Upper Band: {d['upper_band']:.4f} | Lower Band: {d['lower_band']:.4f}")
                    print(f"   Vol Ratio: {d['vol_ratio']:.2f} | ATR: {d['atr']:.4f}")
                    print(f"   Bull Str: {d['bull_strength']:.2f} | Bear Str: {d['bear_strength']:.2f}")
                    print(f"   Long Cond: {d['long_conditions']}")
                    print(f"   Short Cond: {d['short_conditions']}")

                # Check if actual signal (not just debug info)
                if result.get('type') is not None:
                    signal = result
                    print(f"\n{'='*60}")
                    print(f"🎯 SIGNAL DETECTED: {symbol} ({timeframe})")
                    print(f"   Type: {signal['type']}")
                    print(f"   Entry: ${signal['price']:.4f}")
                    print(f"   Stop: ${signal['stop_loss']:.4f}")
                    if signal['target'] is not None:
                        print(f"   Target ({signal['exit_mode']}): ${signal['target']:.4f}")
                    else:
                        print(f"   Target: None (Exit Mode: {signal['exit_mode']})")
                    print(f"{'='*60}\n")

                    # Format signal message
                    message = self.format_signal_message(symbol, timeframe, signal)

                    # Generate and send chart with message as caption
                    if SEND_CHART:
                        try:
                            # Calculate VWAP data for chart
                            df_chart = df.reset_index(drop=True)
                            vwap_df = self.strategy.calculate_vwap(df_chart.copy(), symbol, timeframe)
                            df_chart = pd.concat([df_chart, vwap_df], axis=1)

                            # Generate chart
                            chart_path = generate_chart(df_chart, symbol, timeframe, signal)

                            # Send chart with full message as caption
                            send_telegram_photo(chart_path, message)
                        except Exception as chart_err:
                            print(f"⚠️ Chart generation failed: {chart_err}")
                            # Fallback to text-only
                            send_telegram(message)
                    else:
                        # No chart, send text only
                        send_telegram(message)

                    # Add to signal history for HTML viewer
                    self.add_signal_to_history(symbol, timeframe, signal['type'])

                    # Update last signal time
                    self.last_signals[key] = time.time()

            except Exception as e:
                print(f"❌ Error checking {symbol} {timeframe}: {e}")
                continue
    
    async def monitor(self):
        """Main monitoring loop"""
        print(f"\n🚀 Starting Bybit VWAP Monitor")
        print(f"📊 Monitoring top {TOP_OI_COUNT} OI symbols")
        print(f"⏱️  Timeframes: {', '.join(TIMEFRAMES)}")
        print(f"🔄 Check interval: {CHECK_INTERVAL}s")
        print(f"\n📋 Strategy Parameters:")
        print(f"   Session Timezone: {SESSION_TIMEZONE} (VWAP resets at 00:00)")
        print(f"   Stop Points: {STOP_POINTS}")
        print(f"   Band Entry Mult: {BAND_ENTRY_MULT}")
        print(f"   Exit Mode Long: {EXIT_MODE_LONG}")
        print(f"   Exit Mode Short: {EXIT_MODE_SHORT}")
        print(f"   Min Strength: {MIN_STRENGTH}")
        print(f"   Min Vol Ratio: {MIN_VOL_RATIO}")
        print(f"   Allow Longs: {ALLOW_LONGS}")
        print(f"   Allow Shorts: {ALLOW_SHORTS}")
        print(f"   No Trade 09:00 KST: {NO_TRADE_AROUND_0900}")
        print(f"   Debug Mode: {DEBUG_MODE}")
        print(f"   Send Chart: {SEND_CHART}\n")

        send_telegram(f"""🚀 <b>Bybit VWAP Monitor Started</b>

Monitoring top {TOP_OI_COUNT} OI symbols
Timeframes: {', '.join(TIMEFRAMES)}
Exit Mode Long: {EXIT_MODE_LONG}
Exit Mode Short: {EXIT_MODE_SHORT}""")
        
        while True:
            try:
                # Get top OI symbols
                symbols = await self.get_top_oi_symbols()

                if not symbols:
                    print("⚠️ No symbols found, retrying...")
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                # Process each symbol - check signals and generate live charts
                symbols_data = []
                timeframe = TIMEFRAMES[0]  # Use first timeframe for live charts

                for symbol in symbols:
                    # Check for signals (all timeframes)
                    await self.check_symbol(symbol)

                    # Generate live chart (first timeframe only)
                    try:
                        df = self.fetch_ohlcv(symbol, timeframe)
                        if df is not None and len(df) >= 15:
                            # Calculate VWAP for data
                            df_calc = df.reset_index(drop=True)
                            vwap_df = self.strategy.calculate_vwap(df_calc.copy(), symbol, timeframe)
                            df_calc = pd.concat([df_calc, vwap_df], axis=1)

                            # Get latest values
                            last = df_calc.iloc[-1]
                            symbol_info = {
                                'symbol': symbol,
                                'close': float(last['close']),
                                'vwap': float(last['vwap']),
                                'upper_band': float(last['upper_band']),
                                'lower_band': float(last['lower_band']),
                                'stdev': float(last['stdev'])
                            }
                            symbols_data.append(symbol_info)

                            # Check for current signal
                            result = self.strategy.check_signal(df, symbol, timeframe)
                            if result and result.get('type'):
                                self.live_data['signals'][symbol] = {
                                    'type': result['type'],
                                    'price': result['price']
                                }
                                self.generate_live_chart(df, symbol, timeframe, result)
                            else:
                                # Remove old signal if no longer valid
                                if symbol in self.live_data['signals']:
                                    del self.live_data['signals'][symbol]
                                self.generate_live_chart(df, symbol, timeframe, None)

                    except Exception as e:
                        print(f"⚠️ Error processing {symbol}: {e}")

                    await asyncio.sleep(0.5)  # Rate limiting

                # Update live data for HTML viewer
                self.update_live_data(symbols_data)

                print(f"\n✅ Scan complete. Charts updated. Next scan in {CHECK_INTERVAL}s...")
                await asyncio.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                send_telegram("⏹️ <b>Bybit VWAP Monitor Stopped</b>")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                await asyncio.sleep(CHECK_INTERVAL)


async def main():
    """Entry point"""
    # Check configuration
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ ERROR: Telegram configuration missing!")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")
        return

    # Start chart viewer server in background
    if SEND_CHART:
        server_thread = threading.Thread(target=start_chart_server, daemon=True)
        server_thread.start()

    monitor = BybitMonitor()
    await monitor.monitor()


if __name__ == "__main__":
    asyncio.run(main())