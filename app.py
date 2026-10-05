import threading, time
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_bcrypt import Bcrypt
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sqlite3, os, json, feedparser
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'stocktracker_secret_2024')
bcrypt = Bcrypt(app)
DB = os.path.join(os.path.dirname(__file__), 'stocks.db')

# ── DB Setup ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, password TEXT,
        created TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, symbol TEXT, condition TEXT,
        target REAL, volume_condition TEXT DEFAULT NULL,
        status TEXT DEFAULT 'Active', created TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS alert_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, symbol TEXT, message TEXT, triggered_at TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, symbol TEXT,
        UNIQUE(user_id, symbol)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, symbol TEXT, shares REAL,
        buy_price REAL, added TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS agent_insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, symbol TEXT, insight TEXT,
        insight_type TEXT, created TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

def get_db():
    return sqlite3.connect(DB)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

# ── Stock Helpers ─────────────────────────────────────────────────────────────
def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="30d")
        if hist.empty:
            return None
        current = info.get('currentPrice') or info.get('regularMarketPrice') or float(hist['Close'].iloc[-1])
        prev = info.get('previousClose') or float(hist['Close'].iloc[-2])
        change_pct = round((current - prev) / prev * 100, 2) if prev else 0
        avg_vol = hist['Volume'].mean()
        today_vol = info.get('volume', avg_vol)
        vol_ratio = round(today_vol / avg_vol, 2) if avg_vol else 1
        return {
            'symbol': symbol.upper(),
            'name': info.get('longName', symbol.upper()),
            'price': round(current, 2),
            'change_pct': change_pct,
            'high': round(info.get('dayHigh', 0), 2),
            'low': round(info.get('dayLow', 0), 2),
            'volume': int(today_vol),
            'avg_volume': int(avg_vol),
            'vol_ratio': vol_ratio,
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'N/A'),
            'history': hist
        }
    except:
        return None

def generate_chart(symbol, history, user_id):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    closes = history['Close']
    color = '#00e676' if closes.iloc[-1] >= closes.iloc[0] else '#ff5252'
    ax.plot(history.index, closes, color=color, linewidth=2)
    ax.fill_between(history.index, closes, alpha=0.08, color=color)
    ax.set_title(f'{symbol} — 30 Day Price', color='#ccc', fontsize=12, pad=10)
    ax.tick_params(colors='#555', labelsize=8)
    ax.set_facecolor('#0a0a14')
    fig.patch.set_facecolor('#0a0a14')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e1e30')
    path = f'static/charts/{user_id}_{symbol}_chart.png'
    os.makedirs('static/charts', exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=100)
    plt.close()
    return path

def check_alerts(user_id, symbol, price, change_pct, vol_ratio):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, condition, target, volume_condition FROM alerts WHERE user_id=? AND symbol=? AND status='Active'", (user_id, symbol))
    alerts = c.fetchall()
    triggered = []
    for aid, condition, target, vol_cond in alerts:
        hit = False
        msg = ''
        if condition == 'above' and price >= target:
            hit, msg = True, f"🚨 {symbol} crossed ABOVE ${target}! Now: ${price}"
        elif condition == 'below' and price <= target:
            hit, msg = True, f"🚨 {symbol} dropped BELOW ${target}! Now: ${price}"
        elif condition == 'pct_up' and change_pct >= target:
            hit, msg = True, f"🚨 {symbol} UP {change_pct}% (target: +{target}%)"
        elif condition == 'pct_down' and change_pct <= -target:
            hit, msg = True, f"🚨 {symbol} DOWN {abs(change_pct)}% (target: -{target}%)"
        if hit:
            if vol_cond == '2x' and vol_ratio < 2:
                hit = False
            if vol_cond == '3x' and vol_ratio < 3:
                hit = False
        if hit:
            priority = "🔴 HIGH PRIORITY" if vol_ratio >= 2 else "🟡"
            msg = f"{priority} {msg}"
            c.execute("UPDATE alerts SET status='Triggered' WHERE id=?", (aid,))
            c.execute("INSERT INTO alert_history (user_id, symbol, message, triggered_at) VALUES (?,?,?,?)",
                      (user_id, symbol, msg, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            triggered.append(msg)
    conn.commit()
    conn.close()
    return triggered

def detect_unusual_activity(data):
    alerts = []
    if abs(data['change_pct']) >= 5:
        direction = "surged" if data['change_pct'] > 0 else "dropped"
        alerts.append(f"⚡ Unusual: {data['symbol']} {direction} {abs(data['change_pct'])}%")
    if data['vol_ratio'] >= 2:
        alerts.append(f"📊 Volume {data['vol_ratio']}x above average — unusual activity!")
    return alerts

def get_news(symbol):
    try:
        name = symbol.replace('.NS', '').replace('.BO', '')
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        news = []
        for entry in feed.entries[:6]:
            news.append({
                'title': entry.get('title', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', '')[:16] if entry.get('published') else ''
            })
        return news
    except:
        return []

def get_global_markets():
    symbols = {
        'NIFTY 50': '^NSEI', 'SENSEX': '^BSESN',
        'NASDAQ': '^IXIC', 'S&P 500': '^GSPC',
        'Dow Jones': '^DJI', 'Gold': 'GC=F', 'Oil': 'CL=F'
    }
    result = []
    for name, sym in symbols.items():
        try:
            t = yf.Ticker(sym)
            info = t.info
            price = info.get('regularMarketPrice') or info.get('currentPrice', 0)
            prev = info.get('previousClose', price)
            chg = round((price - prev) / prev * 100, 2) if prev else 0
            result.append({'name': name, 'price': round(price, 2), 'change': chg})
        except:
            result.append({'name': name, 'price': 0, 'change': 0})
    return result

def get_portfolio_summary(user_id):
    conn = get_db()
    rows = conn.execute("SELECT id, symbol, shares, buy_price FROM portfolio WHERE user_id=?", (user_id,)).fetchall()
    conn.close()
    if not rows:
        return None
    total_invested = 0
    total_current = 0
    holdings = []
    sector_map = {}
    for row_id, symbol, shares, buy_price in rows:
        data = get_stock_data(symbol)
        if not data:
            continue
        invested = shares * buy_price
        current = shares * data['price']
        pnl = current - invested
        pnl_pct = round((pnl / invested) * 100, 2) if invested else 0
        total_invested += invested
        total_current += current
        sector = data.get('sector', 'Other')
        sector_map[sector] = sector_map.get(sector, 0) + current
        holdings.append({
            'id': row_id, 'symbol': symbol, 'shares': shares,
            'buy_price': buy_price, 'current_price': data['price'],
            'invested': round(invested, 2), 'current': round(current, 2),
            'pnl': round(pnl, 2), 'pnl_pct': pnl_pct,
            'change_pct': data['change_pct']
        })
    total_pnl = total_current - total_invested
    total_pnl_pct = round((total_pnl / total_invested) * 100, 2) if total_invested else 0
    today_pnl = sum(h['current'] * h['change_pct'] / 100 for h in holdings)

    # Risk Analysis
    risk_score = 5.0
    tech_pct = 0
    if total_current > 0:
        tech_pct = round(sector_map.get('Technology', 0) / total_current * 100, 1)
        if tech_pct > 60: risk_score += 2
        elif tech_pct > 40: risk_score += 1
        if len(holdings) < 3: risk_score += 1.5
        elif len(holdings) > 8: risk_score -= 1
    risk_score = min(10, max(1, round(risk_score, 1)))
    risk_label = 'Low' if risk_score < 4 else 'Medium' if risk_score < 7 else 'High'

    sector_alloc = [{'sector': k, 'pct': round(v / total_current * 100, 1)} for k, v in sector_map.items()] if total_current else []

    return {
        'holdings': holdings,
        'total_invested': round(total_invested, 2),
        'total_current': round(total_current, 2),
        'total_pnl': round(total_pnl, 2),
        'total_pnl_pct': total_pnl_pct,
        'today_pnl': round(today_pnl, 2),
        'risk_score': risk_score,
        'risk_label': risk_label,
        'tech_pct': tech_pct,
        'sector_alloc': sector_alloc,
        'num_stocks': len(holdings)
    }

# ── Auth Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_page'))

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def register():
    d = request.json
    name = d.get('name', '').strip()
    email = d.get('email', '').strip()
    password = d.get('password', '')
    if not all([name, email, password]):
        return jsonify({'error': 'All fields required'}), 400
    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    try:
        conn = get_db()
        conn.execute("INSERT INTO users (name, email, password, created) VALUES (?,?,?,?)",
                     (name, email, hashed, datetime.now().strftime('%Y-%m-%d')))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Account created!'})
    except:
        return jsonify({'error': 'Email already exists'}), 400

@app.route('/api/login', methods=['POST'])
def login():
    d = request.json
    email = d.get('email', '').strip()
    password = d.get('password', '')
    conn = get_db()
    user = conn.execute("SELECT id, name, password FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if not user or not bcrypt.check_password_hash(user[2], password):
        return jsonify({'error': 'Invalid email or password'}), 401
    session['user_id'] = user[0]
    session['user_name'] = user[1]
    return jsonify({'message': 'Login successful'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# ── Main Pages ────────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=session['user_name'])

@app.route('/tracker')
@login_required
def tracker():
    return render_template('index.html', name=session['user_name'])

@app.route('/portfolio')
@login_required
def portfolio_page():
    return render_template('portfolio.html', name=session['user_name'])

# ── API Routes ────────────────────────────────────────────────────────────────
@app.route('/api/dashboard_data')
@login_required
def dashboard_data():
    user_id = session['user_id']
    hour = datetime.now().hour
    greeting = "Good Morning" if hour < 12 else "Good Afternoon" if hour < 17 else "Good Evening"
    return jsonify({'greeting': greeting, 'name': session['user_name']})

@app.route('/api/markets_data')
@login_required
def markets_data():
    return jsonify(get_global_markets())

@app.route('/api/watchlist_prices')
@login_required
def watchlist_prices():
    user_id = session['user_id']
    conn = get_db()
    wl = conn.execute("SELECT symbol FROM watchlist WHERE user_id=?", (user_id,)).fetchall()
    conn.close()
    result = []
    for (sym,) in wl:
        d = get_stock_data(sym)
        if d:
            result.append({'symbol': sym, 'price': d['price'], 'change_pct': d['change_pct']})
    return jsonify(result)

@app.route('/api/portfolio_summary')
@login_required
def portfolio_summary():
    return jsonify(get_portfolio_summary(session['user_id']) or {})

@app.route('/api/get_stock', methods=['POST'])
@login_required
def get_stock():
    user_id = session['user_id']
    symbol = request.json.get('symbol', '').strip().upper()
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        price = getattr(info, 'last_price', None) or getattr(info, 'regular_market_price', None)
        prev  = getattr(info, 'previous_close', None)
        if not price:
            return jsonify({'error': 'Stock not found. Try: AAPL, TCS.NS, RELIANCE.NS'}), 404
        change_pct = round((price - prev) / prev * 100, 2) if prev else 0
        day_high = getattr(info, 'day_high', price)
        day_low  = getattr(info, 'day_low',  price)
        volume   = getattr(info, 'last_volume', 0)
        return jsonify({
            'symbol': symbol, 'name': symbol,
            'price': round(price,2), 'change_pct': change_pct,
            'high': round(day_high,2) if day_high else 0,
            'low':  round(day_low,2)  if day_low  else 0,
            'volume': int(volume) if volume else 0,
            'triggered': [], 'unusual': [], 'news': []
        })
    except:
        return jsonify({'error': 'Stock not found. Try: AAPL, TCS.NS, RELIANCE.NS'}), 404

@app.route('/api/get_stock_full', methods=['POST'])
@login_required
def get_stock_full():
    user_id = session['user_id']
    symbol = request.json.get('symbol', '').strip().upper()
    data = get_stock_data(symbol)
    if not data:
        return jsonify({'error': 'Not found'}), 404
    chart = generate_chart(symbol, data['history'], user_id)
    triggered = check_alerts(user_id, symbol, data['price'], data['change_pct'], data['vol_ratio'])
    unusual = detect_unusual_activity(data)
    news = get_news(symbol)
    return jsonify({
        'symbol': data['symbol'], 'name': data['name'],
        'price': data['price'], 'change_pct': data['change_pct'],
        'high': data['high'], 'low': data['low'],
        'volume': data['volume'], 'avg_volume': data['avg_volume'],
        'vol_ratio': data['vol_ratio'], 'sector': data['sector'],
        'chart': chart, 'triggered': triggered,
        'unusual': unusual, 'news': news
    })

@app.route('/api/set_alert', methods=['POST'])
@login_required
def set_alert():
    d = request.json
    symbol = d.get('symbol', '').upper()
    condition = d.get('condition')
    target = d.get('target')
    vol_cond = d.get('volume_condition') or None
    if not all([symbol, condition, target]):
        return jsonify({'error': 'Missing fields'}), 400
    conn = get_db()
    conn.execute("INSERT INTO alerts (user_id, symbol, condition, target, volume_condition, created) VALUES (?,?,?,?,?,?)",
                 (session['user_id'], symbol, condition, float(target), vol_cond, datetime.now().strftime('%Y-%m-%d %H:%M')))
    conn.commit()
    conn.close()
    return jsonify({'message': f'✅ Alert set for {symbol}'})

@app.route('/api/get_alerts')
@login_required
def get_alerts():
    conn = get_db()
    rows = conn.execute("SELECT id, symbol, condition, target, volume_condition, status, created FROM alerts WHERE user_id=? ORDER BY id DESC",
                        (session['user_id'],)).fetchall()
    conn.close()
    return jsonify([{'id': r[0], 'symbol': r[1], 'condition': r[2], 'target': r[3],
                     'volume_condition': r[4], 'status': r[5], 'created': r[6]} for r in rows])

@app.route('/api/delete_alert', methods=['POST'])
@login_required
def delete_alert():
    conn = get_db()
    conn.execute("DELETE FROM alerts WHERE id=? AND user_id=?", (request.json.get('id'), session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Deleted'})

@app.route('/api/get_history')
@login_required
def get_history():
    conn = get_db()
    rows = conn.execute("SELECT symbol, message, triggered_at FROM alert_history WHERE user_id=? ORDER BY id DESC LIMIT 20",
                        (session['user_id'],)).fetchall()
    conn.close()
    return jsonify([{'symbol': r[0], 'message': r[1], 'time': r[2]} for r in rows])

@app.route('/api/watchlist/add', methods=['POST'])
@login_required
def add_watchlist():
    symbol = request.json.get('symbol', '').upper()
    conn = get_db()
    try:
        conn.execute("INSERT INTO watchlist (user_id, symbol) VALUES (?,?)", (session['user_id'], symbol))
        conn.commit()
    except: pass
    conn.close()
    return jsonify({'message': f'{symbol} added'})

@app.route('/api/watchlist/get')
@login_required
def get_watchlist():
    conn = get_db()
    rows = conn.execute("SELECT symbol FROM watchlist WHERE user_id=?", (session['user_id'],)).fetchall()
    conn.close()
    return jsonify([r[0] for r in rows])

@app.route('/api/watchlist/remove', methods=['POST'])
@login_required
def remove_watchlist():
    symbol = request.json.get('symbol', '').upper()
    conn = get_db()
    conn.execute("DELETE FROM watchlist WHERE user_id=? AND symbol=?", (session['user_id'], symbol))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Removed'})

@app.route('/api/stock_price', methods=['POST'])
@login_required
def stock_price():
    symbol = request.json.get('symbol', '').strip().upper()
    if not symbol:
        return jsonify({'error': 'No symbol'}), 400
    # Try original, then .NS, then .BO
    attempts = [symbol]
    if '.' not in symbol:
        attempts += [symbol + '.NS', symbol + '.BO']
    for sym in attempts:
        try:
            t = yf.Ticker(sym)
            info = t.fast_info
            price = getattr(info, 'last_price', None) or getattr(info, 'regular_market_price', None)
            if not price or price == 0:
                continue
            prev = getattr(info, 'previous_close', price)
            change_pct = round((price - prev) / prev * 100, 2) if prev else 0
            full_info = t.info
            name = full_info.get('longName', sym)
            return jsonify({'price': round(price, 2), 'change_pct': change_pct, 'name': name, 'symbol': sym})
        except:
            continue
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/portfolio/live', methods=['POST'])
@login_required
def portfolio_live():
    symbols = request.json.get('symbols', [])
    result = {}
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            info = t.info
            price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            prev = info.get('previousClose', price)
            change_pct = round((price - prev) / prev * 100, 2) if prev and price else 0
            result[sym] = {'price': round(price, 2), 'change_pct': change_pct}
        except:
            result[sym] = {'price': 0, 'change_pct': 0}
    return jsonify(result)


@app.route('/api/portfolio/add', methods=['POST'])
@login_required
def add_portfolio():
    d = request.json
    symbol = d.get('symbol', '').upper()
    shares = float(d.get('shares', 0))
    buy_price = float(d.get('buy_price', 0))
    if not all([symbol, shares, buy_price]):
        return jsonify({'error': 'Missing fields'}), 400
    conn = get_db()
    conn.execute("INSERT INTO portfolio (user_id, symbol, shares, buy_price, added) VALUES (?,?,?,?,?)",
                 (session['user_id'], symbol, shares, buy_price, datetime.now().strftime('%Y-%m-%d')))
    conn.commit()
    conn.close()
    return jsonify({'message': f'{symbol} added to portfolio'})

@app.route('/api/portfolio/get')
@login_required
def get_portfolio():
    summary = get_portfolio_summary(session['user_id'])
    return jsonify(summary or {})


@app.route('/api/portfolio/delete', methods=['POST'])
@login_required
def delete_portfolio():
    conn = get_db()
    conn.execute("DELETE FROM portfolio WHERE id=? AND user_id=?", (request.json.get('id'), session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Removed'})

@app.route('/api/global_markets')
@login_required
def global_markets():
    return jsonify(get_global_markets())

# ── Stock Detail (30d history + high/low for chart modal) ─────────────────────
@app.route('/api/stock_detail', methods=['POST'])
@login_required
def stock_detail():
    symbol = request.json.get('symbol', '').strip().upper()
    try:
        t = yf.Ticker(symbol)
        info = t.info
        hist = t.history(period='30d')
        if hist.empty:
            return jsonify({'error': 'No data'}), 404
        closes = hist['Close'].tolist()
        highs  = hist['High'].tolist()
        lows   = hist['Low'].tolist()
        dates  = [str(d.date()) for d in hist.index]
        price  = info.get('currentPrice') or info.get('regularMarketPrice') or closes[-1]
        prev   = info.get('previousClose', closes[-2] if len(closes) > 1 else price)
        change_pct = round((price - prev) / prev * 100, 2) if prev else 0
        return jsonify({
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'price': round(price, 2),
            'change_pct': change_pct,
            'high_52w': round(info.get('fiftyTwoWeekHigh', max(highs)), 2),
            'low_52w':  round(info.get('fiftyTwoWeekLow',  min(lows)),  2),
            'day_high': round(info.get('dayHigh', highs[-1]), 2),
            'day_low':  round(info.get('dayLow',  lows[-1]),  2),
            'volume':   info.get('volume', 0),
            'market_cap': info.get('marketCap', 0),
            'sector':   info.get('sector', 'N/A'),
            'pe_ratio': round(info.get('trailingPE', 0) or 0, 2),
            'dates': dates, 'closes': [round(c,2) for c in closes],
            'highs':  [round(h,2) for h in highs],
            'lows':   [round(l,2) for l in lows],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Market Stocks List ────────────────────────────────────────────────────────
MARKET_STOCKS = [
    ('AAPL','Apple'),('MSFT','Microsoft'),('GOOGL','Alphabet'),('AMZN','Amazon'),
    ('NVDA','NVIDIA'),('TSLA','Tesla'),('META','Meta'),('NFLX','Netflix'),
    ('TCS.NS','TCS'),('RELIANCE.NS','Reliance'),('INFY.NS','Infosys'),('HDFCBANK.NS','HDFC Bank'),
    ('WIPRO.NS','Wipro'),('ICICIBANK.NS','ICICI Bank'),('SBIN.NS','SBI'),('BAJFINANCE.NS','Bajaj Finance'),
]

@app.route('/api/market_stocks')
@login_required
def market_stocks():
    result = []
    for sym, name in MARKET_STOCKS:
        try:
            t = yf.Ticker(sym)
            info = t.info
            price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            prev  = info.get('previousClose', price)
            chg   = round((price - prev) / prev * 100, 2) if prev and price else 0
            result.append({'symbol': sym, 'name': name, 'price': round(price,2) if price else 0, 'change': chg})
        except:
            result.append({'symbol': sym, 'name': name, 'price': 0, 'change': 0})
    return jsonify(result)

# ── AI Agent Insights ─────────────────────────────────────────────────────────
def generate_insight(symbol, price, change_pct, vol_ratio, high_52w, low_52w, buy_price=None):
    insights = []
    # Trend
    if change_pct >= 3:
        insights.append(f"🚀 Strong bullish momentum — {symbol} up {change_pct}% today. Consider booking partial profits if holding.")
    elif change_pct <= -3:
        insights.append(f"⚠️ Sharp decline — {symbol} down {abs(change_pct)}% today. Watch for support levels before averaging down.")
    elif 0 < change_pct < 1:
        insights.append(f"📊 {symbol} is consolidating with minor gains. Sideways movement — wait for breakout confirmation.")
    # 52-week position
    if high_52w and price >= high_52w * 0.97:
        insights.append(f"🔝 {symbol} is near its 52-week HIGH (${high_52w}). Resistance zone — momentum traders may book profits here.")
    elif low_52w and price <= low_52w * 1.05:
        insights.append(f"📉 {symbol} is near its 52-week LOW (${low_52w}). Potential value buy zone — high risk, high reward.")
    # Volume
    if vol_ratio >= 2:
        insights.append(f"📊 Volume spike {vol_ratio}x above average on {symbol} — institutional activity detected. Big move likely incoming.")
    # Buy price comparison
    if buy_price:
        gain = round((price - buy_price) / buy_price * 100, 2)
        if gain >= 20:
            insights.append(f"💰 {symbol} is up {gain}% from your buy price ${buy_price}. Consider setting a trailing stop-loss to protect gains.")
        elif gain <= -10:
            insights.append(f"🔴 {symbol} is down {abs(gain)}% from your buy price ${buy_price}. Review your thesis — cut loss or average down?")
        elif 5 <= gain < 20:
            insights.append(f"✅ {symbol} showing healthy +{gain}% gain from your entry. Hold and let it run with a stop at breakeven.")
    return insights[0] if insights else f"📈 {symbol} at ${price} — no significant signals right now. Monitor for breakout above ${round(price*1.03,2)}."

def run_ai_agent():
    """Background agent — runs every 5 minutes, analyzes all portfolio stocks for all users"""
    while True:
        try:
            conn = get_db()
            users = conn.execute("SELECT DISTINCT user_id FROM portfolio").fetchall()
            for (uid,) in users:
                rows = conn.execute("SELECT symbol, buy_price FROM portfolio WHERE user_id=?", (uid,)).fetchall()
                for symbol, buy_price in rows:
                    try:
                        t = yf.Ticker(symbol)
                        info = t.info
                        hist = t.history(period='30d')
                        if hist.empty: continue
                        price = info.get('currentPrice') or info.get('regularMarketPrice') or float(hist['Close'].iloc[-1])
                        prev  = info.get('previousClose', float(hist['Close'].iloc[-2]))
                        change_pct = round((price - prev) / prev * 100, 2) if prev else 0
                        avg_vol = hist['Volume'].mean()
                        today_vol = info.get('volume', avg_vol)
                        vol_ratio = round(today_vol / avg_vol, 2) if avg_vol else 1
                        high_52w = info.get('fiftyTwoWeekHigh', 0)
                        low_52w  = info.get('fiftyTwoWeekLow', 0)
                        insight = generate_insight(symbol, price, change_pct, vol_ratio, high_52w, low_52w, buy_price)
                        # Keep only latest insight per symbol per user
                        conn.execute("DELETE FROM agent_insights WHERE user_id=? AND symbol=?", (uid, symbol))
                        conn.execute("INSERT INTO agent_insights (user_id, symbol, insight, insight_type, created) VALUES (?,?,?,?,?)",
                            (uid, symbol, insight,
                             'bullish' if change_pct > 0 else 'bearish',
                             datetime.now().strftime('%Y-%m-%d %H:%M')))
                        conn.commit()
                        # Auto-alert if big move
                        if abs(change_pct) >= 4:
                            msg = f"🤖 Agent Alert: {symbol} moved {change_pct:+.2f}% — {insight[:80]}"
                            conn.execute("INSERT INTO alert_history (user_id, symbol, message, triggered_at) VALUES (?,?,?,?)",
                                (uid, symbol, msg, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                            conn.commit()
                    except: continue
            conn.close()
        except: pass
        time.sleep(300)  # every 5 minutes

# Start background agent
agent_thread = threading.Thread(target=run_ai_agent, daemon=True)
agent_thread.start()

@app.route('/api/agent_insights')
@login_required
def agent_insights():
    conn = get_db()
    rows = conn.execute(
        "SELECT symbol, insight, insight_type, created FROM agent_insights WHERE user_id=? ORDER BY id DESC",
        (session['user_id'],)
    ).fetchall()
    conn.close()
    return jsonify([{'symbol': r[0], 'insight': r[1], 'type': r[2], 'time': r[3]} for r in rows])

if __name__ == '__main__':
    app.run(debug=True)
