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

@app.route('/api/market_status')
def market_status():
    now = datetime.utcnow()
    # IST = UTC+5:30, NYSE = UTC-5 (EST)
    ist_hour = (now.hour + 5) % 24
    ist_min  = (now.minute + 30) % 60
    ist_time = ist_hour + (1 if now.minute + 30 >= 60 else 0)
    weekday  = now.weekday()  # 0=Mon, 6=Sun
    nse_open  = weekday < 5 and (9 < ist_time < 15 or (ist_time == 9 and ist_min >= 15) or (ist_time == 15 and ist_min <= 30))
    nyse_hour = (now.hour - 5) % 24
    nyse_open = weekday < 5 and 9 <= nyse_hour < 16
    if nse_open and nyse_open:
        label, color = 'NSE & NYSE Open', '#00e676'
    elif nse_open:
        label, color = 'NSE Open', '#00e676'
    elif nyse_open:
        label, color = 'NYSE Open', '#00e676'
    else:
        label, color = 'Markets Closed', '#ff5252'
    return jsonify({'label': label, 'color': color, 'open': nse_open or nyse_open})

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

# ── Technical Indicators ─────────────────────────────────────────────────────
@app.route('/api/indicators', methods=['POST'])
@login_required
def get_indicators():
    symbol = request.json.get('symbol', '').strip().upper()
    indicators = request.json.get('indicators', ['RSI','MACD','MA','EMA','BB'])
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period='90d')
        if hist.empty or len(hist) < 20:
            return jsonify({'error': 'Not enough data'}), 404
        closes = hist['Close']
        result = {'symbol': symbol, 'dates': [str(d.date()) for d in hist.index], 'closes': [round(c,2) for c in closes]}
        if 'RSI' in indicators:
            delta = closes.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            result['rsi'] = [round(v,2) if not pd.isna(v) else None for v in rsi]
            result['rsi_current'] = round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else None
        if 'MACD' in indicators:
            ema12 = closes.ewm(span=12).mean()
            ema26 = closes.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_hist = macd - signal
            result['macd'] = [round(v,4) for v in macd]
            result['macd_signal'] = [round(v,4) for v in signal]
            result['macd_hist'] = [round(v,4) for v in macd_hist]
            result['macd_current'] = round(macd.iloc[-1],4)
            result['macd_signal_current'] = round(signal.iloc[-1],4)
        if 'MA' in indicators:
            ma20 = closes.rolling(20).mean()
            ma50 = closes.rolling(50).mean() if len(closes) >= 50 else pd.Series([None]*len(closes))
            result['ma20'] = [round(v,2) if not pd.isna(v) else None for v in ma20]
            result['ma50'] = [round(v,2) if not pd.isna(v) else None for v in ma50]
        if 'EMA' in indicators:
            ema9  = closes.ewm(span=9).mean()
            ema21 = closes.ewm(span=21).mean()
            result['ema9']  = [round(v,2) for v in ema9]
            result['ema21'] = [round(v,2) for v in ema21]
        if 'BB' in indicators:
            ma20 = closes.rolling(20).mean()
            std20 = closes.rolling(20).std()
            result['bb_upper'] = [round(v,2) if not pd.isna(v) else None for v in (ma20 + 2*std20)]
            result['bb_lower'] = [round(v,2) if not pd.isna(v) else None for v in (ma20 - 2*std20)]
            result['bb_mid']   = [round(v,2) if not pd.isna(v) else None for v in ma20]
        if 'SR' in indicators:
            resistance = round(float(hist['High'].rolling(20).max().iloc[-1]), 2)
            support    = round(float(hist['Low'].rolling(20).min().iloc[-1]),  2)
            result['support'] = support
            result['resistance'] = resistance
        if 'VOL' in indicators:
            avg_vol = hist['Volume'].rolling(20).mean()
            result['volumes'] = [int(v) for v in hist['Volume']]
            result['avg_vol'] = [int(v) if not pd.isna(v) else None for v in avg_vol]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Stock Comparison ──────────────────────────────────────────────────────────
@app.route('/api/compare', methods=['POST'])
@login_required
def compare_stocks():
    symbols = request.json.get('symbols', [])
    if len(symbols) < 2:
        return jsonify({'error': 'Need 2 symbols'}), 400
    result = []
    for sym in symbols[:2]:
        try:
            t = yf.Ticker(sym)
            info = t.info
            hist = t.history(period='1y')
            price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            prev  = info.get('previousClose', price)
            chg   = round((price - prev) / prev * 100, 2) if prev else 0
            yr_start = float(hist['Close'].iloc[0]) if not hist.empty else price
            yr_return = round((price - yr_start) / yr_start * 100, 2) if yr_start else 0
            result.append({
                'symbol': sym.upper(), 'name': info.get('longName', sym),
                'price': round(price, 2), 'change_pct': chg,
                'pe_ratio': round(info.get('trailingPE', 0) or 0, 2),
                'pb_ratio': round(info.get('priceToBook', 0) or 0, 2),
                'roe': round((info.get('returnOnEquity', 0) or 0) * 100, 2),
                'revenue': info.get('totalRevenue', 0),
                'net_income': info.get('netIncomeToCommon', 0),
                'market_cap': info.get('marketCap', 0),
                'div_yield': round((info.get('dividendYield', 0) or 0) * 100, 2),
                'beta': round(info.get('beta', 0) or 0, 2),
                'sector': info.get('sector', 'N/A'),
                'yr_return': yr_return,
                '52w_high': round(info.get('fiftyTwoWeekHigh', 0), 2),
                '52w_low':  round(info.get('fiftyTwoWeekLow', 0), 2),
                'analyst_rating': info.get('recommendationKey', 'N/A').upper(),
            })
        except Exception as e:
            result.append({'symbol': sym.upper(), 'error': str(e)})
    verdict = []
    if len(result) == 2 and 'error' not in result[0] and 'error' not in result[1]:
        a, b = result[0], result[1]
        if a['pe_ratio'] and b['pe_ratio']:
            cheaper = a['symbol'] if a['pe_ratio'] < b['pe_ratio'] else b['symbol']
            verdict.append(f"💰 Better Valuation: {cheaper} (lower P/E)")
        if a['roe'] and b['roe']:
            stronger = a['symbol'] if a['roe'] > b['roe'] else b['symbol']
            verdict.append(f"💪 Stronger Profitability: {stronger} (higher ROE)")
        if a['yr_return'] != b['yr_return']:
            better = a['symbol'] if a['yr_return'] > b['yr_return'] else b['symbol']
            verdict.append(f"📈 Better 1Y Return: {better} ({max(a['yr_return'], b['yr_return'])}%)")
        if a['beta'] and b['beta']:
            safer = a['symbol'] if a['beta'] < b['beta'] else b['symbol']
            verdict.append(f"🛡️ Lower Risk: {safer} (beta {min(a['beta'], b['beta'])})")
    return jsonify({'stocks': result, 'verdict': verdict})

# ── Sector Analysis ───────────────────────────────────────────────────────────
SECTOR_STOCKS = {
    'IT':      ['TCS.NS','INFY.NS','WIPRO.NS','HCLTECH.NS','LTIM.NS'],
    'Banking': ['HDFCBANK.NS','ICICIBANK.NS','SBIN.NS','KOTAKBANK.NS','AXISBANK.NS'],
    'Pharma':  ['SUNPHARMA.NS','DRREDDY.NS','CIPLA.NS'],
    'Auto':    ['TATAMOTORS.NS','MARUTI.NS','BAJAJ-AUTO.NS'],
    'Energy':  ['RELIANCE.NS','ONGC.NS','NTPC.NS'],
}

@app.route('/api/sectors')
@login_required
def sector_analysis():
    result = []
    for sector, stocks in SECTOR_STOCKS.items():
        changes = []
        for sym in stocks:
            try:
                t = yf.Ticker(sym)
                fi = t.fast_info
                price = getattr(fi, 'last_price', 0) or 0
                prev  = getattr(fi, 'previous_close', price) or price
                chg   = round((price - prev) / prev * 100, 2) if prev else 0
                changes.append({'symbol': sym, 'change': chg})
            except: pass
        avg_chg = round(sum(c['change'] for c in changes) / len(changes), 2) if changes else 0
        result.append({'sector': sector, 'change': avg_chg, 'stocks': changes})
    return jsonify(result)

@app.route('/api/sector_stocks', methods=['POST'])
@login_required
def sector_stocks_detail():
    sector = request.json.get('sector', '')
    stocks = SECTOR_STOCKS.get(sector, [])
    result = []
    for sym in stocks:
        try:
            t = yf.Ticker(sym)
            fi = t.fast_info
            price = getattr(fi, 'last_price', 0) or 0
            prev  = getattr(fi, 'previous_close', price) or price
            chg   = round((price - prev) / prev * 100, 2) if prev else 0
            result.append({'symbol': sym.replace('.NS',''), 'full_sym': sym, 'price': round(price,2), 'change': chg})
        except: pass
    return jsonify(result)

# ── Risk Profiler ─────────────────────────────────────────────────────────────
@app.route('/api/risk_profile', methods=['POST'])
@login_required
def risk_profile():
    profile = request.json.get('profile', 'moderate')
    summary = get_portfolio_summary(session['user_id'])
    if not summary:
        return jsonify({'analysis': 'Add stocks to your portfolio first.', 'suggestions': []})
    suggestions = []
    rs = summary['risk_score']
    tech = summary['tech_pct']
    n = summary['num_stocks']
    if profile == 'conservative':
        analysis = f"Your portfolio risk score is {rs}/10 ({summary['risk_label']}). "
        analysis += "⚠️ Higher than recommended for conservative." if rs > 4 else "✅ Aligns with conservative profile."
        if tech > 30: suggestions.append("📉 Reduce IT/Tech below 30% — add Banking or FMCG stocks.")
        if n < 5: suggestions.append("📊 Diversify — hold at least 5-8 stocks across sectors.")
        suggestions.append("🛡️ Add defensive stocks: HINDUNILVR, NESTLEIND, ITC.")
    elif profile == 'moderate':
        analysis = f"Your portfolio risk score is {rs}/10 ({summary['risk_label']}). "
        analysis += "✅ Well balanced." if 4 <= rs <= 7 else ("📈 Can take slightly more risk." if rs < 4 else "⚠️ Consider reducing concentration.")
        if tech > 50: suggestions.append(f"⚖️ Tech at {tech}% — balance with other sectors.")
        suggestions.append("📊 Mix growth (NVDA, TSLA) and value stocks (HDFC, TCS).")
    elif profile == 'aggressive':
        analysis = f"Your portfolio risk score is {rs}/10 ({summary['risk_label']}). "
        analysis += "🚀 High risk, high reward — aligned." if rs >= 7 else "📈 Can increase exposure to high-growth stocks."
        suggestions.append("🚀 Consider: NVDA, AMD, ADANIENT for momentum plays.")
        suggestions.append("⚡ Use stop-losses at -8% to protect against drawdowns.")
    else:
        analysis = "Unknown profile."
    return jsonify({'profile': profile, 'risk_score': rs, 'risk_label': summary['risk_label'],
                    'analysis': analysis, 'suggestions': suggestions,
                    'sector_alloc': summary['sector_alloc'], 'num_stocks': n})

# ── Paper Trading ─────────────────────────────────────────────────────────────
def init_paper_trading():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS paper_portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, symbol TEXT, shares REAL, buy_price REAL, added TEXT
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS paper_cash (
        user_id INTEGER PRIMARY KEY, cash REAL DEFAULT 100000
    )''')
    conn.commit()
    conn.close()

init_paper_trading()

@app.route('/paper')
@login_required
def paper_trading_page():
    return render_template('paper_trading.html')

@app.route('/compare')
@login_required
def compare_page():
    return render_template('compare.html')

@app.route('/api/paper/status')
@login_required
def paper_status():
    uid = session['user_id']
    conn = get_db()
    cash_row = conn.execute("SELECT cash FROM paper_cash WHERE user_id=?", (uid,)).fetchone()
    if not cash_row:
        conn.execute("INSERT INTO paper_cash (user_id, cash) VALUES (?,?)", (uid, 100000))
        conn.commit()
        cash = 100000
    else:
        cash = cash_row[0]
    holdings = conn.execute("SELECT id, symbol, shares, buy_price FROM paper_portfolio WHERE user_id=?", (uid,)).fetchall()
    conn.close()
    total_val = 0
    h_list = []
    for hid, sym, shares, bp in holdings:
        try:
            t = yf.Ticker(sym)
            fi = t.fast_info
            price = getattr(fi, 'last_price', bp) or bp
            val = shares * price
            pnl = val - shares * bp
            total_val += val
            h_list.append({'id': hid, 'symbol': sym, 'shares': shares, 'buy_price': bp,
                           'current_price': round(price,2), 'value': round(val,2),
                           'pnl': round(pnl,2), 'pnl_pct': round(pnl/(shares*bp)*100,2) if bp else 0})
        except: pass
    total_assets = cash + total_val
    return_pct = round((total_assets - 100000) / 100000 * 100, 2)
    return jsonify({'cash': round(cash,2), 'holdings': h_list,
                    'total_value': round(total_val,2), 'total_assets': round(total_assets,2),
                    'return_pct': return_pct})

@app.route('/api/paper/buy', methods=['POST'])
@login_required
def paper_buy():
    uid = session['user_id']
    symbol = request.json.get('symbol','').upper()
    shares = float(request.json.get('shares', 0))
    if not symbol or shares <= 0:
        return jsonify({'error': 'Invalid input'}), 400
    try:
        t = yf.Ticker(symbol)
        fi = t.fast_info
        price = getattr(fi, 'last_price', None)
        if not price: return jsonify({'error': 'Price not found'}), 404
        cost = price * shares
        conn = get_db()
        cash_row = conn.execute("SELECT cash FROM paper_cash WHERE user_id=?", (uid,)).fetchone()
        if not cash_row:
            conn.execute("INSERT INTO paper_cash (user_id, cash) VALUES (?,?)", (uid, 100000))
            cash = 100000
        else:
            cash = cash_row[0]
        if cost > cash:
            conn.close()
            return jsonify({'error': f'Insufficient funds. Need ${cost:.2f}, have ${cash:.2f}'}), 400
        conn.execute("UPDATE paper_cash SET cash=? WHERE user_id=?", (cash - cost, uid))
        conn.execute("INSERT INTO paper_portfolio (user_id, symbol, shares, buy_price, added) VALUES (?,?,?,?,?)",
                     (uid, symbol, shares, round(price,2), datetime.now().strftime('%Y-%m-%d %H:%M')))
        conn.commit()
        conn.close()
        return jsonify({'message': f'✅ Bought {shares} {symbol} @ ${price:.2f}', 'cost': round(cost,2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/paper/sell', methods=['POST'])
@login_required
def paper_sell():
    uid = session['user_id']
    hid = request.json.get('id')
    conn = get_db()
    row = conn.execute("SELECT symbol, shares, buy_price FROM paper_portfolio WHERE id=? AND user_id=?", (hid, uid)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Not found'}), 404
    sym, shares, bp = row
    try:
        t = yf.Ticker(sym)
        fi = t.fast_info
        price = getattr(fi, 'last_price', bp) or bp
        proceeds = price * shares
        pnl = proceeds - shares * bp
        conn.execute("DELETE FROM paper_portfolio WHERE id=?", (hid,))
        conn.execute("UPDATE paper_cash SET cash = cash + ? WHERE user_id=?", (proceeds, uid))
        conn.commit()
        conn.close()
        return jsonify({'message': f'✅ Sold {shares} {sym} @ ${price:.2f}', 'pnl': round(pnl,2)})
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/paper/reset', methods=['POST'])
@login_required
def paper_reset():
    uid = session['user_id']
    conn = get_db()
    conn.execute("DELETE FROM paper_portfolio WHERE user_id=?", (uid,))
    conn.execute("INSERT OR REPLACE INTO paper_cash (user_id, cash) VALUES (?,?)", (uid, 100000))
    conn.commit()
    conn.close()
    return jsonify({'message': '✅ Reset to $100,000'})

@app.route('/api/paper/leaderboard')
@login_required
def paper_leaderboard():
    conn = get_db()
    users = conn.execute("SELECT id, name FROM users").fetchall()
    board = []
    for uid, uname in users:
        cash_row = conn.execute("SELECT cash FROM paper_cash WHERE user_id=?", (uid,)).fetchone()
        if not cash_row: continue
        cash = cash_row[0]
        holdings = conn.execute("SELECT symbol, shares, buy_price FROM paper_portfolio WHERE user_id=?", (uid,)).fetchall()
        total_val = cash
        for sym, shares, bp in holdings:
            try:
                t = yf.Ticker(sym)
                fi = t.fast_info
                price = getattr(fi, 'last_price', bp) or bp
                total_val += shares * price
            except: total_val += shares * bp
        ret = round((total_val - 100000) / 100000 * 100, 2)
        board.append({'name': uname, 'total': round(total_val,2), 'return_pct': ret})
    conn.close()
    board.sort(key=lambda x: x['return_pct'], reverse=True)
    for i, b in enumerate(board): b['rank'] = i + 1
    return jsonify(board)

if __name__ == '__main__':
    app.run(debug=True)
