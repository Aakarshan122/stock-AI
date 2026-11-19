import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def fetch_features(symbol, period='2y'):
    t = yf.Ticker(symbol)
    hist = t.history(period=period)
    if hist.empty or len(hist) < 60:
        raise ValueError(f"Not enough data for {symbol}")

    df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.dropna(inplace=True)

    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']

    df['Vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['Price_change'] = df['Close'].pct_change()
    df['High_Low_pct'] = (df['High'] - df['Low']) / df['Close']

    df['Target_clf'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Target_reg'] = df['Close'].shift(-1)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    'MA10', 'MA20', 'MA50', 'EMA12', 'EMA26', 'MACD',
    'RSI', 'BB_width', 'Vol_ratio', 'Price_change', 'High_Low_pct',
    'Close', 'Volume'
]


# ── Pure numpy StandardScaler ─────────────────────────────────────────────────
class NumpyScaler:
    def fit_transform(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1
        return (X - self.mean_) / self.std_

    def transform(self, X):
        return (X - self.mean_) / self.std_


# ── Pure numpy Linear Regression ─────────────────────────────────────────────
class NumpyLinearRegression:
    def fit(self, X, y):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        self.coef_ = np.linalg.lstsq(Xb, y, rcond=None)[0]

    def predict(self, X):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        return Xb @ self.coef_


# ── Pure numpy Decision Tree (for Random Forest) ─────────────────────────────
class DecisionTree:
    def __init__(self, max_depth=8, min_samples=5, n_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features
        self.tree = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y, minlength=2) / len(y)
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        n_feats = X.shape[1]
        feat_idx = np.random.choice(n_feats, self.n_features or n_feats, replace=False)
        parent_gini = self._gini(y)
        for f in feat_idx:
            thresholds = np.unique(X[:, f])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, f], np.linspace(10, 90, 10))
            for t in thresholds:
                left = y[X[:, f] <= t]
                right = y[X[:, f] > t]
                if len(left) < self.min_samples or len(right) < self.min_samples:
                    continue
                gain = parent_gini - (len(left)/len(y)*self._gini(left) + len(right)/len(y)*self._gini(right))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, f, t
        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples or len(np.unique(y)) == 1:
            vals = np.bincount(y, minlength=2)
            return {'leaf': True, 'class': int(np.argmax(vals)), 'proba': vals / vals.sum()}
        feat, thresh = self._best_split(X, y)
        if feat is None:
            vals = np.bincount(y, minlength=2)
            return {'leaf': True, 'class': int(np.argmax(vals)), 'proba': vals / vals.sum()}
        mask = X[:, feat] <= thresh
        return {
            'leaf': False, 'feat': feat, 'thresh': thresh,
            'left': self._build(X[mask], y[mask], depth + 1),
            'right': self._build(X[~mask], y[~mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['class'], node['proba']
        if x[node['feat']] <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree)[0] for x in X])

    def predict_proba(self, X):
        return np.array([self._predict_one(x, self.tree)[1] for x in X])


# ── Random Forest ─────────────────────────────────────────────────────────────
class NumpyRandomForest:
    def __init__(self, n_trees=40, max_depth=8, min_samples=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        n, f = X.shape
        n_feats = max(1, int(np.sqrt(f)))
        self.trees = []
        feat_counts = np.zeros(f)
        for _ in range(self.n_trees):
            idx = np.random.choice(n, n, replace=True)
            tree = DecisionTree(self.max_depth, self.min_samples, n_feats)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        # Approximate feature importance via usage count
        def count_feats(node, counts):
            if node['leaf']:
                return
            counts[node['feat']] += 1
            count_feats(node['left'], counts)
            count_feats(node['right'], counts)
        for tree in self.trees:
            count_feats(tree.tree, feat_counts)
        total = feat_counts.sum()
        self.feature_importances_ = feat_counts / total if total > 0 else feat_counts

    def predict(self, X):
        votes = np.array([t.predict(X) for t in self.trees])
        return np.round(votes.mean(axis=0)).astype(int)

    def predict_proba(self, X):
        probas = np.array([t.predict_proba(X) for t in self.trees])
        return probas.mean(axis=0)


# ── Metrics ───────────────────────────────────────────────────────────────────
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

def classification_report_dict(y_true, y_pred):
    report = {}
    for cls in [0, 1]:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        report[str(cls)] = {
            'precision': round(float(precision), 3),
            'recall': round(float(recall), 3),
            'f1-score': round(float(f1), 3),
            'support': int(np.sum(y_true == cls))
        }
    return report


# ── Train ─────────────────────────────────────────────────────────────────────
def train_model(symbol):
    print(f"[Trainer] Fetching data for {symbol}...")
    df = fetch_features(symbol)

    X = df[FEATURE_COLS].values.astype(float)
    y_clf = df['Target_clf'].values.astype(int)
    y_reg = df['Target_reg'].values.astype(float)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    yc_train, yc_test = y_clf[:split], y_clf[split:]
    yr_train, yr_test = y_reg[:split], y_reg[split:]

    scaler = NumpyScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"[Trainer] Training Random Forest ({len(X_train)} samples)...")
    clf = NumpyRandomForest(n_trees=40, max_depth=8)
    clf.fit(X_train_s, yc_train)
    yc_pred = clf.predict(X_test_s)
    clf_acc = round(float(accuracy_score(yc_test, yc_pred)) * 100, 2)
    clf_report = classification_report_dict(yc_test, yc_pred)

    print(f"[Trainer] Training Linear Regression...")
    reg = NumpyLinearRegression()
    reg.fit(X_train_s, yr_train)
    yr_pred = reg.predict(X_test_s)
    mae = round(float(mean_absolute_error(yr_test, yr_pred)), 4)
    r2 = round(float(r2_score(yr_test, yr_pred)), 4)

    feat_importance = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in clf.feature_importances_]))

    n = min(30, len(yr_test))
    actual_vs_pred = {
        'dates': [str(df.index[-len(yr_test) + i].date()) for i in range(n)],
        'actual': [round(float(v), 2) for v in yr_test[:n]],
        'predicted': [round(float(v), 2) for v in yr_pred[:n]]
    }

    model_data = {
        'symbol': symbol,
        'clf': clf,
        'reg': reg,
        'scaler': scaler,
        'feature_cols': FEATURE_COLS,
        'metrics': {
            'accuracy': clf_acc,
            'mae': mae,
            'r2': r2,
            'clf_report': clf_report,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'total_samples': len(X),
        },
        'feat_importance': feat_importance,
        'actual_vs_pred': actual_vs_pred,
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_close': round(float(df['Close'].iloc[-1]), 2),
    }

    path = os.path.join(MODELS_DIR, f"{symbol.replace('.', '_')}_model.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"[Trainer] {symbol} — Accuracy: {clf_acc}% | MAE: {mae} | R²: {r2}")
    return model_data


def load_model(symbol):
    path = os.path.join(MODELS_DIR, f"{symbol.replace('.', '_')}_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict(symbol):
    model_data = load_model(symbol)
    if not model_data:
        model_data = train_model(symbol)

    df = fetch_features(symbol)
    latest = df[FEATURE_COLS].iloc[-1].values.astype(float).reshape(1, -1)
    latest_s = model_data['scaler'].transform(latest)

    direction = int(model_data['clf'].predict(latest_s)[0])
    prob = model_data['clf'].predict_proba(latest_s)[0]
    next_price = round(float(model_data['reg'].predict(latest_s)[0]), 2)
    confidence = round(float(max(prob)) * 100, 2)

    return {
        'symbol': symbol,
        'current_price': model_data['last_close'],
        'predicted_price': next_price,
        'direction': 'UP' if direction == 1 else 'DOWN',
        'confidence': confidence,
        'metrics': model_data['metrics'],
        'trained_at': model_data['trained_at'],
    }


if __name__ == '__main__':
    result = train_model('AAPL')
    print(result['metrics'])
