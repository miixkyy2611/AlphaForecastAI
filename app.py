from __future__ import annotations

import math
import os
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request
from matplotlib.ticker import FuncFormatter
from pandas.tseries.offsets import BDay
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__)


@dataclass
class ForecastResult:
    ticker: str
    lookback: int
    horizon: int
    last_close: float
    train_samples: int
    val_samples: int
    test_samples: int
    weights: Dict[str, float]
    metrics: Dict[str, float]
    future_df: pd.DataFrame
    plot_paths: Dict[str, str]
    feature_importance_table: List[Tuple[str, float]]
    headline: str
    explanation: str


def safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: str, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(min_value, min(parsed, max_value))


def percent_fmt(x: float) -> str:
    return f"{x:.2f}%"


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df


def fetch_stock_data(ticker: str, start: str = '2018-01-01') -> pd.DataFrame:
    end = (dt.datetime.now() + dt.timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = normalize_columns(df)
    required = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if df.empty or not required.issubset(df.columns):
        raise ValueError('No se pudieron descargar datos OHLCV completos para ese ticker.')

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    close = df['Close']
    volume = df['Volume'].replace(0, np.nan)

    df['Return_1'] = close.pct_change()
    df['LogReturn'] = np.log(close / close.shift(1))
    df['RangePct'] = (df['High'] - df['Low']) / close
    df['GapPct'] = (df['Open'] - close.shift(1)) / close.shift(1)
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA50'] = close.rolling(50).mean()
    df['EMA12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA26'] = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACDSignal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['RSI14'] = compute_rsi(close, 14)
    df['Volatility20'] = df['LogReturn'].rolling(20).std()
    df['ATR14'] = compute_atr(df, 14)
    rolling_std20 = close.rolling(20).std()
    df['BBUpper'] = df['MA20'] + (2 * rolling_std20)
    df['BBLower'] = df['MA20'] - (2 * rolling_std20)
    bb_width = (df['BBUpper'] - df['BBLower']).replace(0, np.nan)
    df['BBPosition'] = ((close - df['BBLower']) / bb_width).clip(-5, 5)
    df['Momentum5'] = close.pct_change(5)
    df['Momentum10'] = close.pct_change(10)
    df['VolumeZ20'] = ((volume - volume.rolling(20).mean()) / volume.rolling(20).std()).replace([np.inf, -np.inf], np.nan)
    df['VolChange'] = volume.pct_change()
    df['CloseVsMA20'] = close / df['MA20'] - 1
    df['CloseVsMA50'] = close / df['MA50'] - 1
    df['TrendSlope10'] = close.rolling(10).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / max(np.mean(x), 1e-9), raw=False)
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


BASE_FEATURES = [
    'Close', 'Volume', 'Return_1', 'LogReturn', 'RangePct', 'GapPct',
    'MA10', 'MA20', 'MA50', 'EMA12', 'EMA26', 'MACD', 'MACDSignal',
    'RSI14', 'Volatility20', 'ATR14', 'BBPosition', 'Momentum5',
    'Momentum10', 'VolumeZ20', 'VolChange', 'CloseVsMA20', 'CloseVsMA50',
    'TrendSlope10', 'DayOfWeek', 'Month'
]


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    current_close: np.ndarray
    future_close: np.ndarray
    forecast_dates: List[pd.DatetimeIndex]
    anchor_dates: List[pd.Timestamp]
    feature_names: List[str]



def build_supervised_dataset(df: pd.DataFrame, lookback: int, horizon: int) -> DatasetBundle:
    if len(df) < lookback + horizon + 100:
        raise ValueError('No hay suficientes datos históricos para ese lookback y horizonte. Reduce uno de los dos.')

    feature_rows: List[np.ndarray] = []
    target_rows: List[np.ndarray] = []
    current_close_rows: List[float] = []
    future_close_rows: List[np.ndarray] = []
    forecast_dates: List[pd.DatetimeIndex] = []
    anchor_dates: List[pd.Timestamp] = []

    feature_names = []
    for lag in range(lookback, 0, -1):
        for feat in BASE_FEATURES:
            feature_names.append(f'{feat}_lag{lag}')
    summary_feature_names = [
        'close_mean_lookback', 'close_std_lookback', 'close_min_lookback',
        'close_max_lookback', 'volume_mean_lookback', 'return_mean_lookback',
        'return_std_lookback', 'rsi_mean_lookback', 'macd_mean_lookback',
        'trend_slope_lookback'
    ]
    feature_names.extend(summary_feature_names)

    close = df['Close']

    for i in range(lookback, len(df) - horizon):
        window = df.iloc[i - lookback:i]
        future = df.iloc[i:i + horizon]
        current_close = float(close.iloc[i - 1])

        flat_window = window[BASE_FEATURES].to_numpy(dtype=float).reshape(-1)
        close_window = window['Close'].to_numpy(dtype=float)
        volume_window = window['Volume'].to_numpy(dtype=float)
        return_window = window['LogReturn'].to_numpy(dtype=float)
        rsi_window = window['RSI14'].to_numpy(dtype=float)
        macd_window = window['MACD'].to_numpy(dtype=float)

        trend_slope = np.polyfit(np.arange(len(close_window)), close_window, 1)[0] / max(np.mean(close_window), 1e-9)
        summary = np.array([
            close_window.mean(),
            close_window.std(ddof=0),
            close_window.min(),
            close_window.max(),
            volume_window.mean(),
            return_window.mean(),
            return_window.std(ddof=0),
            rsi_window.mean(),
            macd_window.mean(),
            trend_slope,
        ], dtype=float)

        next_closes = future['Close'].to_numpy(dtype=float)
        prev_steps = np.concatenate([[current_close], next_closes[:-1]])
        stepwise_log_returns = np.log(next_closes / prev_steps)

        feature_rows.append(np.concatenate([flat_window, summary]))
        target_rows.append(stepwise_log_returns)
        current_close_rows.append(current_close)
        future_close_rows.append(next_closes)
        forecast_dates.append(future.index)
        anchor_dates.append(df.index[i - 1])

    return DatasetBundle(
        X=np.vstack(feature_rows),
        y=np.vstack(target_rows),
        current_close=np.array(current_close_rows, dtype=float),
        future_close=np.vstack(future_close_rows),
        forecast_dates=forecast_dates,
        anchor_dates=anchor_dates,
        feature_names=feature_names,
    )


@dataclass
class SplitData:
    train: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.DatetimeIndex], List[pd.Timestamp]]
    val: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.DatetimeIndex], List[pd.Timestamp]]
    test: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.DatetimeIndex], List[pd.Timestamp]]



def split_dataset(data: DatasetBundle) -> SplitData:
    n = len(data.X)
    train_end = max(int(n * 0.7), 1)
    val_end = max(int(n * 0.85), train_end + 1)
    if n - val_end < 15:
        raise ValueError('No hay suficientes muestras para evaluación temporal robusta. Usa un horizonte o lookback menor.')

    def take(start: int, end: int):
        return (
            data.X[start:end],
            data.y[start:end],
            data.current_close[start:end],
            data.future_close[start:end],
            data.forecast_dates[start:end],
            data.anchor_dates[start:end],
        )

    return SplitData(
        train=take(0, train_end),
        val=take(train_end, val_end),
        test=take(val_end, n),
    )



def train_models(X_train: np.ndarray, y_train: np.ndarray):
    gbr = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss='squared_error',
            learning_rate=0.035,
            max_depth=6,
            max_iter=350,
            min_samples_leaf=20,
            l2_regularization=0.05,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
    )
    et = ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features='sqrt',
    )

    gbr.fit(X_train, y_train)
    et.fit(X_train, y_train)
    return {'GradientBoosting': gbr, 'ExtraTrees': et}



def predict_weighted(models: Dict[str, object], X: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
    pred = None
    for name, model in models.items():
        current = model.predict(X)
        if pred is None:
            pred = np.zeros_like(current, dtype=float)
        pred += current * weights[name]
    return pred



def inverse_price_paths(current_close: np.ndarray, predicted_log_returns: np.ndarray) -> np.ndarray:
    current = np.asarray(current_close, dtype=float).reshape(-1, 1)
    cumulative = np.cumsum(predicted_log_returns, axis=1)
    return current * np.exp(cumulative)



def compute_metrics(actual_prices: np.ndarray, predicted_prices: np.ndarray) -> Dict[str, float]:
    actual_flat = actual_prices.reshape(-1)
    pred_flat = predicted_prices.reshape(-1)
    one_step_actual = actual_prices[:, 0]
    one_step_pred = predicted_prices[:, 0]

    def rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    mape_flat = np.mean(np.abs((actual_flat - pred_flat) / np.clip(np.abs(actual_flat), 1e-9, None))) * 100
    mape_step = np.mean(np.abs((one_step_actual - one_step_pred) / np.clip(np.abs(one_step_actual), 1e-9, None))) * 100
    directional_accuracy = np.mean(
        np.sign(np.diff(np.column_stack([one_step_actual, actual_prices[:, 1]]), axis=1)[:, 0])
        == np.sign(np.diff(np.column_stack([one_step_pred, predicted_prices[:, 1]]), axis=1)[:, 0])
    ) * 100 if actual_prices.shape[1] > 1 else np.nan

    return {
        'mae_1d': float(mean_absolute_error(one_step_actual, one_step_pred)),
        'rmse_1d': rmse(one_step_actual, one_step_pred),
        'mape_1d': float(mape_step),
        'r2_1d': float(r2_score(one_step_actual, one_step_pred)),
        'mae_all': float(mean_absolute_error(actual_flat, pred_flat)),
        'rmse_all': rmse(actual_flat, pred_flat),
        'mape_all': float(mape_flat),
        'r2_all': float(r2_score(actual_flat, pred_flat)),
        'directional_accuracy_h2': float(directional_accuracy) if not np.isnan(directional_accuracy) else 0.0,
    }



def calculate_weights(models: Dict[str, object], X_val: np.ndarray, y_val: np.ndarray, current_close_val: np.ndarray) -> Dict[str, float]:
    scores = {}
    for name, model in models.items():
        pred_returns = model.predict(X_val)
        pred_prices = inverse_price_paths(current_close_val, pred_returns)
        actual_prices = inverse_price_paths(current_close_val, y_val)
        mae = mean_absolute_error(actual_prices.reshape(-1), pred_prices.reshape(-1))
        scores[name] = max(mae, 1e-6)

    inv = {name: 1.0 / mae for name, mae in scores.items()}
    total = sum(inv.values())
    return {name: value / total for name, value in inv.items()}



def aggregate_feature_importance(models: Dict[str, object], feature_names: List[str]) -> List[Tuple[str, float]]:
    contributions = {name: 0.0 for name in BASE_FEATURES}

    et_model = models['ExtraTrees']
    if hasattr(et_model, 'feature_importances_'):
        importances = et_model.feature_importances_
        for feature_name, importance in zip(feature_names, importances):
            matched = None
            for base in BASE_FEATURES:
                if feature_name.startswith(base + '_'):
                    matched = base
                    break
            if matched is not None:
                contributions[matched] += float(importance)
    total = sum(contributions.values()) or 1.0
    ranked = sorted(((k, v / total) for k, v in contributions.items()), key=lambda x: x[1], reverse=True)
    return ranked[:10]





def apply_plot_theme(fig, ax):
    fig.patch.set_facecolor('#0b1120')
    ax.set_facecolor('#0f172a')
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.tick_params(colors='#dbeafe', labelsize=10)
    ax.xaxis.label.set_color('#cbd5e1')
    ax.yaxis.label.set_color('#cbd5e1')
    ax.title.set_color('#f8fafc')
    ax.grid(color='#334155', alpha=0.30, linewidth=0.8)

def save_plot_backtest(anchor_dates: List[pd.Timestamp], actual_prices: np.ndarray, pred_prices: np.ndarray, ticker: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    apply_plot_theme(fig, ax)
    dates = pd.to_datetime(anchor_dates)
    ax.plot(dates, actual_prices[:, 0], label='Real (1 día adelante)', linewidth=2.4, color='#7dd3fc')
    ax.plot(dates, pred_prices[:, 0], label='Predicción (1 día adelante)', linewidth=2.15, color='#a78bfa')
    ax.set_title(f'{ticker} • Backtest 1 día adelante en test', pad=12, fontsize=14, weight='bold')
    ax.set_xlabel('Fecha ancla')
    ax.set_ylabel('Precio')
    legend = ax.legend(frameon=True, facecolor='#0f172a', edgecolor='#334155')
    for txt in legend.get_texts():
        txt.set_color('#e2e8f0')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    path = os.path.join(STATIC_DIR, 'backtest_plot.png')
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return 'backtest_plot.png'



def save_plot_forecast(last_history: pd.Series, future_df: pd.DataFrame, ticker: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    apply_plot_theme(fig, ax)
    ax.plot(last_history.index, last_history.values, label='Histórico reciente', linewidth=2.4, color='#7dd3fc')
    ax.plot(future_df.index, future_df['PredictedClose'], label='Forecast', linewidth=2.55, color='#f59e0b')
    ax.fill_between(
        future_df.index,
        future_df['Lower80'],
        future_df['Upper80'],
        color='#f59e0b',
        alpha=0.18,
        label='Banda 80%'
    )
    ax.fill_between(
        future_df.index,
        future_df['Lower95'],
        future_df['Upper95'],
        color='#c084fc',
        alpha=0.10,
        label='Banda 95%'
    )
    ax.set_title(f'{ticker} • Forecast multi-step directo', pad=12, fontsize=14, weight='bold')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio')
    legend = ax.legend(frameon=True, facecolor='#0f172a', edgecolor='#334155')
    for txt in legend.get_texts():
        txt.set_color('#e2e8f0')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    path = os.path.join(STATIC_DIR, 'forecast_plot.png')
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return 'forecast_plot.png'



def save_plot_model_weights(weights: Dict[str, float], metrics: Dict[str, float]) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    apply_plot_theme(fig, ax)
    names = list(weights.keys())
    values = [weights[n] * 100 for n in names]
    colors = ['#7dd3fc', '#a78bfa', '#f59e0b', '#34d399']
    bars = ax.bar(names, values, color=colors[:len(names)], edgecolor='#e2e8f0', linewidth=0.3)
    ax.set_title(f'Peso del ensemble • MAE global test {metrics["mae_all"]:.2f}', pad=12, fontsize=13, weight='bold')
    ax.set_ylabel('Peso (%)')
    ax.set_ylim(0, max(values) * 1.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.04, f'{value:.1f}%', ha='center', va='bottom', color='#f8fafc', fontsize=10)
    path = os.path.join(STATIC_DIR, 'weights_plot.png')
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return 'weights_plot.png'



def save_plot_feature_importance(feature_importance_table: List[Tuple[str, float]]) -> str:
    names = [n for n, _ in feature_importance_table][::-1]
    values = [v * 100 for _, v in feature_importance_table][::-1]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    apply_plot_theme(fig, ax)
    colors = ['#60a5fa', '#63b3ff', '#65c1ff', '#67cfff', '#69ddff', '#7ee7ff', '#a78bfa', '#c084fc', '#f59e0b', '#34d399']
    ax.barh(names, values, color=colors[:len(names)], edgecolor='#e2e8f0', linewidth=0.25)
    ax.set_title('Familias de features más influyentes (ExtraTrees)', pad=12, fontsize=13, weight='bold')
    ax.set_xlabel('Importancia relativa (%)')
    path = os.path.join(STATIC_DIR, 'feature_importance_plot.png')
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return 'feature_importance_plot.png'



def build_headline(last_close: float, first_forecast: float, last_forecast: float, horizon: int) -> Tuple[str, str]:
    total_change = ((last_forecast / last_close) - 1) * 100
    initial_change = ((first_forecast / last_close) - 1) * 100

    if total_change > 2:
        tone = 'Sesgo alcista moderado'
    elif total_change < -2:
        tone = 'Sesgo bajista moderado'
    else:
        tone = 'Sesgo lateral / indeciso'

    explanation = (
        f'El modelo estima un movimiento acumulado de {total_change:.2f}% a {horizon} días hábiles, '
        f'con el primer paso en {initial_change:.2f}% vs. el último cierre. '
        'Tómalo como señal cuantitativa experimental, no como recomendación de inversión.'
    )
    return tone, explanation



def run_forecast_pipeline(ticker: str, lookback: int, horizon: int) -> ForecastResult:
    df = fetch_stock_data(ticker)
    dataset = build_supervised_dataset(df, lookback=lookback, horizon=horizon)
    split = split_dataset(dataset)

    X_train, y_train, current_close_train, future_close_train, forecast_dates_train, anchor_train = split.train
    X_val, y_val, current_close_val, future_close_val, forecast_dates_val, anchor_val = split.val
    X_test, y_test, current_close_test, future_close_test, forecast_dates_test, anchor_test = split.test

    models = train_models(X_train, y_train)
    weights = calculate_weights(models, X_val, y_val, current_close_val)

    pred_returns_test = predict_weighted(models, X_test, weights)
    pred_prices_test = inverse_price_paths(current_close_test, pred_returns_test)
    actual_prices_test = inverse_price_paths(current_close_test, y_test)
    metrics = compute_metrics(actual_prices_test, pred_prices_test)

    latest_window = dataset.X[-1:]
    latest_close = float(df['Close'].iloc[-1])
    latest_pred_returns = predict_weighted(models, latest_window, weights)
    future_prices = inverse_price_paths(np.array([latest_close]), latest_pred_returns)[0]

    val_pred_returns = predict_weighted(models, X_val, weights)
    val_pred_prices = inverse_price_paths(current_close_val, val_pred_returns)
    val_actual_prices = inverse_price_paths(current_close_val, y_val)
    val_errors = val_pred_prices - val_actual_prices
    horizon_std = np.std(val_errors, axis=0, ddof=1)
    horizon_std = np.where(np.isnan(horizon_std) | (horizon_std < 1e-9), np.maximum(np.abs(future_prices) * 0.01, 0.5), horizon_std)

    last_date = df.index[-1]
    future_dates = pd.bdate_range(last_date + BDay(1), periods=horizon)
    future_df = pd.DataFrame({
        'PredictedClose': future_prices,
        'PredictedReturnPct': (future_prices / np.r_[latest_close, future_prices[:-1]] - 1) * 100,
        'Lower80': future_prices - 1.28 * horizon_std,
        'Upper80': future_prices + 1.28 * horizon_std,
        'Lower95': future_prices - 1.96 * horizon_std,
        'Upper95': future_prices + 1.96 * horizon_std,
    }, index=future_dates)

    headline, explanation = build_headline(latest_close, future_prices[0], future_prices[-1], horizon)

    feature_importance_table = aggregate_feature_importance(models, dataset.feature_names)

    recent_history = df['Close'].iloc[-max(90, lookback):]
    plot_paths = {
        'backtest': save_plot_backtest(anchor_test, actual_prices_test, pred_prices_test, ticker),
        'forecast': save_plot_forecast(recent_history, future_df, ticker),
        'weights': save_plot_model_weights(weights, metrics),
        'feature_importance': save_plot_feature_importance(feature_importance_table),
    }

    return ForecastResult(
        ticker=ticker.upper(),
        lookback=lookback,
        horizon=horizon,
        last_close=latest_close,
        train_samples=len(X_train),
        val_samples=len(X_val),
        test_samples=len(X_test),
        weights=weights,
        metrics=metrics,
        future_df=future_df,
        plot_paths=plot_paths,
        feature_importance_table=feature_importance_table,
        headline=headline,
        explanation=explanation,
    )


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/forecast', methods=['POST'])
def forecast():
    ticker = (request.form.get('ticker') or 'AAPL').strip().upper()
    lookback = safe_int(request.form.get('lookback', '60'), default=60, min_value=20, max_value=180)
    horizon = safe_int(request.form.get('horizon', '10'), default=10, min_value=3, max_value=30)

    try:
        result = run_forecast_pipeline(ticker=ticker, lookback=lookback, horizon=horizon)
        metrics = {
            'MAE 1D': f"{result.metrics['mae_1d']:.2f}",
            'RMSE 1D': f"{result.metrics['rmse_1d']:.2f}",
            'MAPE 1D': f"{result.metrics['mape_1d']:.2f}%",
            'R² 1D': f"{result.metrics['r2_1d']:.4f}",
            f'MAE {result.horizon}D': f"{result.metrics['mae_all']:.2f}",
            f'RMSE {result.horizon}D': f"{result.metrics['rmse_all']:.2f}",
            f'MAPE {result.horizon}D': f"{result.metrics['mape_all']:.2f}%",
            f'R² {result.horizon}D': f"{result.metrics['r2_all']:.4f}",
            'Directional Acc. H2': f"{result.metrics['directional_accuracy_h2']:.2f}%",
        }
        weights_pretty = {k: f'{v * 100:.1f}%' for k, v in result.weights.items()}
        future_rows = []
        for idx, row in result.future_df.iterrows():
            future_rows.append({
                'date': idx.strftime('%Y-%m-%d'),
                'predicted_close': f"{row['PredictedClose']:.2f}",
                'predicted_return': f"{row['PredictedReturnPct']:.2f}%",
                'lower80': f"{row['Lower80']:.2f}",
                'upper80': f"{row['Upper80']:.2f}",
                'lower95': f"{row['Lower95']:.2f}",
                'upper95': f"{row['Upper95']:.2f}",
            })
        feature_rows = [(name, f'{score * 100:.2f}%') for name, score in result.feature_importance_table]
        return render_template(
            'result.html',
            result=result,
            metrics=metrics,
            weights=weights_pretty,
            future_rows=future_rows,
            feature_rows=feature_rows,
        )
    except Exception as exc:
        error_detail = f'{type(exc).__name__}: {exc}'
        trace = traceback.format_exc(limit=2)
        return render_template('index.html', error=error_detail, trace=trace)


if __name__ == '__main__':
    app.run(debug=True)
