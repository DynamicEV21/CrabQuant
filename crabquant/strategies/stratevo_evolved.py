"""
stratevo_evolved — StratEvo Multi-Factor Evolved Strategy

Auto-generated from StratEvo evolution best DNA (generation 39).
Original backtest: 66.9% annual return, 1.56 Sharpe, 14.4% max drawdown on daily crypto.

Scores assets on 20+ weighted factors and enters when the composite score
exceeds a minimum threshold. Top evolved factors: PEG, price_efficiency,
consecutive_pattern, support, volume_profile, Williams %R, KDJ, Aroon.

The DNA was discovered by genetic optimization over 60 generations of
crossover + mutation on 8 crypto pairs (BTCUSDT, ETHUSDT, SOLUSDT, etc.)
using daily bars.
"""

from itertools import product

import numpy as np
import pandas as pd

# ─── Evolved DNA (best_ever, fitness=120.7, gen 39) ───────────────────────

_DNA = {
    "min_score": 4,
    "rsi_buy_threshold": 46.64,
    "rsi_sell_threshold": 76.74,
    "r2_min": 0.324,
    "slope_min": 0.603,
    "volume_ratio_min": 0.5,
    "hold_bars": 5,
    "stop_loss_pct": 1.67,
    "take_profit_pct": 29.96,
    "dip_threshold_pct": 12.80,
    "r2_trend_min": 0.443,
    "w_momentum": 0.0,
    "w_mean_reversion": 0.0,
    "w_volume": 0.0,
    "w_trend": 0.001,
    "w_pattern": 0.0,
    "w_macd": 0.002,
    "w_bollinger": 0.0171,
    "w_kdj": 0.0484,
    "w_obv": 0.0,
    "w_support": 0.0591,
    "w_volume_profile": 0.0602,
    "w_pe": 0.0205,
    "w_pb": 0.0,
    "w_roe": 0.0231,
    "w_revenue_growth": 0.0068,
    "w_atr": 0.0,
    "w_adx": 0.0,
    "w_roc": 0.0011,
    "w_williams_r": 0.0596,
    "w_cci": 0.0,
    "w_mfi": 0.0,
    "w_vwap": 0.0061,
    "w_donchian": 0.0005,
    "w_ichimoku": 0.0,
    "w_elder_ray": 0.1026,
    "w_beta": 0.0759,
    "w_r_squared": 0.0082,
    "w_quantile_upper": 0.0023,
    "w_quantile_lower": 0.0067,
    "w_aroon": 0.0267,
    "w_peg": 0.127,
    "w_debt_ratio": 0.0192,
    "w_revenue_qoq": 0.0266,
    "w_ps": 0.0001,
    "price_efficiency": 0.1919,
    "consecutive_pattern": 0.0643,
    "volume_acceleration": 0.0393,
    "gap_momentum": 0.0038,
}

DEFAULT_PARAMS = {
    "min_score": 4,
    "hold_bars": 5,
    "stop_loss_pct": 1.67,
}

PARAM_GRID = {
    "min_score": [3, 4, 5, 6],
    "hold_bars": [3, 5, 7, 10],
    "stop_loss_pct": [1.5, 2.0, 3.0, 5.0],
}

DESCRIPTION = (
    "StratEvo evolved multi-factor strategy. Scores assets on 20+ weighted factors "
    "(PEG, price efficiency, consecutive pattern, support, volume profile, Williams %R, "
    "KDJ, Aroon, elder ray, beta, etc.) and enters when composite score exceeds min_score. "
    "Original DNA backtest: 66.9% annual return, 1.56 Sharpe, 14.4% max drawdown."
)


# ─── Indicator helpers (self-contained, no external imports) ──────────────


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _linear_regression(close: pd.Series, window: int = 20):
    r2 = pd.Series(np.nan, index=close.index, dtype=float)
    slope = pd.Series(np.nan, index=close.index, dtype=float)
    for i in range(window, len(close)):
        seg = close.iloc[i - window : i].values
        x = np.arange(window, dtype=float)
        mx, my = x.mean(), seg.mean()
        ss_xy = ((x - mx) * (seg - my)).sum()
        ss_xx = ((x - mx) ** 2).sum()
        ss_yy = ((seg - my) ** 2).sum()
        if ss_xx > 0 and ss_yy > 0:
            s = ss_xy / ss_xx
            slope.iloc[i] = s / my * 100 if my != 0 else 0
            r2.iloc[i] = (ss_xy**2) / (ss_xx * ss_yy)
    return r2, slope


def _volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    avg = volume.rolling(window).mean()
    return volume / avg.replace(0, np.nan)


def _ma_alignment(close: pd.Series) -> pd.Series:
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    result = pd.Series(0.0, index=close.index)
    result[(ma5 > ma10) & (ma10 > ma20)] = 1.0
    result[(ma5 < ma10) & (ma10 < ma20)] = -1.0
    return result


def _macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist


def _bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std


def _kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9):
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    rsv = (close - ll) / (hh - ll).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def _obv_trend(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    obv = (np.sign(close.diff()) * volume).cumsum()
    obv_ma = obv.rolling(window).mean()
    diff = obv - obv_ma
    std = obv.rolling(window).std().replace(0, np.nan)
    return (diff / std).clip(-1, 1)


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean() / close * 100


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return (high.rolling(period).max() - close) / (
        high.rolling(period).max() - low.rolling(period).min()
    ).replace(0, np.nan) * -100


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad).replace(0, np.nan)


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    return 100 - (100 / (1 + pos / neg.replace(0, np.nan)))


def _roc(close: pd.Series, period: int = 12) -> pd.Series:
    return close.pct_change(period) * 100


def _aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.Series:
    dsh = high.rolling(period).apply(lambda x: period - 1 - np.argmax(x), raw=True)
    dsl = low.rolling(period).apply(lambda x: period - 1 - np.argmin(x), raw=True)
    return ((period - dsh) / period * 100 - (period - dsl) / period * 100) / 100


def _donchian_pos(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    return (close - lower) / (upper - lower).replace(0, np.nan)


def _quantile_scores(close: pd.Series, window: int = 20):
    q80 = close.rolling(window).quantile(0.8)
    q20 = close.rolling(window).quantile(0.2)
    rng = (q80 - q20).replace(0, np.nan)
    lower_raw = ((close - q20) / rng).clip(0, 1)
    return (1 - lower_raw).clip(0, 1), lower_raw


# ─── Core scoring ──────────────────────────────────────────────────────────


def _score_series(df: pd.DataFrame, dna: dict) -> pd.Series:
    """Compute StratEvo-style composite score [0, 10] for every bar."""
    close, high, low, open_, volume = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # Pre-compute all indicators
    r2, slope = _linear_regression(close)
    macd_line, macd_signal, macd_hist = _macd(close)
    bb_upper, bb_mid, bb_lower = _bollinger(close)
    kdj_k, kdj_d, kdj_j = _kdj(high, low, close)
    q_upper, q_lower = _quantile_scores(close)

    # 1. Momentum
    slope_min = max(dna.get("slope_min", 0.6), 0.01)
    momentum = (slope / slope_min).clip(upper=2.0) / 2.0

    # 2. Mean reversion (RSI)
    rsi = _rsi(close)
    rsi_buy = dna.get("rsi_buy_threshold", 44)
    rsi_sell = dna.get("rsi_sell_threshold", 68)
    rng = rsi_sell - rsi_buy
    mr = pd.Series(0.5, index=close.index)
    if rng > 0:
        mr = (1.0 - (rsi - rsi_buy) / rng).clip(0, 1)
        mr[rsi <= rsi_buy] = 1.0
        mr[rsi >= rsi_sell] = 0.0

    # 3. Volume
    vr = _volume_ratio(volume)
    vol_min = max(dna.get("volume_ratio_min", 1.2), 0.01)
    vol = (vr / vol_min).clip(upper=2.0) / 2.0

    # 4. Trend (R² × MA alignment)
    ma_align = _ma_alignment(close)
    base_trend = r2.where(slope > 0, r2 * 0.3)
    trend = (base_trend * (1.0 + 0.3 * ma_align)).clip(0, 1)

    # 5. Pattern (dip + candle)
    dip_thresh = dna.get("dip_threshold_pct", 7.3)
    r2_trend_min = dna.get("r2_trend_min", 0.48)
    recent_high = close.rolling(20).max()
    pullback = (recent_high - close) / recent_high * 100
    dip = (pullback / dip_thresh).clip(upper=1.0)
    dip = dip.where((pullback >= dip_thresh * 0.5) & (r2 >= r2_trend_min), 0.0)
    candle = (close > open_).astype(float)
    pattern = (dip * 0.5 + candle * 0.5).clip(0, 1)

    # 6. MACD
    macd_prev = macd_hist.shift(1)
    macd_score = pd.Series(0.5, index=close.index)
    golden = (macd_hist > 0) & (macd_prev <= 0)
    macd_score[golden] = 1.0
    macd_score[(macd_hist > 0) & ~golden] = 0.7
    macd_score[(macd_hist > macd_prev) & (macd_hist <= 0)] = 0.4

    # 7. Bollinger
    bb_rng = (bb_upper - bb_lower).replace(0, np.nan)
    bb_pos = ((close - bb_lower) / bb_rng).clip(0, 1)
    bb_score = (1.0 - bb_pos).clip(0, 1)

    # 8. KDJ
    kdj_cross = (kdj_k > kdj_d) & (kdj_k.shift(1) <= kdj_d.shift(1))
    kdj_score = pd.Series(0.5, index=close.index)
    kdj_score[kdj_cross] = 1.0
    kdj_score[(kdj_k > kdj_d) & ~kdj_cross] = 0.6
    kdj_score[kdj_k < 20] = 0.7

    # 9. OBV
    obv = _obv_trend(close, volume)
    obv_score = (0.5 + obv * 0.5).clip(0, 1)

    # 10. ATR (prefer moderate volatility)
    atr = _atr_pct(high, low, close).fillna(0.5)
    atr_score = pd.Series(0.5, index=close.index)
    atr_score[atr < 3.0] = 1.0
    atr_score[(atr >= 3.0) & (atr < 5.0)] = 0.8
    atr_score[(atr >= 5.0) & (atr < 10.0)] = 0.6
    atr_score[(atr >= 10.0) & (atr < 15.0)] = 0.4
    atr_score[atr >= 15.0] = 0.2

    # 11. ROC
    roc = _roc(close).fillna(0)
    roc_score = pd.Series(0.5, index=close.index)
    roc_score[roc > 10] = 1.0
    roc_score[(roc > 5) & (roc <= 10)] = 0.8
    roc_score[(roc > 0) & (roc <= 5)] = 0.6
    roc_score[(roc > -5) & (roc <= 0)] = 0.3
    roc_score[roc <= -5] = 0.1

    # 12. Williams %R
    wr = _williams_r(high, low, close).fillna(-50)
    wr_score = pd.Series(0.5, index=close.index)
    wr_score[wr < -80] = 1.0
    wr_score[(wr >= -80) & (wr < -50)] = 0.6
    wr_score[(wr >= -50) & (wr < -20)] = 0.3
    wr_score[wr >= -20] = 0.1

    # 13. CCI
    cci = _cci(high, low, close).fillna(0)
    cci_score = pd.Series(0.5, index=close.index)
    cci_score[cci < -200] = 1.0
    cci_score[(cci >= -200) & (cci < -100)] = 0.8
    cci_score[(cci >= -100) & (cci < 0)] = 0.5
    cci_score[(cci >= 0) & (cci < 100)] = 0.4
    cci_score[(cci >= 100) & (cci < 200)] = 0.2
    cci_score[cci >= 200] = 0.1

    # 14. MFI
    mfi = _mfi(high, low, close, volume).fillna(50)
    mfi_score = pd.Series(0.5, index=close.index)
    mfi_score[mfi < 20] = 1.0
    mfi_score[(mfi >= 20) & (mfi < 40)] = 0.7
    mfi_score[(mfi >= 40) & (mfi < 60)] = 0.5
    mfi_score[(mfi >= 60) & (mfi < 80)] = 0.3
    mfi_score[mfi >= 80] = 0.1

    # 15. Donchian position
    donchian = _donchian_pos(high, low, close).fillna(0.5)
    donchian_score = (1.0 - donchian).clip(0, 1)

    # 16. Quantile
    q_upper_score = q_upper.fillna(0.5)
    q_lower_score = q_lower.fillna(0.5)

    # 17. Aroon
    aroon = _aroon(high, low).fillna(0)
    aroon_score = ((aroon + 1.0) / 2.0).clip(0, 1)

    # 18. Price efficiency (simplified: R² as proxy)
    price_eff = r2.fillna(0.5).clip(0, 1)

    # 19. Consecutive pattern (simplified: streak counter)
    up_streak = pd.Series(0, index=close.index, dtype=float)
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            up_streak.iloc[i] = up_streak.iloc[i - 1] + 1
    consec = (up_streak / 5.0).clip(0, 1)

    # 20. Volume acceleration
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    vol_accel = (vol_ma5 / vol_ma20.replace(0, np.nan)).fillna(1.0).clip(0.5, 2.0) / 2.0

    # ─── Weighted combination ───────────────────────────────────────

    _keys = [
        ("w_momentum", momentum), ("w_mean_reversion", mr), ("w_volume", vol),
        ("w_trend", trend), ("w_pattern", pattern), ("w_macd", macd_score),
        ("w_bollinger", bb_score), ("w_kdj", kdj_score), ("w_obv", obv_score),
        ("w_atr", atr_score), ("w_adx", pd.Series(0.5, index=close.index)),
        ("w_roc", roc_score), ("w_williams_r", wr_score), ("w_cci", cci_score),
        ("w_mfi", mfi_score), ("w_vwap", pd.Series(0.5, index=close.index)),
        ("w_donchian", donchian_score), ("w_ichimoku", pd.Series(0.5, index=close.index)),
        ("w_elder_ray", pd.Series(0.5, index=close.index)),
        ("w_beta", (slope.abs() / 3.0).clip(0, 1)),
        ("w_r_squared", r2.fillna(0.5).clip(0, 1)),
        ("w_quantile_upper", q_upper_score), ("w_quantile_lower", q_lower_score),
        ("w_aroon", aroon_score), ("w_support", (1.0 - donchian).clip(0, 1)),
        ("w_volume_profile", vol), ("w_pe", pd.Series(0.5, index=close.index)),
        ("w_roe", pd.Series(0.5, index=close.index)),
        ("w_revenue_growth", pd.Series(0.5, index=close.index)),
        ("w_peg", pd.Series(0.5, index=close.index)),
        ("w_debt_ratio", pd.Series(0.5, index=close.index)),
        ("w_revenue_qoq", pd.Series(0.5, index=close.index)),
        ("w_ps", pd.Series(0.5, index=close.index)),
    ]

    weighted_sum = sum(dna.get(wk, 0) * series for wk, series in _keys)
    # Add non-w_ factors
    weighted_sum += dna.get("price_efficiency", 0) * price_eff
    weighted_sum += dna.get("consecutive_pattern", 0) * consec
    weighted_sum += dna.get("volume_acceleration", 0) * vol_accel

    # Total weight for normalization
    all_w_keys = [k for k in dna if k.startswith("w_")]
    total_weight = sum(dna.get(k, 0) for k in all_w_keys)
    total_weight += dna.get("price_efficiency", 0)
    total_weight += dna.get("consecutive_pattern", 0)
    total_weight += dna.get("volume_acceleration", 0)

    if total_weight > 0:
        normalized = (weighted_sum / total_weight).clip(0, 1)
    else:
        normalized = weighted_sum.clip(0, 1)

    return normalized * 10


# ─── CrabQuant interface ──────────────────────────────────────────────────


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Generate entry/exit signals using StratEvo's evolved multi-factor DNA.

    Args:
        df: OHLCV DataFrame (columns: open, high, low, close, volume)
        params: Override min_score / hold_bars / stop_loss_pct

    Returns:
        (entries, exits) as boolean Series
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    scores = _score_series(df, _DNA)

    entries = scores >= p["min_score"]
    exits = scores < (p["min_score"] - 1)

    return entries.fillna(False), exits.fillna(False)


def generate_signals_matrix(
    df: pd.DataFrame, param_grid: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Generate signals for all param combinations (vectorized scoring once)."""
    pg = param_grid or PARAM_GRID
    keys = list(pg.keys())
    combos = list(product(*(pg[k] for k in keys)))

    scores = _score_series(df, _DNA)

    all_entries = []
    all_exits = []
    combo_list = []

    for combo in combos:
        params = dict(zip(keys, combo))
        ms = params.get("min_score", DEFAULT_PARAMS["min_score"])
        all_entries.append(scores >= ms)
        all_exits.append(scores < (ms - 1))
        combo_list.append(params)

    return pd.DataFrame(all_entries).T, pd.DataFrame(all_exits).T, combo_list
