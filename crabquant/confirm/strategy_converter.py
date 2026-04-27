"""
Strategy Converter

Converts VectorBT strategy logic (vectorized generate_signals) into
backtesting.py Strategy classes (bar-by-bar iteration).

Each converter knows how to:
1. Compute indicators in init() using self.I()
2. Implement entry/exit logic in next()
3. Handle position sizing, stop-losses, edge cases
"""

from backtesting import Strategy
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: rolling function wrappers for self.I()
# backtesting.py self.I() needs callables, not pre-computed arrays.
# ---------------------------------------------------------------------------

def _rolling_max(series_array, window):
    """Rolling max computed incrementally on array slices."""
    out = np.full_like(series_array, np.nan, dtype=float)
    for i in range(window - 1, len(series_array)):
        out[i] = np.nanmax(series_array[i - window + 1:i + 1])
    return out


def _rolling_min(series_array, window):
    """Rolling min computed incrementally."""
    out = np.full_like(series_array, np.nan, dtype=float)
    for i in range(window - 1, len(series_array)):
        out[i] = np.nanmin(series_array[i - window + 1:i + 1])
    return out


def _rolling_mean(series_array, window):
    """Simple rolling mean."""
    out = np.full_like(series_array, np.nan, dtype=float)
    for i in range(window - 1, len(series_array)):
        out[i] = np.nanmean(series_array[i - window + 1:i + 1])
    return out


def _rolling_sum(series_array, window):
    """Simple rolling sum."""
    out = np.full_like(series_array, np.nan, dtype=float)
    for i in range(len(series_array)):
        start = max(0, i - window + 1)
        out[i] = np.nansum(series_array[start:i + 1])
    return out


def _ewm_mean(series_array, span):
    """Exponentially weighted mean matching pandas ewm(span=, adjust=False)."""
    alpha = 2.0 / (span + 1)
    out = np.full_like(series_array, np.nan, dtype=float)
    prev = np.nan
    for i in range(len(series_array)):
        val = series_array[i]
        if np.isnan(val):
            out[i] = prev
        elif np.isnan(prev):
            out[i] = val
            prev = val
        else:
            prev = alpha * val + (1 - alpha) * prev
            out[i] = prev
    return out


def _rsi(close_array, length):
    """RSI using wilder smoothing."""
    delta = np.diff(close_array, prepend=close_array[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # Initial averages
    avg_gain = np.nanmean(gain[1:length + 1]) if length + 1 <= len(gain) else np.nan
    avg_loss = np.nanmean(loss[1:length + 1]) if length + 1 <= len(loss) else np.nan

    out = np.full(len(close_array), np.nan)
    out[:length] = np.nan  # warmup

    if not np.isnan(avg_gain) and avg_loss == 0:
        out[length] = 100.0
    elif not np.isnan(avg_gain) and not np.isnan(avg_loss):
        out[length] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    for i in range(length + 1, len(close_array)):
        avg_gain = (avg_gain * (length - 1) + gain[i]) / length
        avg_loss = (avg_loss * (length - 1) + loss[i]) / length
        if avg_loss == 0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    return out


def _atr(high_array, low_array, close_array, length=14):
    """Average True Range using wilder smoothing."""
    tr = np.full(len(close_array), np.nan)
    for i in range(1, len(close_array)):
        hl = high_array[i] - low_array[i]
        hc = abs(high_array[i] - close_array[i - 1])
        lc = abs(low_array[i] - close_array[i - 1])
        tr[i] = max(hl, hc, lc)
    tr[0] = high_array[0] - low_array[0]

    out = np.full(len(close_array), np.nan)
    if length >= len(tr):
        return out

    out[length] = np.nanmean(tr[1:length + 1])
    for i in range(length + 1, len(tr)):
        out[i] = (out[i - 1] * (length - 1) + tr[i]) / length

    return out


def _adx(high_array, low_array, close_array, length=14):
    """ADX indicator."""
    tr = np.full(len(close_array), np.nan)
    plus_dm = np.full(len(close_array), 0.0)
    minus_dm = np.full(len(close_array), 0.0)

    for i in range(1, len(close_array)):
        up = high_array[i] - high_array[i - 1]
        down = low_array[i - 1] - low_array[i]
        plus_dm[i] = up if up > down and up > 0 else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0
        hl = high_array[i] - low_array[i]
        hc = abs(high_array[i] - close_array[i - 1])
        lc = abs(low_array[i] - close_array[i - 1])
        tr[i] = max(hl, hc, lc)
    tr[0] = high_array[0] - low_array[0]

    # Wilder smoothing
    out = np.full(len(close_array), np.nan)
    if length + 1 > len(close_array):
        return out

    atr_val = np.nanmean(tr[1:length + 1])
    smooth_plus = np.nanmean(plus_dm[1:length + 1])
    smooth_minus = np.nanmean(minus_dm[1:length + 1])

    if atr_val == 0:
        out[length] = 0.0
    else:
        plus_di = 100.0 * smooth_plus / atr_val
        minus_di = 100.0 * smooth_minus / atr_val
        di_sum = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum != 0 else 0.0
        out[length] = dx

    for i in range(length + 1, len(close_array)):
        atr_val = (atr_val * (length - 1) + tr[i]) / length
        smooth_plus = (smooth_plus * (length - 1) + plus_dm[i]) / length
        smooth_minus = (smooth_minus * (length - 1) + minus_dm[i]) / length
        if atr_val == 0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = 100.0 * smooth_plus / atr_val
            minus_di = 100.0 * smooth_minus / atr_val
        di_sum = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum != 0 else 0.0
        if i == length + 1:
            adx_val = dx
        else:
            adx_val = (adx_val * (length - 1) + dx) / length
        out[i] = adx_val

    return out


def _macd(close_array, fast=12, slow=26, signal=9):
    """MACD histogram only."""
    ema_fast = _ewm_mean(close_array, fast)
    ema_slow = _ewm_mean(close_array, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ewm_mean(macd_line, signal)
    hist = macd_line - signal_line
    return hist


def _stoch(high_array, low_array, close_array, k=14, d=3):
    """Stochastic K and D."""
    n = len(close_array)
    stoch_k = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - k + 1)
        hh = np.nanmax(high_array[start:i + 1])
        ll = np.nanmin(low_array[start:i + 1])
        if hh != ll:
            stoch_k[i] = 100.0 * (close_array[i] - ll) / (hh - ll)
        else:
            stoch_k[i] = 50.0
    stoch_d = _rolling_mean(stoch_k, d)
    return stoch_k, stoch_d


def _bbands(close_array, length=20, std_mult=2.0):
    """Bollinger Bands: upper, mid, lower."""
    mid = _rolling_mean(close_array, length)
    upper = np.full_like(close_array, np.nan)
    lower = np.full_like(close_array, np.nan)
    for i in range(len(close_array)):
        start = max(0, i - length + 1)
        window = close_array[start:i + 1]
        vals = window[~np.isnan(window)]
        if len(vals) > 1:
            upper[i] = np.mean(vals) + std_mult * np.std(vals, ddof=0)
            lower[i] = np.mean(vals) - std_mult * np.std(vals, ddof=0)
    return upper, mid, lower


def _roc(close_array, length):
    """Rate of Change as percentage."""
    out = np.full_like(close_array, np.nan)
    for i in range(length, len(close_array)):
        if close_array[i - length] != 0:
            out[i] = (close_array[i] - close_array[i - length]) / close_array[i - length] * 100
    return out


def _sma(series_array, length):
    """Simple moving average."""
    return _rolling_mean(series_array, length)


def _vpt(close_array, volume_array):
    """Volume Price Trend."""
    vpt_values = np.zeros(len(close_array))
    for i in range(1, len(close_array)):
        if close_array[i - 1] != 0:
            vpt_values[i] = vpt_values[i - 1] + volume_array[i] * (
                (close_array[i] - close_array[i - 1]) / close_array[i - 1]
            )
        else:
            vpt_values[i] = vpt_values[i - 1]
    return vpt_values


# ---------------------------------------------------------------------------
# Base class that all converted strategies inherit from
# ---------------------------------------------------------------------------

class CrabQuantBacktest(Strategy):
    """Base class for all converted CrabQuant strategies.

    Subclasses must set:
        _cq_params: dict of strategy parameters
        _cq_position_pct: fraction of portfolio to use per trade (default 0.95)
        _cq_slippage_pct: slippage as fraction (default 0.001)
    """

    _cq_params: dict = {}
    _cq_position_pct: float = 0.95
    _cq_slippage_pct: float = 0.001

    def init(self):
        """Compute all indicators. Override in subclasses."""
        pass

    def next(self):
        """Bar-by-bar logic. Override in subclasses."""
        pass

    def _safe_entry(self, condition: bool):
        """Enter long if condition is True and not already in position."""
        if condition and not self.position:
            # Adjust size for slippage
            size = self._cq_position_pct
            self.buy(size=size)

    def _safe_exit(self, condition: bool):
        """Exit long if condition is True and in position."""
        if condition and self.position:
            self.position.close()

    def _val(self, indicator, idx=-1):
        """Safely get indicator value at index, returns NaN if not ready."""
        v = indicator[idx]
        if np.isnan(v):
            return np.nan
        return v


def _make_strategy_class(name: str, params: dict, position_pct: float = 0.95,
                         slippage_pct: float = 0.001) -> type:
    """
    Factory: create a backtesting.py Strategy class for a given CrabQuant strategy.

    Args:
        name: Strategy name from STRATEGY_REGISTRY
        params: Strategy parameters dict
        position_pct: Fraction of portfolio per trade
        slippage_pct: Slippage fraction

    Returns:
        A Strategy subclass ready for backtesting.Backtest()
    """
    converter = _CONVERTERS.get(name)
    if converter is None:
        raise ValueError(f"No converter registered for strategy '{name}'. "
                         f"Available: {list(_CONVERTERS.keys())}")

    return converter(params, position_pct, slippage_pct)


# ---------------------------------------------------------------------------
# Individual strategy converters
# Each returns a Strategy subclass
# ---------------------------------------------------------------------------

def _convert_rsi_crossover(params, pos_pct, slip_pct):
    """RSI Crossover: fast/slow RSI cross with regime filter."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            self.rsi_fast = self.I(_rsi, c, p["fast_len"])
            self.rsi_slow = self.I(_rsi, c, p["slow_len"])
            self.regime = self.I(_rsi, c, p["regime_len"])

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            rf = self.rsi_fast[-2]
            rs = self.rsi_slow[-2]
            rf_now = self.rsi_fast[-1]
            rs_now = self.rsi_slow[-1]
            reg = self.regime[-1]
            exit_val = p["exit_level"]

            if any(np.isnan(v) for v in [rf, rs, rf_now, rs_now, reg]):
                return

            entry = (rf < rs) and (rf_now > rs_now) and (reg > p["regime_bull"])
            exit_cond = rf_now < exit_val

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_macd_momentum(params, pos_pct, slip_pct):
    """MACD Momentum: histogram momentum shift with trend and volume filter."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            v = self.data.Volume
            self.hist = self.I(_macd, c, p["macd_fast"], p["macd_slow"], p["macd_signal"])
            self.sma = self.I(_sma, c, p["sma_len"])
            self.vol_avg = self.I(_sma, v, p["volume_window"])

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 3:
                return
            h0 = self.hist[-1]
            h1 = self.hist[-2]
            h2 = self.hist[-3]
            sma = self.sma[-1]
            close = self.data.Close[-1]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]

            if any(np.isnan(v) for v in [h0, h1, h2, sma, va]):
                return

            trend_ok = close > sma
            vol_ok = vol > (va * p["volume_mult"])
            mom_cross = (h1 <= 0) and (h0 > 0)
            mom_strong = (h1 < h2) and (h0 > h1) and (h0 > 0)

            entry = trend_ok and vol_ok and (mom_cross or mom_strong)
            exit_cond = h0 < p["exit_hist"]

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_adx_pullback(params, pos_pct, slip_pct):
    """ADX Pullback: ADX trend + pullback to EMA entry, ATR take-profit exit."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            self.adx_val = self.I(_adx, h, l, c, p["adx_len"])
            self.ema = self.I(_ewm_mean, c, p["ema_len"])
            self.atr = self.I(_atr, h, l, c, 14)

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            adx = self.adx_val[-1]
            ema_now = self.ema[-1]
            ema_prev = self.ema[-2]
            close = self.data.Close[-1]
            atr = self.atr[-1]

            if any(np.isnan(v) for v in [adx, ema_now, ema_prev, atr]):
                return

            strong_trend = adx > p["adx_threshold"]
            pullback = (close < ema_now) and (self.data.Close[-2] >= ema_prev)
            take_profit = close > ema_now + atr * p["take_atr"]

            entry = strong_trend and pullback
            exit_cond = take_profit

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_atr_channel_breakout(params, pos_pct, slip_pct):
    """ATR Channel Breakout: Keltner Channel breakout with volume and trend."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.ema = self.I(_ewm_mean, c, p["ema_len"])
            self.atr = self.I(_atr, h, l, c, p["atr_len"])
            self.vol_avg = self.I(_sma, v, 20)
            self.trend_ema = self.I(_ewm_mean, c, 50)

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            close = self.data.Close[-1]
            ema = self.ema[-1]
            atr = self.atr[-1]
            ema_prev = self.ema[-2]
            atr_prev = self.atr[-2]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]
            trend = self.trend_ema[-1]

            if any(np.isnan(v) for v in [ema, atr, ema_prev, atr_prev, va, trend]):
                return

            upper_prev = ema_prev + atr_prev * p["mult"]
            breakout = close > upper_prev
            vol_ok = vol > (va * p["vol_mult"])
            trend_ok = close > trend

            entry = breakout and vol_ok and trend_ok
            exit_cond = close < ema

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_volume_breakout(params, pos_pct, slip_pct):
    """Volume Breakout: Donchian Channel breakout with volume spike."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.donch_high = self.I(_rolling_max, h, p["dc_len"])
            self.donch_low = self.I(_rolling_min, l, p["dc_len"])
            self.vol_avg = self.I(_sma, v, p["vol_len"])

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            close = self.data.Close[-1]
            dh = self.donch_high[-2]
            dl = self.donch_low[-1]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]

            if any(np.isnan(v) for v in [dh, dl, va]):
                return

            breakout = close > dh
            vol_spike = vol > (va * p["vol_mult"])
            dc_mid = (dh + dl) / 2

            entry = breakout and vol_spike
            exit_cond = close < dc_mid

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_multi_rsi_confluence(params, pos_pct, slip_pct):
    """Multi-RSI Confluence: three timeframe RSI oversold + turning up."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            v = self.data.Volume
            self.rsi1 = self.I(_rsi, c, p["rsi1"])
            self.rsi2 = self.I(_rsi, c, p["rsi2"])
            self.rsi3 = self.I(_rsi, c, p["rsi3"])
            self.vol_avg = self.I(_sma, v, 20)

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            r1 = self.rsi1[-1]
            r1_prev = self.rsi1[-2]
            r2 = self.rsi2[-1]
            r3 = self.rsi3[-1]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]

            if any(np.isnan(v) for v in [r1, r1_prev, r2, r3, va]):
                return

            all_oversold = (r1 < p["thresh"]) and (r2 < p["thresh"]) and (r3 < p["thresh"])
            turning = r1 > r1_prev
            vol_ok = vol > (va * p["vol_mult"])

            entry = all_oversold and turning and vol_ok
            exit_cond = r1 > p["exit_thresh"]

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_ema_ribbon_reversal(params, pos_pct, slip_pct):
    """EMA Ribbon: 10/20/30/50 EMA alignment with RSI dip entry."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            self.ema10 = self.I(_ewm_mean, c, 10)
            self.ema20 = self.I(_ewm_mean, c, 20)
            self.ema30 = self.I(_ewm_mean, c, 30)
            self.ema50 = self.I(_ewm_mean, c, 50)
            self.rsi = self.I(_rsi, c, 14)

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            e10 = self.ema10[-1]
            e20 = self.ema20[-1]
            e30 = self.ema30[-1]
            e50 = self.ema50[-1]
            rsi = self.rsi[-1]

            if any(np.isnan(v) for v in [e10, e20, e30, e50, rsi]):
                return

            aligned = (e10 > e20) and (e20 > e30) and (e30 > e50)
            dip = rsi < p["dip_level"]

            entry = aligned and dip
            exit_cond = (rsi > 60) or not (e10 > e20)

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_bollinger_squeeze(params, pos_pct, slip_pct):
    """Bollinger Squeeze: BB squeeze followed by breakout with volume."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            v = self.data.Volume
            self.bbu, self.bbm, self.bbl = self.I(_bbands, c, p["bb_len"], p["bb_std"])
            self.vol_avg = self.I(_sma, v, 20)

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < p["squeeze_len"] + 2:
                return
            close = self.data.Close[-1]
            bbu = self.bbu[-1]
            bbm = self.bbm[-1]
            bbl = self.bbl[-1]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]

            if any(np.isnan(v) for v in [bbu, bbm, bbl, va]):
                return

            # Compute BB width and its rolling average for squeeze detection
            if bbm == 0:
                return
            bb_width_now = (bbu - bbl) / bbm
            # Look back for squeeze
            widths = []
            for j in range(p["squeeze_len"]):
                idx = -(j + 1)
                bbl_v = self.bbl[idx]
                bbu_v = self.bbu[idx]
                bbm_v = self.bbm[idx]
                if np.isnan(bbl_v) or np.isnan(bbu_v) or np.isnan(bbm_v) or bbm_v == 0:
                    return
                widths.append((bbu_v - bbl_v) / bbm_v)
            avg_width = sum(widths) / len(widths)
            squeeze = widths[-2] < avg_width * p["squeeze_mult"] if len(widths) >= 2 else False

            breakout_up = close > bbu
            vol_ok = vol > (va * p["vol_mult"])

            entry = squeeze and breakout_up and vol_ok
            exit_cond = close < bbm

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_ichimoku_trend(params, pos_pct, slip_pct):
    """Ichimoku Trend: Tenkan/Kijun cross above cloud."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            h = self.data.High
            l = self.data.Low
            self.tenkan = self.I(_rolling_mean, (h + l) / 2, 9)
            self.kijun = self.I(_rolling_mean, (h + l) / 2, 26)
            # Span A/B shifted 26 bars — we compute full arrays and index
            self.span_a = self.I(lambda x: _rolling_mean((self.data.High.values + self.data.Low.values) / 2, 9), self.data.Close)
            self.span_b = self.I(lambda x: _rolling_mean((self.data.High.values + self.data.Low.values) / 2, 52), self.data.Close)

        def next(self):
            i = len(self.data) - 1
            if i < 52:
                return
            close = self.data.Close[-1]
            tenkan = self.tenkan[-1]
            tenkan_prev = self.tenkan[-2]
            kijun = self.kijun[-1]
            kijun_prev = self.kijun[-2]

            # Span A shifted 26 bars back
            span_a_idx = i - 26
            span_b_idx = i - 26
            if span_a_idx < 0:
                return
            sa = self.span_a[span_a_idx]
            sb = self.span_b[span_b_idx]

            if any(np.isnan(v) for v in [tenkan, tenkan_prev, kijun, kijun_prev, sa, sb]):
                return

            above_cloud = close > sa
            tk_cross = (tenkan_prev < kijun_prev) and (tenkan > kijun)

            entry = above_cloud and tk_cross
            exit_cond = close < sa

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_invented_momentum_rsi_atr(params, pos_pct, slip_pct):
    """Momentum RSI ATR: ROC+RSI pullback entry in uptrend, ATR trailing stop."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.roc = self.I(_roc, c, p["roc_len"])
            self.ema = self.I(_ewm_mean, c, p["ema_len"])
            self.atr = self.I(_atr, h, l, c, p["atr_len"])
            self.adx_val = self.I(_adx, h, l, c, 14)

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 3:
                return
            close = self.data.Close[-1]
            rsi = self.rsi[-1]
            rsi_prev = self.rsi[-2]
            rsi_prev2 = self.rsi[-3] if i >= 2 else np.nan
            roc = self.roc[-1]
            ema = self.ema[-1]
            atr = self.atr[-1]
            adx = self.adx_val[-1]

            if any(np.isnan(v) for v in [rsi, rsi_prev, roc, ema, atr, adx]):
                return

            trend_ok = close > ema
            momentum_ok = roc > p["roc_threshold"]
            adx_ok = adx > 20

            rsi_turning = (rsi_prev < p["rsi_pullback"]) and (rsi >= p["rsi_pullback"])
            rsi_rising = rsi > rsi_prev
            rsi_in_zone = p["rsi_pullback"] <= rsi <= 60
            rsi_prev2_ok = not np.isnan(rsi_prev2) and (rsi_prev2 < p["rsi_pullback"])
            rsi_entry = rsi_turning or (rsi_in_zone and rsi_rising and rsi_prev2_ok)

            atr_stop = ema - (p["atr_exit_mult"] * atr)
            atr_exit = close < atr_stop
            rsi_exit = rsi > p["rsi_overbought"]

            entry = trend_ok and momentum_ok and adx_ok and rsi_entry
            exit_cond = atr_exit or rsi_exit

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_invented_momentum_rsi_stoch(params, pos_pct, slip_pct):
    """Simple RSI oversold with volume spike entry."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            v = self.data.Volume
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.vol_avg = self.I(_sma, v, p["volume_window"])

        def next(self):
            p = self._cq_params
            rsi = self.rsi[-1]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]

            if any(np.isnan(v) for v in [rsi, va]):
                return

            entry = (rsi < p["rsi_oversold"]) and (vol > (va * p["volume_mult"]))
            exit_cond = rsi > 70

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_vpt_crossover(params, pos_pct, slip_pct):
    """VPT Crossover: VPT signal cross with RSI and volume filter."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            v = self.data.Volume
            self.vpt = self.I(_vpt, c, v)
            self.vpt_signal = self.I(lambda x: _rolling_mean(
                _vpt(self.data.Close.values, self.data.Volume.values),
                p["vpt_signal_len"]
            ), self.data.Close)
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.vol_sma = self.I(_sma, v, p["vol_sma_len"])

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            vpt_now = self.vpt[-1]
            vpt_prev = self.vpt[-2]
            sig_now = self.vpt_signal[-1]
            sig_prev = self.vpt_signal[-2]
            rsi = self.rsi[-1]
            vol = self.data.Volume[-1]
            vs = self.vol_sma[-1]

            if any(np.isnan(v) for v in [vpt_now, vpt_prev, sig_now, sig_prev, rsi, vs]):
                return

            entry = (vpt_prev <= sig_prev) and (vpt_now > sig_now) and (rsi > p["rsi_entry"]) and (vol > vs)
            exit_cond = ((vpt_prev >= sig_prev) and (vpt_now < sig_now)) or (rsi > p["rsi_exit"])

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_roc_ema_volume(params, pos_pct, slip_pct):
    """ROC + EMA + Volume entry with ATR trailing stop exit."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.roc = self.I(_roc, c, p["roc_len"])
            self.ema = self.I(_ewm_mean, c, p["ema_len"])
            self.vol_sma = self.I(_sma, v, p["vol_sma_len"])
            self.atr = self.I(_atr, h, l, c, p["atr_len"])
            self.trailing_high = self.I(_rolling_max, c, p["trailing_len"])

        def next(self):
            p = self._cq_params
            close = self.data.Close[-1]
            roc = self.roc[-1]
            ema = self.ema[-1]
            vol = self.data.Volume[-1]
            vs = self.vol_sma[-1]
            atr = self.atr[-1]
            rmax = self.trailing_high[-1]

            if any(np.isnan(v) for v in [roc, ema, vs, atr, rmax]):
                return

            atr_stop = rmax - atr * p["atr_mult"]

            entry = (roc > 0) and (close > ema) and (vol > vs)
            exit_cond = close < atr_stop

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_bb_stoch_macd(params, pos_pct, slip_pct):
    """BB + Stochastic + MACD triple confluence mean reversion."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            self.bbu, self.bbm, self.bbl = self.I(_bbands, c, p["bb_len"], p["bb_std"])
            self.stoch_k, self.stoch_d = self.I(_stoch, h, l, c, p["stoch_k"], p["stoch_d"])
            self.macd_h = self.I(_macd, c, p["macd_fast"], p["macd_slow"], p["macd_signal"])

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            close = self.data.Close[-1]
            bbm = self.bbm[-1]
            bbu = self.bbu[-1]
            sk = self.stoch_k[-1]
            sd = self.stoch_d[-1]
            mh = self.macd_h[-1]
            mh_prev = self.macd_h[-2]

            if any(np.isnan(v) for v in [bbm, bbu, sk, sd, mh, mh_prev]):
                return

            entry = (close < bbm) and (sk < 20) and (sk > sd) and (mh > mh_prev)
            exit_cond = ((sk > 80) and (sk < sd)) or (close > bbu)

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_rsi_regime_dip(params, pos_pct, slip_pct):
    """RSI Regime Dip: long RSI regime + short RSI dip timing."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            self.regime_rsi = self.I(_rsi, c, p["regime_len"])
            self.timing_rsi = self.I(_rsi, c, p["timing_len"])

        def next(self):
            p = self._cq_params
            regime = self.regime_rsi[-1]
            timing = self.timing_rsi[-1]

            if any(np.isnan(v) for v in [regime, timing]):
                return

            bullish = regime > p["regime_bull"]
            entry = bullish and (timing < p["dip_level"])
            exit_cond = (timing > p["recovery_level"]) or (not bullish)

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_ema_crossover(params, pos_pct, slip_pct):
    """EMA Crossover: fast/slow EMA golden/death cross."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            self.ema_fast = self.I(_ewm_mean, c, p["fast_len"])
            self.ema_slow = self.I(_ewm_mean, c, p["slow_len"])

        def next(self):
            i = len(self.data) - 1
            if i < 2:
                return
            ef = self.ema_fast[-1]
            ef_prev = self.ema_fast[-2]
            es = self.ema_slow[-1]
            es_prev = self.ema_slow[-2]

            if any(np.isnan(v) for v in [ef, ef_prev, es, es_prev]):
                return

            entry = (ef_prev < es_prev) and (ef > es)
            exit_cond = (ef_prev > es_prev) and (ef < es)

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_injected_momentum_atr_volume(params, pos_pct, slip_pct):
    """Injected Momentum ATR Volume: ROC + volume + RSI regime + ATR trailing stop."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.roc = self.I(_roc, c, p["roc_len"])
            self.vol_sma = self.I(_sma, v, p["vol_sma_len"])
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.atr = self.I(_atr, h, l, c, p["atr_len"])
            self.ema_short = self.I(_ewm_mean, c, p["ema_short_len"])
            self.ema_long = self.I(_ewm_mean, c, p["ema_long_len"])

        def next(self):
            p = self._cq_params
            close = self.data.Close[-1]
            roc = self.roc[-1]
            vol = self.data.Volume[-1]
            vs = self.vol_sma[-1]
            rsi = self.rsi[-1]
            atr = self.atr[-1]
            es = self.ema_short[-1]
            el = self.ema_long[-1]

            if any(np.isnan(v) for v in [roc, vs, rsi, atr, es, el]):
                return

            is_uptrend = es > el
            long_momentum = roc > p["roc_threshold"]
            vol_ratio = vol / vs if vs > 0 else 0
            volume_spike = vol_ratio > p["vol_threshold"]
            rsi_healthy = ((is_uptrend and (rsi > p["rsi_min_uptrend"])) or
                          (not is_uptrend and (rsi < p["rsi_max_downtrend"])))

            trailing_stop = close - (atr * p["atr_mult"])

            entry = long_momentum and volume_spike and rsi_healthy
            exit_cond = (close <= trailing_stop) and self.position

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_informed_simple_adaptive(params, pos_pct, slip_pct):
    """Informed Simple Adaptive: ADX regime + RSI extremes with volume."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.adx_val = self.I(_adx, h, l, c, p["adx_len"])
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.vol_avg = self.I(_sma, v, p["volume_window"])

        def next(self):
            p = self._cq_params
            close = self.data.Close[-1]
            adx = self.adx_val[-1]
            rsi = self.rsi[-1]
            rsi_prev = self.rsi[-2]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]

            if any(np.isnan(v) for v in [adx, rsi, rsi_prev, va]):
                return

            trending = adx > p["adx_threshold"]
            volume_spike = vol > (va * p["volume_mult"])

            if trending:
                entry = (rsi < p["rsi_overbought"]) and volume_spike
                exit_cond = rsi > p["rsi_overbought"]
            else:
                entry = ((rsi < p["rsi_oversold"]) or (rsi > p["rsi_overbought"])) and volume_spike
                exit_cond = (rsi > 50) and (rsi_prev <= 50)

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_invented_momentum_confluence(params, pos_pct, slip_pct):
    """Momentum Confluence: ROC + volume + RSI + ADX with ATR exits."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.roc = self.I(_roc, c, p["roc_len"])
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.adx_val = self.I(_adx, h, l, c, p["adx_len"])
            self.atr = self.I(_atr, h, l, c, p["atr_len"])
            self.vol_avg = self.I(_sma, v, p["vol_sma_len"])

        def next(self):
            p = self._cq_params
            close = self.data.Close[-1]
            roc = self.roc[-1]
            rsi = self.rsi[-1]
            rsi_prev = self.rsi[-2]
            adx = self.adx_val[-1]
            atr = self.atr[-1]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]

            if any(np.isnan(v) for v in [roc, rsi, rsi_prev, adx, atr, va]):
                return

            momentum_up = roc > 0
            volume_expansion = vol > (va * p["volume_mult"])
            atr_stop = close - (atr * p["atr_mult"])

            entry = momentum_up and volume_expansion and (rsi < p["rsi_overbought"])
            exit_cond = (close < atr_stop) or (rsi > p["rsi_overbought"] and rsi_prev <= p["rsi_overbought"])

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_invented_rsi_volume_atr(params, pos_pct, slip_pct):
    """RSI Volume ATR: RSI oversold cross + volume spike + optional MACD filter."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.rsi_prev = self.I(lambda x: np.roll(x, 1), self.rsi) if False else None
            self.vol_avg = self.I(_sma, v, p["volume_ma_len"])
            self.atr = self.I(_atr, h, l, c, p["atr_len"])
            if p.get("macd_filter", False):
                self.macd_h = self.I(_macd, c, p["macd_fast"], p["macd_slow"], p["macd_signal"])
            else:
                self.macd_h = None

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            close = self.data.Close[-1]
            low = self.data.Low[-1]
            rsi_now = self.rsi[-1]
            rsi_prev = self.rsi[-2]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]
            atr = self.atr[-1]

            if any(np.isnan(v) for v in [rsi_now, rsi_prev, va, atr]):
                return

            rsi_cross_up = (rsi_prev <= p["rsi_oversold"]) and (rsi_now > p["rsi_oversold"])
            volume_spike = vol > (va * p["volume_spike_mult"])
            macd_ok = True
            if self.macd_h is not None:
                mh = self.macd_h[-1]
                if np.isnan(mh):
                    return
                macd_ok = mh > 0

            atr_stop = close - (atr * p["atr_mult"])

            entry = rsi_cross_up and volume_spike and macd_ok
            exit_cond = (rsi_now < p["rsi_overbought"] and rsi_prev >= p["rsi_overbought"]) or (low <= atr_stop)

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_invented_volume_adx_ema(params, pos_pct, slip_pct):
    """Volume ADX EMA: OBV crossover + ADX trend + EMA direction + ATR/RSI exits."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.obv = self.I(_vpt, c, v)  # Use VPT as OBV proxy (similar cumulative volume logic)
            self.obv_fast = self.I(_ewm_mean, c, p["obv_fast"])  # Simplified: EMA of close as OBV proxy
            self.obv_slow = self.I(_ewm_mean, c, p["obv_slow"])
            self.adx_val = self.I(_adx, h, l, c, p["adx_len"])
            self.ema = self.I(_ewm_mean, c, p["ema_len"])
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.atr = self.I(_atr, h, l, c, 14)

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            close = self.data.Close[-1]
            obv_f = self.obv_fast[-1]
            obv_f_prev = self.obv_fast[-2]
            obv_s = self.obv_slow[-1]
            adx = self.adx_val[-1]
            ema = self.ema[-1]
            rsi = self.rsi[-1]
            atr = self.atr[-1]

            if any(np.isnan(v) for v in [obv_f, obv_f_prev, obv_s, adx, ema, rsi, atr]):
                return

            obv_cross = (obv_f_prev <= obv_s) and (obv_f > obv_s)
            strong_trend = adx > p["adx_threshold"]
            above_ema = close > ema
            atr_stop = close - (p["atr_mult"] * atr)

            entry = obv_cross and strong_trend and above_ema
            exit_cond = (close < atr_stop) or (rsi > p["rsi_overbought"])

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_invented_volume_breakout_adx(params, pos_pct, slip_pct):
    """Volume Breakout ADX: volume spike + ADX trend + SMA trend direction."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.vol_avg = self.I(_sma, v, p["vol_sma_len"])
            self.adx_val = self.I(_adx, h, l, c, p["adx_len"])
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.atr = self.I(_atr, h, l, c, p["atr_len"])
            self.sma_fast = self.I(_sma, c, p["sma_fast"])
            self.sma_slow = self.I(_sma, c, p["sma_slow"])

        def next(self):
            p = self._cq_params
            close = self.data.Close[-1]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]
            adx = self.adx_val[-1]
            rsi = self.rsi[-1]
            atr = self.atr[-1]
            sf = self.sma_fast[-1]
            ss = self.sma_slow[-1]

            if any(np.isnan(v) for v in [va, adx, rsi, atr, sf, ss]):
                return

            vol_spike = vol > (va * p["vol_mult"])
            strong_trend = adx > p["adx_threshold"]
            trend_up = sf > ss
            atr_stop = close - (p["atr_mult"] * atr)

            entry = vol_spike and strong_trend and trend_up and (rsi < 70)
            exit_cond = (close < atr_stop) or (rsi > 80)

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_invented_volume_momentum_trend(params, pos_pct, slip_pct):
    """Volume Momentum Trend: volume breakout + ADX + RSI + ATR trailing stop."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.vol_avg = self.I(_sma, v, p["volume_sma_len"])
            self.adx_val = self.I(_adx, h, l, c, p["adx_len"])
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.atr = self.I(_atr, h, l, c, p["atr_len"])

        def next(self):
            p = self._cq_params
            close = self.data.Close[-1]
            open_ = self.data.Open[-1]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]
            adx = self.adx_val[-1]
            rsi = self.rsi[-1]
            atr = self.atr[-1]

            if any(np.isnan(v) for v in [va, adx, rsi, atr]):
                return

            green_candle = close > open_
            vol_breakout = vol > (va * p["volume_mult"])
            strong_trend = adx > p["adx_threshold"]
            rsi_ok = rsi > p["rsi_oversold"]
            atr_stop = close - (p["atr_mult"] * atr)

            entry = green_candle and strong_trend and rsi_ok and vol_breakout
            exit_cond = (close < atr_stop) or (rsi > p["rsi_overbought"])

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_invented_volume_roc_atr_trend(params, pos_pct, slip_pct):
    """Volume ROC ATR Trend: volume spike + ROC momentum + EMA trend + RSI/ATR exits."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            h = self.data.High
            l = self.data.Low
            v = self.data.Volume
            self.roc = self.I(_roc, c, p["roc_len"])
            self.ema = self.I(_ewm_mean, c, p["ema_len"])
            self.vol_avg = self.I(_sma, v, p["vol_sma_len"])
            self.atr = self.I(_atr, h, l, c, p["atr_len"])
            self.rsi = self.I(_rsi, c, p["rsi_len"])
            self.trailing_high = self.I(_rolling_max, c, p["atr_len"] * 3)

        def next(self):
            p = self._cq_params
            close = self.data.Close[-1]
            roc = self.roc[-1]
            ema = self.ema[-1]
            vol = self.data.Volume[-1]
            va = self.vol_avg[-1]
            atr = self.atr[-1]
            rsi = self.rsi[-1]
            rmax = self.trailing_high[-1]

            if any(np.isnan(v) for v in [roc, ema, va, atr, rsi, rmax]):
                return

            vol_spike = vol > (va * p["volume_mult"])
            momentum = roc > 0
            above_ema = close > ema
            atr_stop = rmax - (p["atr_mult"] * atr)

            entry = vol_spike and momentum and above_ema
            exit_cond = (close < atr_stop) or (rsi > p["rsi_overbought"])

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


def _convert_invented_vpt_roc_ema(params, pos_pct, slip_pct):
    """VPT ROC EMA: VPT signal + ROC momentum + EMA trend."""
    class S(CrabQuantBacktest):
        _cq_params = params
        _cq_position_pct = pos_pct
        _cq_slippage_pct = slip_pct

        def init(self):
            p = self._cq_params
            c = self.data.Close
            v = self.data.Volume
            self.vpt = self.I(_vpt, c, v)
            self.vpt_signal = self.I(lambda x: _rolling_mean(
                _vpt(self.data.Close.values, self.data.Volume.values),
                p["vpt_len"]
            ), self.data.Close)
            self.roc = self.I(_roc, c, p["roc_len"])
            self.ema = self.I(_ewm_mean, c, p["ema_len"])

        def next(self):
            p = self._cq_params
            i = len(self.data) - 1
            if i < 2:
                return
            close = self.data.Close[-1]
            vpt = self.vpt[-1]
            vpt_prev = self.vpt[-2]
            sig = self.vpt_signal[-1]
            sig_prev = self.vpt_signal[-2]
            roc = self.roc[-1]
            ema = self.ema[-1]

            if any(np.isnan(v) for v in [vpt, vpt_prev, sig, sig_prev, roc, ema]):
                return

            vpt_bullish = vpt > sig
            roc_bullish = roc > p["roc_threshold"]
            above_ema = close > ema
            vpt_bearish = (vpt_prev >= sig_prev) and (vpt < sig)
            roc_bearish = roc < -p["roc_threshold"]
            below_ema = close < ema

            entry = vpt_bullish and roc_bullish and above_ema
            exit_cond = vpt_bearish and roc_bearish and below_ema

            self._safe_entry(entry)
            self._safe_exit(exit_cond)

    return S


# ---------------------------------------------------------------------------
# Registry of all converters
# ---------------------------------------------------------------------------

_CONVERTERS = {
    "rsi_crossover": _convert_rsi_crossover,
    "macd_momentum": _convert_macd_momentum,
    "adx_pullback": _convert_adx_pullback,
    "atr_channel_breakout": _convert_atr_channel_breakout,
    "volume_breakout": _convert_volume_breakout,
    "multi_rsi_confluence": _convert_multi_rsi_confluence,
    "ema_ribbon_reversal": _convert_ema_ribbon_reversal,
    "bollinger_squeeze": _convert_bollinger_squeeze,
    "ichimoku_trend": _convert_ichimoku_trend,
    "invented_momentum_rsi_atr": _convert_invented_momentum_rsi_atr,
    "invented_momentum_rsi_stoch": _convert_invented_momentum_rsi_stoch,
    "vpt_crossover": _convert_vpt_crossover,
    "roc_ema_volume": _convert_roc_ema_volume,
    "bb_stoch_macd": _convert_bb_stoch_macd,
    "rsi_regime_dip": _convert_rsi_regime_dip,
    "ema_crossover": _convert_ema_crossover,
    "injected_momentum_atr_volume": _convert_injected_momentum_atr_volume,
    "informed_simple_adaptive": _convert_informed_simple_adaptive,
    "invented_momentum_confluence": _convert_invented_momentum_confluence,
    "invented_rsi_volume_atr": _convert_invented_rsi_volume_atr,
    "invented_volume_adx_ema": _convert_invented_volume_adx_ema,
    "invented_volume_breakout_adx": _convert_invented_volume_breakout_adx,
    "invented_volume_momentum_trend": _convert_invented_volume_momentum_trend,
    "invented_volume_roc_atr_trend": _convert_invented_volume_roc_atr_trend,
    "invented_vpt_roc_ema": _convert_invented_vpt_roc_ema,
}


def convert_strategy(strategy_name: str, params: dict,
                     position_pct: float = 0.95,
                     slippage_pct: float = 0.001) -> type:
    """
    Convert a CrabQuant strategy to a backtesting.py Strategy class.

    Args:
        strategy_name: Name from STRATEGY_REGISTRY
        params: Strategy parameters dict
        position_pct: Fraction of portfolio per trade (default 0.95)
        slippage_pct: Slippage as fraction (default 0.001 = 0.1%)

    Returns:
        A backtesting.py Strategy subclass
    """
    return _make_strategy_class(strategy_name, params, position_pct, slippage_pct)
