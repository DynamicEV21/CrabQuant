import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "regime_len": 50,
    "regime_bull": 50,
    "rsi_len": 7,
    "rsi_dip": 35,
    "rsi_exit": 60,
    "vol_mult": 1.1,
    "ema_len": 50,
}

DESCRIPTION = (
    "Regime-filtered pullback reversion for SPY/QQQ/IWM. "
    "Bullish regime when RSI-50 > 50 and price above EMA-50. "
    "Enters on RSI-7 dip below 35 with volume above 1.1x avg. "
    "Exits when RSI-7 recovers above 60. "
    "Captures mean reversion within established uptrends."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]
    volume = df["volume"]

    rsi_regime = cached_indicator("rsi", close, length=p["regime_len"])
    rsi_fast = cached_indicator("rsi", close, length=p["rsi_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])
    vol_avg = volume.rolling(20).mean()

    bullish = (rsi_regime > p["regime_bull"]) & (close > ema)
    dip = rsi_fast < p["rsi_dip"]
    vol_confirm = volume > vol_avg * p["vol_mult"]

    entries = (bullish & dip & vol_confirm).fillna(False)
    exits = (rsi_fast > p["rsi_exit"]).fillna(False)

    return entries, exits