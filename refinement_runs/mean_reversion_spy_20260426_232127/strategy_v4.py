import pandas as pd
from crabquant.indicator_cache import cached_indicator

DEFAULT_PARAMS = {
    "regime_len": 50,
    "regime_bull": 45,
    "rsi_len": 7,
    "rsi_dip": 42,
    "rsi_exit": 60,
    "ema_len": 50,
}

DESCRIPTION = (
    "Regime-filtered pullback reversion for SPY/QQQ/IWM. "
    "Bullish regime when RSI-50 > 45 and price above EMA-50. "
    "Enters on RSI-7 dip below 42 (no volume filter). "
    "Exits when RSI-7 recovers above 60. "
    "Relaxed entry vs v3 to increase trade frequency."
)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df["close"]

    rsi_regime = cached_indicator("rsi", close, length=p["regime_len"])
    rsi_fast = cached_indicator("rsi", close, length=p["rsi_len"])
    ema = cached_indicator("ema", close, length=p["ema_len"])

    bullish = (rsi_regime > p["regime_bull"]) & (close > ema)
    dip = rsi_fast < p["rsi_dip"]

    entries = (bullish & dip).fillna(False)
    exits = (rsi_fast > p["rsi_exit"]).fillna(False)

    return entries, exits