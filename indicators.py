"""기술적 지표 계산."""
import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    v = df["Volume"] if "Volume" in df else pd.Series(0.0, index=df.index)

    for n in (5, 20, 60, 120):
        df[f"MA{n}"] = c.rolling(n).mean()
    df["MA60_slope"] = df["MA60"] / df["MA60"].shift(5) - 1

    # RSI(14) — Wilder
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # MACD(12, 26, 9)
    df["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # 볼린저밴드(20, 2σ)와 밴드폭
    std = c.rolling(20).std()
    df["BB_upper"] = df["MA20"] + 2 * std
    df["BB_lower"] = df["MA20"] - 2 * std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["MA20"]

    # ATR(14)
    prev = c.shift()
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["ATR_pct"] = df["ATR"] / c

    # 스토캐스틱(14, 3, 3)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    df["STO_K"] = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()
    df["STO_D"] = df["STO_K"].rolling(3).mean()

    # ADX(14)
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / df["ATR"]
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / df["ATR"]
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["PDI"], df["MDI"] = pdi, mdi
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # 거래량·OBV
    df["VOL_MA20"] = v.rolling(20).mean()
    df["OBV"] = (np.sign(c.diff()).fillna(0) * v).cumsum()

    # 고점·모멘텀
    df["HH20_prev"] = h.rolling(20).max().shift(1)
    df["HH252"] = c.rolling(252, min_periods=120).max()
    df["RET60"] = c.pct_change(60)
    df["RET120"] = c.pct_change(120)
    return df
