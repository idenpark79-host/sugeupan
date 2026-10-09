"""시세 데이터 수집 — yfinance / FinanceDataReader, --demo 시 가상 데이터."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume"]
PERIOD_DAYS = {"3mo": 95, "6mo": 185, "1y": 370, "2y": 740, "3y": 1110}
FDR_ALIAS = {"^KS11": "KS11", "^KQ11": "KQ11", "^GSPC": "US500", "^IXIC": "IXIC", "KRW=X": "USD/KRW"}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in COLS if c in df.columns]].dropna(subset=["Close"])
    if "Volume" not in df:
        df["Volume"] = 0.0
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _fetch_yfinance(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    return _normalize(df) if not df.empty else df


def _fetch_fdr(ticker: str, period: str) -> pd.DataFrame:
    import FinanceDataReader as fdr
    sym = FDR_ALIAS.get(ticker, ticker.split(".")[0])
    start = (pd.Timestamp.today() - pd.Timedelta(days=PERIOD_DAYS.get(period, 370))).strftime("%Y-%m-%d")
    return _normalize(fdr.DataReader(sym, start))


def demo_series(ticker: str, n: int) -> pd.DataFrame:
    """네트워크 없이 동작 확인용 가상 시세 (종목별 고정 시드)."""
    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    drift, vol = rng.uniform(-0.0004, 0.0010), rng.uniform(0.012, 0.028)
    ret = rng.normal(drift, vol, n)
    # 추세 지속 구간을 섞어 신호가 생기도록
    for _ in range(n // 120):
        s = rng.integers(0, n - 40)
        ret[s:s + 40] += rng.normal(0, 0.003)
    base = rng.uniform(800, 18_000) if ticker.startswith("^") else rng.uniform(5_000, 400_000)
    close = base * np.exp(np.cumsum(ret))
    open_ = np.r_[base, close[:-1]] * (1 + rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.lognormal(13, 0.45, n) * (1 + 3 * np.abs(ret) / vol * 0.2)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=idx)


def fetch(ticker: str, period: str = "1y", demo: bool = False) -> pd.DataFrame:
    if demo:
        return demo_series(ticker, int(PERIOD_DAYS.get(period, 370) / 365 * 252))
    errors = []
    korean = ticker.split(".")[0].isdigit() or ticker in FDR_ALIAS
    sources = [_fetch_fdr, _fetch_yfinance] if korean else [_fetch_yfinance, _fetch_fdr]
    for src in sources:
        try:
            df = src(ticker, period)
            if df is not None and len(df) > 30:
                return df
        except Exception as e:
            errors.append(f"{src.__name__}: {e}")
    raise RuntimeError(f"{ticker} 데이터 수집 실패 — " + "; ".join(errors or ["데이터 없음"]))
