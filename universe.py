"""스캔 대상 종목(코스피·코스닥, 시가총액 기준) 목록과 일괄 시세 수집."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
import hashlib
import time

import numpy as np
import pandas as pd

import data

CACHE = Path(__file__).resolve().parent / "cache"


def _pick(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"열을 찾을 수 없음: {names} / 실제: {list(df.columns)}")


def get_universe(min_marcap: float, markets: list[str], demo: bool = False) -> pd.DataFrame:
    """Code, Name, Market, Marcap(원) 반환. 우선주·스팩·리츠 제외."""
    if demo:
        rng = np.random.default_rng(7)
        n = 60
        return pd.DataFrame({
            "Code": [f"9{i:05d}" for i in range(n)],
            "Name": [f"데모종목{i + 1:02d}" for i in range(n)],
            "Market": rng.choice(["KOSPI", "KOSDAQ"], n, p=[0.7, 0.3]),
            "Marcap": rng.uniform(1e12, 8e13, n)})

    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"universe_{date.today():%Y%m%d}.csv"
    if cache.exists():
        u = pd.read_csv(cache, dtype={"Code": str})
    else:
        try:
            import FinanceDataReader as fdr
            raw = fdr.StockListing("KRX")
            u = pd.DataFrame({
                "Code": raw[_pick(raw, "Code", "Symbol", "종목코드")].astype(str).str.zfill(6),
                "Name": raw[_pick(raw, "Name", "종목명")],
                "Market": raw[_pick(raw, "Market", "시장구분")].astype(str),
                "Marcap": pd.to_numeric(raw[_pick(raw, "Marcap", "시가총액")], errors="coerce")})
        except Exception:
            u = krx_listing()[["Code", "Name", "Market", "Marcap"]]
        u.to_csv(cache, index=False)

    mk = u["Market"].str.upper()
    u = u[mk.apply(lambda m: any(m.startswith(x.upper()) for x in markets))]
    u = u[u["Marcap"] >= min_marcap]
    u = u[u["Code"].str.endswith("0")]                       # 우선주 제외
    u = u[~u["Name"].str.contains("스팩|리츠|우$|우B$", regex=True)]
    return u.sort_values("Marcap", ascending=False).reset_index(drop=True)


def krx_listing(day: str | None = None) -> pd.DataFrame:
    """pykrx(KRX 로그인)로 코스피·코스닥 전 종목 당일 시세·시총 조회."""
    from pykrx import stock
    day = day or stock.get_nearest_business_day_in_a_week()
    rows = []
    for mk in ("KOSPI", "KOSDAQ"):
        df = stock.get_market_ohlcv(day, market=mk)
        need = [c for c in ("시가총액", "거래대금") if c not in df.columns]
        if need:                                          # pykrx 버전에 따라 시세표에 이미 포함됨
            cap = stock.get_market_cap(day, market=mk)
            df = df.join(cap[[c for c in need if c in cap.columns]], how="left")
        if "등락률" not in df.columns:
            df["등락률"] = 0.0
        df["Market"] = mk
        rows.append(df)
    df = pd.concat(rows)
    df.index = df.index.astype(str)
    pct = df["등락률"].astype(float)
    close = df["종가"].astype(float)
    out = pd.DataFrame({
        "Code": df.index, "Name": [stock.get_market_ticker_name(t) for t in df.index], "Market": df["Market"].values,
        "Open": df["시가"].values, "High": df["고가"].values, "Low": df["저가"].values, "Close": close.values,
        "Change": (close - close / (1 + pct / 100)).round().values, "ChangePct": pct.values,
        "Volume": df["거래량"].values, "Amount": df["거래대금"].values, "Marcap": df["시가총액"].values})
    out.attrs["day"] = day
    return out.reset_index(drop=True)


def _fetch_one(code: str, years: int, demo: bool) -> pd.DataFrame:
    if demo:
        return data.demo_series(code, 252 * years)
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{code}.csv"
    if f.exists() and date.fromtimestamp(f.stat().st_mtime) == date.today():
        return pd.read_csv(f, index_col=0, parse_dates=True)
    import FinanceDataReader as fdr
    start = (pd.Timestamp.today() - pd.DateOffset(years=years, days=10)).strftime("%Y-%m-%d")
    for attempt in range(3):
        try:
            df = data._normalize(fdr.DataReader(code, start))
            df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
            df.to_csv(f)
            return df
        except Exception:
            time.sleep(1 + attempt)
    raise RuntimeError(f"{code} 수집 실패")


def fetch_all(codes: list[str], years: int, demo: bool = False, workers: int = 8,
              progress=None) -> dict[str, pd.DataFrame]:
    out, total, done = {}, len(codes), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, c, years, demo): c for c in codes}
        for fu in as_completed(futs):
            done += 1
            try:
                df = fu.result()
                if len(df) >= 200:
                    out[futs[fu]] = df
            except Exception:
                pass
            if progress:
                progress(done, total)
            elif done % 25 == 0 or done == total:
                print(f"    시세 수집 {done}/{total}", flush=True)
    return out


# ── 투자자별 매매동향 (pykrx, KRX 로그인 필요) ───────────────────
INST_PARTS = ["금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금"]


def normalize_flow(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("KRX에서 투자자별 데이터를 받지 못했습니다 (KRX 로그인 확인)")
    out = pd.DataFrame(index=pd.to_datetime(df.index))
    v = lambda c: df[c].values if c in df else 0
    out["개인"] = v("개인")
    out["외국인"] = (v("외국인") + v("기타외국인")) if "외국인" in df else v("외국인합계")
    out["기관합계"] = sum(v(c) for c in INST_PARTS) if "연기금" in df else v("기관합계")
    for c in ["연기금", "금융투자", "투신", "사모", "보험", "은행", "기타법인"]:
        out[c] = v(c)
    return out.astype(float)


def _demo_flow(code: str, idx) -> pd.DataFrame:
    r = np.random.default_rng(int(hashlib.md5(f"flow{code}".encode()).hexdigest()[:8], 16))
    px = data.demo_series(code, len(idx)).Close.pct_change().fillna(0).values
    frg = (px * 3 + r.normal(0, 0.02, len(idx))) * 1e10
    inst = (px * 1.5 + r.normal(0, 0.02, len(idx))) * 6e9
    pen = (px * 1.0 + r.normal(0, 0.02, len(idx))) * 2e9
    return pd.DataFrame({"개인": -(frg + inst), "외국인": frg, "기관합계": inst, "연기금": pen}, index=idx)


def _fetch_flow_one(code: str, years: int, demo: bool, index=None) -> pd.DataFrame:
    if demo:
        return _demo_flow(code, index)
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"flow_{code}.csv"
    if f.exists() and date.fromtimestamp(f.stat().st_mtime) == date.today():
        return pd.read_csv(f, index_col=0, parse_dates=True)
    from pykrx import stock
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years, days=10)
    for attempt in range(3):
        try:
            df = normalize_flow(stock.get_market_trading_value_by_date(
                start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code, detail=True))
            df.to_csv(f)
            return df
        except Exception:
            time.sleep(1 + attempt)
    raise RuntimeError(f"{code} 매매동향 수집 실패")


def fetch_flows(prices: dict, years: int, demo: bool = False, workers: int = 4,
                progress=None) -> dict[str, pd.DataFrame]:
    out, total, done = {}, len(prices), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_flow_one, c, years, demo, df.index): c for c, df in prices.items()}
        for fu in as_completed(futs):
            done += 1
            try:
                out[futs[fu]] = fu.result()
            except Exception:
                pass
            if progress:
                progress(done, total)
            elif done % 50 == 0 or done == total:
                print(f"    매매동향 수집 {done}/{total}", flush=True)
    return out
