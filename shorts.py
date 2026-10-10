"""공매도 — 거래 비중·잔고 비율 수집, 종목별 '과거 비슷한 수준일 때' 성과, 전 종목 검증.

자료: 한국거래소 개별종목 공매도 종합정보(공매도 거래량·잔고수량, 잔고는 2016년 7월부터 공시).
공매도 금지 기간(2020.3.16~2021.5.2 전 종목, 2023.11.6~2025.3.30 전 종목)과
공매도 거래가 없던 구간(대형주 외 금지 연장 등)은 통계에서 뺀다.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"
START = pd.Timestamp("2016-07-01")
BANS = [(pd.Timestamp("2020-03-16"), pd.Timestamp("2021-05-02")), (pd.Timestamp("2023-11-06"), pd.Timestamp("2025-03-30"))]
H = 20


def _fetch_one(code: str) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"short_{code}.csv"
    if f.exists() and date.fromtimestamp(f.stat().st_mtime) == date.today():
        return pd.read_csv(f, index_col=0, parse_dates=True)
    from pykrx.stock import get_shorting_status_by_date
    end, a, parts = pd.Timestamp.today(), START, []
    while a < end:                                              # 2년 단위로 나눠 조회
        b = min(a + pd.DateOffset(years=2), end)
        for k in range(3):
            try:
                df = get_shorting_status_by_date(a.strftime("%Y%m%d"), b.strftime("%Y%m%d"), code)
                if df is not None and len(df):
                    df.index = pd.to_datetime(df.index)
                    parts.append(df[["거래량", "잔고수량"]].astype(float).rename(columns={"거래량": "sv", "잔고수량": "bal"}))
                break
            except Exception:
                time.sleep(1 + k)
        a = b + pd.Timedelta(days=1)
    if not parts:
        raise RuntimeError(f"{code} 공매도 수집 실패")
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(f)
    return df


def _sample(code: str, px: pd.DataFrame, shares: float) -> pd.DataFrame:
    """미리보기·데모용 예시 자료 (실제 공매도 자료 아님)."""
    r = np.random.default_rng(int("".join(ch for ch in code if ch.isdigit()) or 1) % 2**32)
    p = px[px.index >= START]
    ret20 = p.Close.pct_change(20).fillna(0).values
    base = r.uniform(0.2, 2.5)
    walk = np.cumsum(r.normal(0, 0.04, len(p)))
    ratio = np.clip(base * np.exp(walk - walk.mean()) * (1 + 0.8 * np.clip(ret20, -0.5, 1)), 0.02, 12) / 100
    bal = pd.Series(ratio * shares, index=p.index).rolling(5, min_periods=1).mean()
    sv = (p.Volume * np.clip(r.normal(0.04, 0.025, len(p)), 0, 0.3)).values
    df = pd.DataFrame({"sv": sv, "bal": bal.values}, index=p.index)
    for a, b in BANS:
        df.loc[(df.index >= a) & (df.index <= b), ["sv", "bal"]] = 0.0
    return df


def fetch(codes, prices: dict, sample: bool, shares: dict | None = None, workers: int = 4, log=print) -> dict:
    out = {}
    if sample:
        for c in codes:
            if c in prices and (shares or {}).get(c):
                out[c] = _sample(c, prices[c], shares[c])
        return out
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fs = {ex.submit(_fetch_one, c): c for c in codes}
        for k, fu in enumerate(as_completed(fs), 1):
            try:
                out[fs[fu]] = fu.result()
            except Exception:
                pass
            if k % 100 == 0 or k == len(fs):
                log(f"    공매도 수집 {k}/{len(fs)}")
    return out


def _frame(sh: pd.DataFrame, px: pd.DataFrame, shares: float) -> pd.DataFrame:
    p = px[px.index >= START][["Close", "Volume"]]
    d = p.join(sh, how="left")
    tail = d.index > sh.index.max()                                             # 잔고는 2영업일 늦게 공시 — 최신 값으로 이어 씀
    d["bal"] = d.bal.ffill().fillna(0)
    d["sv"] = d.sv.fillna(0).where(~tail)
    d["r"] = np.where(d.Volume > 0, d.sv / d.Volume * 100, np.nan)            # 공매도 거래 비중 %
    d["r20"] = d.r.rolling(20, min_periods=10).mean()
    d["b"] = d.bal / shares * 100                                               # 잔고 비율 % (현재 상장주식수 기준)
    d["bch"] = (d.bal / d.bal.shift(H).where(d.bal.shift(H) > 0) - 1) * 100     # 잔고 20일 변화 %
    ban = np.zeros(len(d), bool)
    for a, b in BANS:
        ban |= (d.index >= a) & (d.index <= b)
    d["ok"] = ~ban & (d.sv.fillna(0).rolling(60, min_periods=20).sum() > 0) & (d.bal > 0)
    d["fwd"] = d.Close.shift(-H) / d.Close - 1
    return d


def analyze(sh: pd.DataFrame, px: pd.DataFrame, shares: float, mkt: pd.Series | None) -> dict | None:
    """종목 한 개: 차트용 시계열 + 자기 과거에서 잔고 비율 구간별 다음 1개월 성과."""
    if sh is None or not len(sh) or not shares:
        return None
    d = _frame(sh, px, shares)
    if not d.ok.iloc[-60:].any():
        last = d.iloc[-250:]
        return {"d": last.index.strftime("%Y-%m-%d").tolist(), "r": [None] * len(last), "b": [None] * len(last), "off": True}
    if mkt is not None:
        m = mkt.reindex(d.index).ffill()
        d["ex"] = d.fwd - (m.shift(-H) / m - 1)
    else:
        d["ex"] = d.fwd
    v = d[d.ok & d.b.notna()]
    now = d.iloc[-1]
    hist = v[v.fwd.notna()]
    out = {}
    if len(hist) >= 200:
        edges = np.unique(np.quantile(hist.b, [0, .2, .4, .6, .8, 1]))
        if len(edges) >= 4:
            k = np.clip(np.searchsorted(edges, hist.b, side="right") - 1, 0, len(edges) - 2)
            rows = []
            for i in range(len(edges) - 1):
                g = hist[k == i]
                rows.append([round(float(edges[i]), 2), round(float(edges[i + 1]), 2), round(float(g.fwd.mean()) * 100, 2),
                             round(float(g.ex.mean()) * 100, 2), round(float((g.fwd > 0).mean()) * 100, 1), int(len(g))])
            out["q"] = rows
            out["qi"] = int(np.clip(np.searchsorted(edges, now.b, side="right") - 1, 0, len(edges) - 2))
            out["pc"] = round(float((hist.b < now.b).mean()) * 100, 1)
    last = d.iloc[-250:]
    f = lambda s, k=2: [None if not np.isfinite(x) else round(float(x), k) for x in s]
    out.update({"d": last.index.strftime("%Y-%m-%d").tolist(), "r": f(last.r.where(last.ok)), "b": f(last.b.where(last.ok), 3),
                "now": {"b": round(float(now.b), 3), "r20": None if not np.isfinite(now.r20) else round(float(now.r20), 2),
                        "bch": None if not np.isfinite(now.bch) else round(float(now.bch), 1),
                        "amt": round(float(now.bal * now.Close / 1e8), 1)},
                "from": str(v.index.min().date()) if len(v) else None})
    return out


def study(shorts: dict, prices: dict, shares: dict, log=print) -> dict:
    """전 종목: 같은 날 다른 종목 대비 잔고 비율·잔고 증가·거래 비중이 높을 때 다음 1개월 성과(같은 날 평균 대비)."""
    fr = []
    for c, sh in shorts.items():
        if c not in prices or not shares.get(c):
            continue
        d = _frame(sh, prices[c], shares[c])
        d = d[d.ok & d.fwd.notna()][["b", "bch", "r20", "fwd"]]
        if len(d) > 60:
            fr.append(d.assign(c=c))
    if not fr:
        return {"ok": False}
    A = pd.concat(fr)
    A["ex"] = A.fwd - A.groupby(level=0).fwd.transform("mean")
    n_day = A.groupby(level=0).c.transform("size")
    A = A[n_day >= 30]
    for k in ("b", "bch", "r20"):
        A["k_" + k] = A.groupby(level=0)[k].rank(pct=True)
    G = [("잔고 비율 상위 10%", A.k_b >= .9), ("잔고 비율 하위 50%", A.k_b < .5),
         ("잔고 1개월 급증 (상위 10%)", A.k_bch >= .9), ("잔고 1개월 급감 (하위 10%)", A.k_bch <= .1),
         ("공매도 거래 비중 상위 10%", A.k_r20 >= .9)]
    rows = [[lab, round(float(A.ex[m].mean()) * 100, 2), round(float((A.ex[m] > 0).mean()) * 100, 1), int(m.sum())] for lab, m in G]
    log("    공매도 검증: " + " · ".join(f"{r[0]} {r[1]:+.2f}%p" for r in rows))
    return {"ok": True, "rows": rows, "stocks": int(A.c.nunique()), "from": str(A.index.min().date()), "to": str(A.index.max().date())}
