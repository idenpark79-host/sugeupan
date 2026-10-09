"""앱 데이터 계층 — 실제 데이터(FinanceDataReader·pykrx·Google 뉴스 RSS) / 데모 데이터."""
from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import streamlit as st

import data as rawdata
import universe

INVESTORS = ["개인", "외국인", "기관합계", "연기금", "금융투자", "투신", "사모", "보험", "은행", "기타법인"]
INST_PARTS = ["금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금"]
INDEX_SYMBOLS = {"KOSPI": "^KS11", "KOSDAQ": "^KQ11", "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "원/달러": "KRW=X"}
DEMO_REAL = {"005380": ("현대차", "KOSPI"), "000270": ("기아", "KOSPI"),
             "005930": ("삼성전자", "KOSPI"), "000660": ("SK하이닉스", "KOSPI")}


class KrxLoginError(RuntimeError):
    pass


def _rng(*keys):
    return np.random.default_rng(int(hashlib.md5("|".join(map(str, keys)).encode()).hexdigest()[:8], 16))


def _ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


# ── 종목 목록·현재가 ──────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def listing(demo: bool) -> pd.DataFrame:
    if demo:
        u = universe.get_universe(0, ["KOSPI", "KOSDAQ"], demo=True)
        extra = pd.DataFrame({"Code": list(DEMO_REAL), "Name": [v[0] for v in DEMO_REAL.values()],
                              "Market": [v[1] for v in DEMO_REAL.values()], "Marcap": [6e13, 4e13, 4.5e14, 1.5e14]})
        r = _rng("small")
        small = pd.DataFrame({"Code": [f"8{i:05d}" for i in range(80)],
                              "Name": [f"데모소형{i + 1:02d}" for i in range(80)],
                              "Market": r.choice(["KOSPI", "KOSDAQ"], 80), "Marcap": r.uniform(5e10, 9e11, 80)})
        u = pd.concat([u, extra, small], ignore_index=True)
        rows = []
        for c in u.Code:
            h = rawdata.demo_series(c, 300)
            last, prev = h.iloc[-1], h.iloc[-2]
            rows.append((last.Close, last.Close - prev.Close, (last.Close / prev.Close - 1) * 100,
                         last.Open, last.High, last.Low, last.Volume, last.Volume * last.Close))
        u[["Close", "Change", "ChangePct", "Open", "High", "Low", "Volume", "Amount"]] = rows
        return u
    import FinanceDataReader as fdr
    raw = fdr.StockListing("KRX")
    pick = lambda *ns: next(n for n in ns if n in raw.columns)
    df = pd.DataFrame({
        "Code": raw[pick("Code", "Symbol")].astype(str).str.zfill(6),
        "Name": raw[pick("Name")], "Market": raw[pick("Market")].astype(str),
        "Close": pd.to_numeric(raw[pick("Close")], errors="coerce"),
        "Change": pd.to_numeric(raw[pick("Changes", "Change")], errors="coerce"),
        "ChangePct": pd.to_numeric(raw[pick("ChagesRatio", "ChangesRatio", "ChangeRatio")], errors="coerce"),
        "Open": pd.to_numeric(raw[pick("Open")], errors="coerce"),
        "High": pd.to_numeric(raw[pick("High")], errors="coerce"),
        "Low": pd.to_numeric(raw[pick("Low")], errors="coerce"),
        "Volume": pd.to_numeric(raw[pick("Volume")], errors="coerce"),
        "Amount": pd.to_numeric(raw[pick("Amount")], errors="coerce"),
        "Marcap": pd.to_numeric(raw[pick("Marcap")], errors="coerce")})
    df = df[df.Market.str.upper().str.startswith(("KOSPI", "KOSDAQ"))]
    df["Market"] = df.Market.str.upper().str.replace(" GLOBAL", "", regex=False)
    return df.reset_index(drop=True)


# ── 시세 이력 ────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def history(code: str, days: int, demo: bool) -> pd.DataFrame:
    n = int(days / 365 * 252) + 5
    if demo:
        return rawdata.demo_series(code, max(n, 160))
    import FinanceDataReader as fdr
    start = (date.today() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    df = rawdata._normalize(fdr.DataReader(code, start))
    return df[df.Volume > 0] if df.Volume.sum() > 0 else df


@st.cache_data(ttl=600, show_spinner=False)
def index_history(name: str, days: int, demo: bool) -> pd.DataFrame:
    sym = INDEX_SYMBOLS[name]
    if demo:
        return rawdata.demo_series(sym, int(days / 365 * 252) + 5)
    period = "1y" if days <= 370 else "2y" if days <= 740 else "3y"
    return rawdata.fetch(sym, period).iloc[-(int(days / 365 * 252) + 5):]


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D":
        return df
    rule = {"W": "W-FRI", "M": "ME"}[freq]
    return df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"}).dropna()


# ── 투자자별 매매동향 (pykrx, KRX 로그인 필요) ───────────────────
def _pykrx():
    try:
        from pykrx import stock
        return stock
    except Exception as e:
        raise KrxLoginError(f"pykrx 사용 불가: {e}")


def _normalize_flow(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return universe.normalize_flow(df)
    except RuntimeError as e:
        raise KrxLoginError(str(e))


@st.cache_data(ttl=1800, show_spinner=False)
def investor_flow(target: str, days: int, demo: bool) -> pd.DataFrame:
    """target: 종목코드 또는 KOSPI/KOSDAQ. 일별 순매수 대금(원)."""
    if demo:
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=int(days / 365 * 252) + 2)
        r = _rng("flow", target)
        scale = 3e11 if target in ("KOSPI", "KOSDAQ") else 5e9
        frg = r.normal(0, scale, len(idx)) + np.sin(np.arange(len(idx)) / 9) * scale * 0.6
        pen = r.normal(0, scale * 0.25, len(idx)) + np.cos(np.arange(len(idx)) / 13) * scale * 0.2
        fin = r.normal(0, scale * 0.4, len(idx))
        etc_inst = r.normal(0, scale * 0.3, len(idx))
        corp = r.normal(0, scale * 0.1, len(idx))
        inst = pen + fin + etc_inst
        df = pd.DataFrame({"개인": -(frg + inst + corp), "외국인": frg, "기관합계": inst, "연기금": pen,
                           "금융투자": fin, "투신": etc_inst * 0.4, "사모": etc_inst * 0.2,
                           "보험": etc_inst * 0.25, "은행": etc_inst * 0.15, "기타법인": corp}, index=idx)
        return df
    stock = _pykrx()
    end = date.today()
    start = end - timedelta(days=days)
    df = stock.get_market_trading_value_by_date(_ymd(start), _ymd(end), target, detail=True)
    return _normalize_flow(df)


@st.cache_data(ttl=1800, show_spinner=False)
def net_purchase_rank(market: str, investor: str, days: int, demo: bool) -> pd.DataFrame:
    """기간 내 투자자별 순매수 대금 순위. 반환: Code, Name, 순매수대금, 매수대금, 매도대금"""
    if demo:
        lst = listing(True)
        lst = lst[lst.Market == market] if market != "ALL" else lst
        r = _rng("rank", market, investor, days)
        v = r.normal(0, 1, len(lst)) * lst.Marcap.values * 2e-4 * np.sqrt(days)
        buy = np.abs(v) * r.uniform(1.5, 4, len(lst))
        return pd.DataFrame({"Code": lst.Code.values, "Name": lst.Name.values, "순매수대금": v,
                             "매수대금": buy, "매도대금": buy - v}).sort_values("순매수대금", ascending=False)
    stock = _pykrx()
    end = date.today()
    start = end - timedelta(days=days)
    df = stock.get_market_net_purchases_of_equities(_ymd(start), _ymd(end), market, investor)
    if df is None or df.empty:
        raise KrxLoginError("KRX에서 데이터를 받지 못했습니다. 설정에서 KRX 계정을 확인하세요.")
    return pd.DataFrame({"Code": df.index.astype(str), "Name": df["종목명"],
                         "순매수대금": df["순매수거래대금"], "매수대금": df["매수거래대금"],
                         "매도대금": df["매도거래대금"]}).sort_values("순매수대금", ascending=False)


@st.cache_data(ttl=3600, show_spinner=False)
def foreign_holding(code: str, days: int, demo: bool) -> pd.DataFrame:
    """외국인 지분율(%) 추이."""
    if demo:
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=int(days / 365 * 252) + 2)
        r = _rng("fh", code)
        return pd.DataFrame({"지분율": np.clip(r.uniform(10, 50) + np.cumsum(r.normal(0, 0.08, len(idx))), 0, 100)},
                            index=idx)
    stock = _pykrx()
    end = date.today()
    df = stock.get_exhaustion_rates_of_foreign_investment(_ymd(end - timedelta(days=days)), _ymd(end), code)
    if df is None or df.empty:
        raise KrxLoginError("외국인 지분율을 받지 못했습니다.")
    df.index = pd.to_datetime(df.index)
    return df[["지분율"]].astype(float)


@st.cache_data(ttl=3600, show_spinner=False)
def fundamentals(code: str, demo: bool) -> pd.DataFrame:
    """PER·PBR·EPS·BPS·배당수익률(DIV)·주당배당금(DPS) 1년 추이."""
    if demo:
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=250)
        r = _rng("fund", code)
        eps, bps = r.uniform(500, 15000), r.uniform(10000, 200000)
        close = rawdata.demo_series(code, 250).Close.values
        return pd.DataFrame({"EPS": eps, "BPS": bps, "PER": close / eps, "PBR": close / bps,
                             "DPS": eps * 0.25, "DIV": eps * 0.25 / close * 100}, index=idx)
    stock = _pykrx()
    end = date.today()
    df = stock.get_market_fundamental_by_date(_ymd(end - timedelta(days=370)), _ymd(end), code)
    if df is None or df.empty:
        raise KrxLoginError("투자지표를 받지 못했습니다.")
    df.index = pd.to_datetime(df.index)
    return df.astype(float)


# ── 뉴스 ────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def news(query: str, demo: bool, limit: int = 15) -> list[dict]:
    if demo:
        return [{"title": f"[데모] {query} 관련 기사 제목 예시 {i + 1}", "link": "https://news.google.com",
                 "source": "데모뉴스", "time": pd.Timestamp.now() - pd.Timedelta(hours=3 * i)} for i in range(6)]
    q = urllib.parse.quote(f"{query} 주가")
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        root = ET.fromstring(r.read())
    items = []
    for it in root.iter("item"):
        title = it.findtext("title", "")
        src = it.findtext("source", "")
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        try:
            t = parsedate_to_datetime(it.findtext("pubDate", "")).astimezone().replace(tzinfo=None)
        except Exception:
            t = None
        items.append({"title": title, "link": it.findtext("link", ""), "source": src, "time": t})
        if len(items) >= limit:
            break
    return items
