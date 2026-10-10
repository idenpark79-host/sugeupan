"""웹사이트용 데이터 생성 — site/data/*.json

GitHub Actions가 매일 장 마감 후 실행합니다 (KRX_ID / KRX_PW 는 저장소 Secrets).
  python site_build.py           # 실제 데이터 (KRX 로그인 필요)
  python site_build.py --demo    # 가상 데이터
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import ssl
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

import analysis
import backtest
import commentary
import data as rawdata
import indicators
import model
import patterns
import track
import strategy
import universe

OUT = ROOT / "site" / "data"
KST = timezone(timedelta(hours=9))
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
SITE = {"chart_min_marcap": 1e11, "flow_min_marcap": 3e11, **CFG.get("site", {})}
INV4 = ["개인", "외국인", "기관합계", "연기금"]
DEMO_REAL = {"005930": "삼성전자", "000660": "SK하이닉스", "005380": "현대차", "000270": "기아",
             "035420": "NAVER", "051910": "LG화학", "006400": "삼성SDI", "068270": "셀트리온",
             "105560": "KB금융", "012330": "현대모비스", "035720": "카카오", "247540": "에코프로비엠",
             "086520": "에코프로", "028300": "HLB", "196170": "알테오젠", "042700": "한미반도체"}


def log(*a):
    print(datetime.now(KST).strftime("%H:%M:%S"), *a, flush=True)


def r1(x):
    return None if x is None or x != x else round(float(x), 1)


def r2(x):
    return None if x is None or x != x else round(float(x), 2)


def dump(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def pmap(fn, items, workers, label):
    out, done = {}, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fu in as_completed(futs):
            done += 1
            try:
                out[futs[fu]] = fu.result()
            except Exception:
                pass
            if done % 100 == 0 or done == len(items):
                log(f"  {label} {done}/{len(items)}")
    return out


# ── 원천 데이터 ───────────────────────────────────────────────
class Source:
    def __init__(self, demo: bool):
        self.demo = demo
        if not demo:
            from pykrx import stock
            self.k = stock
            self.day = stock.get_nearest_business_day_in_a_week()

    # 전 종목 시세·지표
    def listing(self) -> pd.DataFrame:
        if self.demo:
            u = universe.get_universe(0, ["KOSPI", "KOSDAQ"], demo=True)
            r = np.random.default_rng(3)
            real = pd.DataFrame({"Code": list(DEMO_REAL), "Name": list(DEMO_REAL.values()),
                                 "Market": ["KOSPI"] * 10 + ["KOSPI", "KOSDAQ", "KOSDAQ", "KOSDAQ", "KOSDAQ", "KOSPI"],
                                 "Marcap": r.uniform(2e13, 4e14, len(DEMO_REAL))})
            small = pd.DataFrame({"Code": [f"8{i:05d}" for i in range(140)],
                                  "Name": [f"데모소형{i + 1:03d}" for i in range(140)],
                                  "Market": r.choice(["KOSPI", "KOSDAQ"], 140), "Marcap": r.uniform(3e10, 9e11, 140)})
            u = pd.concat([real, u, small], ignore_index=True)
            rows = []
            for c in u.Code:
                h = rawdata.demo_series(c, 2520)
                a, b = h.iloc[-1], h.iloc[-2]
                rows.append((a.Open, a.High, a.Low, a.Close, a.Close - b.Close, (a.Close / b.Close - 1) * 100,
                             a.Volume, a.Volume * a.Close))
            u[["Open", "High", "Low", "Close", "Change", "ChangePct", "Volume", "Amount"]] = rows
            u["PER"] = r.uniform(4, 40, len(u))
            u["PBR"] = r.uniform(0.3, 5, len(u))
            u["DIV"] = r.uniform(0, 5, len(u))
            u["FRG"] = r.uniform(1, 55, len(u))
            self.day = pd.Timestamp.today().strftime("%Y%m%d")
            return u
        u = universe.krx_listing(self.day)
        try:
            f = self.k.get_market_fundamental(self.day, market="ALL")
            f.index = f.index.astype(str)
            u = u.join(f[["PER", "PBR", "DIV"]], on="Code")
        except Exception as e:
            log("투자지표 실패", e)
        try:
            fr = self.k.get_exhaustion_rates_of_foreign_investment(self.day, "ALL")
            fr.index = fr.index.astype(str)
            u = u.join(fr[["지분율"]].rename(columns={"지분율": "FRG"}), on="Code")
        except Exception as e:
            log("외국인 지분율 실패", e)
        return u

    def sectors(self, lst: pd.DataFrame) -> dict:
        if self.demo:
            names = ["전기·전자", "운송장비·부품", "화학", "제약", "금융", "IT 서비스", "유통", "기계·장비", "건설", "음식료·담배", "철강·금속", "통신"]
            r = np.random.default_rng(11)
            fixed = {"005930": "전기·전자", "000660": "전기·전자", "005380": "운송장비·부품", "000270": "운송장비·부품",
                     "035420": "IT 서비스", "035720": "IT 서비스", "051910": "화학", "006400": "전기·전자",
                     "068270": "제약", "105560": "금융", "012330": "운송장비·부품", "247540": "전기·전자",
                     "086520": "화학", "028300": "제약", "196170": "제약", "042700": "기계·장비"}
            return {c: fixed.get(c, names[r.integers(len(names))]) for c in lst.Code}
        out = {}
        for mk in ("KOSPI", "KOSDAQ"):
            try:
                df = self.k.get_market_sector_classifications(self.day, mk)
                out.update({str(c): str(v) for c, v in zip(df.index, df["업종명"])})
            except Exception as e:
                log("업종 분류 실패", mk, e)
        return out

    def index(self, name: str, days=400) -> pd.DataFrame:
        sym = {"KOSPI": "^KS11", "KOSDAQ": "^KQ11", "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "원/달러": "KRW=X"}[name]
        if self.demo:
            return rawdata.demo_series(sym, 270)
        if name in ("KOSPI", "KOSDAQ"):
            try:
                end = pd.Timestamp(self.day)
                df = self.k.get_index_ohlcv((end - pd.Timedelta(days=days)).strftime("%Y%m%d"), self.day,
                                            "1001" if name == "KOSPI" else "2001")
                df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
                df.index = pd.to_datetime(df.index)
                return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            except Exception:
                pass
        return rawdata.fetch(sym, "1y")

    def flow(self, target: str, days: int) -> pd.DataFrame:
        if self.demo:
            idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=int(days / 365 * 252))
            if target in ("KOSPI", "KOSDAQ"):
                r = np.random.default_rng(len(target))
                s = 3e11
                frg = r.normal(0, s, len(idx)) + np.sin(np.arange(len(idx)) / 9) * s * .6
                pen = r.normal(0, s * .25, len(idx)) + np.cos(np.arange(len(idx)) / 13) * s * .2
                inst = pen + r.normal(0, s * .5, len(idx))
                return pd.DataFrame({"개인": -(frg + inst), "외국인": frg, "기관합계": inst, "연기금": pen}, index=idx)
            return universe._demo_flow(target, idx)
        end = pd.Timestamp(self.day)
        df = self.k.get_market_trading_value_by_date((end - pd.Timedelta(days=days)).strftime("%Y%m%d"),
                                                     self.day, target, detail=True)
        return universe.normalize_flow(df)

    # 학습용 장기 시세·수급, 지수, 종목 상세 시세
    sample_flow = False

    def scan_prices(self, codes, years):
        return universe.fetch_all(codes, years, self.demo)

    def scan_flows(self, prices, years):
        return universe.fetch_flows(prices, years, self.demo)

    def kospi(self, years):
        return rawdata.demo_series("^KS11", 252 * years) if self.demo else rawdata.fetch("^KS11", f"{years}y")

    def hist(self, code):
        if self.demo:
            return rawdata.demo_series(code, 2520).iloc[-330:]
        import FinanceDataReader as fdr
        start = (pd.Timestamp(self.day) - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        return rawdata._normalize(fdr.DataReader(code, start))

    def rank(self, market: str, investor: str, start: str) -> pd.DataFrame:
        if self.demo:
            lst = self._lst[self._lst.Market == market]
            r = np.random.default_rng(abs(hash((market, investor, start))) % 2**32)
            v = r.normal(0, 1, len(lst)) * lst.Marcap.values * 3e-4
            return pd.DataFrame({"Code": lst.Code.values, "net": v})
        df = self.k.get_market_net_purchases_of_equities(start, self.day, market, investor)
        return pd.DataFrame({"Code": df.index.astype(str), "net": df["순매수거래대금"].astype(float).values})


# ── 빌드 단계 ────────────────────────────────────────────────
def build_market(src: Source, lst: pd.DataFrame):
    log("지수·시장 수급")
    indices, kospi_dates = [], None
    for name in ["KOSPI", "KOSDAQ", "S&P 500", "NASDAQ", "원/달러"]:
        try:
            h = src.index(name)
            c = h.Close.dropna()
            if name == "KOSPI":
                kospi_dates = c.index
            indices.append({"name": name, "last": r2(c.iloc[-1]), "chg": r2(c.iloc[-1] - c.iloc[-2]),
                            "pct": r2((c.iloc[-1] / c.iloc[-2] - 1) * 100),
                            "spark": [r2(v) for v in c.iloc[-60:]],
                            "y1": r2((c.iloc[-1] / c.iloc[max(-len(c), -250)] - 1) * 100)})
        except Exception as e:
            log("지수 실패", name, e)
    breadth = {}
    for mk in ("KOSPI", "KOSDAQ"):
        m = lst[lst.Market == mk]
        breadth[mk] = {"up": int((m.ChangePct > 0).sum()), "flat": int((m.ChangePct == 0).sum()),
                       "down": int((m.ChangePct < 0).sum()), "amount": r1(m.Amount.sum() / 1e8),
                       "limitUp": int((m.ChangePct >= 29.5).sum()), "limitDown": int((m.ChangePct <= -29.5).sum())}
    flows = {}
    for mk in ("KOSPI", "KOSDAQ"):
        try:
            f = src.flow(mk, 370).iloc[-250:]
            flows[mk] = {"dates": f.index.strftime("%Y-%m-%d").tolist(),
                         **{i: [r1(v / 1e8) for v in f[i]] for i in INV4}}
        except Exception as e:
            log("시장 수급 실패", mk, e)
    dump("market.json", {"indices": indices, "breadth": breadth, "flows": flows})
    return kospi_dates


def build_listing(lst: pd.DataFrame, detail: set, flowset: set, metrics: dict, sectors: dict,
                  views: dict | None = None, ai_today: dict | None = None):
    log("종목 목록", len(lst))
    rows = []
    views, ai_today = views or {}, ai_today or {}
    TONE = {"강세": 1, "중립": 0, "약세": -1}
    for r in lst.itertuples():
        mt = metrics.get(r.Code, {})
        vw, ai = views.get(r.Code), ai_today.get(r.Code, {})
        rows.append([r.Code, r.Name, "P" if r.Market == "KOSPI" else "Q", r2(r.Close), r2(r.Change), r2(r.ChangePct),
                     int(r.Volume or 0), r1(r.Amount / 1e8), r1(r.Marcap / 1e8),
                     r2(getattr(r, "PER", None)), r2(getattr(r, "PBR", None)), r2(getattr(r, "DIV", None)),
                     r2(getattr(r, "FRG", None)), (1 if r.Code in detail else 0) + (2 if r.Code in flowset else 0),
                     r2(r.Open), r2(r.High), r2(r.Low), sectors.get(r.Code, "기타")]
                    + [(int(mt[k]) if k in ("align", "nh", "nl") or k.endswith("S") else r2(mt[k]))
                       if mt.get(k) is not None and mt.get(k) == mt.get(k) else None for k in MET_FIELDS]
                    + [TONE.get(vw["tone"]) if vw else None, vw["head"] if vw else None,
                       ai["20"][0] if "20" in ai else None, ai["5"][0] if "5" in ai else None])
    dump("stocks.json", {"fields": ["code", "name", "mkt", "price", "chg", "pct", "vol", "amt", "cap", "per", "pbr",
                                    "div", "frg", "has", "open", "high", "low", "sector"] + MET_FIELDS
                         + ["tone", "head", "ai20", "ai5"], "rows": rows})


def build_ranks(src: Source, lst: pd.DataFrame, kospi_dates):
    log("투자자별 순매수 순위")
    src._lst = lst
    names = dict(zip(lst.Code, lst.Name))
    price = lst.set_index("Code")
    dates = list(kospi_dates.strftime("%Y%m%d")) if kospi_dates is not None else []
    out = {}
    for mk in ("KOSPI", "KOSDAQ"):
        out[mk] = {}
        for inv in ("외국인", "기관합계", "연기금", "개인"):
            out[mk][inv] = {}
            for n in (1, 5, 20):
                start = dates[-n] if len(dates) >= n else src.day
                try:
                    df = src.rank(mk, inv, start).dropna()
                    df = df[df.Code.isin(names)]
                    df = df.sort_values("net", ascending=False)
                    pick = pd.concat([df.head(30), df.tail(30).iloc[::-1]])
                    out[mk][inv][str(n)] = [[c, names[c], r1(v / 1e8), r2(price.at[c, "ChangePct"])]
                                            for c, v in zip(pick.Code, pick.net)]
                except Exception as e:
                    log("순위 실패", mk, inv, n, e)
                if not (src.demo or src.sample_flow):
                    time.sleep(0.3)
    dump("rank.json", out)


def build_details(src: Source, lst: pd.DataFrame, detail: list, flowset: set, scan_prices: dict, scan_flows: dict,
                  pstats: dict | None = None, ai_today: dict | None = None):
    log("종목 상세", len(detail), "/ 수급", len(flowset))

    def get_hist(code):
        if code in scan_prices:
            return scan_prices[code].iloc[-330:]
        return src.hist(code)

    hists = pmap(get_hist, detail, 8, "시세")

    def get_flow(code):
        if code in scan_flows:
            return scan_flows[code].iloc[-130:]
        if src.demo or src.sample_flow:
            return src.flow(code, 190).reindex(hists[code].index[-130:]).fillna(0)
        f = src.flow(code, 190)
        time.sleep(0.2)
        return f

    flows = pmap(get_flow, [c for c in detail if c in flowset and c in hists], 4, "수급")
    names = dict(zip(lst.Code, lst.Name))
    metrics, views = {}, {}
    ai_today = ai_today or {}
    for code, h in hists.items():
        h = h[h.Close > 0].iloc[-330:]
        if len(h) < 30:
            continue
        d = indicators.add_indicators(h)
        metrics[code] = _metrics(d, flows.get(code))
        try:
            vw = commentary.view(d, pstats, flows.get(code))
            views[code] = vw
        except Exception as e:
            vw = None
            log("견해 실패", code, e)
        obj = {"ohlcv": [[t.strftime("%Y-%m-%d"), r2(o), r2(hi), r2(lo), r2(c), int(v)]
                         for t, o, hi, lo, c, v in zip(h.index, h.Open, h.High, h.Low, h.Close, h.Volume)]}
        try:
            res = analysis.analyze(code, names.get(code, code), d, {})
            obj["tech"] = {"trend": res.trend, "score": res.score, "align": res.alignment, "rsi": r1(res.rsi),
                           "atr": r2(d.ATR_pct.iloc[-1] * 100),
                           "ret": {k: r2(v) for k, v in res.returns.items()},
                           "signals": [[n, s] for n, s in res.signals]}
        except Exception:
            pass
        if code in flows:
            f = flows[code].iloc[-130:]
            obj["flow"] = {"dates": f.index.strftime("%Y-%m-%d").tolist(),
                           **{i: [r2(v / 1e8) for v in f[i]] for i in INV4}}
        if vw:
            obj["view"] = vw
        if code in ai_today:
            obj["ai"] = ai_today[code]
        dump(f"s/{code}.json", obj)
    return metrics, views


def _streak(x):
    k = 0
    for v in x[::-1]:
        if not v or (k and (v > 0) != (k > 0)):
            break
        k += 1 if v > 0 else -1
    return k


def _metrics(d: pd.DataFrame, f) -> dict:
    """조건 검색·순위용 종목 지표."""
    c = d.Close
    ret = lambda n: (c.iloc[-1] / c.iloc[-1 - n] - 1) * 100 if len(c) > n else None
    hi = c.iloc[-250:].max()
    lo = c.iloc[-250:].min()
    last = d.iloc[-1]
    mas = [last.get(f"MA{n}") for n in (5, 20, 60, 120)]
    align = 1 if all(pd.notna(mas)) and mas == sorted(mas, reverse=True) else -1 if all(pd.notna(mas)) and mas == sorted(mas) else 0
    m = {"r1w": ret(5), "r1m": ret(21), "r3m": ret(63), "r6m": ret(126), "r1y": ret(245),
         "rsi": last.RSI, "hi52": (c.iloc[-1] / hi - 1) * 100, "lo52": (c.iloc[-1] / lo - 1) * 100,
         "volr": last.Volume / last.VOL_MA20 if last.VOL_MA20 else None, "align": align,
         "d20": (c.iloc[-1] / last.MA20 - 1) * 100 if pd.notna(last.MA20) else None,
         "nh": int(d.High.iloc[-1] >= d.High.iloc[-250:].max()), "nl": int(d.Low.iloc[-1] <= d.Low.iloc[-250:].min()),
         "atr": last.ATR_pct * 100}
    if f is not None and len(f):
        for key, col in (("frg", "외국인"), ("inst", "기관합계"), ("pen", "연기금")):
            x = f[col].fillna(0).values
            m[key + "1"], m[key + "5"], m[key + "20"] = x[-1] / 1e8, x[-5:].sum() / 1e8, x[-20:].sum() / 1e8
            m[key + "S"] = _streak(x)
    return m


MET_FIELDS = ["r1w", "r1m", "r3m", "r6m", "r1y", "rsi", "hi52", "lo52", "volr", "align", "d20", "nh", "nl", "atr",
              "frg1", "frg5", "frg20", "frgS", "inst1", "inst5", "inst20", "instS", "pen1", "pen5", "pen20", "penS"]


def write_index():
    """app.html(본문)을 완전한 HTML 문서로 감싸 GitHub Pages용 index.html 생성."""
    body = (ROOT / "site" / "app.html").read_text(encoding="utf-8")
    (ROOT / "site" / "index.html").write_text(
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<meta name="theme-color" content="#F5F6F8"><style>:root{padding-top:env(safe-area-inset-top,0px)}'
        'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>'
        '</head><body>' + body + '</body></html>', encoding="utf-8")


def build_patterns(pstats: dict | None):
    """차트 패턴 성적표 — 패턴 뒤 1주·1개월 실제 성과 (전 종목 · 10년)."""
    if not pstats:
        dump("patterns.json", {"ok": False})
        return
    rows = [[k, n, b, d, pstats.get(k, {}).get("n", 0), pstats.get(k, {}).get("5"), pstats.get(k, {}).get("20")]
            for k, n, b, d in patterns.PATTERNS]
    dump("patterns.json", {"ok": True, "base": pstats.get("_base"), "rows": rows})


def _ai_today(results: dict) -> dict:
    """종목별 오늘 점수 {코드: {"20": [백분위, 기대수익%, 확률%]}}."""
    out = {}
    for H, r in results.items():
        for code, (p, e, w) in r.today.items():
            out.setdefault(code, {})[str(H)] = [round(p * 100, 1), r2(e * 100), r1(w * 100)]
    return out


def _why(p, vw, H):
    """모델 근거가 2개 미만이면 과거 성적이 좋은 현재 패턴, 변동폭 순으로 보충 — 근거 없는 추천은 없게."""
    why = list(p["why"])
    if len(why) < 2 and vw:
        j = 5 if H >= 20 else 4
        for x in vw.get("pats", []):
            st = x[j]
            if x[3] > 0 and st and st[1] is not None and st[1] >= 0.3 and st[3] >= 300:
                why.append(f"{x[1]} · 과거 {model.HZ.get(H, '')} 시장 대비 {st[1]:+.1f}%p")
                if len(why) >= 2:
                    break
    if len(why) < 2:
        why.append(f"하루 평균 변동 {p['atrp'] * 100:.1f}% · 목표 도달 여유")
    return why[:3]


def build_rec(results: dict, pn, lst, src, pstats: dict, flows: dict, prices: dict, demo: bool):
    """1주·1개월 추천(rec.json)과 누적 성적(track.json)."""
    log("추천·누적 성적")
    if not results:
        dump("rec.json", {"ok": False})
        dump("track.json", {"h": {}})
        return
    names = dict(zip(lst.Code, lst.Name))
    rec = {"ok": True, "nStocks": int(len(pn.meta)), "usedFlows": bool(flows), "sampleFlow": bool(src.sample_flow), "h": {}}
    for H, r in sorted(results.items(), key=lambda x: -x[0]):
        picks = []
        for p in r.picks:
            d = p["df"]
            try:
                vw = commentary.view(d, pstats, flows.get(p["code"]) if flows else None)
            except Exception:
                vw = None
            try:
                ps = commentary.perspective(d, vw)
            except Exception:
                ps = None
            c = p["close"]
            picks.append({"code": p["code"], "name": names.get(p["code"], p["name"]), "close": r2(c), "chg": r2(p["chg"]),
                          "pct": r2(p["pct"]), "exp": r2(p["exp"] * 100), "prob": r1(p["prob"] * 100), "grade": p["grade"],
                          "score": r1(p["score"] * 100), "target": round(p["target"]), "stop": round(p["stop"]),
                          "tgtPct": r2((p["target"] / c - 1) * 100), "stpPct": r2((p["stop"] / c - 1) * 100),
                          "atr": r2(p["atrp"] * 100), "why": _why(p, vw, H), "view": vw, "type": ps,
                          "st": {k: [r2((x["target"] / c - 1) * 100), r2((x["stop"] / c - 1) * 100), round(x["target"]), round(x["stop"]),
                                     r1(x["prob"] * 100), r2(x["exp"] * 100)] for k, x in p.get("styles", {}).items()}})
        g = {k: [r2(v[0] * 100), r1(v[1] * 100), v[2]] for k, v in r.grades.items()}
        rec["h"][str(H)] = {
            "H": H, "label": model.HZ.get(H, f"{H}일"), "kt": r.kt, "ks": r.ks, "gate": r.gate, "regime": r.regime_ok,
            "picks": picks,
            "model": {"baseAvg": r2(r.base[0] * 100), "baseWin": r1(r.base[1] * 100), "topAvg": r2(r.top[0] * 100),
                      "topWin": r1(r.top[1] * 100), "topHit": r1(r.top[2] * 100), "topN": r.top_n, "ic": r2(r.ic * 100),
                      "grades": g, "off": [r2(r.off[0] * 100), r1(r.off[1] * 100)] if r.off else None,
                      "deciles": [[r2(a * 100), r1(b * 100), n] for a, b, n in r.deciles],
                      "monthly": [[m, r2(a * 100), r1(w * 100), k] for m, a, w, k in r.monthly],
                      "folds": r.folds, "years": round(r.years, 1), "nTrain": r.n_train, "nOos": r.n_oos,
                      "oos": [r.oos_period[0].strftime("%Y-%m-%d"), r.oos_period[1].strftime("%Y-%m-%d")],
                      "minWin": r.min_win, "atrCap": r.atr_cap,
                      "styles": {k: [a, b_, r2(c * 100), r1(d * 100), r1(e * 100)] for k, (a, b_, c, d, e) in (r.styles or {}).items()},
                      "exits": [[a, b, r2(c * 100), r1(d * 100), r2(e * 100), r1(f * 100)] for a, b, c, d, e, f in r.exit_table]}}
    dump("rec.json", rec)

    day = pd.Timestamp(src.day).strftime("%Y-%m-%d")
    path = (OUT / "_track_demo.csv") if (demo or src.sample_flow) else (ROOT / "track" / "picks.csv")
    tr = track.load(path)
    need = set(tr[tr.status == "open"].code) - set(prices)
    extra = {}
    if need and not demo and not src.sample_flow:
        import FinanceDataReader as fdr
        start = (pd.Timestamp(day) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        for c in need:
            try:
                extra[c] = rawdata._normalize(fdr.DataReader(c, start))
            except Exception:
                pass
    tr = track.update(tr, {**prices, **extra})
    for H, r in results.items():
        tr = track.add_today(tr, day, [dict(p, name=names.get(p["code"], p["name"])) for p in r.picks], h=H, n=10)
    if not (demo or src.sample_flow):
        path.parent.mkdir(exist_ok=True)
        tr.to_csv(path, index=False)
    else:
        path.unlink(missing_ok=True)
    dump("track.json", track.summary(tr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--offline", help="공개 일별 시세(marcap) 폴더 — 로컬 미리보기용")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--max-detail", type=int, default=0, help="상세 데이터 종목 수 제한 (미리보기용)")
    args = ap.parse_args()
    write_index()
    t0 = time.time()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    if args.offline:
        sys.path.insert(0, args.offline)
        from offline import OfflineSource
        src = OfflineSource()
    else:
        src = Source(args.demo)
    lst = src.listing()
    lst = lst[lst.Close > 0].reset_index(drop=True)
    kd = build_market(src, lst)
    build_ranks(src, lst, kd)

    results, pn, pstats, scan_prices, scan_flows = {}, None, None, {}, {}
    if not args.no_ai:
        log("학습용 데이터 수집")
        sc = CFG.get("scan", {})
        try:
            u = lst[lst.Marcap >= sc.get("min_marcap", 1e12)][["Code", "Name", "Market", "Marcap"]]
            u = u[u.Code.str.endswith("0") & ~u.Name.str.contains("스팩|리츠")].sort_values(
                "Marcap", ascending=False).reset_index(drop=True)
            years = sc.get("history_years", 10)
            scan_prices = src.scan_prices(list(u.Code), years)
            scan_flows = src.scan_flows(scan_prices, years)
            kospi = src.kospi(years)
            pn = strategy.build_panel(scan_prices, u, kospi["Close"], scan_flows)
            log("차트 패턴 통계")
            pstats = patterns.stats(pn.frames)
            log("기대 수익률 모델 (1주·1개월 워크포워드 학습)")
            results = model.run(pn, sc, kospi["Close"], log=log)
            log("포트폴리오 백테스트")
            try:
                dump("bt.json", backtest.run(pn, results, kospi["Close"], sc, log=log))
            except Exception as e:
                import traceback
                traceback.print_exc()
                log("백테스트 실패", e)
                dump("bt.json", {"ok": False})
        except Exception as e:
            import traceback
            traceback.print_exc()
            log("AI 스캔 실패", e)
    build_patterns(pstats)
    build_rec(results, pn, lst, src, pstats, scan_flows, scan_prices, args.demo)
    ai_today = _ai_today(results)

    detail = lst[lst.Marcap >= SITE["chart_min_marcap"]].sort_values("Marcap", ascending=False).Code.tolist()
    if args.max_detail:
        keep = (set(detail[:args.max_detail]) | set(DEMO_REAL)
                | {p["code"] for r in results.values() for p in r.picks})
        detail = [c for c in detail if c in keep]
    flowset = set(lst[lst.Marcap >= SITE["flow_min_marcap"]].Code)
    metrics, views = build_details(src, lst, detail, flowset, scan_prices, scan_flows, pstats, ai_today)
    have = {p.stem for p in (OUT / "s").glob("*.json")}
    have_flow = {c for c in have if c in flowset}
    build_listing(lst, have, have_flow, metrics, src.sectors(lst), views, ai_today)
    now = datetime.now(KST)
    dump("meta.json", {"updated": now.strftime("%Y-%m-%d %H:%M"), "asof": pd.Timestamp(src.day).strftime("%Y-%m-%d"),
                       "demo": args.demo, "sampleFlow": bool(src.sample_flow)})
    log(f"완료 {time.time() - t0:.0f}초 · {sum(f.stat().st_size for f in OUT.rglob('*.json')) / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
