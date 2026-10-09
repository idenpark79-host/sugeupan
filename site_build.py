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
import data as rawdata
import indicators
import model
import scanner
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
                h = rawdata.demo_series(c, 756)
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


def build_listing(lst: pd.DataFrame, detail: set, flowset: set):
    log("종목 목록", len(lst))
    rows = []
    for r in lst.itertuples():
        rows.append([r.Code, r.Name, "P" if r.Market == "KOSPI" else "Q", r2(r.Close), r2(r.Change), r2(r.ChangePct),
                     int(r.Volume or 0), r1(r.Amount / 1e8), r1(r.Marcap / 1e8),
                     r2(getattr(r, "PER", None)), r2(getattr(r, "PBR", None)), r2(getattr(r, "DIV", None)),
                     r2(getattr(r, "FRG", None)), (1 if r.Code in detail else 0) + (2 if r.Code in flowset else 0),
                     r2(r.Open), r2(r.High), r2(r.Low)])
    dump("stocks.json", {"fields": ["code", "name", "mkt", "price", "chg", "pct", "vol", "amt", "cap", "per", "pbr",
                                    "div", "frg", "has", "open", "high", "low"], "rows": rows})


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
                if not src.demo:
                    time.sleep(0.3)
    dump("rank.json", out)


def build_details(src: Source, lst: pd.DataFrame, detail: list, flowset: set, scan_prices: dict, scan_flows: dict):
    log("종목 상세", len(detail), "/ 수급", len(flowset))

    def get_hist(code):
        if code in scan_prices:
            return scan_prices[code].iloc[-330:]
        if src.demo:
            return rawdata.demo_series(code, 756).iloc[-330:]
        import FinanceDataReader as fdr
        start = (pd.Timestamp(src.day) - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        return rawdata._normalize(fdr.DataReader(code, start))

    hists = pmap(get_hist, detail, 8, "시세")

    def get_flow(code):
        if code in scan_flows:
            return scan_flows[code].iloc[-130:]
        if src.demo:
            return src.flow(code, 190).reindex(hists[code].index[-130:]).fillna(0)
        f = src.flow(code, 190)
        time.sleep(0.2)
        return f

    flows = pmap(get_flow, [c for c in detail if c in flowset and c in hists], 4, "수급")
    names = dict(zip(lst.Code, lst.Name))
    for code, h in hists.items():
        h = h[h.Close > 0].iloc[-330:]
        if len(h) < 30:
            continue
        d = indicators.add_indicators(h)
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
        dump(f"s/{code}.json", obj)


def build_ai(res, lst):
    log("AI 추천")
    names = dict(zip(lst.Code, lst.Name))
    if res is None or not res.strategies:
        dump("ai.json", {"ok": False})
        return
    strat_idx = {id(s): i for i, s in enumerate(res.strategies)}
    best = res.strategies[0]
    r = best.rets * 100
    edges = np.linspace(np.percentile(r, 1), np.percentile(r, 99), 33)
    cnt, _ = np.histogram(np.clip(r, edges[0], edges[-1]), bins=edges)
    dump("ai.json", {
        "ok": True, "horizon": res.horizon, "nStocks": res.n_stocks, "nTested": res.n_tested,
        "period": [res.period[0].strftime("%Y-%m"), res.period[1].strftime("%Y-%m")],
        "usedFlows": bool(getattr(res, "used_flows", False)),
        "baseline": {k: r2(v * 100) if k != "n" else int(v) for k, v in res.baseline.items()},
        "strategies": [{"name": s.name, "conds": [[strategy.COND_NAME[k], strategy.COND_DESC[k]] for k in s.combo],
                        "n": int(s.n), "win": r1(s.win * 100), "avg": r2(s.avg * 100), "hit": r1(s.hit * 100),
                        "winTr": r1(s.win_tr * 100), "winTe": r1(s.win_te * 100), "kt": s.kt, "ks": s.ks}
                       for s in res.strategies],
        "hist": {"edges": [r2(e) for e in edges], "counts": cnt.tolist(), "mean": r2(r.mean())},
        "picks": [{"code": p.code, "name": names.get(p.code, p.name), "close": r2(p.close), "chg": r2(p.chg),
                   "pct": r2(p.chg_pct), "target": round(p.target), "stop": round(p.stop),
                   "s": strat_idx.get(id(p.strategy), 0), "ago": int(p.days_ago),
                   "ownN": int(p.own_n), "ownWin": r1(p.own_win * 100) if p.own_n else None}
                  for p in res.picks]})


def write_index():
    """app.html(본문)을 완전한 HTML 문서로 감싸 GitHub Pages용 index.html 생성."""
    body = (ROOT / "site" / "app.html").read_text(encoding="utf-8")
    (ROOT / "site" / "index.html").write_text(
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<meta name="theme-color" content="#F5F6F8"><style>:root{padding-top:env(safe-area-inset-top,0px)}'
        'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>'
        '</head><body>' + body + '</body></html>', encoding="utf-8")


def build_model(mr, res, lst, src, prices: dict, demo: bool):
    """확률 추천(rec.json)과 누적 성적(track.json)."""
    log("확률 추천·누적 성적")
    if mr is None:
        dump("rec.json", {"ok": False})
        dump("track.json", {"total": 0})
        return
    names = dict(zip(lst.Code, lst.Name))
    dump("rec.json", {
        "ok": True, "horizon": mr.horizon, "kt": mr.kt, "ks": mr.ks, "auc": r2(mr.auc),
        "base": r1(mr.base_win * 100), "top": r1(mr.top_win * 100), "topAvg": r2(mr.top_avg * 100),
        "topHit": r1(mr.top_hit * 100), "topN": mr.top_n, "nTrain": mr.n_train, "nTest": mr.n_test,
        "test": [mr.test_period[0].strftime("%Y-%m-%d"), mr.test_period[1].strftime("%Y-%m-%d")],
        "nStocks": len(prices), "usedFlows": bool(getattr(res, "used_flows", False)),
        "calib": [[r1(a * 100), r1(b * 100), n] for a, b, n in mr.calib],
        "monthly": [[m, r1(w * 100), r2(a * 100), n] for m, w, a, n in mr.monthly],
        "picks": [{"code": p["code"], "name": names.get(p["code"], p["name"]), "close": r2(p["close"]),
                   "chg": r2(p["chg"]), "pct": r2(p["pct"]), "prob": r1(p["prob"] * 100),
                   "exp": r2(p["exp"] * 100), "hitp": r1(p["hitp"] * 100),
                   "target": round(p["target"]), "stop": round(p["stop"]), "atr": r2(p["atrp"] * 100),
                   "why": [[a, b] for a, b in p["why"]]} for p in mr.picks]})

    day = pd.Timestamp(src.day).strftime("%Y-%m-%d")
    path = (OUT / "_track_demo.csv") if demo else (ROOT / "track" / "picks.csv")
    tr = track.load(path)
    if demo and tr.empty and mr.recent:                     # 미리보기: 검증 구간 추천으로 예시 성적 생성
        tr = pd.concat([tr, pd.DataFrame(mr.recent).assign(status="open")], ignore_index=True)
    need = set(tr[tr.status == "open"].code) - set(prices)
    extra = {}
    if need and not demo:
        import FinanceDataReader as fdr
        start = (pd.Timestamp(day) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        for c in need:
            try:
                extra[c] = rawdata._normalize(fdr.DataReader(c, start))
            except Exception:
                pass
    tr = track.update(tr, {**prices, **extra}, horizon=mr.horizon)
    tr = track.add_today(tr, day, [{**p, "code": p["code"]} for p in mr.picks], n=10)
    if not demo:
        path.parent.mkdir(exist_ok=True)
        tr.to_csv(path, index=False)
    else:
        path.unlink(missing_ok=True)
    dump("track.json", track.summary(tr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--max-detail", type=int, default=0, help="상세 데이터 종목 수 제한 (미리보기용)")
    args = ap.parse_args()
    write_index()
    t0 = time.time()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    src = Source(args.demo)
    lst = src.listing()
    lst = lst[lst.Close > 0].reset_index(drop=True)
    kd = build_market(src, lst)
    build_ranks(src, lst, kd)

    res, mr, scan_prices, scan_flows = None, None, {}, {}
    if not args.no_ai:
        log("AI 스캔")
        sc = CFG.get("scan", {})
        try:
            u = lst[lst.Marcap >= sc.get("min_marcap", 1e12)][["Code", "Name", "Market", "Marcap"]]
            u = u[u.Code.str.endswith("0") & ~u.Name.str.contains("스팩|리츠")].sort_values(
                "Marcap", ascending=False).reset_index(drop=True)
            years = sc.get("history_years", 3)
            scan_prices = universe.fetch_all(list(u.Code), years, args.demo)
            scan_flows = universe.fetch_flows(scan_prices, years, args.demo)
            kospi = (rawdata.demo_series("^KS11", 252 * years) if args.demo
                     else rawdata.fetch("^KS11", f"{years}y"))
            pn = strategy.build_panel(scan_prices, u, kospi["Close"], scan_flows)
            res = strategy.mine(pn, sc)
            res.used_flows = bool(scan_flows)
            log("상승 확률 모델")
            mr = model.run(pn, sc, top_n=SITE.get("rec_count", 20), log=log)
        except Exception as e:
            import traceback
            traceback.print_exc()
            log("AI 스캔 실패", e)
    build_ai(res, lst)
    build_model(mr, res, lst, src, scan_prices, args.demo)

    detail = lst[lst.Marcap >= SITE["chart_min_marcap"]].sort_values("Marcap", ascending=False).Code.tolist()
    if args.max_detail:
        keep = (set(detail[:args.max_detail]) | {p.code for p in (res.picks if res else [])} | set(DEMO_REAL)
                | {p["code"] for p in (mr.picks if mr else [])})
        detail = [c for c in detail if c in keep]
    flowset = set(lst[lst.Marcap >= SITE["flow_min_marcap"]].Code)
    build_details(src, lst, detail, flowset, scan_prices, scan_flows)
    have = {p.stem for p in (OUT / "s").glob("*.json")}
    have_flow = {c for c in have if c in flowset}
    build_listing(lst, have, have_flow)
    now = datetime.now(KST)
    dump("meta.json", {"updated": now.strftime("%Y-%m-%d %H:%M"), "asof": pd.Timestamp(src.day).strftime("%Y-%m-%d"),
                       "demo": args.demo})
    log(f"완료 {time.time() - t0:.0f}초 · {sum(f.stat().st_size for f in OUT.rglob('*.json')) / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
