"""복합 기법 탐색·백테스트·추천 엔진

1) 23개 기본 조건(추세·모멘텀·눌림·반전·거래량·캔들·시장국면)을 종목·일자별로 계산
2) 1~3개 조건의 모든 조합을 '신호'로 보고, 신호 다음날 시가에 매수 →
   목표가(ATR 배수) 도달 시 익절 / 손절가 이탈 시 손절 / 20거래일 경과 시 종가 청산
   (거래비용 반영) 으로 과거 전 종목에 적용
3) 학습 구간(앞 65%)·검증 구간(뒤 35%) 모두에서 수익이 유지된 조합만 남기고,
   표본 수까지 고려한 승률 하한(Wilson)으로 순위 → 가장 확률이 높은 기법 선정
4) 상위 기법마다 목표가·손절가 배수를 다시 최적화
5) 오늘 신호가 막 발생한 종목을 기법 승률 순으로 추천, 매도가(목표·손절) 산출
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math

import numpy as np
import pandas as pd

import indicators

# ── 기본 조건 정의 ─────────────────────────────────────────────
# (키, 이름, 분류, 설명)
CONDITIONS = [
    ("align", "이평선 정배열", "추세", "5일>20일>60일>120일선"),
    ("ma60up", "60일선 상승 위", "추세", "종가가 상승 중인 60일선 위"),
    ("ma20_60", "20일선>60일선", "추세", "중기 상승 구조"),
    ("ma120", "120일선 위", "추세", "장기 추세 상승"),
    ("adx", "강한 추세(ADX)", "추세", "ADX 25 이상, +DI>-DI"),
    ("mom3m", "3개월 +10% 이상", "모멘텀", "60거래일 수익률 10% 이상"),
    ("rs", "상대강도 상위 30%", "모멘텀", "6개월 수익률이 전체 종목 중 상위 30%"),
    ("hi52", "52주 고점 5% 이내", "모멘텀", "신고가 부근"),
    ("brk20", "20일 신고가 돌파", "돌파", "직전 20일 최고가 상향 돌파"),
    ("sqz", "변동성 수축 후 돌파", "돌파", "밴드폭 수축 직후 볼린저 상단 돌파"),
    ("rsi_pb", "RSI 눌림(40~55)", "눌림", "상승 중 일시 조정 구간"),
    ("near20", "20일선 지지", "눌림", "종가가 20일선 -3%~+2%"),
    ("rsi_os", "RSI 과매도(<30)", "반전", "단기 과매도"),
    ("sto", "스토캐스틱 반등", "반전", "%K가 25 이하에서 %D 상향 돌파"),
    ("bb_rev", "볼린저 하단 반등", "반전", "장중 하단 이탈 후 밴드 안에서 마감"),
    ("gc", "골든크로스(5/20)", "신호", "5일선이 20일선 상향 돌파"),
    ("macd_x", "MACD 상향 돌파", "신호", "MACD가 시그널선 상향 돌파"),
    ("macd_h", "MACD 히스토그램 반등", "신호", "0 아래에서 2일 연속 개선"),
    ("vol", "거래량 급증 양봉", "거래량", "20일 평균 2배 이상 + 상승 마감"),
    ("obv", "OBV 20일 신고가", "거래량", "매집 흐름"),
    ("bigc", "장대양봉", "캔들", "+3% 이상, 윗꼬리 짧음"),
    ("up3", "3일 연속 상승", "캔들", "단기 매수세 지속"),
    ("mkt", "시장 상승국면", "시장", "코스피가 60일선 위"),
    ("frg", "외국인 순매수 지속", "수급", "외국인 5일 누적 순매수 + 당일 순매수"),
    ("inst", "기관 순매수 지속", "수급", "기관 5일 누적 순매수 + 당일 순매수"),
    ("pen", "연기금 순매수", "수급", "연기금 5일 누적 순매수"),
    ("dual", "외국인·기관 동반 매수", "수급", "당일 외국인과 기관이 모두 순매수"),
]
FLOW_KEYS = {"frg", "inst", "pen", "dual"}
COND_NAME = {k: n for k, n, *_ in CONDITIONS}
COND_DESC = {k: d for k, _, _, d in CONDITIONS}


def _conditions(d: pd.DataFrame) -> pd.DataFrame:
    c, o, h, l = d["Close"], d["Open"], d["High"], d["Low"]
    p = d.shift(1)
    sqz_recent = (d["BB_width"] <= d["BB_width"].rolling(120).quantile(0.2)).rolling(5).max().astype(bool)
    r = pd.DataFrame(index=d.index)
    r["align"] = (d.MA5 > d.MA20) & (d.MA20 > d.MA60) & (d.MA60 > d.MA120)
    r["ma60up"] = (c > d.MA60) & (d.MA60_slope > 0)
    r["ma20_60"] = d.MA20 > d.MA60
    r["ma120"] = c > d.MA120
    r["adx"] = (d.ADX > 25) & (d.PDI > d.MDI)
    r["mom3m"] = d.RET60 > 0.10
    r["hi52"] = c >= 0.95 * d.HH252
    r["brk20"] = c > d.HH20_prev
    r["sqz"] = sqz_recent & (c > d.BB_upper)
    r["rsi_pb"] = d.RSI.between(40, 55)
    r["near20"] = (c / d.MA20 - 1).between(-0.03, 0.02)
    r["rsi_os"] = d.RSI < 30
    r["sto"] = (d.STO_K > d.STO_D) & (p.STO_K <= p.STO_D) & (p.STO_K < 25)
    r["bb_rev"] = (l < d.BB_lower) & (c > d.BB_lower)
    r["gc"] = (d.MA5 > d.MA20) & (p.MA5 <= p.MA20)
    r["macd_x"] = (d.MACD > d.MACD_signal) & (p.MACD <= p.MACD_signal)
    r["macd_h"] = (d.MACD_hist < 0) & (d.MACD_hist > p.MACD_hist) & (p.MACD_hist > d.MACD_hist.shift(2))
    r["vol"] = (d.Volume > 2 * d.VOL_MA20) & (c > o) & (c > p.Close)
    r["obv"] = d.OBV >= d.OBV.rolling(20).max()
    r["bigc"] = ((c - o) / o > 0.03) & ((h - c) < (c - o) * 0.3)
    r["up3"] = (c > p.Close) & (p.Close > d.Close.shift(2)) & (d.Close.shift(2) > d.Close.shift(3))
    r["mkt"] = d["MKT_UP"].astype(bool)
    if "F_frg" in d:                                   # 투자자별 매매동향이 있을 때만
        f5, i5, p5 = (d[k].rolling(5).sum() for k in ("F_frg", "F_inst", "F_pen"))
        r["frg"] = (f5 > 0) & (d.F_frg > 0)
        r["inst"] = (i5 > 0) & (d.F_inst > 0)
        r["pen"] = p5 > 0
        r["dual"] = (d.F_frg > 0) & (d.F_inst > 0)
    else:
        for k in FLOW_KEYS:
            r[k] = False
    return r.fillna(False)


# ── 패널 구성 ─────────────────────────────────────────────────
@dataclass
class Panel:
    meta: pd.DataFrame            # sid → Code, Name, Market, Marcap
    frames: dict                  # code → 지표 포함 DataFrame
    date: np.ndarray
    sid: np.ndarray
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    atrp: np.ndarray
    cond: np.ndarray              # (N, K) bool
    keys: list
    warm: np.ndarray              # 지표 계산이 충분한 행
    feat: np.ndarray = None       # (N, F) 연속형 특징 — 확률 모델 입력
    feat_names: list = None


def _features(d: pd.DataFrame, mkt_ret20: pd.Series) -> dict:
    """확률 모델용 연속형 특징 (모두 당일 종가까지의 정보만 사용)."""
    c, v = d.Close, d.Volume
    amt20 = (c * v).rolling(20).mean().replace(0, np.nan)
    band = (d.BB_upper - d.BB_lower).replace(0, np.nan)
    f = {
        "rsi": d.RSI, "sto_k": d.STO_K, "adx": d.ADX, "dmi": d.PDI - d.MDI,
        "macd_h": d.MACD_hist / c * 100, "macd_h_chg": d.MACD_hist.diff() / c * 100,
        "bb_pos": (c - d.BB_lower) / band, "bb_w": d.BB_width, "atrp": d.ATR_pct * 100,
        "vol_r": v / d.VOL_MA20.replace(0, np.nan), "vol_r5": v.rolling(5).mean() / d.VOL_MA20.replace(0, np.nan),
        "d_ma5": c / d.MA5 - 1, "d_ma20": c / d.MA20 - 1, "d_ma60": c / d.MA60 - 1, "d_ma120": c / d.MA120 - 1,
        "ma60_slope": d.MA60_slope, "ma20_slope": d.MA20 / d.MA20.shift(5) - 1,
        "ret1": c.pct_change(), "ret5": c.pct_change(5), "ret20": c.pct_change(20),
        "ret60": d.RET60, "ret120": d.RET120, "hi52": c / d.HH252 - 1,
        "lo52": c / c.rolling(252, min_periods=120).min() - 1,
        "obv_z": (d.OBV - d.OBV.rolling(20).mean()) / d.VOL_MA20.replace(0, np.nan),
        "gap": d.Open / c.shift() - 1, "body": (c - d.Open) / d.Open,
        "upper_tail": (d.High - np.maximum(c, d.Open)) / c,
        "mkt_ret20": mkt_ret20.reindex(d.index, method="ffill"),
    }
    if "F_frg" in d:
        for k, col in (("frg", "F_frg"), ("inst", "F_inst"), ("pen", "F_pen")):
            x = d[col].fillna(0)
            f[f"{k}1"] = x / amt20
            f[f"{k}5"] = x.rolling(5).sum() / amt20
            f[f"{k}20"] = x.rolling(20).sum() / amt20
    return f


def build_panel(prices: dict, meta: pd.DataFrame, market_close: pd.Series, flows: dict | None = None) -> Panel:
    mkt_up = (market_close > market_close.rolling(60).mean())
    mkt_ret20 = market_close.pct_change(20)
    keys = [k for k, *_ in CONDITIONS]
    parts, frames = [], {}
    code_to_sid = {c: i for i, c in enumerate(meta["Code"])}
    for code, df in prices.items():
        if code not in code_to_sid:
            continue
        d = indicators.add_indicators(df)
        d["MKT_UP"] = mkt_up.reindex(d.index, method="ffill").fillna(False)
        if flows and code in flows:
            f = flows[code].reindex(d.index)
            d["F_frg"], d["F_inst"], d["F_pen"] = f["외국인"], f["기관합계"], f["연기금"]
        frames[code] = d
        cd = _conditions(d)
        part = pd.DataFrame({"date": d.index, "sid": code_to_sid[code],
                             "O": d.Open.values, "H": d.High.values, "L": d.Low.values, "C": d.Close.values,
                             "atrp": d.ATR_pct.values, "RET120": d.RET120.values,
                             "warm": d.MA120.notna().values & (np.arange(len(d)) >= 130)})
        for k in keys:
            if k != "rs":
                part[k] = cd[k].values
        for k, v in _features(d, mkt_ret20).items():
            part["f_" + k] = np.asarray(v, dtype=float)
        parts.append(part)
    P = pd.concat(parts, ignore_index=True).sort_values(["sid", "date"], kind="stable").reset_index(drop=True)
    P["rs"] = (P.groupby("date")["RET120"].rank(pct=True) >= 0.7).fillna(False)
    P["f_rs"] = P.groupby("date")["RET120"].rank(pct=True)
    P["f_rs20"] = P.groupby("date")["f_ret20"].rank(pct=True)
    fcols = [c for c in P.columns if c.startswith("f_")]
    X = np.hstack([P[fcols].values.astype(np.float32), P[keys].values.astype(np.float32)])
    X[~np.isfinite(X)] = np.nan
    return Panel(meta=meta, frames=frames, date=P["date"].values, sid=P["sid"].values,
                 O=P.O.values, H=P.H.values, L=P.L.values, C=P.C.values,
                 atrp=P.atrp.values, cond=P[keys].values.astype(bool), keys=keys, warm=P.warm.values,
                 feat=X, feat_names=[c[2:] for c in fcols] + ["c_" + k for k in keys])


# ── 거래 시뮬레이션 ─────────────────────────────────────────────
def simulate(pn: Panel, kt: float, ks: float, horizon: int, cost: float, atr_cap: float | None = None):
    """모든 (종목, 일자)에서 신호가 났다고 가정한 거래 결과.
    atr_cap: 목표·손절 계산에 쓰는 일평균 변동폭 상한 (변동성 큰 종목의 손절폭이 과도해지지 않게)
    반환: ret(순수익률), hit(목표가 도달), valid(미래 데이터 충분)"""
    atrp = np.minimum(pn.atrp, atr_cap) if atr_cap else pn.atrp
    N = len(pn.C)
    idx = np.arange(N)
    e = idx + 1
    valid = (idx + horizon < N)
    valid[valid] &= pn.sid[(idx + horizon)[valid]] == pn.sid[valid]
    valid &= np.isfinite(atrp)
    entry = np.full(N, np.nan)
    entry[valid] = pn.O[e[valid]]
    valid &= entry > 0
    tgt = entry * (1 + kt * atrp)
    stp = entry * (1 - ks * atrp)
    ret = np.full(N, np.nan)
    hit = np.zeros(N, bool)
    open_ = valid.copy()
    for k in range(1, horizon + 1):
        j = idx + k
        m = open_ & (j < N)
        jj = j[m]
        stop_hit = pn.L[jj] <= stp[m]          # 같은 날 둘 다 닿으면 손절로 간주(보수적)
        tgt_hit = (~stop_hit) & (pn.H[jj] >= tgt[m])
        mi = idx[m]
        with np.errstate(invalid="ignore", divide="ignore"):
            oj = np.where(pn.O[jj] > 0, pn.O[jj], np.nan)          # 시가 누락일은 목표·손절가로 체결 가정
            r = np.where(stop_hit, np.fmin(oj, stp[m]) / entry[m] - 1,
                 np.where(tgt_hit, np.fmax(oj, tgt[m]) / entry[m] - 1, np.nan))
        done = stop_hit | tgt_hit
        ret[mi[done]] = r[done]
        hit[mi[tgt_hit]] = True
        open_[mi[done]] = False
    last = open_ & valid
    ret[last] = pn.C[(idx + horizon)[last]] / entry[last] - 1
    ret = ret - cost
    return ret, hit, valid


def _wilson_lb(p, n, z=1.64):
    if n == 0:
        return 0.0
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - adj) / den


def _events(pn: Panel, mask: np.ndarray, cooldown: int = 5) -> np.ndarray:
    """조건 충족이 '새로 시작된' 날만 신호로 인정 (직전 cooldown일 동안 미충족)."""
    cs = np.cumsum(mask.astype(np.int32))
    prev = np.zeros_like(cs)
    prev[cooldown + 1:] = cs[cooldown:-1] - cs[:-cooldown - 1]
    same = np.zeros(len(mask), bool)
    same[cooldown + 1:] = pn.sid[cooldown + 1:] == pn.sid[:-cooldown - 1]
    return mask & (prev == 0) & same & pn.warm


@dataclass
class Strategy:
    combo: tuple
    n: int
    win: float
    avg: float
    hit: float
    n_tr: int
    win_tr: float
    n_te: int
    win_te: float
    avg_te: float
    score: float
    avg_tr: float = 0.0
    sd: float = 0.0
    sd_tr: float = 0.0
    kt: float = 2.0
    ks: float = 1.5
    rets: np.ndarray = field(default=None, repr=False)

    @property
    def name(self):
        return " + ".join(COND_NAME[k] for k in self.combo)


@dataclass
class ScanResult:
    baseline: dict
    strategies: list
    picks: list
    n_stocks: int
    period: tuple
    horizon: int
    n_tested: int


def _stats(ev, ret, hit, train_mask):
    r = ret[ev]
    n = len(r)
    if n == 0:
        return None
    w = r > 0
    tr, te = train_mask[ev], ~train_mask[ev]
    return dict(n=n, win=w.mean(), avg=r.mean(), hit=hit[ev].mean(),
                n_tr=int(tr.sum()), win_tr=w[tr].mean() if tr.any() else 0,
                n_te=int(te.sum()), win_te=w[te].mean() if te.any() else 0,
                avg_te=r[te].mean() if te.any() else 0, avg_tr=r[tr].mean() if tr.any() else 0,
                sd=r.std() if n > 1 else 0, sd_tr=r[tr].std() if tr.sum() > 1 else 0)


def _lcb(s: dict) -> float:
    """기대 수익률의 보수적 하한 (표본이 적거나 변동이 크면 낮아짐)."""
    return s["avg"] - 1.64 * s["sd"] / math.sqrt(max(s["n"], 1))


def _lcb_tr(s: dict) -> float:
    """학습 구간만으로 계산한 하한 — 백테스트용 (검증 구간을 전혀 보지 않음)."""
    return s["avg_tr"] - 1.64 * s["sd_tr"] / math.sqrt(max(s["n_tr"], 1))


def _dedupe(results, n):
    top, seen, sigs = [], [], set()
    for s in results:
        if any(set(s.combo) >= set(t) or set(s.combo) <= set(t) for t in seen):
            continue
        sig = (s.n, round(s.win, 4), round(s.avg, 5))      # 결과가 똑같은 조합(같은 신호) 제외
        if sig in sigs:
            continue
        sigs.add(sig)
        top.append(s)
        seen.append(s.combo)
        if len(top) >= n:
            break
    return top


def mine(pn: Panel, cfg: dict) -> ScanResult:
    H, cost = cfg.get("horizon_days", 20), cfg.get("cost_pct", 0.25) / 100
    kt0, ks0 = cfg.get("default_target_atr", 2.0), cfg.get("default_stop_atr", 1.5)
    ret, hit, valid = simulate(pn, kt0, ks0, H, cost)
    usable = valid & pn.warm & np.isfinite(ret)
    dates = np.unique(pn.date[usable])
    split = dates[int(len(dates) * 0.65)]
    train = pn.date < split

    base = dict(n=int(usable.sum()), win=float((ret[usable] > 0).mean()),
                avg=float(np.nanmean(ret[usable])), hit=float(hit[usable].mean()))

    min_tr, min_te = cfg.get("min_train_signals", 60), cfg.get("min_test_signals", 30)
    K = len(pn.keys)
    results, results_tr = [], []
    tested = 0
    active = [i for i in range(K) if pn.cond[:, i].any()]       # 데이터가 없는 조건(수급 등)은 제외
    for r in range(1, cfg.get("max_combo", 3) + 1):
        for combo in combinations(active, r):
            tested += 1
            m = pn.cond[:, combo].all(axis=1)
            ev = _events(pn, m) & usable
            if ev.sum() < min_tr + min_te:
                continue
            s = _stats(ev, ret, hit, train)
            if s["n_tr"] < min_tr or s["n_te"] < min_te:
                continue
            if s["avg_tr"] > 0:                               # 백테스트용: 학습 구간만 보고 고른 후보
                results_tr.append(Strategy(tuple(pn.keys[i] for i in combo), score=_lcb_tr(s), **s))
            if s["avg_te"] <= 0 or s["avg_tr"] <= 0:          # 학습·검증 구간 모두 기대 수익이 플러스여야 채택
                continue
            results.append(Strategy(tuple(pn.keys[i] for i in combo), score=_lcb(s), **s))

    results.sort(key=lambda s: s.score, reverse=True)
    # 거의 같은 조합(상위 조합을 포함하는 부분집합) 중복 제거
    top = _dedupe(results, cfg.get("top_strategies", 10))
    results_tr.sort(key=lambda s: s.score, reverse=True)
    top_bt = _dedupe(results_tr, cfg.get("top_strategies", 10))
    top_bt = [next((t for t in top if t.combo == x.combo), x) for x in top_bt]   # 같은 조합은 같은 객체 공유

    # 상위 기법별 목표가·손절가 배수 최적화 (학습 구간 기대수익 최대)
    grid_t = cfg.get("exit_grid", {}).get("target_atr", [1.5, 2.0, 2.5, 3.0, 4.0])
    grid_s = cfg.get("exit_grid", {}).get("stop_atr", [1.0, 1.5, 2.0, 2.5])
    sims = {(kt, ks): simulate(pn, kt, ks, H, cost) for kt in grid_t for ks in grid_s}
    for s in top + [x for x in top_bt if all(x is not t for t in top)]:
        m = pn.cond[:, [pn.keys.index(k) for k in s.combo]].all(axis=1)
        best = None
        for (kt, ks), (r2, h2, v2) in sims.items():
            ev = _events(pn, m) & v2 & pn.warm & np.isfinite(r2)
            tr = ev & train
            if tr.sum() < min_tr:
                continue
            exp_tr = r2[tr].mean()
            if best is None or exp_tr > best[0]:
                best = (exp_tr, kt, ks)
        if best:
            s.kt, s.ks = best[1], best[2]
        r2, h2, v2 = sims.get((s.kt, s.ks), (ret, hit, valid))
        ev = _events(pn, m) & v2 & pn.warm & np.isfinite(r2)
        st = _stats(ev, r2, h2, train)
        for k, v in st.items():
            setattr(s, k, v)
        s.rets = r2[ev]
        s.score = _lcb(st)
    top_bt = [s for s in top_bt if s.avg_tr > 0]
    top_bt.sort(key=lambda s: s.avg_tr, reverse=True)
    # 매도 기준 재최적화 후 결과가 같아진 조합 제거
    uniq, sigs = [], set()
    for s in top:
        sig = (s.n, round(s.win, 4), round(s.avg, 5))
        if sig not in sigs:
            sigs.add(sig)
            uniq.append(s)
    top = uniq
    # 기대 수익률(검증 구간) 높은 순
    top = [s for s in top if s.avg_te > 0 and s.avg_tr > 0]
    top.sort(key=lambda s: (s.avg_te, s.avg), reverse=True)

    picks = recommend(pn, top, sims, cfg)
    d0, d1 = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    out = ScanResult(base, top, picks, len(pn.frames), (d0, d1), H, tested)
    out.split = split
    out.bt_strategies = top_bt
    return out


# ── 오늘의 추천 ───────────────────────────────────────────────
@dataclass
class Pick:
    code: str
    name: str
    market: str
    marcap: float
    close: float
    chg: float
    chg_pct: float
    strategy: Strategy
    rank: int
    days_ago: int
    target: float
    stop: float
    own_n: int
    own_win: float
    df: pd.DataFrame = field(repr=False, default=None)


def recommend(pn: Panel, strategies: list, sims: dict, cfg: dict) -> list:
    fresh = cfg.get("signal_fresh_days", 3)
    last_date = pn.date.max()
    last_idx = {}
    for i in np.flatnonzero(np.r_[pn.sid[1:] != pn.sid[:-1], True]):
        if pn.date[i] == last_date:
            last_idx[pn.sid[i]] = i
    taken = {}
    for rank, s in enumerate(strategies, 1):
        m = pn.cond[:, [pn.keys.index(k) for k in s.combo]].all(axis=1)
        ev = _events(pn, m)
        r2, h2, v2 = sims[(s.kt, s.ks)]
        for sid, i in last_idx.items():
            if sid in taken or not m[i]:
                continue
            lo = max(i - fresh + 1, 0)
            hits = np.flatnonzero(ev[lo:i + 1] & (pn.sid[lo:i + 1] == sid))
            if not len(hits):
                continue
            own = np.flatnonzero(ev & v2 & (pn.sid == sid) & np.isfinite(r2))
            taken[sid] = (s, rank, i - (lo + hits[-1]), own, r2)
    picks = []
    for sid, (s, rank, ago, own, r2) in taken.items():
        row = pn.meta.iloc[sid]
        df = pn.frames[row.Code]
        c, pc = df.Close.iloc[-1], df.Close.iloc[-2]
        atrp = df.ATR_pct.iloc[-1]
        picks.append(Pick(row.Code, row.Name, row.Market, row.Marcap, c, c - pc, (c / pc - 1) * 100,
                          s, rank, ago, c * (1 + s.kt * atrp), c * (1 - s.ks * atrp),
                          len(own), float((r2[own] > 0).mean()) if len(own) else float("nan"), df))
    picks.sort(key=lambda p: (p.strategy.avg_te, p.strategy.avg), reverse=True)
    return picks[:cfg.get("recommend_count", 10)]
