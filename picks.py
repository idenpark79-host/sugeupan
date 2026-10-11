"""추천종목 — 단기(1주)·스윙(1개월)·중장기(3개월).

점수는 하나(투자의견 점수: 추세·섹터·수급·유명 트레이더 기법으로 1개월 시장 대비 수익을 예측)만 쓴다 —
기간별 별도 모델은 검증에서 이 점수보다 낫지 않고 실행마다 결과가 흔들렸다. 그래서 추천과 투자의견이 항상 일치한다.
· 공통: 20·60일선 위(상승 추세 확인), 낙폭과대 반등형 제외
· 단기: 매수 이상 + 거래대금 급증(5일 평균이 60일 평균의 1.5배 이상) 중 점수 상위 10
· 스윙: 매수 이상 중 점수 상위 10
· 중장기: 매수 이상 + 추세 템플릿 8조건 중 7개 이상 중 점수 상위 10
손절가는 1.5 × 일간 변동성 × √기간(종가 기준, 기간별 상한 10·20·25%), 목표가는 손절폭의 2배(손익비 2:1).
검증: 학습에 쓰지 않은 기간에 매일 같은 규칙으로 고른 종목을 다음 날 시가에 사서 목표가·손절가·기간 만료로 판 결과.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HZ = {5: ("단기", "1주"), 20: ("스윙", "1개월"), 60: ("중장기", "3개월")}
RISK = {5: (0.04, 0.10), 20: (0.08, 0.20), 60: (0.12, 0.25)}   # 손절폭 하한·상한 — 상한이 좁으면 변동 큰 종목이 정상 등락에 손절돼 성과가 크게 줄었다(검증)
K_T1 = 0.5   # 1차 목표 = 0.5 × 일간 변동성 × √기간 — 과거 추천 종목의 약 60%가 기간 안에 한 번 이상 닿은 수준
K_STOP, RR = 1.5, 2.0   # 손절폭 = 1.5 × 일간 변동성 × √기간, 목표 = 손절폭 × 2 (과거 검증에서 그냥 보유와 비슷한 성과를 지키는 가장 좁은 손절)
SMOOTH = {5: 3, 20: 5, 60: 5}
COST = 0.0025
TOPN = 10
PER_THEME = 3            # 한 업종에서 최대 3종목 — 한 업종에 몰리면 같은 날 같이 빠질 위험이 커서
MIN_AMT = 2e9
MIN_OP = 0.75            # 투자의견 매수 이상
FLOW_G = {"frg": "외국인", "inst": "기관", "pen": "연기금", "trust": "투신", "pef": "사모"}


def _trend_ok(F):
    """추세 확인: 주가가 20일·60일 이동평균선 위 + 낙폭과대(3개월 −15% 이하 또는 1개월 −8% 이하) 반등 기대 종목 제외.
    검증: 떨어지는 칼날을 잡는 반등형 종목을 빼도 평균 수익은 비슷하거나 높았고(중장기 대형주는 시장 대비 −0.2%p → +2%p대), 근거가 기술적 추세로 일관된다."""
    return ((F["d20"] > 0) & (F["d60"] > 0) & ~((F["r60"] < -0.15) | (F["r20"] < -0.08))).values


STYLE = {5: lambda F: (_trend_ok(F) & (F["amtr"] >= 1.5).values).astype("float32"),
         20: lambda F: _trend_ok(F).astype("float32"),
         60: lambda F: (_trend_ok(F) & (F["mm"] >= 7).values).astype("float32")}
RULE = {5: "투자의견 매수 이상 + 20·60일선 위 + 최근 5일 거래대금이 3개월 평균의 1.5배 이상인 종목 중 점수 순 (낙폭과대 반등형 제외)",
        20: "투자의견 매수 이상 + 20·60일선 위(상승 추세 확인) 종목 중 점수 순 (낙폭과대 반등형 제외)",
        60: "투자의견 매수 이상 + 추세 템플릿 8조건 중 7개 이상(장기 정배열) 종목 중 점수 순 (낙폭과대 반등형 제외)"}


def tick(p: float) -> float:
    for lim, t in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if p < lim:
            return round(p / t) * t
    return round(p / 1000) * 1000


def _walk(c, y, Hh):
    X, d_i, starts, fold, nd, rng = c["X"], c["d_i"], c["starts"], c["fold"], c["nd"], c["rng"]
    lab = np.isfinite(y)
    lo, hi = np.nanpercentile(y[lab], [1, 99])
    yc = np.clip(y, lo, hi)
    pred = np.full(len(y), np.nan)
    for s0 in starts:
        s1 = min(s0 + fold, nd)
        tr = np.flatnonzero(lab & (d_i <= s0 - Hh - 1))
        te = np.flatnonzero((d_i >= s0) & (d_i < s1))
        if len(tr) < 5000 or not len(te):
            continue
        it = tr if len(tr) <= 300_000 else rng.choice(tr, 300_000, replace=False)
        pred[te] = c["reg"]().fit(X[it], yc[it]).predict(X[te])
    it = np.flatnonzero(lab)
    it = it if len(it) <= 400_000 else rng.choice(it, 400_000, replace=False)
    today = np.flatnonzero(d_i == nd - 1)
    return pred, today, c["reg"]().fit(X[it], yc[it]).predict(X[today])


def _smooth(d_i, ci, pred, today, p_today, k, nd):
    P = pd.DataFrame({"d": d_i, "c": ci, "p": pred})
    P = P[np.isfinite(P.p)]
    P["p0"] = P.groupby("d").p.rank(pct=True)
    P = P.sort_values(["c", "d"])
    P["s"] = P.groupby("c").p0.transform(lambda s: s.rolling(k, min_periods=1).mean())
    P["pct"] = P.groupby("d").s.rank(pct=True)
    p0 = pd.Series(p_today).rank(pct=True).values
    prev = P[(P.d >= nd - k) & (P.d < nd - 1)].groupby("c").p0.apply(list)
    ct = ci[today]
    sm = np.array([np.mean((prev.get(c_, []) or [])[-(k - 1):] + [p0[j]]) for j, c_ in enumerate(ct)])
    return P[["d", "c", "pct"]], pd.Series(pd.Series(sm).rank(pct=True).values, index=ct)


def _sim(A, d, cc, Hh, risk):
    """다음 날 시가 매수 → 장중 목표가 도달 시 매도, 종가가 손절가 아래면 다음 날 시가에 매도, 아니면 기간 만료 종가에 매도."""
    O, Hi, Lo, C = A
    entry = O[d + 1, cc]
    idx = d[:, None] + np.arange(1, Hh + 1)[None, :]
    o, h, cl = O[idx, cc[:, None]], Hi[idx, cc[:, None]], C[idx, cc[:, None]]
    stop, tgt = entry * (1 - risk), entry * (1 + RR * risk)
    with np.errstate(invalid="ignore"):
        hs, ht = cl <= stop[:, None], h >= tgt[:, None]
    big = Hh + 1
    fs = np.where(hs.any(1), hs.argmax(1), big)
    ft = np.where(ht.any(1), ht.argmax(1), big)
    r = np.arange(len(d))
    stopped = (fs < big) & (fs < ft)
    hit = (ft < big) & (ft <= fs)
    px_s = np.where(fs + 1 <= Hh - 1, o[r, np.minimum(fs + 1, Hh - 1)], cl[r, np.minimum(fs, Hh - 1)])
    px_t = np.fmax(o[r, np.minimum(ft, Hh - 1)], tgt)
    last = C[d + Hh, cc]
    ret = np.where(stopped, px_s, np.where(hit, px_t, last)) / entry - 1 - COST
    return ret, stopped, hit, last / entry - 1 - COST


def _sim_half(A, d, cc, Hh, risk, t1):
    """분할 매도: 1차 목표에 닿으면 절반 매도 + 나머지 손절가를 매수가(본전)로 올림 → 2차 목표·본전 손절·기간 만료 중 먼저."""
    O, Hi, Lo, C = A
    e = O[d + 1, cc]
    idx = d[:, None] + np.arange(1, Hh + 1)[None, :]
    o, h, cl = O[idx, cc[:, None]], Hi[idx, cc[:, None]], C[idx, cc[:, None]]
    out = np.full(len(d), np.nan)
    for j in range(len(d)):
        if not np.isfinite(e[j]):
            continue
        st, T1, T2, half, res, w = e[j] * (1 - risk[j]), e[j] * (1 + t1[j]), e[j] * (1 + RR * risk[j]), False, 0.0, 1.0
        for k in range(Hh):
            if not np.isfinite(cl[j, k]):
                continue
            if not half and h[j, k] >= T1:
                res, w, half, st = 0.5 * (max(o[j, k], T1) / e[j] - 1), 0.5, True, e[j]
            if h[j, k] >= T2:
                out[j] = res + w * (max(o[j, k], T2) / e[j] - 1)
                break
            if cl[j, k] <= st:
                px = o[j, k + 1] if k + 1 < Hh and np.isfinite(o[j, k + 1]) else cl[j, k]
                out[j] = res + w * (px / e[j] - 1)
                break
        else:
            out[j] = res + w * (C[d[j] + Hh, cc[j]] / e[j] - 1)
    return out - COST


LV_MA, LV_DMIN, LV_DMAX, LV_T2, LV_FAR = False, 0.5, 1.3, 1.5, 1.6   # 검증: 이평선보다 실제 저점이 손절 근거로 나았고, 너무 가까운 저항을 2차 목표로 쓰면 수익이 줄었다
LOOK = {5: 60, 20: 120, 60: 250}      # 지지·저항을 찾는 기간(봉)
PIVK = {5: 3, 20: 5, 60: 8}           # 고점·저점: 앞뒤 k봉 중 가장 높은(낮은) 봉


def _levels(A, d, c, Hh, s, dates=None):
    """손절가·1차·2차 목표가를 차트의 지지·저항에서 정한다.
    손절: 아래쪽 가장 가까운 지지(최근 저점·20/60/120일선) 1% 아래 — 단, 기간에 맞는 흔들림 폭 안에 있을 때만
    1차 목표: 위쪽 가장 가까운 저항(최근 고점·52주 최고가) 바로 아래 / 2차 목표: 그다음 저항, 없으면 손절폭의 2배"""
    from numpy.lib.stride_tricks import sliding_window_view as swv
    O, Hi, Lo, C = A
    a = max(0, d - LOOK[Hh] + 1)
    h, l, cl = Hi[a:d + 1, c], Lo[a:d + 1, c], C[a:d + 1, c]
    px = C[d, c]
    if not np.isfinite(px) or len(h) < 30:
        return None
    k = PIVK[Hh]
    sv = (s if np.isfinite(s) else 0.02) * np.sqrt(Hh)
    lo_c, hi_c = RISK[Hh]
    fmt = lambda i: pd.Timestamp(dates[a + i]).strftime("%m.%d") if dates is not None else ""
    sup, res = [], []
    if len(h) > 2 * k + 1:
        wh, wl = swv(h, 2 * k + 1), swv(l, 2 * k + 1)
        with np.errstate(invalid="ignore"):
            ph = np.flatnonzero(h[k:-k] >= np.nanmax(wh, 1)) + k
            pl = np.flatnonzero(l[k:-k] <= np.nanmin(wl, 1)) + k
        res += [(float(h[i]), f"{fmt(i)} 고점") for i in ph if h[i] > px * 1.005]
        sup += [(float(l[i]), f"{fmt(i)} 저점") for i in pl if l[i] < px * 0.995]
    if LV_MA:
        for n in (20, 60, 120):
            if d + 1 >= n:
                m = float(np.nanmean(C[d - n + 1:d + 1, c]))
                if np.isfinite(m) and m < px * 0.995:
                    sup.append((m, f"{n}일 이동평균선"))
    lowall = float(np.nanmin(l))
    if lowall < px * 0.995:
        sup.append((lowall, f"최근 {LOOK[Hh] // 20}개월 최저가"))
    h52 = float(np.nanmax(Hi[max(0, d - 249):d + 1, c]))
    if h52 > px * 1.005:
        res.append((h52, "52주 최고가"))
    # 손절
    dmin, dmax = max(lo_c * 0.6, LV_DMIN * sv), hi_c * LV_DMAX
    dist = lambda lv: 1 - lv * 0.99 / px
    if LV_MA is False:          # 이동평균선은 실제 저점 지지가 없을 때만
        mas = []
        for n in ((5, 10, 20, 60) if Hh == 5 else (10, 20, 60, 120)):
            if d + 1 >= n:
                m = float(np.nanmean(C[d - n + 1:d + 1, c]))
                if np.isfinite(m) and m < px * 0.995:
                    mas.append((m, f"{n}일 이동평균선"))
    else:
        mas = []
    tiers = [[x for x in sup if dmin <= dist(x[0]) <= dmax], [x for x in mas if dmin <= dist(x[0]) <= dmax],
             [x for x in sup + mas if dmax < dist(x[0]) <= hi_c * LV_FAR],
             [x for x in sup + mas if lo_c * 0.5 <= dist(x[0]) < dmin]]
    lo10 = float(np.nanmin(Lo[max(0, d - 9):d + 1, c]))
    if lo10 < px * 0.985:
        tiers.append([(lo10, "최근 2주 최저가")])
    pick_ = next((sorted(t, key=lambda x: -x[0])[0] for t in tiers if t), None)
    if pick_:
        stop, sw = pick_[0] * 0.99, ("sup", pick_[1], pick_[0])
    else:
        r = float(min(max(1.5 * sv, lo_c * 0.6), hi_c * LV_DMAX))
        stop, sw = px * (1 - r), ("vol", None, None)
    risk = 1 - stop / px
    # 목표
    rs = sorted(set((round(lv, 2), w) for lv, w in res if lv / px - 1 >= max(0.25 * sv, 0.02)))
    if rs:
        t1, t1w = rs[0][0] * 0.995, ("res", rs[0][1], rs[0][0])
        rest = [x for x in rs[1:] if x[0] * 0.995 >= t1 * 1.03]
    else:
        t1, t1w, rest = px * (1 + max(risk, 0.5 * sv)), ("free", None, None), []
    rest = [x for x in rest if x[0] * 0.995 / px - 1 >= LV_T2 * risk]
    if rest:
        t2, t2w = rest[0][0] * 0.995, ("res", rest[0][1], rest[0][0])
    else:
        t2, t2w = max(px * (1 + 2 * risk), t1 * 1.05), ("rr", None, None)
    if t2 < t1 * 1.03:
        t2, t2w = t1 * 1.05, ("rr", None, None)
    return {"stop": stop, "t1": t1, "t2": t2, "sw": sw, "t1w": t1w, "t2w": t2w, "risk": risk}


def _sim_lv(A, d, cc, Hh, stop, tgt):
    """절대 가격 손절·목표로 모의 매매 (규칙은 _sim과 같음)."""
    O, Hi, Lo, C = A
    entry = O[d + 1, cc]
    idx = d[:, None] + np.arange(1, Hh + 1)[None, :]
    o, h, cl = O[idx, cc[:, None]], Hi[idx, cc[:, None]], C[idx, cc[:, None]]
    with np.errstate(invalid="ignore"):
        hs, ht = cl <= stop[:, None], h >= tgt[:, None]
    big = Hh + 1
    fs = np.where(hs.any(1), hs.argmax(1), big)
    ft = np.where(ht.any(1), ht.argmax(1), big)
    r = np.arange(len(d))
    stopped = (fs < big) & (fs < ft)
    hit = (ft < big) & (ft <= fs)
    px_s = np.where(fs + 1 <= Hh - 1, o[r, np.minimum(fs + 1, Hh - 1)], cl[r, np.minimum(fs, Hh - 1)])
    px_t = np.fmax(o[r, np.minimum(ft, Hh - 1)], tgt)
    last = C[d + Hh, cc]
    ret = np.where(stopped, px_s, np.where(hit, px_t, last)) / entry - 1 - COST
    return ret, stopped, hit, last / entry - 1 - COST


def _risk(sig, Hh):
    lo, hi = RISK[Hh]
    return np.clip(K_STOP * sig * np.sqrt(Hh), lo, hi)


SEG = {"L": ("대형주", "시가총액 1조 원 이상"), "S": ("중소형주", "시가총액 1조 원 미만")}


def _val(V, dates):
    V = V.copy()
    t1 = {"t1": round(float(V.t1.mean()) * 100, 1), "t1base": round(float(V.t1b.mean()) * 100, 1)} if "t1" in V else {}
    if "r2" in V and V.r2.notna().any():
        t1.update({"win2": round(float((V.r2 > 0).mean()) * 100, 1), "avg2": round(float(V.r2.mean()) * 100, 2)})
    V["mo"] = pd.to_datetime(dates[V.d.values]).strftime("%Y-%m")
    mon = V.groupby("mo").agg(r=("r", "mean"), mk=("mk", "mean"))
    return {"n": int(len(V)), "avg": round(float(V.r.mean()) * 100, 2), "mkt": round(float(V.mk.mean()) * 100, 2),
            "win": round(float((V.r > 0).mean()) * 100, 1), "hitT": round(float(V.t.mean()) * 100, 1),
            "hitS": round(float(V.s.mean()) * 100, 1), "hold": round(float(V.h.mean()) * 100, 2),
            "ew": round(float(V.ew.mean()) * 100, 2), "up": round(float(V.r[V.r > 0].mean()) * 100, 1),
            "dn": round(float(V.r[V.r <= 0].mean()) * 100, 1),
            "beat": round(float((mon.r > mon.mk).mean()) * 100, 1), "months": int(len(mon)),
            "from": str(pd.Timestamp(dates[int(V.d.min())]).date()) if len(V) else None,
            "to": str(pd.Timestamp(dates[int(V.d.max())]).date()) if len(V) else None, **t1}


def build(c: dict, log=print) -> dict:
    C, O, Hh_, L = c["C"], c["O"], c["Hh"], c["L"]
    codes, dates, nd, F, th = c["codes"], c["dates"], c["nd"], c["F"], c["th"]
    A = tuple(np.asarray(x.values, dtype="float64") for x in (O, Hh_, L, C))
    amt60 = c["amt60"].values
    sig = c["vol60"].values
    midx = c["midx"].values
    big = (c["w"].values >= 1e12)                  # 추정 시가총액 1조 원 이상 = 대형주
    oos0 = c["starts"][0]
    thc = np.array([th.get(x, "기타") for x in codes], dtype=object)
    P20 = c["P"][["d", "c", "pct"]]
    op_today = pd.Series(c["pct_today"], index=c["c_today"])
    t_last = nd - 1
    out = {}
    for Hh in (5, 20, 60):
        flt = STYLE[Hh](F)
        fw = A[3][Hh + 1:] / A[0][1:-Hh] - 1
        fmax = Hh_.iloc[::-1].rolling(Hh, min_periods=1).max().iloc[::-1].shift(-1).values      # 다음 날~H일 뒤 최고가
        reach = np.full_like(fmax, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            reach[:-1] = (fmax[:-1] / A[0][1:] - 1 >= K_T1 * sig[:-1] * np.sqrt(Hh)).astype(float)
        reach[nd - Hh - 1:] = np.nan
        elig = amt60[:-Hh - 1] >= MIN_AMT
        Q0 = P20[(P20.d >= oos0) & (P20.d + Hh < nd - 1) & (P20.pct >= MIN_OP)]
        if flt is not None:
            Q0 = Q0[np.nan_to_num(flt[Q0.d.values, Q0.c.values]) > 0]
        Q0 = Q0[amt60[Q0.d.values, Q0.c.values] >= MIN_AMT]
        res = {"H": Hh, "label": HZ[Hh][0], "hold": HZ[Hh][1], "rule": RULE[Hh], "picks": [], "segs": {}}
        allV = []
        for sg, (sname, sdesc) in SEG.items():
            inseg = big[Q0.d.values, Q0.c.values] if sg == "L" else ~big[Q0.d.values, Q0.c.values]
            Q = Q0[inseg].sort_values(["d", "pct"], ascending=[True, False])
            tq = thc[Q.c.values]
            Q = Q[(Q.assign(th_=tq).groupby(["d", "th_"]).cumcount().values < PER_THEME) | (tq == "기타")].groupby("d").head(TOPN)
            d, cc = Q.d.values.astype(int), Q.c.values.astype(int)
            LV = [_levels(A, int(d_), int(c_), Hh, sig[d_, c_]) for d_, c_ in zip(d, cc)]
            okl = np.array([x is not None for x in LV])
            d, cc, LV = d[okl], cc[okl], [x for x in LV if x is not None]
            stp_ = np.array([x["stop"] for x in LV]); t1_ = np.array([x["t1"] for x in LV]); t2_ = np.array([x["t2"] for x in LV])
            ret, st_, hit, hold = _sim_lv(A, d, cc, Hh, stp_, t2_)
            hmax = np.nanmax(A[1][d[:, None] + np.arange(1, Hh + 1)[None, :], cc[:, None]], axis=1)
            rch = (hmax >= t1_).astype(float)
            rk_ = 1 - stp_ / A[3][d, cc]
            ret2 = _sim_half(A, d, cc, Hh, rk_, t1_ / A[0][d + 1, cc] - 1)
            mk = midx[d + Hh] / midx[d] - 1
            segm = big[:-Hh - 1] if sg == "L" else ~big[:-Hh - 1]
            ew_day = np.nanmean(np.where(elig & segm, fw, np.nan), axis=1)
            t1b_day = np.nanmean(np.where(elig & segm & np.isfinite(sig[:-Hh - 1]), reach[:-Hh - 1], np.nan), axis=1)
            m = np.isfinite(ret) & np.isfinite(mk)
            V = pd.DataFrame({"d": d[m], "r": ret[m], "s": st_[m], "t": hit[m], "h": hold[m], "mk": mk[m], "ew": ew_day[d][m],
                              "t1": rch[m], "t1b": t1b_day[d][m], "r2": ret2[m],
                              "rp": rk_[m], "t1p": (t1_ / A[3][d, cc] - 1)[m], "t2p": (t2_ / A[3][d, cc] - 1)[m]})
            allV.append(V)
            val = _val(V, dates) if len(V) else {}
            val["name"], val["desc"] = sname, sdesc
            res["segs"][sg] = val
            if len(V):
                log(f"    추천 {HZ[Hh][0]}·{sname} 검증: 평균 {val['avg']:+.2f}% · 같은 규모 일반 종목 {val['ew']:+.2f}% · 지수 {val['mkt']:+.2f}% · "
                    f"수익 마감 {val['win']}% (분할 매도 시 {val.get('win2')}%, 평균 {val.get('avg2')}%) · 1차 목표 도달 {val.get('t1')}% (같은 규모 {val.get('t1base')}%) · 목표 {val['hitT']}% · 손절 {val['hitS']}% · {val['n']:,}건")
            cand = [(ci_, float(p)) for ci_, p in op_today.items()
                    if p >= MIN_OP and amt60[t_last, ci_] >= MIN_AMT and np.isfinite(A[3][t_last, ci_])
                    and (big[t_last, ci_] if sg == "L" else not big[t_last, ci_])
                    and (flt is None or np.nan_to_num(flt[t_last, ci_]) > 0)]
            cand.sort(key=lambda x: -x[1])
            cnt, sel = {}, []
            for ci_, p in cand:
                k_ = thc[ci_]
                if cnt.get(k_, 0) < PER_THEME or k_ == "기타":
                    cnt[k_] = cnt.get(k_, 0) + 1
                    sel.append((ci_, p))
            for rank, (ci_, p) in enumerate(sel[:TOPN], 1):
                code = codes[ci_]
                close = float(A[3][t_last, ci_])
                s0 = float(sig[t_last, ci_]) if np.isfinite(sig[t_last, ci_]) else 0.02
                lv = _levels(A, t_last, ci_, Hh, s0, dates)
                if lv is None:
                    continue
                info = _story(c, code, ci_, Hh, val)
                st_p, t1_p, t2_p = tick(lv["stop"]), tick(lv["t1"]), tick(lv["t2"])
                lv.update(stop=st_p, t1=t1_p, t2=t2_p, risk=1 - st_p / close)
                res["picks"].append({"code": code, "name": c["names"].get(code, code), "close": close, "size": sg, "rank": rank,
                                     "t1": t1_p, "t1Pct": round((t1_p / close - 1) * 100, 1),
                                     "target": t2_p, "stop": st_p,
                                     "tgtPct": round((t2_p / close - 1) * 100, 1), "stpPct": round((st_p / close - 1) * 100, 1),
                                     "score": round(p * 100, 1), "theme": th.get(code, "기타"),
                                     "plan": _plan2(Hh, s0, close, lv, val), "prob": val.get("win", 50) / 100, "exp": val.get("avg", 0) / 100, **info})
        res["val"] = _val(pd.concat(allV), dates)
        out[str(Hh)] = res
    try:
        out["warn"] = warn(c, A, P20, op_today, big, amt60, log)
    except Exception as e:  # 하락 주의는 부가 정보 — 실패해도 추천은 그대로
        log(f"    하락 주의 실패: {e}")
    return out


WARN_MAX = 0.25    # 퀀트 스코어 하위 25% 안에서
WARN_N = 10


def warn(c, A, P20, op_today, big, amt60, log=print) -> dict:
    """하락 주의 — 중소형주 중 퀀트 스코어 최하위 10종목.
    검증(학습에 쓰지 않은 기간): 과거 같은 조건 종목은 1개월 뒤 약 3분의 2, 3개월 뒤 약 70%가 산 가격보다 낮았다.
    대형주는 같은 방법으로 하락을 맞히는 비율이 절반 수준이라 목록에서 뺐다."""
    nd, dates, codes = c["nd"], c["dates"], c["codes"]
    O, C = A[0], A[3]
    oos0, t_last = c["starts"][0], nd - 1
    val = {}
    for Hh in (20, 60):
        fw = np.full_like(C, np.nan)
        fw[:-Hh - 1] = C[Hh + 1:] / O[1:-Hh] - 1
        Q = P20[(P20.d >= oos0) & (P20.d + Hh < nd - 1) & (P20.pct <= WARN_MAX)]
        Q = Q[(amt60[Q.d.values, Q.c.values] >= MIN_AMT) & ~big[Q.d.values, Q.c.values]]
        Q = Q.sort_values(["d", "pct"]).groupby("d").head(WARN_N)
        d, cc = Q.d.values.astype(int), Q.c.values.astype(int)
        r = fw[d, cc]
        segm = (amt60[:-Hh - 1] >= MIN_AMT) & ~big[:-Hh - 1]
        ew = np.nanmean(np.where(segm, fw[:-Hh - 1], np.nan), axis=1)[d]
        m = np.isfinite(r) & np.isfinite(ew)
        r, ew, d = r[m], ew[m], d[m]
        mo = pd.Series(r - ew).groupby(pd.to_datetime(dates[d]).strftime("%Y-%m")).mean()
        val[str(Hh)] = {"n": int(len(r)), "fell": round(float((r < 0).mean()) * 100, 1), "big": round(float((r < -0.1).mean()) * 100, 1),
                        "up10": round(float((r > 0.1).mean()) * 100, 1), "avg": round(float(r.mean()) * 100, 2),
                        "ew": round(float(ew.mean()) * 100, 2), "under": round(float((mo < 0).mean()) * 100, 1), "months": int(len(mo)),
                        "from": str(pd.Timestamp(dates[int(d.min())]).date()), "to": str(pd.Timestamp(dates[int(d.max())]).date())}
        v = val[str(Hh)]
        log(f"    하락 주의 {HZ[Hh][1]} 검증: 하락 {v['fell']}% · 10% 넘게 하락 {v['big']}% · 평균 {v['avg']:+.2f}% (같은 규모 평균 {v['ew']:+.2f}%) · {v['n']:,}건")
    cand = [(ci_, float(p)) for ci_, p in op_today.items()
            if p <= WARN_MAX and amt60[t_last, ci_] >= MIN_AMT and not big[t_last, ci_] and np.isfinite(C[t_last, ci_])]
    cand.sort(key=lambda x: x[1])
    L = []
    for rank, (ci_, p) in enumerate(cand[:WARN_N], 1):
        code = codes[ci_]
        L.append({"code": code, "name": c["names"].get(code, code), "close": float(C[t_last, ci_]), "rank": rank,
                  "score": round(p * 100, 1), "theme": c["th"].get(code, "기타"), **_wstory(c, code)})
    return {"rule": "중소형주(시가총액 1조 원 미만) 중 퀀트 스코어가 가장 낮은 10종목", "val": val, "list": L}


def _wstory(c, code):
    """하락 주의 근거 — 냉정하게, 숫자로."""
    F, t, th, tnow, flows = c["F"], c["nd"] - 1, c["th"], c["tnow"], c.get("flows") or {}
    s = []
    Cc = c["C"][code].dropna()
    if len(Cc) >= 221:
        m200 = Cc.rolling(200).mean()
        px, a, ap = Cc.iloc[-1], m200.iloc[-1], m200.iloc[-21]
        if px < a and a < ap:
            s.append(f"장기 하락 추세예요. 주가({px:,.0f}원)가 200일 이동평균선({a:,.0f}원) 아래에 있고, 200일선도 한 달 전보다 내려가고 있어요.")
        elif px < a:
            s.append(f"주가가 200일 이동평균선({a:,.0f}원) 아래에 있어요. 지난 1년 가까이 산 사람들의 평균 가격보다 낮아, 반등 때마다 팔려는 물량이 나오기 쉬워요.")
    tk = _tech(c, code)
    dead = [r for r in tk if r[0].startswith("데드크로스")]
    if dead:
        s.append(f"{dead[0][0]}가 {dead[0][2]}했어요. 짧은 기간 평균 가격이 긴 기간 평균 가격 아래로 내려갔다는 뜻으로, 하락 흐름이 굳어지는 신호예요.")
    ma = {n: Cc.rolling(n).mean().iloc[-1] for n in (5, 20, 60, 120)} if len(Cc) >= 120 else None
    if ma and ma[5] < ma[20] < ma[60] < ma[120]:
        s.append("이동평균선이 역배열(5 < 20 < 60 < 120일선)이에요. 짧게 보든 길게 보든 최근에 산 사람일수록 더 싸게 샀다는 뜻 — 모든 구간이 내리막이에요.")
    fl = flows.get(code)
    if fl is not None and len(fl):
        sel = []
        for g in ("외국인", "연기금", "투신", "기관합계"):
            if g in fl:
                m3 = float(fl[g].iloc[-60:].sum()) / 1e8
                if m3 < 0:
                    sel.append((g, m3))
        if sel:
            sel.sort(key=lambda x: x[1])
            s.append(f"{sel[0][0]}이 최근 3개월 동안 {-sel[0][1]:,.0f}억 원을 순매도했어요." + (f" {'·'.join(x[0] for x in sel[1:3])}도 팔고 있어요." if len(sel) > 1 else ""))
    r20 = _v(F, "r20", t, code)
    if r20 is not None and r20 >= 0.2:
        s.append(f"최근 한 달 {r20 * 100:+.0f}% 급등했지만 추세·수급이 받쳐 주지 않아요. 과거 이런 종목은 상승분을 되돌리는 경우가 많았어요.")
    lo52 = _v(F, "lo52", t, code)
    if lo52 is not None and lo52 <= 0.05:
        s.append("52주 최저가 근처에 있어요. 바닥처럼 보여도, 신저가를 갱신하는 종목은 더 내려가는 경우가 많아요.")
    tn = tnow.get(th.get(code))
    if tn and tn["rank"] > len(tnow) * 0.75:
        s.append(f"'{th.get(code)}' 업종 자체가 약해요({len(tnow)}개 업종 중 {tn['rank']}위).")
    if not s:
        s.append("추세·수급·업종을 함께 본 퀀트 스코어가 전체 종목 중 가장 낮은 쪽이에요.")
    tr = _trend(c, code)
    mm = sum(r[1] for r in tr) if tr else None
    ev = [{"k": "추세 템플릿", "v": f"{mm}/8 충족" if mm is not None else "-", "ok": (mm or 0) >= 6, "rows": tr or []},
          {"k": "기술적 신호", "v": f"{sum(r[1] for r in tk)}/{len(tk)} 긍정" if tk else "-", "ok": bool(tk) and sum(r[1] for r in tk) >= len(tk) * 0.6, "rows": tk}]
    return {"story": s[:5], "ev": ev}


def _plan2(Hh, s0, px, lv, val):
    """지지·저항 근거 설명."""
    w = lambda x: f"{x:,.0f}원"
    sw, t1w, t2w = lv["sw"], lv["t1w"], lv["t2w"]
    if sw[0] == "sup":
        stop = (f"{sw[1]}({w(sw[2])})이 아래쪽 지지선이에요. 최근에 사는 힘이 들어와 주가가 버틴 자리라, 종가가 이보다 1% 더 아래({w(lv['stop'])})로 마감하면 "
                f"지지가 깨졌다고 보고 다음 날 정리해요. 산 가격 대비 {-lv['risk'] * 100:.1f}%예요.")
    else:
        stop = (f"가까운 아래쪽에 {HZ[Hh][1]} 보유에 맞는 지지선이 없어서, 이 종목이 {HZ[Hh][1]} 동안 보통 흔들리는 폭(하루 평균 {s0 * 100:.1f}% 움직임 기준)을 넘는 "
                f"{w(lv['stop'])}에 손절가를 뒀어요.")
    if t1w[0] == "res":
        t1 = f"{t1w[1]}({w(t1w[2])})이 위쪽 첫 저항이에요. 과거에 여기서 팔려는 물량이 나와 막혔던 가격이라 바로 아래 {w(lv['t1'])}을 1차 목표로 잡았어요. 닿으면 일부를 팔아 수익을 확정하세요."
    else:
        t1 = f"최근 고점 위로 올라선 상태라 위쪽에 막힐 가격(저항)이 없어요. 그래서 손절폭과 같은 만큼 오른 {w(lv['t1'])}을 1차 목표로 잡았어요."
    if t2w[0] == "res":
        t2 = f"그 위 다음 저항은 {t2w[1]}({w(t2w[2])})이에요. 1차 저항을 넘으면 이 가격까지 열려 있어요."
    else:
        t2 = f"1차 목표 위로 손절폭의 1.5배 이상 떨어진 뚜렷한 저항이 없어서, 손절할 때 잃는 폭의 2배({w(lv['t2'])})를 2차 목표로 잡았어요."
    t2 += f" 과거 같은 방법으로 고른 종목은 {val.get('t1', 0):.0f}%가 {HZ[Hh][1]} 안에 1차 목표에 닿았고, {val.get('hitT', 0):.0f}%가 2차 목표까지 갔어요."
    return {"stop": stop, "t1": t1, "target": t2}


def _plan(Hh, s0, r_, val, t1p=None):
    move = K_STOP * s0 * np.sqrt(Hh) * 100
    lo, hi = RISK[Hh]
    cap = "" if lo * 100 < move < hi * 100 else (f" 다만 {HZ[Hh][1]} 보유 기준 손실은 최대 {hi * 100:.0f}%까지만 감수하도록 제한했어요." if move >= hi * 100
                                                  else f" 움직임이 작은 종목이라 최소 {lo * 100:.0f}% 폭은 두었어요.")
    return {"stop": f"이 종목은 하루에 평균 {s0 * 100:.1f}% 정도 오르내려요. {HZ[Hh][1]}이면 보통 {move:.0f}% 안팎까지 흔들릴 수 있어서, "
                    f"그보다 더 내려가 종가가 손절가 아래로 마감하면 '예상이 틀렸다'고 보고 다음 날 정리하는 가격이에요.{cap}",
            "t1": (f"이 종목이 {HZ[Hh][1]} 동안 보통 움직이는 폭의 절반({t1p * 100:.0f}%) 위예요. 과거 같은 방법으로 고른 종목은 {val.get('t1', 0):.0f}%가 "
                   f"기간 안에 한 번 이상 이 높이에 닿았어요(같은 규모 일반 종목은 {val.get('t1base', 0):.0f}%). 닿으면 일부를 팔아 수익을 확정하는 자리예요.") if t1p else "",
            "target": f"손절할 때 잃는 폭({r_ * 100:.0f}%)의 2배예요. 잃을 때보다 벌 때 2배 크게 가져가자는 원칙이에요. "
                      f"과거 같은 방법에서 2차 목표에 먼저 닿은 경우는 {val.get('hitT', 0):.0f}%였고, 닿지 않으면 {HZ[Hh][1]} 뒤 그때 가격으로 정리했어요."}


def _v(F, k, t, code):
    try:
        x = float(F[k][code].iloc[t])
        return x if np.isfinite(x) else None
    except Exception:
        return None


FLOW_G = ["외국인", "연기금", "투신", "사모", "금융투자", "보험", "기관합계"]


def _trend(c, code):
    """추세 템플릿 8조건(절대 기준 7 + 상대강도 1) — 이동평균선 용어로."""
    C, Hh_, L = c["C"][code].dropna(), c["Hh"][code].dropna(), c["L"][code].dropna()
    if len(C) < 210:
        return None
    m50, m150, m200 = (C.rolling(n).mean() for n in (50, 150, 200))
    px, a50, a150, a200, a200p = C.iloc[-1], m50.iloc[-1], m150.iloc[-1], m200.iloc[-1], m200.iloc[-21]
    lo, hi = L.iloc[-250:].min(), Hh_.iloc[-250:].max()
    r12 = c["C"].iloc[-1] / c["C"].iloc[-251] - 1 if len(c["C"]) > 251 else None
    rk = float((r12 < r12[code]).mean()) if r12 is not None and np.isfinite(r12.get(code, np.nan)) else None
    n = lambda x: f"{x:,.0f}원"
    rows = [
        ("주가가 150일·200일 이동평균선 위", px > a150 and px > a200, f"현재 {n(px)} · 150일선 {n(a150)} · 200일선 {n(a200)}"),
        ("150일선이 200일선 위 (장기 정배열)", a150 > a200, f"150일선 {n(a150)} · 200일선 {n(a200)}"),
        ("200일선이 1개월 전보다 상승 (장기 추세 상승 전환)", a200 > a200p, f"1개월 전 {n(a200p)} → 현재 {n(a200)}"),
        ("50일선이 150일·200일선 위 (중장기 정배열)", a50 > a150 and a50 > a200, f"50일선 {n(a50)}"),
        ("주가가 50일선 위", px > a50, f"50일선 대비 {(px / a50 - 1) * 100:+.1f}%"),
        ("52주 최저가 대비 30% 이상 상승", px >= lo * 1.3, f"52주 최저 {n(lo)} 대비 {(px / lo - 1) * 100:+.0f}%"),
        ("52주 최고가 대비 25% 이내 (신고가권 근접)", px >= hi * 0.75, f"52주 최고 {n(hi)} 대비 {(px / hi - 1) * 100:+.0f}%"),
        ("52주 상대강도(RS) 상위 30% — 다른 종목과 비교", rk is not None and rk >= 0.7,
         (f"상위 {max(1, round((1 - rk) * 100))}%" if rk >= 0.5 else f"하위 {max(1, round(rk * 100))}%") if rk is not None else "-"),
    ]
    return [[a, bool(b), x] for a, b, x in rows]


def _tech(c, code):
    """기술적 신호 — 정배열, 골든크로스, MACD, RSI, 볼린저밴드, 거래량."""
    C = c["C"][code].dropna()
    V = c["V"][code].reindex(C.index) if "V" in c else None
    if len(C) < 130:
        return []
    ma = {n: C.rolling(n).mean() for n in (5, 20, 60, 120)}
    rows = []
    al = ma[5].iloc[-1] > ma[20].iloc[-1] > ma[60].iloc[-1] > ma[120].iloc[-1]
    rows.append(["이동평균선 정배열 (5 > 20 > 60 > 120일선)", bool(al), " > ".join(f"{n}일 {ma[n].iloc[-1]:,.0f}" for n in (5, 20, 60, 120))])
    for a, b in ((5, 20), (20, 60), (60, 120)):
        d = (ma[a] - ma[b]).iloc[-21:]
        up = np.flatnonzero((d.values[1:] > 0) & (d.values[:-1] <= 0))
        dn = np.flatnonzero((d.values[1:] < 0) & (d.values[:-1] >= 0))
        if len(up) and (not len(dn) or up[-1] > dn[-1]):
            rows.append([f"골든크로스 ({a}일선이 {b}일선 상향 돌파)", True, f"{d.index[up[-1] + 1]:%m.%d} 발생"])
        elif len(dn):
            rows.append([f"데드크로스 ({a}일선이 {b}일선 하향 돌파)", False, f"{d.index[dn[-1] + 1]:%m.%d} 발생"])
    e12, e26 = C.ewm(span=12, adjust=False).mean(), C.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sigl = macd.ewm(span=9, adjust=False).mean()
    rows.append(["MACD가 시그널선 위 (상승 모멘텀)", bool(macd.iloc[-1] > sigl.iloc[-1]), f"MACD {macd.iloc[-1]:,.1f} · 시그널 {sigl.iloc[-1]:,.1f}"])
    dlt = C.diff()
    g = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l_ = (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = float(100 - 100 / (1 + g.iloc[-1] / l_.iloc[-1])) if l_.iloc[-1] > 0 else 100.0
    rows.append(["RSI 50~70 (힘은 있으나 과열 아님)", bool(50 <= rsi <= 70), f"RSI {rsi:.0f}" + (" · 70 이상 과열권" if rsi > 70 else " · 50 미만 약세권" if rsi < 50 else "")])
    m20, sd = ma[20].iloc[-1], C.rolling(20).std().iloc[-1]
    pb = (C.iloc[-1] - (m20 - 2 * sd)) / (4 * sd) if sd > 0 else 0.5
    rows.append(["볼린저밴드 중심선 위", bool(C.iloc[-1] > m20), f"밴드 내 위치 {pb * 100:.0f}% (0%=하단, 100%=상단)"])
    return rows


def _valuation(c, code):
    """재무 — PER·PBR·EPS·BPS·배당을 같은 업종 중앙값과 비교."""
    fund = c.get("fund")
    if not fund:
        return None
    df = fund[max(fund)]
    if code not in df.index:
        return None
    C = c["C"].ffill().iloc[-1]
    th = c["th"]
    t = th.get(code, "기타")
    def ratios(cd):
        if cd not in df.index or not np.isfinite(C.get(cd, np.nan)):
            return None, None, None
        b, e, d = (float(df.loc[cd].get(k, np.nan)) for k in ("BPS", "EPS", "DPS"))
        px = float(C[cd])
        return (px / e if e > 0 else None), (px / b if b > 0 else None), (d / px * 100 if np.isfinite(d) and d >= 0 else None)
    per, pbr, dy = ratios(code)
    mem = [cd for cd, tt in th.items() if tt == t and cd != code]
    pe_s = [x for x in (ratios(cd)[0] for cd in mem) if x and x < 200]
    pb_s = [x for x in (ratios(cd)[1] for cd in mem) if x and x < 50]
    mpe, mpb = (float(np.median(pe_s)) if len(pe_s) >= 3 else None), (float(np.median(pb_s)) if len(pb_s) >= 3 else None)
    row = df.loc[code]
    n_ = lambda x, k=1: "-" if x is None else f"{x:,.{k}f}"
    rows = [["PER (주가 ÷ 주당순이익)", bool(per and mpe and per < mpe), f"{n_(per)}배 · 업종 중앙값 {n_(mpe)}배" if per else "적자(주당순이익 0 이하)라 계산 불가"],
            ["PBR (주가 ÷ 주당순자산)", bool(pbr and mpb and pbr < mpb), f"{n_(pbr, 2)}배 · 업종 중앙값 {n_(mpb, 2)}배"],
            ["EPS (주당순이익)", bool(row.get("EPS", 0) > 0), f"{row.get('EPS', float('nan')):,.0f}원"],
            ["BPS (주당순자산)", bool(row.get("BPS", 0) > 0), f"{row.get('BPS', float('nan')):,.0f}원"],
            ["배당수익률", bool(dy and dy >= 2), f"{n_(dy, 2)}%"]]
    cheap_pb = bool(pbr and mpb and pbr < mpb * 0.8)
    cheap_pe = bool(per and mpe and per < mpe * 0.8)
    rich = bool((pbr and mpb and pbr > mpb * 1.5) or (per and mpe and per > mpe * 1.5))
    v = "업종 대비 저평가" if (cheap_pb or cheap_pe) else "업종 대비 고평가" if rich else "업종 평균 수준"
    story = ""
    if cheap_pb or cheap_pe:
        parts = ([f"PER {per:.1f}배(업종 중앙값 {mpe:.1f}배)"] if cheap_pe else []) + ([f"PBR {pbr:.2f}배(업종 중앙값 {mpb:.2f}배)"] if cheap_pb else [])
        story = f"같은 업종보다 싸게 거래되고 있어요 — {' · '.join(parts)}. 버는 돈과 가진 재산에 비해 주가가 낮다는 뜻이라, 오를 때 여유가 있어요."
    text = ("PER은 회사가 1년에 버는 돈의 몇 배에 주가가 거래되는지, PBR은 회사가 가진 순자산의 몇 배인지를 뜻해요. 업종마다 적정 수준이 달라서 같은 업종 중앙값과 비교했어요. "
            + ("다만 업종보다 비싸다는 것은 그만큼 성장 기대가 이미 주가에 반영돼 있다는 뜻이기도 해요." if rich else ""))
    return {"v": v, "cheap": cheap_pb or cheap_pe, "rich": rich, "rows": rows, "text": text, "story": story}


def _story(c, code, ci_, Hh, val):
    """초등학생도 알아듣게 — 왜 오를 거라고 봤는지 + 근거 4가지(누르면 자세히)."""
    F, t, th, tnow, flows = c["F"], c["nd"] - 1, c["th"], c["tnow"], c.get("flows") or {}
    s = []
    # 1) 큰손
    fl, buyers, frows = flows.get(code), [], []
    if fl is not None and len(fl):
        for g in FLOW_G:
            if g in fl:
                m1, m3 = float(fl[g].iloc[-20:].sum()) / 1e8, float(fl[g].iloc[-60:].sum()) / 1e8
                dd = int((fl[g].iloc[-20:] > 0).sum())
                frows.append([g, round(m1), round(m3), dd])
                if g != "기관합계" and m3 > 0 and m1 > 0:
                    buyers.append((g, m1, m3, dd))
    buyers.sort(key=lambda x: -x[2])
    if buyers:
        b = buyers[0]
        txt = (f"{b[0]}이 최근 3개월 동안 {b[2]:,.0f}억 원을 순매수했고, 최근 한 달(20거래일) 중 {b[3]}일을 사들였어요.")
        if len(buyers) > 1:
            txt += f" {'·'.join(x[0] for x in buyers[1:3])}도 함께 순매수 중이에요."
        s.append(txt)
    # 2) 추세
    tr = _trend(c, code)
    mm = sum(r[1] for r in tr) if tr else None
    if mm is not None and mm >= 7:
        s.append("최근 몇 달 동안 계단을 오르듯 꾸준히 올라왔어요. 50일·150일·200일 평균 가격이 모두 위를 향하고, 지금 가격도 그 평균들보다 위에 있어요. "
                 "오르는 흐름이 이어지고 있다는 뜻이에요.")
    elif mm is not None and mm >= 5:
        s.append("길게 보면 오르는 흐름 쪽에 있어요. 평균 가격들이 대체로 위를 향하고 있어요.")
    # 2-1) 이동평균선 신호 — 골든크로스·정배열
    tk = _tech(c, code)
    gc = [r for r in tk if r[0].startswith("골든크로스")]
    if gc:
        g0 = gc[-1]
        s.append(f"{g0[0]}가 {g0[2]}했어요. 짧은 기간 평균 가격이 더 긴 기간 평균 가격 위로 올라섰다는 뜻으로, 최근에 산 사람들이 이익을 보고 있는 상승 전환 신호예요.")
    elif tk and tk[0][1] and (mm or 0) < 7:
        s.append("이동평균선이 정배열(5 > 20 > 60 > 120일선)이에요. 짧게 보든 길게 보든 최근에 산 사람일수록 더 비싸게 샀는데도 계속 사고 있다는 뜻 — 모든 구간이 오르막이에요.")
    # 3) 신고가 돌파
    t55 = _v(F, "t55", t, code)
    if t55 is not None and t55 >= 0:
        s.append("최근 석 달(55거래일) 중 가장 높은 가격을 새로 넘었어요. 그동안 이 가격에서 팔고 나가려던 사람들의 물량을 다 받아내고 올라섰다는 뜻이에요.")
    # 4) 거래 증가
    amt = c["amt60"]
    a60 = float(amt[code].iloc[t]) / 1e8 if np.isfinite(amt[code].iloc[t]) else None
    amtr = _v(F, "amtr", t, code)
    a5 = a60 * amtr if (a60 and amtr) else None
    hi52 = _v(F, "hi52", t, code)
    if amtr and amtr >= 1.5:
        near = hi52 is not None and hi52 >= -0.05
        s.append(f"최근 5일 동안 하루 평균 {a5:,.0f}억 원어치가 거래됐어요. 지난 3개월 하루 평균({a60:,.0f}억 원)의 {amtr:.1f}배예요. "
                 + ("1년 중 가장 비싼 가격 근처인데도 사려는 사람이 몰리고 있어요." if near else "갑자기 관심이 몰리고 있다는 뜻이에요."))
    # 5) 업종
    tn = tnow.get(th.get(code))
    if tn and tn["rank"] <= max(3, len(tnow) // 5):
        s.append(f"'{th.get(code)}' 업종 전체가 요즘 시장에서 힘이 센 편이에요({len(tnow)}개 업종 중 {tn['rank']}위). "
                 "업종 전체에 돈이 들어오면 그 안의 종목도 함께 오르기 쉬워요.")
    # 6) 눌림
    d20 = _v(F, "d20", t, code)
    if d20 is not None and -0.04 <= d20 <= 0.03 and (mm or 0) >= 5:
        s.append("많이 오른 뒤 최근 한 달 평균 가격 근처까지 내려와 쉬고 있어요. 오르는 주식을 비싸지 않게 살 수 있는 자리예요.")
    rs = _v(F, "k_rs60", t, code)
    if len(s) < 3 and rs is not None and rs >= 0.8:
        s.append(f"같은 업종 다른 주식들보다 최근 3개월 동안 더 많이 올랐어요(전체 종목 중 상위 {max(1, round((1 - rs) * 100))}%).")
    if not s:
        s.append("추세·업종·거래량을 함께 본 점수가 전체 종목 중 위쪽에 있어요.")
    s = s[:5]
    past = (f"과거에 같은 방법으로 추천한 종목을 {HZ[Hh][1]} 동안 들고 있었다면 평균 {val.get('avg', 0):+.1f}%였어요"
            f"(같은 기간 같은 규모 일반 종목 평균 {val.get('ew', 0):+.1f}%).")
    vr = _valuation(c, code)
    if vr and vr["cheap"]:
        s.append(vr["story"])
    tops = max(3, len(tnow) // 4)
    ev = [{"k": "수급", "v": (buyers[0][0] + (f" 외 {len(buyers) - 1}" if len(buyers) > 1 else "") + " 순매수") if buyers else ("자료 없음" if not frows else "주요 주체 매수 없음"),
           "ok": bool(buyers), "rows": frows},
          {"k": "추세 템플릿", "v": f"{mm}/8 충족" if mm is not None else "-", "ok": (mm or 0) >= 6, "rows": tr or []},
          {"k": "기술적 신호", "v": f"{sum(r[1] for r in tk)}/{len(tk)} 긍정" if tk else "-", "ok": bool(tk) and sum(r[1] for r in tk) >= len(tk) * 0.6, "rows": tk},
          {"k": "업종", "v": f"{tn['rank']}위 / {len(tnow)}" if tn else "-", "ok": bool(tn and tn["rank"] <= tops),
           "text": (f"'{th.get(code)}' 업종 지수 최근 1개월 {tn['r1m'] * 100:+.1f}%, 3개월 {tn['r3m'] * 100:+.1f}%. 업종 순위는 시장 대비 1·3개월 수익률, "
                    f"20일 평균 가격 위에 있는 종목 비율, 거래대금 쏠림, 1년 최고가 근처 종목 비율을 합쳐 매겨요.") if tn else ""},
          {"k": "거래 증가", "v": f"{amtr:.1f}배" if amtr else "-", "ok": bool(amtr and amtr >= 1.2),
           "text": (f"최근 5일 하루 평균 거래대금 {a5:,.0f}억 원 ÷ 지난 3개월 하루 평균 {a60:,.0f}억 원 = {amtr:.1f}배. "
                    "1.2배 이상이면 평소보다 관심이 늘어난 것으로 봐요.") if a5 else ""}]
    if vr:
        ev.insert(3, {"k": "밸류에이션", "v": vr["v"], "ok": vr["cheap"], "rows": vr["rows"], "text": vr["text"]})
    return {"story": s, "past": past, "ev": ev}
