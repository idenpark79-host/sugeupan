"""추천종목 — 단기(1주)·스윙(1개월)·중장기(3개월).

점수는 하나(투자의견 점수: 추세·섹터·수급·유명 트레이더 기법으로 1개월 시장 대비 수익을 예측)만 쓴다 —
기간별 별도 모델은 검증에서 이 점수보다 낫지 않고 실행마다 결과가 흔들렸다. 그래서 추천과 투자의견이 항상 일치한다.
· 단기: 매수 이상 + 거래대금 급증(5일 평균이 60일 평균의 1.5배 이상) 중 점수 상위 10
· 스윙: 매수 이상 중 점수 상위 10
· 중장기: 매수 이상 + 미너비니 추세 템플릿 8조건 중 7개 이상 중 점수 상위 10
손절가는 1.5 × 일간 변동성 × √기간(종가 기준, 기간별 상한 10·20·25%), 목표가는 손절폭의 2배(손익비 2:1).
검증: 학습에 쓰지 않은 기간에 매일 같은 규칙으로 고른 종목을 다음 날 시가에 사서 목표가·손절가·기간 만료로 판 결과.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HZ = {5: ("단기", "1주"), 20: ("스윙", "1개월"), 60: ("중장기", "3개월")}
RISK = {5: (0.04, 0.10), 20: (0.08, 0.20), 60: (0.12, 0.25)}   # 손절폭 하한·상한 — 상한이 좁으면 변동 큰 종목이 정상 등락에 손절돼 성과가 크게 줄었다(검증)
K_STOP, RR = 1.5, 2.0   # 손절폭 = 1.5 × 일간 변동성 × √기간, 목표 = 손절폭 × 2 (과거 검증에서 그냥 보유와 비슷한 성과를 지키는 가장 좁은 손절)
SMOOTH = {5: 3, 20: 5, 60: 5}
COST = 0.0025
TOPN = 10
MIN_AMT = 2e9
MIN_OP = 0.75            # 투자의견 매수 이상
FLOW_G = {"frg": "외국인", "inst": "기관", "pen": "연기금", "trust": "투신", "pef": "사모"}


STYLE = {5: lambda F: (F["amtr"] >= 1.5).values.astype("float32"),
         20: lambda F: None,
         60: lambda F: (F["mm"] >= 7).values.astype("float32")}
RULE = {5: "투자의견 매수 이상 + 최근 거래가 평소의 1.5배 이상으로 늘어난 종목 중 점수 순",
        20: "투자의견 매수 이상 종목 중 점수 순",
        60: "투자의견 매수 이상 + 미너비니 추세 템플릿(8조건 중 7개 이상)을 만족하는 종목 중 점수 순"}


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


def _risk(sig, Hh):
    lo, hi = RISK[Hh]
    return np.clip(K_STOP * sig * np.sqrt(Hh), lo, hi)


def build(c: dict, log=print) -> dict:
    C, O, Hh_, L = c["C"], c["O"], c["Hh"], c["L"]
    codes, dates, nd, F, th = c["codes"], c["dates"], c["nd"], c["F"], c["th"]
    ri, ci, d_i = c["ri"], c["ci"], c["d_i"]
    A = tuple(np.asarray(x.values, dtype="float64") for x in (O, Hh_, L, C))
    amt60 = c["amt60"].values
    sig = c["vol60"].values
    midx = c["midx"].values
    oos0 = c["starts"][0]
    # 투자의견(1개월) 점수 — 추천 필터
    P20 = c["P"][["d", "c", "pct"]]
    op_today = pd.Series(c["pct_today"], index=c["c_today"])
    ok = c["ok"]
    out = {}
    for Hh in (5, 20, 60):
        flt = STYLE[Hh](F)                        # 날짜×종목 조건 (없으면 None)
        Q = P20.assign(ph=P20.pct)
        Q = Q[(Q.d >= oos0) & (Q.d + Hh < nd - 1) & (Q.pct >= MIN_OP)]
        if flt is not None:
            Q = Q[np.nan_to_num(flt[Q.d.values, Q.c.values]) > 0]
        Q = Q[amt60[Q.d.values, Q.c.values] >= MIN_AMT]
        Q = Q.sort_values(["d", "ph"], ascending=[True, False]).groupby("d").head(TOPN)
        d, cc = Q.d.values.astype(int), Q.c.values.astype(int)
        rk = _risk(sig[d, cc], Hh)
        ret, st_, hit, hold = _sim(A, d, cc, Hh, rk)
        mk = midx[d + Hh] / midx[d] - 1
        fw = A[3][Hh + 1:] / A[0][1:-Hh] - 1
        elig = amt60[:-Hh - 1] >= MIN_AMT
        ew_day = np.nanmean(np.where(elig, fw, np.nan), axis=1)
        ewv = ew_day[d]
        m = np.isfinite(ret) & np.isfinite(mk)
        V = pd.DataFrame({"d": d[m], "r": ret[m], "s": st_[m], "t": hit[m], "h": hold[m], "mk": mk[m], "ew": ewv[m]})
        V["mo"] = pd.to_datetime(dates[V.d.values]).strftime("%Y-%m")
        mon = V.groupby("mo").agg(r=("r", "mean"), mk=("mk", "mean"))
        val = {"n": int(len(V)), "avg": round(float(V.r.mean()) * 100, 2), "mkt": round(float(V.mk.mean()) * 100, 2),
               "win": round(float((V.r > 0).mean()) * 100, 1), "hitT": round(float(V.t.mean()) * 100, 1),
               "hitS": round(float(V.s.mean()) * 100, 1), "hold": round(float(V.h.mean()) * 100, 2),
               "ew": round(float(V.ew.mean()) * 100, 2), "up": round(float(V.r[V.r > 0].mean()) * 100, 1),
               "dn": round(float(V.r[V.r <= 0].mean()) * 100, 1),
               "beat": round(float((mon.r > mon.mk).mean()) * 100, 1), "months": int(len(mon)),
               "from": str(pd.Timestamp(dates[int(V.d.min())]).date()) if len(V) else None,
               "to": str(pd.Timestamp(dates[int(V.d.max())]).date()) if len(V) else None,
               "monthly": [[k, round(float(x.r) * 100, 2), round(float(x.mk) * 100, 2)] for k, x in mon.iterrows()][-36:]}
        log(f"    추천 {HZ[Hh][0]} 검증: 평균 {val['avg']:+.2f}% (시장 {val['mkt']:+.2f}%) · 상승 {val['win']}% · "
            f"일반 종목 {val['ew']:+.2f}% · 목표가 도달 {val['hitT']}% · 손절 {val['hitS']}% · {val['n']:,}건")
        # 오늘 추천
        t_last = nd - 1
        cand = [(code_i, float(p)) for code_i, p in op_today.items()
                if p >= MIN_OP and amt60[t_last, code_i] >= MIN_AMT and np.isfinite(A[3][t_last, code_i])
                and (flt is None or np.nan_to_num(flt[t_last, code_i]) > 0)]
        cand.sort(key=lambda x: -x[1])
        picks = []
        for code_i, p in cand[:TOPN]:
            code = codes[code_i]
            close = float(A[3][t_last, code_i])
            r_ = float(_risk(sig[t_last, code_i], Hh)) if np.isfinite(sig[t_last, code_i]) else RISK[Hh][0]
            ev, story = _story(c, code, code_i, Hh, val)
            picks.append({"code": code, "name": c["names"].get(code, code), "close": close,
                          "target": tick(close * (1 + RR * r_)), "stop": tick(close * (1 - r_)),
                          "tgtPct": round(RR * r_ * 100, 1), "stpPct": round(-r_ * 100, 1),
                          "score": round(p * 100, 1),
                          "theme": th.get(code, "기타"), "story": story, "ev": ev,
                          "prob": val["win"] / 100, "exp": val["avg"] / 100})
        out[str(Hh)] = {"H": Hh, "label": HZ[Hh][0], "hold": HZ[Hh][1], "rule": RULE[Hh], "picks": picks, "val": val}
    return out


def _v(F, k, t, code):
    try:
        x = float(F[k][code].iloc[t])
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _story(c, code, ci_, Hh, val):
    """초등학생도 알아듣게 — 왜 오를 것 같은지 2~3문장 + 근거 4가지."""
    F, t, th, tnow, flows = c["F"], c["nd"] - 1, c["th"], c["tnow"], c.get("flows") or {}
    s = []
    fl = flows.get(code)
    buy = []
    if fl is not None and len(fl):
        for g, nm in (("외국인", "외국인"), ("연기금", "연기금"), ("투신", "투신(펀드)"), ("사모", "사모펀드"), ("기관합계", "기관")):
            if g in fl:
                v = float(fl[g].iloc[-20:].sum()) / 1e8
                if v > 0 and (g != "기관합계" or not buy):
                    buy.append((nm, v))
    big = [b for b in buy if b[0] != "기관"]
    if len(big) >= 2:
        s.append(f"{'·'.join(b[0] for b in big[:2])} 같은 큰손들이 최근 한 달 동안 이 주식을 사 모으고 있어요 "
                 f"({', '.join(f'{b[0]} +{b[1]:,.0f}억' for b in big[:2])}).")
    elif buy and buy[0][1] >= 30:
        s.append(f"{buy[0][0]}이 최근 한 달 동안 {buy[0][1]:,.0f}억 원어치를 샀어요.")
    mm = _v(F, "mm", t, code)
    if mm is not None and mm >= 7:
        s.append(f"주가가 계단처럼 꾸준히 오르는 모양이에요. 유명 트레이더 마크 미너비니가 보는 8가지 조건 중 {int(mm)}개를 만족해요.")
    t55 = _v(F, "t55", t, code)
    if t55 is not None and t55 >= 0:
        s.append("최근 55일 중 가장 높은 가격을 넘어섰어요. '터틀 트레이딩'에서 사라는 신호로 보는 자리예요.")
    tn = tnow.get(th.get(code))
    top = max(3, len(tnow) // 5)
    if tn and tn["rank"] <= top and len(s) < 3:
        s.append(f"{th.get(code)} 업종이 요즘 시장에서 힘이 센 편이에요({len(tnow)}개 업종 중 {tn['rank']}위).")
    onl = _v(F, "onl", t, code)
    amtr = _v(F, "amtr", t, code)
    if onl and len(s) < 3:
        s.append(f"1년 중 가장 비싼 가격 근처인데도 거래가 평소보다 {amtr or 1.5:.1f}배 많아요. 사려는 사람이 몰린다는 뜻이에요.")
    d20 = _v(F, "d20", t, code)
    if len(s) < 3 and d20 is not None and -0.04 <= d20 <= 0.03 and (mm or 0) >= 5:
        s.append("많이 오른 뒤 한 달 평균 가격 근처에서 쉬고 있어서, 지금 사도 비싸게 사는 부담이 덜해요.")
    rs = _v(F, "k_rs60", t, code)
    if len(s) < 2 and rs is not None and rs >= 0.8:
        s.append("같은 업종 주식들보다 최근 3개월 동안 더 많이 올랐어요.")
    if not s:
        s.append("추세·업종·거래량을 함께 본 점수가 전체 종목 중 위쪽에 있어요.")
    s = s[:3]
    w10 = int(round(val["win"] / 10))
    mult = f", 일반 종목 평균({val['ew']:+.1f}%)의 {val['avg'] / val['ew']:.1f}배" if val["ew"] > 0.1 and val["avg"] >= val["ew"] * 1.5 else ""
    s.append(f"과거에 같은 방법으로 고른 종목은 {HZ[Hh][1]} 뒤 평균 {val['avg']:+.1f}%였어요{mult}. "
             f"10번 중 {w10}번은 수익으로 끝났고, 오를 때는 평균 {val['up']:+.0f}%, 내릴 때는 평균 {val['dn']:.0f}%였어요.")
    ev = [["큰손 매수", f"{len(big)}곳" if big else "없음", len(big) >= 2],
          ["추세 점수", f"{int(mm)}/8" if mm is not None else "-", (mm or 0) >= 6],
          ["업종 순위", f"{tn['rank']}위" if tn else "-", bool(tn and tn["rank"] <= max(3, len(tnow) // 4))],
          ["거래량", f"{amtr:.1f}배" if amtr else "-", bool(amtr and amtr >= 1.2)]]
    return ev, s
