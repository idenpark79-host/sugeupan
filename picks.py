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


SEG = {"L": ("대형주", "시가총액 1조 원 이상"), "S": ("중소형주", "시가총액 1조 원 미만")}


def _val(V, dates):
    V = V.copy()
    V["mo"] = pd.to_datetime(dates[V.d.values]).strftime("%Y-%m")
    mon = V.groupby("mo").agg(r=("r", "mean"), mk=("mk", "mean"))
    return {"n": int(len(V)), "avg": round(float(V.r.mean()) * 100, 2), "mkt": round(float(V.mk.mean()) * 100, 2),
            "win": round(float((V.r > 0).mean()) * 100, 1), "hitT": round(float(V.t.mean()) * 100, 1),
            "hitS": round(float(V.s.mean()) * 100, 1), "hold": round(float(V.h.mean()) * 100, 2),
            "ew": round(float(V.ew.mean()) * 100, 2), "up": round(float(V.r[V.r > 0].mean()) * 100, 1),
            "dn": round(float(V.r[V.r <= 0].mean()) * 100, 1),
            "beat": round(float((mon.r > mon.mk).mean()) * 100, 1), "months": int(len(mon)),
            "from": str(pd.Timestamp(dates[int(V.d.min())]).date()) if len(V) else None,
            "to": str(pd.Timestamp(dates[int(V.d.max())]).date()) if len(V) else None}


def build(c: dict, log=print) -> dict:
    C, O, Hh_, L = c["C"], c["O"], c["Hh"], c["L"]
    codes, dates, nd, F, th = c["codes"], c["dates"], c["nd"], c["F"], c["th"]
    A = tuple(np.asarray(x.values, dtype="float64") for x in (O, Hh_, L, C))
    amt60 = c["amt60"].values
    sig = c["vol60"].values
    midx = c["midx"].values
    big = (c["w"].values >= 1e12)                  # 추정 시가총액 1조 원 이상 = 대형주
    oos0 = c["starts"][0]
    P20 = c["P"][["d", "c", "pct"]]
    op_today = pd.Series(c["pct_today"], index=c["c_today"])
    t_last = nd - 1
    out = {}
    for Hh in (5, 20, 60):
        flt = STYLE[Hh](F)
        fw = A[3][Hh + 1:] / A[0][1:-Hh] - 1
        elig = amt60[:-Hh - 1] >= MIN_AMT
        Q0 = P20[(P20.d >= oos0) & (P20.d + Hh < nd - 1) & (P20.pct >= MIN_OP)]
        if flt is not None:
            Q0 = Q0[np.nan_to_num(flt[Q0.d.values, Q0.c.values]) > 0]
        Q0 = Q0[amt60[Q0.d.values, Q0.c.values] >= MIN_AMT]
        res = {"H": Hh, "label": HZ[Hh][0], "hold": HZ[Hh][1], "rule": RULE[Hh], "picks": [], "segs": {}}
        allV = []
        for sg, (sname, sdesc) in SEG.items():
            inseg = big[Q0.d.values, Q0.c.values] if sg == "L" else ~big[Q0.d.values, Q0.c.values]
            Q = Q0[inseg].sort_values(["d", "pct"], ascending=[True, False]).groupby("d").head(TOPN)
            d, cc = Q.d.values.astype(int), Q.c.values.astype(int)
            ret, st_, hit, hold = _sim(A, d, cc, Hh, _risk(sig[d, cc], Hh))
            mk = midx[d + Hh] / midx[d] - 1
            segm = big[:-Hh - 1] if sg == "L" else ~big[:-Hh - 1]
            ew_day = np.nanmean(np.where(elig & segm, fw, np.nan), axis=1)
            m = np.isfinite(ret) & np.isfinite(mk)
            V = pd.DataFrame({"d": d[m], "r": ret[m], "s": st_[m], "t": hit[m], "h": hold[m], "mk": mk[m], "ew": ew_day[d][m]})
            allV.append(V)
            val = _val(V, dates) if len(V) else {}
            val["name"], val["desc"] = sname, sdesc
            res["segs"][sg] = val
            if len(V):
                log(f"    추천 {HZ[Hh][0]}·{sname} 검증: 평균 {val['avg']:+.2f}% · 같은 규모 일반 종목 {val['ew']:+.2f}% · 지수 {val['mkt']:+.2f}% · "
                    f"수익 마감 {val['win']}% · 목표 {val['hitT']}% · 손절 {val['hitS']}% · {val['n']:,}건")
            cand = [(ci_, float(p)) for ci_, p in op_today.items()
                    if p >= MIN_OP and amt60[t_last, ci_] >= MIN_AMT and np.isfinite(A[3][t_last, ci_])
                    and (big[t_last, ci_] if sg == "L" else not big[t_last, ci_])
                    and (flt is None or np.nan_to_num(flt[t_last, ci_]) > 0)]
            cand.sort(key=lambda x: -x[1])
            for rank, (ci_, p) in enumerate(cand[:TOPN], 1):
                code = codes[ci_]
                close = float(A[3][t_last, ci_])
                s0 = float(sig[t_last, ci_]) if np.isfinite(sig[t_last, ci_]) else 0.02
                r_ = float(_risk(s0, Hh))
                info = _story(c, code, ci_, Hh, val)
                res["picks"].append({"code": code, "name": c["names"].get(code, code), "close": close, "size": sg, "rank": rank,
                                     "target": tick(close * (1 + RR * r_)), "stop": tick(close * (1 - r_)),
                                     "tgtPct": round(RR * r_ * 100, 1), "stpPct": round(-r_ * 100, 1),
                                     "score": round(p * 100, 1), "theme": th.get(code, "기타"),
                                     "plan": _plan(Hh, s0, r_, val), "prob": val.get("win", 50) / 100, "exp": val.get("avg", 0) / 100, **info})
        res["val"] = _val(pd.concat(allV), dates)
        out[str(Hh)] = res
    return out


def _plan(Hh, s0, r_, val):
    move = K_STOP * s0 * np.sqrt(Hh) * 100
    lo, hi = RISK[Hh]
    cap = "" if lo * 100 < move < hi * 100 else (f" 다만 {HZ[Hh][1]} 보유 기준 손실은 최대 {hi * 100:.0f}%까지만 감수하도록 제한했어요." if move >= hi * 100
                                                  else f" 움직임이 작은 종목이라 최소 {lo * 100:.0f}% 폭은 두었어요.")
    return {"stop": f"이 종목은 하루에 평균 {s0 * 100:.1f}% 정도 오르내려요. {HZ[Hh][1]}이면 보통 {move:.0f}% 안팎까지 흔들릴 수 있어서, "
                    f"그보다 더 내려가 종가가 손절가 아래로 마감하면 '예상이 틀렸다'고 보고 다음 날 정리하는 가격이에요.{cap}",
            "target": f"손절할 때 잃는 폭({r_ * 100:.0f}%)의 2배를 목표로 잡았어요. 잃을 때보다 벌 때 2배 크게 가져가자는 원칙이에요. "
                      f"과거 같은 방법에서 목표가에 먼저 닿은 경우는 {val.get('hitT', 0):.0f}%였고, 닿지 않으면 {HZ[Hh][1]} 뒤 그때 가격으로 정리했어요."}


def _v(F, k, t, code):
    try:
        x = float(F[k][code].iloc[t])
        return x if np.isfinite(x) else None
    except Exception:
        return None


FLOW_NM = [("외국인", "외국인", "해외 연기금·펀드·증권사"), ("연기금", "연기금", "국민연금 같은 연금"), ("투신", "투신", "자산운용사 펀드"),
           ("사모", "사모펀드", "소수 큰손 펀드"), ("금융투자", "금융투자", "증권사 자기 돈"), ("보험", "보험", "보험사"),
           ("기관합계", "기관 합계", "국내 기관 전체")]


def _trend(c, code):
    """추세 8가지 체크 — 쉬운 설명과 실제 값."""
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
        ("지금 가격이 150일·200일 평균 가격보다 높다", px > a150 and px > a200, f"현재 {n(px)} · 150일 평균 {n(a150)} · 200일 평균 {n(a200)}"),
        ("150일 평균이 200일 평균보다 높다", a150 > a200, "중간 흐름이 긴 흐름보다 강하다는 뜻"),
        ("200일 평균 가격이 한 달 전보다 올랐다", a200 > a200p, f"한 달 전 {n(a200p)} → 지금 {n(a200)}"),
        ("50일 평균이 150일·200일 평균보다 높다", a50 > a150 and a50 > a200, f"50일 평균 {n(a50)}"),
        ("지금 가격이 50일 평균보다 높다", px > a50, f"50일 평균 대비 {(px / a50 - 1) * 100:+.1f}%"),
        ("1년 중 가장 쌌던 때보다 30% 이상 올라 있다", px >= lo * 1.3, f"1년 최저 {n(lo)} 대비 {(px / lo - 1) * 100:+.0f}%"),
        ("1년 중 가장 비쌌던 가격에서 25% 안쪽이다", px >= hi * 0.75, f"1년 최고 {n(hi)} 대비 {(px / hi - 1) * 100:+.0f}%"),
        ("1년 상승률이 전체 종목 중 상위 30% 안이다", rk is not None and rk >= 0.7, (f"상위 {max(1, round((1 - rk) * 100))}%" if rk >= 0.5 else f"하위 {max(1, round(rk * 100))}%") if rk is not None else "-"),
    ]
    return [[a, bool(b), x] for a, b, x in rows]


def _story(c, code, ci_, Hh, val):
    """초등학생도 알아듣게 — 왜 오를 거라고 봤는지 + 근거 4가지(누르면 자세히)."""
    F, t, th, tnow, flows = c["F"], c["nd"] - 1, c["th"], c["tnow"], c.get("flows") or {}
    s = []
    # 1) 큰손
    fl, buyers, frows = flows.get(code), [], []
    if fl is not None and len(fl):
        for g, nm, desc in FLOW_NM:
            if g in fl:
                m1, m3 = float(fl[g].iloc[-20:].sum()) / 1e8, float(fl[g].iloc[-60:].sum()) / 1e8
                frows.append([nm, desc, round(m1), round(m3)])
                if g != "기관합계" and m3 > 0 and m1 > 0:
                    buyers.append((nm, desc, m1, m3))
    buyers.sort(key=lambda x: -x[3])
    if buyers:
        b = buyers[0]
        txt = f"{b[0]}({b[1]})이 최근 3개월 동안 이 주식을 {b[3]:,.0f}억 원어치 사 모았고, 최근 한 달에만 {b[2]:,.0f}억 원을 더 샀어요."
        if len(buyers) > 1:
            txt += f" {'·'.join(x[0] for x in buyers[1:3])}도 함께 사고 있어요."
        s.append(txt + " 돈이 많은 큰손이 꾸준히 산다는 건 그만큼 이 회사를 좋게 본다는 뜻이에요.")
    # 2) 추세
    tr = _trend(c, code)
    mm = sum(r[1] for r in tr) if tr else None
    if mm is not None and mm >= 7:
        s.append("최근 몇 달 동안 계단을 오르듯 꾸준히 올라왔어요. 50일·150일·200일 평균 가격이 모두 위를 향하고, 지금 가격도 그 평균들보다 위에 있어요. "
                 "오르는 흐름이 이어지고 있다는 뜻이에요.")
    elif mm is not None and mm >= 5:
        s.append("길게 보면 오르는 흐름 쪽에 있어요. 평균 가격들이 대체로 위를 향하고 있어요.")
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
            f"(같은 기간 같은 규모 일반 종목 평균 {val.get('ew', 0):+.1f}%). 추천 100번 중 {val.get('win', 0):.0f}번은 판 가격이 산 가격보다 높았고, "
            f"그때는 평균 {val.get('up', 0):+.0f}%, 나머지는 평균 {val.get('dn', 0):.0f}%였어요.")
    tops = max(3, len(tnow) // 4)
    ev = [{"k": "큰손 매수", "v": (buyers[0][0] + (f" 외 {len(buyers) - 1}" if len(buyers) > 1 else "")) if buyers else ("자료 없음" if not frows else "없음"),
           "ok": bool(buyers), "rows": frows},
          {"k": "추세", "v": f"{mm}/8 충족" if mm is not None else "-", "ok": (mm or 0) >= 6, "rows": tr or []},
          {"k": "업종", "v": f"{tn['rank']}위 / {len(tnow)}" if tn else "-", "ok": bool(tn and tn["rank"] <= tops),
           "text": (f"'{th.get(code)}' 업종 지수 최근 1개월 {tn['r1m'] * 100:+.1f}%, 3개월 {tn['r3m'] * 100:+.1f}%. 업종 순위는 시장 대비 1·3개월 수익률, "
                    f"20일 평균 가격 위에 있는 종목 비율, 거래대금 쏠림, 1년 최고가 근처 종목 비율을 합쳐 매겨요.") if tn else ""},
          {"k": "거래 증가", "v": f"{amtr:.1f}배" if amtr else "-", "ok": bool(amtr and amtr >= 1.2),
           "text": (f"최근 5일 하루 평균 거래대금 {a5:,.0f}억 원 ÷ 지난 3개월 하루 평균 {a60:,.0f}억 원 = {amtr:.1f}배. "
                    "1.2배 이상이면 평소보다 관심이 늘어난 것으로 봐요.") if a5 else ""}]
    return {"story": s, "past": past, "ev": ev}
