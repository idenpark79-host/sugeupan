"""섹터 동향·주도섹터·주도주와 종목별 투자의견(강력매수~강력매도).

· 섹터 지수: 테마 구성 종목의 시가총액 가중 일간 수익률(현재 시가총액 × 주가 비율로 과거 비중 추정)
· 주도섹터 점수: 시장 대비 1·3개월 수익률, 20일선 위 종목 비율, 거래대금 쏠림, 52주 고점권 종목 비율의 순위 평균
· 투자의견: 종목 추세·모멘텀·변동성·거래대금 + 소속 섹터 강도 + 섹터 내 상대강도로 1개월 시장 대비 수익을 예측하는
  모델(워크포워드, 각 시점 이전 데이터로만 학습) → 매일 백분위 상위 5% 강력매수 · 25% 매수 · 중간 50% 중립 · 하위 25% 매도 · 5% 강력매도
· 검증: 학습에 쓰지 않은 기간에서 의견별 다음 1개월 시장 대비 수익·적중률, 주도섹터의 다음 1개월 성과
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import themes as TH

LV = {2: "강력매수", 1: "매수", 0: "중립", -1: "매도", -2: "강력매도"}
CUT = (0.95, 0.75, 0.25, 0.05)
H = 20


def level(p):
    p = np.asarray(p, dtype=float)
    return np.where(p >= CUT[0], 2, np.where(p >= CUT[1], 1, np.where(p >= CUT[2], 0, np.where(p >= CUT[3], -1, -2))))


def _wide(prices: dict, codes: list):
    cols = {k: {} for k in ("O", "H", "L", "C", "V")}
    for c in codes:
        d = prices.get(c)
        if d is None or len(d) < 60:
            continue
        for k, col in (("O", "Open"), ("H", "High"), ("L", "Low"), ("C", "Close"), ("V", "Volume")):
            cols[k][c] = d[col].astype("float32")
    W = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
    for k in W:
        W[k] = W[k].where(W[k] > 0) if k != "V" else W[k]
    idx = W["C"].index
    n_ok = W["C"].notna().sum(axis=1)
    keep = n_ok >= max(20, 0.3 * n_ok.max())                  # 거래일만 (휴장·데이터 공백 제외)
    return {k: v.loc[keep[keep].index] for k, v in W.items()}


def _rsi(C, n=14):
    d = C.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l_ = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l_.replace(0, np.nan))


def _index(ret: pd.DataFrame, w: pd.DataFrame, cols) -> pd.Series:
    r, ww = ret[cols], w[cols].shift(1)
    m = r.notna() & ww.notna()
    num = (r.where(m, 0) * ww.where(m, 0)).sum(axis=1)
    den = ww.where(m, 0).sum(axis=1)
    return (num / den.replace(0, np.nan)).fillna(0)


def build(prices: dict, cap: dict, names: dict, krx: dict | None = None, log=print, min_amt=3e8, flows: dict | None = None):
    codes = [c for c in prices if c in cap]
    W = _wide(prices, codes)
    O, Hh, L, C, V = W["O"], W["H"], W["L"], W["C"], W["V"]
    codes = list(C.columns)
    dates = C.index
    log(f"    투자의견·섹터: {len(codes)}종목 × {len(dates)}일")
    th = TH.assign(codes, names, krx)
    ret = C.pct_change(fill_method=None).clip(-0.5, 1.0)
    last = C.ffill().iloc[-1]
    w = C.ffill() * (pd.Series(cap).reindex(codes) / last)       # 추정 시가총액
    amt = C * V
    amt5, amt20, amt60 = amt.rolling(5, min_periods=3).mean(), amt.rolling(20, min_periods=10).mean(), amt.rolling(60, min_periods=30).mean()

    # ── 시장·섹터 지수 ──
    mret = _index(ret, w, codes)
    midx = (1 + mret).cumprod()
    groups = {}
    for c, t in th.items():
        groups.setdefault(t, []).append(c)
    T = {t: m for t, m in groups.items() if t != "기타" and len(m) >= 3}
    tidx = pd.DataFrame({t: (1 + _index(ret, w, m)).cumprod() for t, m in T.items()})
    ma20 = C.rolling(20, min_periods=15).mean()
    ma60 = C.rolling(60, min_periods=40).mean()
    above20, above60 = (C > ma20).where(C.notna() & ma20.notna()), (C > ma60).where(C.notna() & ma60.notna())
    hi250 = Hh.rolling(250, min_periods=120).max()
    near_hi = (C >= hi250 * 0.95).where(hi250.notna())
    tot5, tot60 = amt5.sum(axis=1), amt60.sum(axis=1)
    TF = {}
    for t, m in T.items():
        TF[t] = pd.DataFrame({
            "r5": tidx[t].pct_change(5) - midx.pct_change(5), "r20": tidx[t].pct_change(20) - midx.pct_change(20),
            "r60": tidx[t].pct_change(60) - midx.pct_change(60), "a20": tidx[t].pct_change(20), "a60": tidx[t].pct_change(60),
            "br20": above20[m].mean(axis=1), "br60": above60[m].mean(axis=1),
            "amt": (amt5[m].sum(axis=1) / tot5) / (amt60[m].sum(axis=1) / tot60).replace(0, np.nan),
            "hi": near_hi[m].mean(axis=1)})
    # 주도섹터 점수: 섹터 간 순위 평균 (매일)
    panel = pd.concat({t: f for t, f in TF.items()}, axis=1)
    sc = sum(panel.xs(k, axis=1, level=1).rank(axis=1, pct=True) for k in ("r20", "r60", "br20", "amt", "hi")) / 5
    srank = sc.rank(axis=1, ascending=False)

    # ── 종목 특징 ──
    F = {
        "r5": C.pct_change(5, fill_method=None), "r20": C.pct_change(20, fill_method=None), "r60": C.pct_change(60, fill_method=None),
        "r120": C.pct_change(120, fill_method=None), "mom": C.shift(20) / C.shift(250) - 1,
        "hi52": C / hi250 - 1, "lo52": C / L.rolling(250, min_periods=120).min() - 1,
        "d20": C / ma20 - 1, "d60": C / ma60 - 1, "d120": C / C.rolling(120, min_periods=80).mean() - 1,
        "vol20": ret.rolling(20, min_periods=15).std(), "amtr": amt5 / amt60, "liq": np.log(amt60.clip(lower=1)),
        "maxr": ret.rolling(20, min_periods=15).max(), "rsi": _rsi(C), "up20": (ret > 0).astype("float32").where(ret.notna()).rolling(20, min_periods=15).mean(),
    }
    tcol = pd.Series(th).reindex(codes)
    for k in ("r5", "r20", "r60", "a20", "a60", "br20", "br60", "amt", "hi"):
        F["t_" + k] = pd.DataFrame({c: (TF[tcol[c]][k] if tcol[c] in TF else np.nan) for c in codes}, index=dates)
    F["t_score"] = pd.DataFrame({c: (sc[tcol[c]] if tcol[c] in TF else np.nan) for c in codes}, index=dates)
    F["rs20"] = F["r20"] - F["t_a20"]
    F["rs60"] = F["r60"] - F["t_a60"]
    F["m_r20"] = pd.DataFrame(np.repeat(midx.pct_change(20).values[:, None], len(codes), axis=1), index=dates, columns=codes)
    F["m_br"] = pd.DataFrame(np.repeat(above20.mean(axis=1).values[:, None], len(codes), axis=1), index=dates, columns=codes)
    # 유명 트레이더 기법 — 미너비니 추세 템플릿(8조건), 와인스타인 2단계, 터틀 55일 돌파, 오닐 신고가+거래량
    ma50, ma150, ma200 = (C.rolling(n, min_periods=int(n * 0.8)).mean() for n in (50, 150, 200))
    lo250 = L.rolling(250, min_periods=120).min()
    rs12 = (C / C.shift(250) - 1).rank(axis=1, pct=True)
    crit = [(C > ma150) & (C > ma200), ma150 > ma200, ma200 > ma200.shift(20), (ma50 > ma150) & (ma50 > ma200),
            C > ma50, C >= lo250 * 1.3, C >= hi250 * 0.75, rs12 >= 0.7]
    F["mm"] = sum(x.astype("float32") for x in crit).where(ma200.notna())
    F["st2"] = ((C > ma150) & (ma150 > ma150.shift(10))).astype("float32").where(ma150.notna())
    F["t55"] = C / Hh.shift(1).rolling(55, min_periods=40).max() - 1
    v50 = V.rolling(50, min_periods=30).mean()
    F["onl"] = ((C >= hi250 * 0.95) & (V.rolling(5, min_periods=3).mean() >= v50 * 1.5)).astype("float32").where(hi250.notna())
    # 수급 — 투자자별 20일·5일 순매수 / 같은 기간 거래대금 (외국인·기관·연기금·투신·사모·금융투자)
    if flows:
        for g, k in (("외국인", "frg"), ("기관합계", "inst"), ("연기금", "pen"), ("투신", "trust"), ("사모", "pef"), ("금융투자", "fin")):
            net = pd.DataFrame({c: flows[c][g] for c in codes if c in flows and g in flows[c]})
            if net.empty:
                continue
            net = net.reindex(index=dates, columns=codes)
            F[f"f_{k}20"] = net.rolling(20, min_periods=10).sum() / (amt60 * 20)
            F[f"f_{k}5"] = net.rolling(5, min_periods=3).sum() / (amt60 * 5)
        if "f_frg20" in F:
            F["f_smart"] = sum((F[f"f_{k}20"] > 0).astype("float32") for k in ("frg", "pen", "trust") if f"f_{k}20" in F).where(F["f_frg20"].notna())
    RANK = ["r5", "r20", "r60", "r120", "mom", "hi52", "d20", "d60", "vol20", "amtr", "liq", "rsi", "rs20", "rs60"]
    for k in RANK:
        F["k_" + k] = F[k].rank(axis=1, pct=True)
    names_f = list(F)

    # 목표: 다음 날 시가 매수 → 20거래일 뒤 종가, 같은 날 전 종목 평균 대비
    fwd = C.shift(-H) / O.shift(-1) - 1
    fwd = fwd.where(np.isfinite(fwd) & (fwd.abs() < 3))
    ok = (amt60 >= min_amt) & C.notna() & F["mom"].notna()
    fx = fwd.where(ok)
    fx = fx.sub(fx.mean(axis=1), axis=0)

    # 긴 형태로
    ri, ci = np.nonzero(ok.values)
    X = np.column_stack([F[k].values[ri, ci].astype("float32") for k in names_f])
    X[~np.isfinite(X)] = np.nan
    y = fx.values[ri, ci]
    d_i = ri
    log(f"    학습 행 {len(ri):,} · 특징 {len(names_f)}개")

    from sklearn.ensemble import HistGradientBoostingRegressor
    reg = lambda: HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=400,
                                                l2_regularization=1.0, random_state=0)
    lo, hi_ = np.nanpercentile(y[np.isfinite(y)], [1, 99])
    yc = np.clip(y, lo, hi_)
    rng = np.random.default_rng(0)
    nd = len(dates)
    oos_days = min(756, int(nd * 0.45))
    fold = 126
    starts = list(range(nd - oos_days, nd, fold))
    pred = np.full(len(ri), np.nan)
    lab = np.isfinite(y)
    for s0 in starts:
        s1 = min(s0 + fold, nd)
        tr = np.flatnonzero(lab & (d_i <= s0 - H - 1))
        te = np.flatnonzero((d_i >= s0) & (d_i < s1))
        if len(tr) < 5000 or not len(te):
            continue
        it = tr if len(tr) <= 300_000 else rng.choice(tr, 300_000, replace=False)
        pred[te] = reg().fit(X[it], yc[it]).predict(X[te])
        log(f"    투자의견 워크포워드 {pd.Timestamp(dates[s0]):%Y.%m}~{pd.Timestamp(dates[s1 - 1]):%Y.%m} 학습 {len(it):,}")
    P = pd.DataFrame({"d": d_i, "c": ci, "p": pred, "y": y})
    P = P[np.isfinite(P.p)]
    P["pct0"] = P.groupby("d").p.rank(pct=True)
    P = P.sort_values(["c", "d"])
    P["pct"] = P.groupby("c").pct0.transform(lambda s: s.rolling(5, min_periods=1).mean())   # 5거래일 평균 — 의견이 하루 만에 뒤집히지 않게
    P["pct"] = P.groupby("d").pct.rank(pct=True)
    P["lv"] = level(P.pct.values)

    # ── 검증 ──
    V2 = P[np.isfinite(P.y)].copy()
    raw = fwd.values[V2.d.values, V2.c.values]
    V2["raw"] = raw
    val = {}
    for lv in (2, 1, 0, -1, -2):
        s = V2[V2.lv == lv]
        val[str(lv)] = [round(float(s.y.mean()) * 100, 2), round(float((s.y > 0).mean()) * 100, 1),
                        round(float(np.nanmean(s.raw)) * 100, 2), round(float((s.raw > 0).mean()) * 100, 1), int(len(s))]
    V2["m"] = pd.to_datetime(dates[V2.d.values]).to_period("M").astype(str)
    mon = V2.groupby(["m", "lv"]).y.mean().unstack()
    monthly = [[m, round(float(r.get(2, np.nan)) * 100, 2), round(float(r.get(-2, np.nan)) * 100, 2)] for m, r in mon.iterrows()]
    consist = float(((mon.get(2) - mon.get(-2)) > 0).mean()) if 2 in mon and -2 in mon else None
    dec = V2.groupby(pd.cut(V2.pct, np.linspace(0, 1, 11), include_lowest=True), observed=True).y.mean()
    # ── 1개월 예상 범위: 점수 10분위별 실제 1개월 수익률 분포(60일 변동성 단위) — 앞 2/3로 만든 범위를 뒤 1/3에서 검증 ──
    vol60 = ret.rolling(60, min_periods=40).std().clip(0.005, 0.08)
    V2["z"] = V2.raw / vol60.values[V2.d.values, V2.c.values]
    V3 = V2[np.isfinite(V2.z)]
    QS = [.1, .25, .5, .75, .9]
    kk, zz, dd = np.minimum((V3.pct.values * 10).astype(int), 9), V3.z.values, V3.d.values
    qt = lambda m: np.array([np.quantile(zz[m & (kk == k)], QS) if (m & (kk == k)).sum() > 200 else np.quantile(zz[m], QS) for k in range(10)])
    ud = np.unique(dd)
    cutd = ud[int(len(ud) * 2 / 3)]
    late = dd >= cutd
    q1 = qt(~late)
    fcv = {"H": H, "c80": round(float(((zz >= q1[kk, 0]) & (zz <= q1[kk, 4]))[late].mean()) * 100, 1),
           "c50": round(float(((zz >= q1[kk, 1]) & (zz <= q1[kk, 3]))[late].mean()) * 100, 1),
           "up": round(float((zz > q1[kk, 2])[late].mean()) * 100, 1), "n": int(late.sum()),
           "from": str(pd.Timestamp(dates[cutd]).date()), "to": str(pd.Timestamp(dates[int(ud[-1])]).date())}
    QT = qt(np.ones(len(zz), bool))
    z1, z99 = np.quantile(zz, [.01, .99])
    QM = np.array([float(np.mean(np.clip(zz[kk == k], z1, z99))) if (kk == k).any() else 0.0 for k in range(10)])   # 분위별 평균(극단값 1% 제한)
    log(f"    1개월 예상 범위 검증: 80% 범위 적중 {fcv['c80']}% · 50% 범위 적중 {fcv['c50']}% ({fcv['n']:,}건)")
    log("    투자의견 검증(다음 1개월 시장 대비): " + " · ".join(f"{LV[int(k)]} {v[0]:+.2f}%p" for k, v in val.items()))

    # 주도섹터 검증: 매주 순위 상위 3 / 하위 3 섹터의 다음 1개월 시장 대비
    tf20 = tidx.shift(-H) / tidx - 1
    mf20 = midx.shift(-H) / midx - 1
    rel_f = tf20.sub(mf20, axis=0)
    rows = []
    for i in range(260, nd - H, 5):
        r_ = srank.iloc[i].dropna()
        if len(r_) < 8:
            continue
        top, bot = r_.nsmallest(3).index, r_.nlargest(3).index
        rows.append((dates[i], float(rel_f.iloc[i][top].mean()), float(rel_f.iloc[i][bot].mean())))
    tv = pd.DataFrame(rows, columns=["d", "top", "bot"]).dropna()
    oos0 = dates[starts[0]]
    tv_o = tv[tv.d >= oos0]
    tval = {"all": [round(tv.top.mean() * 100, 2), round(tv.bot.mean() * 100, 2), round(float((tv.top > tv.bot).mean()) * 100, 1), len(tv)],
            "oos": [round(tv_o.top.mean() * 100, 2), round(tv_o.bot.mean() * 100, 2), round(float((tv_o.top > tv_o.bot).mean()) * 100, 1), len(tv_o)],
            "from": str(pd.Timestamp(tv.d.min()).date()) if len(tv) else None}
    log(f"    주도섹터 상위3 다음 1개월 시장 대비 {tval['all'][0]:+.2f}%p · 하위3 {tval['all'][1]:+.2f}%p · 상위>하위 {tval['all'][2]:.0f}%")

    # ── 최종 모델 → 오늘 ──
    it = np.flatnonzero(lab)
    it = it if len(it) <= 400_000 else rng.choice(it, 400_000, replace=False)
    m_final = reg().fit(X[it], yc[it])
    t_last = nd - 1
    today = np.flatnonzero(ri == t_last)
    p_today = m_final.predict(X[today])
    c_today = ci[today]
    p0 = pd.Series(pd.Series(p_today).rank(pct=True).values, index=c_today)
    prev = P[P.d >= nd - 5][P.d < nd - 1].groupby("c").pct0.apply(list) if len(P) else pd.Series(dtype=object)
    sm = np.array([np.mean((prev.get(c_, []) or [])[-4:] + [p0[c_]]) for c_ in c_today])
    pct_today = pd.Series(sm).rank(pct=True).values

    # 설명용 근사(요인 순위 → 점수)
    EXPL = ["t_r20", "t_r60", "t_br20", "t_amt", "t_score", "rs60", "r20", "r60", "mom", "hi52", "d20", "vol20", "amtr", "rsi"]
    from sklearn.linear_model import Ridge
    samp = P.sample(min(len(P), 300_000), random_state=1)
    Zs = pd.DataFrame({k: F[k].values[samp.d.values, samp.c.values] for k in EXPL})
    Zs = Zs.groupby(samp.d.values).rank(pct=True) - 0.5
    okz = Zs.notna().all(axis=1).values
    rg = Ridge(alpha=10).fit(Zs.values[okz], samp.pct.values[okz] - 0.5)
    coef = dict(zip(EXPL, rg.coef_))
    Zt = pd.DataFrame({k: F[k].iloc[t_last] for k in EXPL}).rank(pct=True) - 0.5

    # 의견 이력(최근 120거래일, 워크포워드 예측)
    recent = P[P.d >= nd - 120]
    hist = {}
    for c_, g in recent.groupby("c"):
        s = g.sort_values("d")
        ch, prev = [], None
        for d_, lv in zip(s.d.values, s.lv.values):
            if lv != prev:
                ch.append([str(pd.Timestamp(dates[d_]).date()), int(lv)])
                prev = lv
        hist[codes[c_]] = ch

    # ── 섹터 표 ──
    tnow = {}
    for t, m in T.items():
        f = TF[t].iloc[-1]
        ti = tidx[t]
        mem = [c for c in m if pd.notna(C[c].iloc[-1])]
        lead = []
        if mem:
            df = pd.DataFrame({"r60": F["r60"][mem].iloc[-1], "r20": F["r20"][mem].iloc[-1], "amt": amt20[mem].iloc[-1],
                               "cap": pd.Series(cap).reindex(mem), "hi": F["hi52"][mem].iloc[-1]})
            rk = df.rank(pct=True)
            ls = (rk.cap * 2 + rk.amt * 2 + rk.r60 + rk.r20 + rk.hi).sort_values(ascending=False)   # 규모·거래대금 우선, 그다음 강도
            lead = list(ls.index[:5])
        series = ti.iloc[-250:]
        tnow[t] = {
            "n": len(mem), "cap": float(pd.Series(cap).reindex(mem).sum()),
            "r1d": float(ti.pct_change().iloc[-1]), "r1w": float(ti.pct_change(5).iloc[-1]), "r1m": float(ti.pct_change(20).iloc[-1]),
            "r3m": float(ti.pct_change(60).iloc[-1]), "r6m": float(ti.pct_change(120).iloc[-1]), "r1y": float(ti.pct_change(240).iloc[-1]),
            "rel1m": float(f.r20), "rel3m": float(f.r60), "br20": float(f.br20), "br60": float(f.br60), "amt": float(f.amt), "hi": float(f.hi),
            "score": float(sc[t].iloc[-1]), "rank": int(srank[t].iloc[-1]),
            "rank1w": int(srank[t].iloc[-6]) if nd > 6 and pd.notna(srank[t].iloc[-6]) else None,
            "rank1m": int(srank[t].iloc[-21]) if nd > 21 and pd.notna(srank[t].iloc[-21]) else None,
            "amtShare": float(amt5[mem].iloc[-1].sum() / tot5.iloc[-1]) if mem else 0.0,
            "lead": lead, "members": mem,
            "series": [[str(pd.Timestamp(d_).date()), round(float(v / series.iloc[0]), 4)] for d_, v in series.items()],
            "scoreHist": [[str(pd.Timestamp(d_).date()), int(v)] for d_, v in srank[t].iloc[-60:].dropna().items()],
        }
    mser = midx.iloc[-250:]

    # ── 종목별 결과 ──
    vals = {k: F[k].iloc[t_last] for k in set(EXPL) | {"t_a20", "t_a60", "r5"}}
    out = {}
    for j, c_ in enumerate(c_today):
        code = codes[c_]
        p_ = float(pct_today[j])
        lv = int(level([p_])[0])
        t = th.get(code, "기타")
        z = Zt.loc[code] if code in Zt.index else None
        why = _why(code, t, lv, coef, z, vals, tnow.get(t))
        out[code] = {"lv": lv, "lab": LV[lv], "score": round(p_ * 100, 1), "why": why, "theme": t,
                     "r1m": round(float(vals["r20"].get(code, np.nan)) * 100, 2) if np.isfinite(vals["r20"].get(code, np.nan)) else None,
                     "r3m": round(float(vals["r60"].get(code, np.nan)) * 100, 2) if np.isfinite(vals["r60"].get(code, np.nan)) else None,
                     "tRank": tnow[t]["rank"] if t in tnow else None, "hist": hist.get(code, [])[-12:],
                     "val": val[str(lv)][:2]}
        v_ = vol60[code].iloc[t_last]
        if np.isfinite(v_):
            kq = min(int(p_ * 10), 9)
            out[code]["fc"] = [round(float(x * v_) * 100, 2) for x in QT[kq]] + [round(float(QM[kq] * v_) * 100, 2)]
    meta = {"asof": str(pd.Timestamp(dates[-1]).date()), "n": len(out), "oos": [str(pd.Timestamp(dates[starts[0]]).date()), str(pd.Timestamp(dates[-1]).date())],
            "val": val, "fc": fcv, "monthly": monthly[-36:], "consist": round(consist * 100, 1) if consist is not None else None,
            "dec": [round(float(v) * 100, 2) for v in dec.values], "tval": tval, "cut": CUT,
            "market": [[str(pd.Timestamp(d_).date()), round(float(v / mser.iloc[0]), 4)] for d_, v in mser.items()],
            "mret": {"r1d": float(midx.pct_change().iloc[-1]), "r1w": float(midx.pct_change(5).iloc[-1]),
                     "r1m": float(midx.pct_change(20).iloc[-1]), "r3m": float(midx.pct_change(60).iloc[-1])}}
    pk = None
    try:
        import picks as PK
        pk = PK.build(dict(locals()), log=log)
    except Exception:
        import traceback
        traceback.print_exc()
    return {"ops": out, "themes": tnow, "theme_of": th, "meta": meta, "picks": pk}


def _pp(x, d=1):
    return f"{x * 100:+.{d}f}".replace("-", "−")


def _why(code, t, lv, coef, z, v, tinfo):
    """의견 근거 — 의견 방향으로 기여한 요인 상위 3개 (값 포함). 중립은 강점 1·부담 1."""
    if z is None:
        return []
    g = lambda k: float(v[k].get(code, np.nan)) if k in v else np.nan
    bull, bear = [], []
    for k, cf in coef.items():
        zk = z.get(k, np.nan)
        if not np.isfinite(zk):
            continue
        contrib = cf * zk
        txt = _phrase(k, zk >= 0, contrib > 0, g, t, tinfo)
        if txt:
            (bull if contrib > 0 else bear).append((abs(contrib), txt))
    bull.sort(reverse=True)
    bear.sort(reverse=True)
    pick = bull[:3] if lv > 0 else bear[:3] if lv < 0 else bull[:1] + bear[:1]
    res = []
    for _, s_ in pick:
        if s_ not in res:
            res.append(s_)
    return res


def _phrase(k, high, bull, g, t, ti):
    """요인 값(high: 같은 날 상위권) × 의견 방향(bull) → 문장. 의미 있는 크기가 아니면 None."""
    tn = t if t != "기타" else "소속 업종"
    q = lambda hb, hs, lb, ls: (hb if bull else hs) if high else (lb if bull else ls)
    if k in ("t_r20", "t_r60"):
        r, per, th_ = g(k), "1개월" if k == "t_r20" else "3개월", 0.01 if k == "t_r20" else 0.02
        if not np.isfinite(r) or abs(r) < th_:
            return None
        return q(f"{tn} {per} 시장 대비 {_pp(r)}%p · 섹터 강세", f"{tn} {per} 시장 대비 {_pp(r)}%p 급등 · 섹터 과열 부담",
                 f"{tn} {per} 시장 대비 {_pp(r)}%p 부진 · 순환매 기대", f"{tn} {per} 시장 대비 {_pp(r)}%p · 섹터 약세")
    if k == "t_br20":
        b = g(k)
        if not np.isfinite(b) or 0.35 < b < 0.6:
            return None
        p_ = f"{b * 100:.0f}%"
        return q(f"{tn} 종목 {p_}가 20일선 위 · 상승 확산", f"{tn} 종목 {p_}가 20일선 위 · 과열권",
                 f"{tn} 20일선 위 종목 {p_} · 바닥권", f"{tn} 20일선 위 종목 {p_} · 섹터 위축")
    if k == "t_amt":
        a = g(k)
        if not np.isfinite(a) or 0.85 < a < 1.2:
            return None
        return q(f"{tn} 거래대금 비중 평소 {a:.1f}배 · 자금 유입", f"{tn} 거래대금 비중 평소 {a:.1f}배 · 과열 신호",
                 f"{tn} 거래대금 비중 평소 {a:.1f}배 · 매물 소화", f"{tn} 거래대금 비중 평소 {a:.1f}배 · 관심 이탈")
    if k == "t_score":
        if not ti:
            return None
        r = ti["rank"]
        if 6 <= r <= 24:
            return None
        return q(f"주도섹터 {r}위 ({tn})", f"주도섹터 {r}위 ({tn}) · 과열 부담", f"{tn} 섹터 순위 {r}위 · 소외 섹터 반등 기대", f"{tn} 섹터 순위 {r}위 · 소외")
    if k == "rs60":
        r = g(k)
        if not np.isfinite(r) or abs(r) < 0.05:
            return None
        return q(f"섹터 대비 3개월 {_pp(r)}%p · 섹터 내 주도", f"섹터 대비 3개월 {_pp(r)}%p · 단기 차익 부담",
                 f"섹터 대비 3개월 {_pp(r)}%p · 키 맞추기 기대", f"섹터 대비 3개월 {_pp(r)}%p · 섹터 내 소외")
    if k == "r20":
        r = g(k)
        if not np.isfinite(r) or -0.08 < r < 0.08:
            return None
        return q(f"1개월 {_pp(r, 0)}% 상승 탄력", f"1개월 {_pp(r, 0)}% 급등 · 단기 과열", f"1개월 {_pp(r, 0)}% 급락 · 반등 여지", f"1개월 {_pp(r, 0)}% 하락 · 추세 약화")
    if k == "r60":
        r = g(k)
        if not np.isfinite(r) or -0.15 < r < 0.15:
            return None
        return q(f"3개월 {_pp(r, 0)}% 상승 추세", f"3개월 {_pp(r, 0)}% 급등 · 차익 매물 부담", f"3개월 {_pp(r, 0)}% · 낙폭 과대", f"3개월 {_pp(r, 0)}% 하락 추세")
    if k == "mom":
        r = g(k)
        if not np.isfinite(r) or abs(r) < 0.2:
            return None
        return q(f"12개월 모멘텀 {_pp(r, 0)}%", f"12개월 {_pp(r, 0)}% 급등 · 가격 부담", f"12개월 {_pp(r, 0)}% · 저점권", f"12개월 {_pp(r, 0)}% · 장기 약세")
    if k == "hi52":
        r = g(k)
        if not np.isfinite(r) or -0.35 < r < -0.05:
            return None
        return q(f"52주 고점 {_pp(r, 1)}% · 신고가권", f"52주 고점 {_pp(r, 1)}% · 차익 실현 부담",
                 f"52주 고점 대비 {_pp(r, 0)}% · 가격 매력", f"52주 고점 대비 {_pp(r, 0)}% · 장기 하락")
    if k == "d20":
        r = g(k)
        if not np.isfinite(r) or -0.08 < r < 0.08:
            return None
        return q(f"20일선 위 {_pp(r, 0)}% · 단기 강세", f"20일선 이격 {_pp(r, 0)}% · 과열", f"20일선 아래 {_pp(r, 0)}% · 평균 회귀 여지", f"20일선 아래 {_pp(r, 0)}% · 단기 약세")
    if k == "vol20":
        r = g(k)
        if not np.isfinite(r) or 0.018 < r < 0.04:
            return None
        p_ = f"{r * 100:.1f}%"
        return q(f"일 변동성 {p_} · 반등 탄력", f"일 변동성 {p_} · 위험 확대", f"일 변동성 {p_} · 안정적 흐름", f"일 변동성 {p_} · 모멘텀 부재")
    if k == "amtr":
        r = g(k)
        if not np.isfinite(r) or 0.6 < r < 1.5:
            return None
        return q(f"거래대금 평소 {r:.1f}배 · 관심 유입", f"거래대금 평소 {r:.1f}배 · 단기 과열", f"거래대금 평소 {r:.1f}배 · 매도세 진정", f"거래대금 평소 {r:.1f}배 · 관심 감소")
    if k == "rsi":
        r = g(k)
        if not np.isfinite(r) or 30 < r < 70:
            return None
        return q(f"RSI {r:.0f} 강세", f"RSI {r:.0f} 과열", f"RSI {r:.0f} 침체 · 반등 여지", f"RSI {r:.0f} 약세")
    return None
