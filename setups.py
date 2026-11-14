"""매매 기법(셋업) 기반 추천 — 한국 시장에서 쓰이는 차트 기법을 매일 10년 실데이터로 다시 검증하고,
지금 시장 국면에서 '아무 날이나 산 경우(무작위 진입)'보다 확실히 나았던 기법의 오늘 신호만 추천한다.

- 매수: 신호가 뜬 다음 날 시가
- 청산: 기법마다 정해진 규칙(예: 종가가 5일선 위로 올라오면 다음 날 시가 매도) + 손절 상한
- 손절 상한: 단기 기법 −10%, 20일 이상 기법 −15% (ATR 기준 손절이 이보다 넓으면 상한으로 좁힘)
- 비용: 0.3% (수수료·세금·슬리피지)
"""
from __future__ import annotations
import numpy as np, pandas as pd

COST = 0.003

# 청산 규칙: (종류, 최대 보유일, 손절 ATR 배수, 손절 상한)
RULES = {
    "mr5": ("ma5", 10, 3.0, 0.10, "종가가 5일선 위로 올라오면 다음 날 시가 매도 · 최대 10거래일"),
    "h20": ("hold", 20, 3.0, 0.15, "20거래일 보유 후 매도"),
    "tr10": ("ma10", 30, 2.5, 0.15, "종가가 10일선 아래로 내려가면 다음 날 시가 매도 · 최대 30거래일"),
}

# 기법: 이름, 설명, 청산 규칙, 순위 키(작을수록 우선이면 음수)
SETUPS = {
    "dip": {"name": "낙폭과대 반등", "rule": "mr5",
            "desc": "대형주가 20일선 대비 12% 이상 떨어졌을 때 사서 5일선 회복 시 판다 (이격도 매매). 이격도 85% 이하·외국인 순매수·거래 감소가 겹친 종목을 먼저 고른다"},
    "rsi2": {"name": "과매도 반등", "rule": "mr5",
             "desc": "200일선 위 상승 추세 종목이 2일 RSI 10 아래로 급락했을 때 사서 5일선 회복 시 판다"},
    "pull": {"name": "기준봉 눌림목", "rule": "mr5",
             "desc": "최근 10일 안에 거래량 3배 장대양봉(기준봉)이 나온 뒤, 기준봉 시가 위에서 거래가 줄며 5일선 아래로 눌렸을 때"},
    "high52": {"name": "52주 신고가 돌파", "rule": "h20",
               "desc": "거래대금이 평소의 1.5배 이상 늘며 1년 최고가를 양봉으로 넘었을 때 사서 20거래일 보유"},
    "flow": {"name": "외국인·기관 쌍끌이", "rule": "h20",
             "desc": "최근 5일 중 4일 이상 외국인과 기관이 함께 순매수하고 20일선 위에 있을 때"},
}


def _panel(prices: dict, caps: dict, flows: dict | None):
    codes = [c for c in prices if prices[c] is not None and len(prices[c]) > 260]
    C = pd.DataFrame({c: prices[c].Close for c in codes}).sort_index()
    idx = C.index
    g = lambda k: pd.DataFrame({c: prices[c][k] for c in codes}).reindex(idx)
    O, H, L, V = g("Open"), g("High"), g("Low"), g("Volume")
    C = C.where(C > 0); O = O.where(O > 0).fillna(C)
    H = H.where(H > 0).fillna(C); L = L.where(L > 0).fillna(C)
    cap = np.array([caps.get(c, 0) or 0 for c in codes], float)
    P = {"codes": codes, "idx": idx, "C": C, "O": O, "H": H, "L": L, "V": V, "cap": cap}
    if flows:
        P["fr"] = pd.DataFrame({c: flows[c]["외국인"] for c in flows if "외국인" in flows[c]}).reindex(idx).reindex(columns=codes)
        P["ins"] = pd.DataFrame({c: flows[c]["기관합계"] for c in flows if "기관합계" in flows[c]}).reindex(idx).reindex(columns=codes)
    return P


def _rsi(C, n):
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _signals(P):
    C, O, H, L, V, cap = P["C"], P["O"], P["H"], P["L"], P["V"], P["cap"]
    idx = C.index
    BIG = np.broadcast_to(cap >= 1e12, C.shape)
    ma = {n: C.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60, 200)}
    amt = C * V
    amt20 = amt.rolling(20, min_periods=15).mean()
    vol20 = V.rolling(20, min_periods=15).mean()
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()]).groupby(level=0).max().reindex(idx)
    atr = tr.rolling(14, min_periods=10).mean()
    rsi2 = _rsi(C, 2)
    ret1 = C.pct_change()
    body = (C - O) / O
    liq = amt20 >= 3e9
    dis = C / ma[20]
    hi250p = H.rolling(250, min_periods=200).max().shift(1)
    kb = (ret1 >= .07) & (body >= .05) & (V >= vol20.shift(1) * 3)
    kb_recent = kb.rolling(10, min_periods=1).max().astype(bool) & ~kb
    kb_open = O.where(kb).ffill(limit=10)
    kb_vol = V.where(kb).ffill(limit=10)
    S = {
        "dip": liq & BIG & (dis <= .88),
        "rsi2": liq & (C > ma[200]) & (rsi2 < 10),
        "pull": liq & (ma[20] > ma[60]) & kb_recent & (C < ma[5]) & (C > kb_open) & (V < kb_vol / 3),
        "high52": liq & (C > hi250p) & (amt >= amt20.shift(1) * 1.5) & (body > 0),
    }
    if "fr" in P:
        both = ((P["fr"] > 0) & (P["ins"] > 0)).astype(float).where(P["fr"].notna())
        S["flow"] = liq & (both.rolling(5).sum() >= 4) & (C > ma[20])
    rng = np.random.default_rng(7)
    S["_base"] = liq & pd.DataFrame(rng.random(C.shape) < 0.01, index=idx, columns=C.columns)
    # 시장 국면 (시가총액 가중 지수)
    w = np.nan_to_num(cap)
    r = (ret1.fillna(0) * w).sum(axis=1) / (C.notna() * w).sum(axis=1).replace(0, np.nan)
    kos = (1 + r.fillna(0)).cumprod()
    k20, k60 = kos.rolling(20).mean(), kos.rolling(60).mean()
    breadth = (C > ma[20]).sum(axis=1) / C.notna().sum(axis=1)
    kvolp = kos.pct_change().rolling(20).std().rolling(250, min_periods=120).rank(pct=True)
    gate = ((kos > k60) & (kos > k20) & (breadth >= .4) & (kvolp < .85)).values
    risk_off = ((kos < k60) & (kos < k20)).values
    regime = np.where(gate, "on", np.where(risk_off, "off", "mid"))
    q = (dis <= .85).astype(int) + (V < vol20).astype(int)
    if "fr" in P:
        q = q + (P["fr"] > 0).astype(int)
    return {"S": S, "q": q, "fr": P.get("fr"), "vol20": vol20, "ma": ma, "atr": atr, "amt20": amt20, "dis": dis, "rsi2": rsi2, "regime": regime,
            "kb_open": kb_open, "hi250p": hi250p}


def _exits(P, X, d, c, rule):
    """다음 날 시가 매수 후 청산 규칙으로 판 수익률·보유일 (손절 상한 반영)."""
    kind, hmax, ks, cap_s = RULES[rule][:4]
    Cn, On, Hn, Ln = (P[k].values.astype(float) for k in ("C", "O", "H", "L"))
    nd = Cn.shape[0]
    atrn = X["atr"].values
    e = On[d + 1, c]
    a = np.nan_to_num(atrn[d, c], nan=np.nanmedian(atrn))
    stop = np.maximum(e - ks * a, e * (1 - cap_s))
    w = np.minimum(d[:, None] + np.arange(1, hmax + 1)[None, :], nd - 1)
    cl, op, lo = Cn[w, c[:, None]], On[w, c[:, None]], Ln[w, c[:, None]]
    hs = cl <= stop[:, None]
    if kind == "ma5":
        ht = cl > X["ma"][5].values[w, c[:, None]]
    elif kind == "ma10":
        ht = cl < X["ma"][10].values[w, c[:, None]]
    else:
        ht = np.zeros_like(hs)
    B = hmax + 1
    fs = np.where(hs.any(1), hs.argmax(1), B)
    ft = np.where(ht.any(1), ht.argmax(1), B)
    r = np.arange(len(d))
    nxt = lambda k: np.where(k + 1 < hmax, op[r, np.minimum(k + 1, hmax - 1)], cl[r, np.minimum(k, hmax - 1)])
    stopped = (fs < B) & (fs <= ft)
    hit = (ft < B) & ~stopped
    px = np.where(stopped, nxt(fs), np.where(hit, nxt(ft), cl[:, -1]))
    days = np.where(stopped, fs + 2, np.where(hit, ft + 2, hmax))
    return px / e - 1 - COST, np.minimum(days, hmax)


def _stat(r, dy):
    if len(r) < 40:
        return None
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": int(len(r)), "win": round(float((r > 0).mean()) * 100, 1), "avg": round(float(r.mean()) * 100, 2),
            "pf": round(float(g / l), 2) if l > 0 else None, "days": round(float(dy.mean()), 1),
            "worst": round(float(np.percentile(r, 5)) * 100, 1)}


def build(prices: dict, caps: dict, names: dict, flows: dict | None = None, ops: dict | None = None, log=print, top=12):
    P = _panel(prices, caps, flows)
    X = _signals(P)
    nd = len(P["idx"])
    reg = X["regime"]
    years = P["idx"].year.values
    now_reg = str(reg[-1])
    amt20 = X["amt20"].values
    book, today = {}, []
    base_cache = {}
    for key, sig in X["S"].items():
        m = sig.fillna(False).values.copy()
        last = m[-1].copy()
        m[:250] = False
        m[nd - 31:] = False
        d, c = np.nonzero(m)
        if len(d) < 40:
            continue
        df = pd.DataFrame({"d": d, "c": c, "a": amt20[d, c]}).sort_values(["d", "a"], ascending=[True, False]).groupby("d").head(20)
        d, c = df.d.values, df.c.values
        rules = [SETUPS[key]["rule"]] if key != "_base" else list(RULES)
        for rule in rules:
            r, dy = _exits(P, X, d, c, rule)
            ok = np.isfinite(r)
            rg = reg[d]
            st = {"all": _stat(r[ok], dy[ok]), "now": _stat(r[ok & (rg == now_reg)], dy[ok & (rg == now_reg)]),
                  "recent": _stat(r[ok & (years[d] >= years[-1] - 1)], dy[ok & (years[d] >= years[-1] - 1)])}
            yy = {}
            for y in np.unique(years[d]):
                s = _stat(r[ok & (years[d] == y)], dy[ok & (years[d] == y)])
                if s:
                    yy[str(y)] = s["avg"]
            st["years"] = yy
            if key == "_base":
                base_cache[rule] = st
            else:
                book[key] = {**{k: SETUPS[key][k] for k in ("name", "desc", "rule")}, "exit": RULES[rule][4], "stat": st,
                             "today": [P["codes"][j] for j in np.flatnonzero(last)]}
    # 우위 판정: 지금 국면(표본이 적으면 전체)에서 같은 청산 규칙의 무작위 진입보다 평균 0.3%p 이상, 승률도 높고, 손익비 1.1 이상
    for key, b in book.items():
        base = base_cache.get(b["rule"], {})
        s = b["stat"]["now"] if (b["stat"]["now"] and b["stat"]["now"]["n"] >= 60) else b["stat"]["all"]
        bs = base.get("now") if (s is b["stat"]["now"] and base.get("now")) else base.get("all")
        b["basis"] = "now" if s is b["stat"]["now"] else "all"
        b["base"] = bs
        lose_y = sum(1 for v in b["stat"]["years"].values() if v < 0)
        b["lose_years"] = lose_y
        b["years_n"] = len(b["stat"]["years"])
        b["edge"] = round(s["avg"] - bs["avg"], 2) if (s and bs) else None
        b["ok"] = bool(s and bs and s["avg"] - bs["avg"] >= 0.3 and s["win"] >= bs["win"] and (s["pf"] or 0) >= 1.1
                       and lose_y <= max(2, b["years_n"] // 4))
        log(f"    {b['name']}: {'채택' if b['ok'] else '제외'} · {b['basis']} {s} · 기준 {bs} · 손실 연도 {lose_y}/{b['years_n']} · 오늘 {len(b['today'])}")
    # 오늘 신호 → 추천
    C, ma, atr = P["C"], X["ma"], X["atr"]
    j_of = {c: j for j, c in enumerate(P["codes"])}
    seen = set()
    order = sorted([k for k, b in book.items() if b["ok"]], key=lambda k: -(book[k]["edge"] or 0))
    for key in order:
        b = book[key]
        kind, hmax, ks, cap_s = RULES[b["rule"]][:4]
        cand = []
        for code in b["today"]:
            j = j_of[code]
            px = float(C.iloc[-1, j])
            a = float(atr.iloc[-1, j]) if np.isfinite(atr.iloc[-1, j]) else px * .03
            stop = max(px - ks * a, px * (1 - cap_s))
            m5, m20 = float(ma[5].iloc[-1, j]), float(ma[20].iloc[-1, j])
            if kind == "ma5":
                tgt, tlab = max(m5, px * 1.01), "5일선"
                t2, t2lab = (m20, "20일선") if m20 > tgt * 1.01 else (None, None)
            elif kind == "hold":
                tgt, tlab, t2, t2lab = px + 2 * a, "ATR 2배", None, None
            else:
                tgt, tlab, t2, t2lab = None, None, None, None
            why = []
            if key == "dip":
                why.append(f"20일선 대비 {(X['dis'].iloc[-1, j] - 1) * 100:.1f}% (이격도 {X['dis'].iloc[-1, j] * 100:.0f}%)")
                if X["dis"].iloc[-1, j] <= .85:
                    why.append("이격도 85% 이하 — 과거 이 깊이에서 반등 성적이 더 좋았음")
                if X["fr"] is not None and X["fr"].iloc[-1, j] > 0:
                    why.append("급락 중에도 오늘 외국인 순매수")
                if P["V"].iloc[-1, j] < X["vol20"].iloc[-1, j]:
                    why.append("거래량이 20일 평균보다 줄어 투매가 잦아드는 모습")
            elif key == "rsi2":
                why.append(f"2일 RSI {X['rsi2'].iloc[-1, j]:.0f} · 200일선 위")
            elif key == "pull":
                why.append(f"기준봉 시가 {X['kb_open'].iloc[-1, j]:,.0f}원 위에서 눌림")
            elif key == "high52":
                why.append(f"1년 최고가 {X['hi250p'].iloc[-1, j]:,.0f}원 돌파")
            op = (ops or {}).get(code) or {}
            score = op.get("score")
            cand.append({"code": code, "name": names.get(code, code), "setup": key, "close": round(px, 2),
                         "stop": round(stop), "stop_pct": round((stop / px - 1) * 100, 1),
                         "target": round(tgt) if tgt else None, "target_lab": tlab,
                         "target_pct": round((tgt / px - 1) * 100, 1) if tgt else None,
                         "target2": round(t2) if t2 else None, "target2_lab": t2lab,
                         "why": why, "score": score, "amt": float(amt20[-1, j]), "q": int(X["q"].iloc[-1, j])})
        if key == "dip":
            cand.sort(key=lambda x: (-x["q"], -(x["score"] or 0), -x["amt"]))
        else:
            cand.sort(key=lambda x: (-(x["score"] or 0), -x["amt"]))
        k = 0
        for x in cand:
            if x["code"] in seen:
                continue
            seen.add(x["code"]); x.pop("amt"); x.pop("q")
            today.append(x); k += 1
            if k >= 5:
                break
    today = today[:top]
    for b in book.values():
        b["count"] = len(b.pop("today"))
    return {"asof": str(P["idx"][-1].date()), "regime": now_reg,
            "regime_lab": {"on": "상승장", "mid": "혼조", "off": "하락장"}[now_reg],
            "book": book, "base": base_cache, "picks": today,
            "rules": {k: v[4] for k, v in RULES.items()}}
