"""기술적 분석 리포트 — 국면(가격조정·기간조정 등), 피보나치 되돌림, 매물대, 이동평균, 거래량·캔들, 시나리오.

입력은 지표가 붙은 일봉(indicators.add_indicators 결과). 모든 판단은 마지막 봉까지의 정보만 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIB = (0.236, 0.382, 0.5, 0.618, 0.786)
PHASE = {   # 키: (이름, 방향)
    "uptrend": ("상승 추세", 1),
    "price_corr": ("가격조정", 0),
    "price_corr_rb": ("가격조정 후 반등", 1),
    "time_corr": ("기간조정", 0),
    "base": ("바닥 다지기", 0),
    "downtrend": ("하락 추세", -1),
    "box": ("박스권", 0),
    "mixed": ("변동성 확대", 0),
}


def won(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def wr(v: float) -> str:
    return won(rt(v))


def pc(x: float, d: int = 0) -> str:
    return f"{x * 100:+.{d}f}%".replace("-", "−")


def tick(p: float) -> int:
    for lim, t in ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500)):
        if p < lim:
            return t
    return 1_000


def rt(p: float) -> float:
    """호가 단위로 반올림."""
    t = tick(p)
    return float(round(p / t) * t)


def _date(d, i) -> str:
    return pd.Timestamp(d.index[i]).strftime("%m.%d")


# ── 스윙: 최근 고점·그 이전 저점·고점 이후 저점 ────────────────────
def swing(d: pd.DataFrame, look: int = 120) -> dict:
    n = len(d)
    h, l, c = d.High.values, d.Low.values, d.Close.values
    a = max(0, n - look)
    ih = a + int(np.flatnonzero(h[a:] == h[a:].max())[-1])
    b = max(0, ih - look)
    il = b + int(np.argmin(l[b:ih + 1]))
    H, L, C = float(h[ih]), float(l[il]), float(c[-1])
    if ih < n - 1:
        ill = ih + 1 + int(np.argmin(l[ih + 1:]))
        LL = float(l[ill])
    else:
        ill, LL = ih, float(l[ih])
    return {"ih": ih, "il": il, "ill": ill, "H": H, "L": L, "LL": LL, "C": C, "days": n - 1 - ih,
            "dd": C / H - 1, "up": H / L - 1 if L > 0 else 0.0, "retr": (H - C) / (H - L) if H > L else 0.0,
            "rng": H / LL - 1 if LL > 0 else 0.0, "rb": C / LL - 1 if LL > 0 else 0.0}


# ── 매물대 (가격대별 거래대금) ─────────────────────────────────────
def volume_profile(d: pd.DataFrame, look: int = 120, bins: int = 24) -> dict:
    x = d.iloc[-look:]
    lo, hi = float(x.Low.min()), float(x.High.max())
    if not hi > lo:
        return {}
    edges = np.linspace(lo, hi, bins + 1)
    w = np.zeros(bins)
    for L_, H_, C_, V_ in zip(x.Low.values, x.High.values, x.Close.values, x.Volume.values):
        if not (V_ > 0 and H_ >= L_):
            continue
        amt = V_ * C_
        if H_ == L_:
            w[min(bins - 1, int((C_ - lo) / (hi - lo) * bins))] += amt
            continue
        ov = np.clip(np.minimum(edges[1:], H_) - np.maximum(edges[:-1], L_), 0, None)
        if ov.sum() > 0:
            w += amt * ov / ov.sum()
    share = w / w.sum() if w.sum() > 0 else w
    c = float(d.Close.iloc[-1])
    heavy = share >= max(1.25 / bins, np.quantile(share, 0.7))

    def zone(direction):
        idx = range(bins) if direction > 0 else range(bins - 1, -1, -1)
        z = None
        for i in idx:
            inside = (edges[i] > c) if direction > 0 else (edges[i + 1] < c)
            if not inside:
                continue
            if heavy[i]:
                z = [i, i] if z is None else [min(z[0], i), max(z[1], i)]
            elif z is not None:
                break
        if z is None:
            return None
        return [rt(float(edges[z[0]])), rt(float(edges[z[1] + 1])), float(share[z[0]:z[1] + 1].sum())]

    poc = int(np.argmax(share))
    return {"bins": [[round(float(edges[i]), 2), round(float(edges[i + 1]), 2), round(float(share[i]), 4)] for i in range(bins)],
            "poc": rt(float((edges[poc] + edges[poc + 1]) / 2)), "above": zone(1), "below": zone(-1)}


# ── 국면 판정 ────────────────────────────────────────────────────
def phase(d: pd.DataFrame, sw: dict) -> dict:
    x = d.iloc[-1]
    c = float(x.Close)
    ma = {k: float(x[f"MA{k}"]) if pd.notna(x.get(f"MA{k}")) else np.nan for k in (5, 20, 60, 120)}
    s60 = float(d.MA60.iloc[-1] / d.MA60.iloc[-6] - 1) if len(d) > 70 else 0.0
    n, dd, up, rng = sw["days"], sw["dd"], sw["up"], sw["rng"]
    H, LL = sw["H"], sw["LL"]
    hd = _date(d, sw["ih"])
    x20 = d.iloc[-20:]
    r20 = float(x20.High.max() / x20.Low.min() - 1)
    lo52 = float(d.Low.iloc[-250:].min())
    x60 = d.iloc[-60:]
    lo60, hi60 = float(x60.Low.min()), float(x60.High.max())
    s20 = float(d.MA20.iloc[-1] / d.MA20.iloc[-6] - 1) if len(d) > 26 else 0.0
    corr = LL / H - 1                     # 고점 이후 최대 하락폭
    if dd >= -0.06 and ma[20] > ma[60] and s60 > 0:
        k = "uptrend"
        desc = f"고점 {wr(H)}원 부근({pc(dd, 1)}) · 20일선이 60일선 위에서 상승"
    elif up >= 0.15 and dd <= -0.12 and sw["rb"] < 0.07 and n <= 60:
        k = "price_corr"
        desc = f"고점 {wr(H)}원({hd}) 대비 {pc(dd, 1)} · {n}영업일째 · " + (f"상승폭의 {sw['retr'] * 100:.0f}% 되돌림" if sw['retr'] < 0.97 else "직전 상승분 대부분 반납")
    elif up >= 0.15 and corr <= -0.12 and sw["rb"] >= 0.07 and n <= 90 and dd <= -0.05:
        k = "price_corr_rb"
        desc = f"고점 {wr(H)}원({hd}) 대비 {pc(corr, 0)} 조정 후 저점 대비 {pc(sw['rb'], 1)} 반등 · 현재 고점 대비 {pc(dd, 1)}"
    elif up >= 0.15 and n >= 12 and rng <= 0.16:
        k = "time_corr"
        desc = f"고점 {wr(H)}원({hd}) 이후 {n}영업일째 {wr(LL)}~{wr(H)}원 횡보 · 가격 대신 시간으로 조정"
    elif dd <= -0.25 and c < ma[120] and r20 <= 0.14 and abs(s20) < 0.02:
        k = "base"
        desc = f"고점 대비 {pc(dd, 0)} 하락 후 최근 20일 {wr(float(x20.Low.min()))}~{wr(float(x20.High.max()))}원 횡보"
    elif ma[20] < ma[60] and s60 < 0 and dd <= -0.15:
        k = "downtrend"
        desc = f"20일선이 60일선 아래에서 하락 · 고점 대비 {pc(dd, 0)}"
        if c <= lo52 * 1.05:
            desc += " · 52주 저가권"
    elif hi60 / lo60 - 1 <= 0.25:
        k = "box"
        desc = f"최근 60일 {wr(lo60)}~{wr(hi60)}원 박스권 (폭 {(hi60 / lo60 - 1) * 100:.0f}%)"
    else:
        k = "mixed"
        desc = f"최근 60일 {wr(lo60)}~{wr(hi60)}원 · 고점 대비 {pc(dd, 0)} · 방향성 탐색"
    return {"key": k, "name": PHASE[k][0], "bias": PHASE[k][1], "desc": desc, "days": n,
            "from": str(pd.Timestamp(d.index[sw["ih"]]).date()), "hi": H, "lo": LL}


# ── 피보나치 ─────────────────────────────────────────────────────
def fibonacci(d: pd.DataFrame, sw: dict, ph: dict) -> dict | None:
    c = sw["C"]
    if ph["key"] in ("uptrend", "price_corr", "price_corr_rb", "time_corr", "box", "mixed") and sw["up"] >= 0.15:
        lo, hi = sw["L"], sw["H"]
        lv = [[r, rt(hi - (hi - lo) * r)] for r in FIB]
        kind, a, b = "retr", sw["il"], sw["ih"]
        now = sw["retr"]
    elif ph["key"] in ("downtrend", "base") and sw["H"] / sw["LL"] - 1 >= 0.2 and sw["ill"] > sw["ih"]:
        hi, lo = sw["H"], sw["LL"]
        lv = [[r, rt(lo + (hi - lo) * r)] for r in FIB]
        kind, a, b = "rebound", sw["ih"], sw["ill"]
        now = (c - lo) / (hi - lo)
    else:
        return None
    near = min(lv, key=lambda z: abs(z[1] / c - 1))
    note = None
    if abs(near[1] / c - 1) <= 0.025:
        note = (f"상승폭의 {near[0] * 100:.1f}% 되돌림({won(near[1])}원) 지지 테스트" if kind == "retr"
                else f"하락폭의 {near[0] * 100:.1f}% 반등({won(near[1])}원) 저항 테스트")
    elif kind == "retr":
        nxt = [z for z in lv if z[1] < c]
        if nxt:
            note = f"되돌림 {now * 100:.0f}% · 다음 지지 {nxt[0][0] * 100:.1f}%({won(nxt[0][1])}원)"
    else:
        nxt = [z for z in lv if z[1] > c]
        if nxt:
            note = f"반등 {now * 100:.0f}% · 1차 저항 {nxt[0][0] * 100:.1f}%({won(nxt[0][1])}원)"
    return {"kind": kind, "a": [str(pd.Timestamp(d.index[a]).date()), round(float(lo if kind == 'retr' else hi), 2)],
            "b": [str(pd.Timestamp(d.index[b]).date()), round(float(hi if kind == 'retr' else lo), 2)],
            "levels": [[r, round(float(p), 2)] for r, p in lv], "now": round(float(now), 3), "note": note}


# ── 지지·저항 (겹치는 근거 묶기) ───────────────────────────────────
def key_levels(d: pd.DataFrame, fib: dict | None, vp: dict, sw: dict, extra: list | None = None) -> dict:
    import patterns as P
    c = float(d.Close.iloc[-1])
    atrp = float(d.ATR_pct.iloc[-1]) if np.isfinite(d.ATR_pct.iloc[-1]) else 0.02
    gap = max(0.012, atrp * 0.4)
    cand = []
    x = d.iloc[-120:]
    kp = 8
    ph_, pl_ = P._pivots(x.High.values, x.Low.values, kp)
    for t in np.flatnonzero(ph_):
        p = float(x.High.values[t - kp])
        cand.append((p, "전고점" if p > c else "직전 저항선"))
    for t in np.flatnonzero(pl_):
        p = float(x.Low.values[t - kp])
        cand.append((p, "전저점" if p < c else "직전 지지선"))
    for n in (20, 60, 120):
        v = d[f"MA{n}"].iloc[-1]
        if np.isfinite(v):
            cand.append((float(v), f"{n}일선"))
    if fib:
        for r, p in fib["levels"]:
            if r in (0.382, 0.5, 0.618):
                cand.append((float(p), f"피보나치 {r * 100:.1f}%"))
    for z, nm in ((vp.get("above"), "상단 매물대"), (vp.get("below"), "하단 매물대")):
        if z:
            cand.append(((z[0] + z[1]) / 2, nm))
    for p, nm in (extra or []):
        cand.append((float(p), nm))
    y = d.iloc[-250:]
    cand.append((float(y.High.max()), "52주 최고가"))
    cand.append((float(y.Low.min()), "52주 최저가"))

    def merge(side):
        pts = sorted([z for z in cand if (z[0] > c * (1 + gap) if side > 0 else z[0] < c * (1 - gap))],
                     key=lambda z: z[0], reverse=side < 0)
        out = []
        for p, nm in pts:
            if out and abs(p / out[-1][0] - 1) <= 0.012:
                if nm not in out[-1][1]:
                    out[-1][1].append(nm)
                continue
            out.append([p, [nm]])
            if len(out) >= 3:
                break
        return [[rt(p), "·".join(nm[:3])] for p, nm in out]

    return {"support": merge(-1), "resist": merge(1)}


# ── 이동평균·거래량·캔들·보조지표 ─────────────────────────────────
def ma_state(d: pd.DataFrame) -> dict:
    x = d.iloc[-1]
    c = float(x.Close)
    m = {k: float(x[f"MA{k}"]) for k in (5, 20, 60, 120) if pd.notna(x.get(f"MA{k}"))}
    if len(m) < 4:
        return {"text": "이동평균 산출 기간 부족", "conv": None}
    if m[5] > m[20] > m[60] > m[120]:
        arr = "정배열"
    elif m[5] < m[20] < m[60] < m[120]:
        arr = "역배열"
    else:
        arr = "혼조"
    sp = (max(m[5], m[20], m[60]) - min(m[5], m[20], m[60])) / c
    s20 = float(d.MA20.iloc[-1] / d.MA20.iloc[-6] - 1) if len(d) > 26 else 0
    trend20 = "상승" if s20 > 0.005 else "하락" if s20 < -0.005 else "보합"
    t = f"{arr} · 20일선 {trend20} · 주가 20일선 {pc(c / m[20] - 1, 1)}"
    if sp <= 0.03:
        t += f" · 5·20·60일선 {sp * 100:.1f}% 이내 수렴"
    return {"text": t, "arr": arr, "conv": round(sp, 4), "s20": trend20}


def volume_state(d: pd.DataFrame, ph_key: str) -> str:
    v = d.Volume
    vm = float(d.VOL_MA20.iloc[-1]) if d.VOL_MA20.iloc[-1] else np.nan
    if not np.isfinite(vm) or vm <= 0:
        return "-"
    r1 = float(v.iloc[-1]) / vm
    r5 = float(v.iloc[-5:].mean()) / vm
    if r1 >= 2:
        t = f"당일 거래량 20일 평균의 {r1:.1f}배 · 수급 유입"
    elif r5 <= 0.6:
        t = f"최근 5일 거래량 20일 평균의 {r5 * 100:.0f}% · 거래량 바닥권"
    else:
        t = f"당일 거래량 20일 평균의 {r1:.1f}배"
    if ph_key in ("price_corr", "time_corr") and r5 < 0.8:
        t += " · 조정 중 거래 감소로 매도 압력 약화"
    elif ph_key in ("price_corr",) and r5 >= 1.3:
        t += " · 조정 중 거래 증가로 매물 출회 지속"
    return t


def candle_state(d: pd.DataFrame) -> str:
    x = d.iloc[-1]
    o, h, l, c = (float(x[k]) for k in ("Open", "High", "Low", "Close"))
    atr = float(x.ATR) if np.isfinite(x.ATR) else (h - l)
    rng_ = h - l
    if rng_ <= 0:
        return "보합"
    body = abs(c - o)
    up_t, lo_t = h - max(c, o), min(c, o) - l
    vr = float(x.Volume / x.VOL_MA20) if x.VOL_MA20 else 1
    if body >= atr and c > o:
        return "장대양봉" + (f" · 거래량 {vr:.1f}배 동반" if vr >= 1.5 else "")
    if body >= atr and c < o:
        return "장대음봉" + (f" · 거래량 {vr:.1f}배 동반" if vr >= 1.5 else "")
    if up_t >= 0.5 * rng_ and rng_ >= 0.8 * atr:
        return f"윗꼬리 긴 캔들 · {won(h)}원 부근 매물 출회"
    if lo_t >= 0.5 * rng_ and rng_ >= 0.8 * atr:
        return f"아랫꼬리 긴 캔들 · {won(l)}원 부근 저가 매수 유입"
    if body <= 0.1 * rng_:
        return "도지형 · 매수·매도 균형"
    return ("양봉" if c > o else "음봉" if c < o else "보합") + " · 변동 제한적"


def osc_state(d: pd.DataFrame) -> str:
    x = d.iloc[-1]
    rsi = float(x.RSI) if pd.notna(x.RSI) else np.nan
    t = []
    if np.isfinite(rsi):
        t.append(f"RSI {rsi:.0f}" + (" 과매수" if rsi >= 70 else " 과매도" if rsi <= 30 else ""))
    if pd.notna(x.MACD) and pd.notna(x.MACD_signal):
        t.append("MACD 시그널 상회" if x.MACD > x.MACD_signal else "MACD 시그널 하회")
    if pd.notna(x.get("STO_K")):
        k = float(x.STO_K)
        if k >= 80 or k <= 20:
            t.append(f"스토캐스틱 {k:.0f}")
    return " · ".join(t) or "-"


# ── 빗각(추세선) · 기준봉 ──────────────────────────────────────────
def _ds(d, i):
    return str(pd.Timestamp(d.index[i]).date())


def _trendline(h, l, c, side, look=200, k=5, tol=0.004):
    """추세선(빗각) — 하락: 고점 2개 이상을 잇고 그 사이 어떤 고가도 선을 넘지 않는 선, 상승: 저점 대칭.
    후보 가운데 '선에 닿은 고점(저점) 수'가 많고 기간이 긴 선을 고른다. 이후 종가가 선을 1% 넘게 벗어나면 돌파(이탈)."""
    n = len(c)
    s0 = max(0, n - look)
    src = h if side < 0 else l
    piv = [i for i in range(s0 + k, n - 2) if (src[i] == src[max(s0, i - k):min(n, i + k + 1)].max() if side < 0
                                               else src[i] == src[max(s0, i - k):min(n, i + k + 1)].min())]
    if len(piv) < 2:
        return None
    best = None
    for ai in range(len(piv)):
        i = piv[ai]
        for j in piv[ai + 1:]:
            if j - i < 8:
                continue
            p1, p2 = src[i], src[j]
            if (side < 0 and p2 >= p1) or (side > 0 and p2 <= p1):
                continue
            sl = (p2 - p1) / (j - i)
            xs = np.arange(i, n)
            ln = p1 + sl * (xs - i)
            if np.any(ln <= 0):
                continue
            # 두 점 사이에서는 고가(저가)가 선을 넘으면 안 됨
            seg = src[i:j + 1]
            lj = ln[: j - i + 1]
            if (side < 0 and np.any(seg > lj * (1 + tol))) or (side > 0 and np.any(seg < lj * (1 - tol))):
                continue
            # 이후 돌파: 종가 기준 1% 이상
            after = c[j + 1:]
            la = ln[j - i + 1:]
            br = np.flatnonzero(after > la * 1.01) if side < 0 else np.flatnonzero(after < la * 0.99)
            brk = int(j + 1 + br[0]) if len(br) else None
            end = brk if brk is not None else n - 1
            seg2 = src[j:end]
            l2 = ln[j - i:end - i]
            if len(seg2) and ((side < 0 and np.any(seg2 > l2 * 1.02)) or (side > 0 and np.any(seg2 < l2 * 0.98))):
                continue
            touch = sum(1 for q in piv if i <= q <= end and abs(src[q] / (p1 + sl * (q - i)) - 1) <= 0.015)
            score = touch * 100 + (end - i)
            if best is None or score > best["score"]:
                best = {"i1": i, "p1": float(p1), "i2": j, "p2": float(p2), "s": float(sl), "brk": brk, "touch": touch, "score": score}
    return best


def diagonal(d: pd.DataFrame) -> dict:
    """빗각(추세선): 하락 빗각 = 낮아지는 고점들을 이은 선(저항), 상승 빗각 = 높아지는 저점들을 이은 선(지지)."""
    n = len(d)
    c_ = d.Close.values.astype(float)
    c = float(c_[-1])
    h, l = d.High.values.astype(float), d.Low.values.astype(float)
    lines, text, lv = [], [], []
    for side in (-1, 1):
        L = _trendline(h, l, c_, side)
        if not L:
            continue
        val = lambda i: L["p1"] + L["s"] * (i - L["i1"])
        brk = L["brk"]
        if brk is not None and n - 1 - brk > 40:            # 오래전에 깨진 선은 표시하지 않음
            continue
        now = val(n - 1)
        if brk is None and ((side > 0 and now < c * 0.75) or (side < 0 and now > c * 1.3) or now <= 0):
            continue                                        # 지금 가격과 너무 먼 선은 의미가 약함
        endi = n - 1 if brk is None else min(n - 1, brk + 10)
        nm = "하락 빗각" if side < 0 else "상승 빗각"
        anc = f"{_date(d, L['i1'])}·{_date(d, L['i2'])} {'고점' if side < 0 else '저점'} 연결, 접점 {L['touch']}회"
        xs = np.arange(L["i1"], endi + 1)
        base = L["p1"] + L["s"] * (xs - L["i1"])
        src_o = l if side < 0 else h
        off = float((src_o[L["i1"]:endi + 1] - base).min()) if side < 0 else float((src_o[L["i1"]:endi + 1] - base).max())
        lines.append({"kind": "dn" if side < 0 else "up", "name": nm, "a": [_ds(d, L["i1"]), round(L["p1"], 2)],
                      "b": [_ds(d, endi), round(val(endi), 2)], "touch": L["touch"],
                      "ch": [[_ds(d, L["i1"]), round(L["p1"] + off, 2)], [_ds(d, endi), round(val(endi) + off, 2)]]
                      if abs((val(endi) + off) / c - 1) <= 0.3 else None,
                      "brk": _ds(d, brk) if brk is not None else None})
        gap = c / now - 1
        if side < 0:
            if brk is not None:
                st = f"{_date(d, brk)} 종가 돌파" + (" 후 선 위 안착 · 저항이 지지로 바뀌는지 확인" if gap >= 0 else " 후 재이탈 · 돌파 실패 유의")
                if gap >= 0:
                    lv.append((now, "돌파한 하락 빗각", -1))
            elif gap >= -0.03:
                st = f"주가 {pc(gap, 1)} · 저항선 근접, 돌파 여부 주목"
                lv.append((now, "하락 빗각", 1))
            else:
                st = f"주가 {pc(gap, 0)} 아래 · 하락 추세 유지"
                lv.append((now, "하락 빗각", 1))
        else:
            if brk is not None:
                st = f"{_date(d, brk)} 종가 이탈" + (" 후 회복" if gap >= 0 else " · 상승 추세 훼손")
                if gap < 0:
                    lv.append((now, "이탈한 상승 빗각", 1))
            elif gap <= 0.03:
                st = f"주가 {pc(gap, 1)} · 지지선 테스트 중"
                lv.append((now, "상승 빗각", -1))
            else:
                st = f"주가 {pc(gap, 0)} 위 · 상승 추세 유지"
                lv.append((now, "상승 빗각", -1))
        text.append(f"{nm} {wr(now)}원({anc}) · {st}")
    import patterns as P
    fl, _, _ = P.diag(d)
    last = None
    for k in ("tl_dn_brk", "tl_up_brk", "tl_up_sup", "tl_dn_rej"):
        hits = np.flatnonzero(fl[k][-5:])
        if len(hits):
            ago = 4 - int(hits[-1])
            if last is None or ago < last[1]:
                last = (k, ago)
    return {"lines": lines, "text": text, "levels": lv, "recent": last}


def reference_candle(d: pd.DataFrame) -> dict | None:
    """최근 60영업일 안의 기준봉(거래량 3배·+8% 장대양봉)과 현재 위치."""
    import patterns as P
    fl, cur = P.base_candle(d)
    if not cur:
        return None
    n = len(d)
    c = float(d.Close.iloc[-1])
    i = cur["i"]
    o, m, hh = cur["o"], cur["m"], cur["h"]
    if c > hh:
        st = f"고가 {wr(hh)}원 돌파 · 추가 상승 시도"
    elif c >= m:
        st = f"중심값 {wr(m)}원 위 · 기준봉 매수세 유지"
    elif c >= o:
        st = f"중심값 {wr(m)}원 아래 · 시가 {wr(o)}원 지지 여부 관건"
    else:
        st = f"시가 {wr(o)}원 이탈 · 기준봉 매수세 무력화"
    return {"d": _ds(d, i), "o": round(o, 2), "m": round(m, 2), "h": round(hh, 2), "vr": round(float(cur["vr"]), 1),
            "chg": round(float(cur["c"] / d.Close.values[i - 1] - 1), 4), "ago": n - 1 - i,
            "text": f"{_date(d, i)} 거래량 {cur['vr']:.1f}배 · {pc(cur['c'] / d.Close.values[i - 1] - 1, 1)} 장대양봉 · {st}"}


# ── 일목균형표·장기선·평단가·엔벨로프·갭·상한가·거래량 이력 ───────────
def extras(d: pd.DataFrame) -> dict:
    import patterns as P
    x = P.ext_indicators(d)
    n = len(d)
    c = float(d.Close.iloc[-1])
    h, l, o, cl, v = (d[k].values.astype(float) for k in ("High", "Low", "Open", "Close", "Volume"))
    rows, lv, ov = [], [], {}
    last = x.iloc[-1]
    # 일목균형표
    top, bot = last.CLOUD_TOP, last.CLOUD_BOT
    if np.isfinite(top) and np.isfinite(bot):
        pos = "구름대 위" if c > top else "구름대 아래" if c < bot else "구름대 안"
        tk, kj = float(last.TENKAN), float(last.KIJUN)
        tkr = "전환선 > 기준선" if tk > kj else "전환선 < 기준선" if tk < kj else "전환선 = 기준선"
        chk = "후행스팬 주가 위" if n > 26 and c > cl[-27] else "후행스팬 주가 아래"
        fut = "양운" if float(last.SPAN_A) >= float(last.SPAN_B) else "음운"
        thick = (top - bot) / c
        rows.append(["일목균형표", f"{pos} · {tkr} · {chk} · 26일 뒤 {fut}" + (f" · 구름 두께 {thick * 100:.1f}%" if thick > 0 else "")])
        lv += [(top, "구름대 상단"), (bot, "구름대 하단"), (kj, "기준선")]
        ov["ichi"] = {"top": round(float(top), 2), "bot": round(float(bot), 2), "tenkan": round(tk, 2), "kijun": round(kj, 2)}
    # 장기 이평 (120·240)
    m120, m240 = float(d.MA120.iloc[-1]) if pd.notna(d.MA120.iloc[-1]) else np.nan, float(last.MA240) if pd.notna(last.MA240) else np.nan
    parts = []
    for nm, m in (("120일선", m120), ("240일선(1년선)", m240)):
        if np.isfinite(m):
            parts.append(f"{nm} {wr(m)}원 {('상회' if c >= m else '하회')} {pc(c / m - 1, 1)}")
            lv.append((m, nm.split("(")[0]))
    if parts:
        rows.append(["장기 이평", " · ".join(parts)])
    # 평단가(120일 VWAP)
    vw = float(last.VWAP120) if pd.notna(last.VWAP120) else np.nan
    if np.isfinite(vw):
        rows.append(["120일 평단가", f"{wr(vw)}원 · 주가 {pc(c / vw - 1, 1)} · 최근 6개월 매수자 평균 " + ("수익 구간" if c >= vw else "손실 구간")])
        lv.append((vw, "120일 평단가"))
        ov["vwap"] = round(vw, 2)
    # 엔벨로프(20, ±20%)
    ma20 = float(d.MA20.iloc[-1])
    if np.isfinite(ma20):
        el, eh = ma20 * 0.8, ma20 * 1.2
        rows.append(["엔벨로프", f"20일선 ±20% · 하단 {wr(el)}원 · 상단 {wr(eh)}원 · 현재 20일선 {pc(c / ma20 - 1, 1)}"])
    # 갭 (최근 250일): 미체결 = 이후 가격이 갭 구간을 다 채우지 않음. 박스는 남은 구간만.
    gaps, filled = [], []
    for t in range(max(1, n - 250), n):
        if l[t] > h[t - 1] * 1.002:
            lo_, hi_ = h[t - 1], l[t]
            rest = l[t:].min()
            if rest > lo_:
                gaps.append(("상승 갭", lo_, min(hi_, rest), t, None))
            elif n - t <= 120:
                f_ = t + int(np.flatnonzero(l[t:] <= lo_)[0])
                filled.append(("상승 갭", lo_, hi_, t, f_))
        if h[t] < l[t - 1] * 0.998:
            lo_, hi_ = h[t], l[t - 1]
            rest = h[t:].max()
            if rest < hi_:
                gaps.append(("하락 갭", max(lo_, rest), hi_, t, None))
            elif n - t <= 120:
                f_ = t + int(np.flatnonzero(h[t:] >= hi_)[0])
                filled.append(("하락 갭", lo_, hi_, t, f_))
    if gaps:
        near = sorted(gaps, key=lambda g: abs((g[1] + g[2]) / 2 / c - 1))[:3]
        rows.append(["미체결 갭", f"{len(gaps)}개 · 가까운 순 " + " / ".join(f"{k} {wr(a)}~{wr(b)}원({_date(d, t)})" for k, a, b, t, _ in near)])
        for k, a, b, t, _ in near[:2]:
            lv.append((a if k == "상승 갭" else b, f"{k} {'하단' if k == '상승 갭' else '상단'}"))
    if gaps or filled:
        ov["gaps"] = [[k, round(float(a), 2), round(float(b), 2), _ds(d, t), _ds(d, f) if f is not None else None]
                      for k, a, b, t, f in sorted(gaps, key=lambda g: -g[3])[:12] + filled[-8:]]
    # 상·하한가, 거래량 이력, 연속 등락
    chg = cl[1:] / cl[:-1] - 1
    lu = [i + 1 for i in range(max(0, n - 61), n - 1) if chg[i] >= 0.295]
    ld = [i + 1 for i in range(max(0, n - 61), n - 1) if chg[i] <= -0.295]
    if lu or ld:
        rows.append(["상·하한가", " · ".join(([f"상한가 {len(lu)}회(최근 {_date(d, lu[-1])})"] if lu else []) +
                                            ([f"하한가 {len(ld)}회(최근 {_date(d, ld[-1])})"] if ld else [])) + " · 최근 60일"])
    y = slice(max(0, n - 250), n)
    iv = int(np.argmax(v[y])) + y.start
    vm = float(np.nanmean(v[max(0, iv - 20):iv])) if iv > 0 else np.nan
    rows.append(["거래량 이력", f"1년 최대 거래량 {_date(d, iv)}" + (f" · 평소 {v[iv] / vm:.1f}배" if np.isfinite(vm) and vm > 0 else "") +
                 (f" · 전일 대비 {pc(cl[iv] / cl[iv - 1] - 1, 1)}" if iv > 0 else "") + f" · 당일 거래량은 그날의 {v[-1] / v[iv] * 100:.0f}%"])
    k = 0
    for t in range(n - 1, 0, -1):
        s_ = np.sign(cl[t] - cl[t - 1])
        if s_ == 0 or (k and np.sign(k) != s_):
            break
        k += int(s_)
    # 보조지표 상세
    ind = []
    if pd.notna(last.CCI):
        ind.append(f"CCI {last.CCI:.0f}")
    if pd.notna(last.MFI):
        ind.append(f"MFI {last.MFI:.0f}" + (" 과매도" if last.MFI <= 20 else " 과매수" if last.MFI >= 80 else ""))
    if pd.notna(last.WR):
        ind.append(f"Williams %R {last.WR:.0f}")
    if pd.notna(d.ADX.iloc[-1]):
        ind.append(f"ADX {d.ADX.iloc[-1]:.0f}({'+DI' if d.PDI.iloc[-1] > d.MDI.iloc[-1] else '−DI'} 우위)")
    if pd.notna(last.SAR):
        ind.append(f"SAR {'매수' if last.SAR < c else '매도'} 국면")
    return {"rows": rows, "levels": lv, "ov": ov, "streak": k, "ind": " · ".join(ind)}


# ── 종합 ─────────────────────────────────────────────────────────
def report(d: pd.DataFrame) -> dict:
    if len(d) < 130:
        return {}
    sw = swing(d)
    ph = phase(d, sw)
    vp = volume_profile(d)
    fib = fibonacci(d, sw, ph)
    dg = diagonal(d)
    bc = reference_candle(d)
    ex = extras(d)
    ext = [(p, nm) for p, nm, _ in dg["levels"]] + ex["levels"]
    if bc:
        ext += [(bc["h"], "기준봉 고가"), (bc["m"], "기준봉 중심값"), (bc["o"], "기준봉 시가")]
    lv = key_levels(d, fib, vp, sw, ext)
    ma = ma_state(d)
    vol = volume_state(d, ph["key"])
    cdl = candle_state(d)
    osc = osc_state(d)
    c = sw["C"]
    S, R = lv["support"], lv["resist"]

    # 헤드라인
    k = ph["key"]
    tail = None
    DG = {"tl_dn_brk": "하락 빗각 돌파", "tl_up_brk": "상승 빗각 이탈", "tl_up_sup": "상승 빗각 지지", "tl_dn_rej": "하락 빗각 저항"}
    if dg["recent"] and dg["recent"][1] <= 2:
        tail = DG[dg["recent"][0]]
    elif bc and bc["ago"] >= 2 and abs(c / bc["m"] - 1) <= 0.02:
        tail = "기준봉 중심값 지지 테스트"
    elif fib and fib["note"] and "테스트" in fib["note"]:
        tail = f"피보나치 {fib['note'].split('의 ')[1].split('(')[0]} 구간"
    elif S and abs(S[0][0] / c - 1) <= 0.03:
        tail = f"{S[0][1].split('·')[0]} 지지 테스트"
    elif R and abs(R[0][0] / c - 1) <= 0.03:
        r0 = R[0][1].split('·')[0]
        tail = f"{r0} 회복 시도" if r0 == "직전 지지선" else f"{r0} 돌파 시도"
    if k == "time_corr" and ma.get("conv") is not None and ma["conv"] <= 0.03:
        tail = "이평선 수렴, 방향성 결정 임박"
    head = {
        "uptrend": "상승 추세 지속",
        "price_corr": "가격조정 진행",
        "price_corr_rb": "가격조정 후 기술적 반등",
        "time_corr": "기간조정 진행",
        "base": "바닥권 횡보",
        "downtrend": "하락 추세 지속",
        "box": "박스권 등락",
        "mixed": "변동성 확대 구간",
    }[k] + (f", {tail}" if tail else "")

    # 종합 의견 (개조식)
    summ = [ph["desc"]]
    if fib and fib["note"]:
        summ.append(fib["note"])
    vz = vp.get("above")
    if vz and vz[0] <= c * 1.2:
        summ.append(f"상단 매물대 {won(vz[0])}~{won(vz[1])}원 (최근 6개월 거래대금의 {vz[2] * 100:.0f}%) 소화 필요")
    for t_ in dg["text"]:
        if any(w in t_ for w in ("돌파", "이탈", "근접", "테스트")):
            summ.insert(1, t_)
            break
    if bc and bc["ago"] <= 40:
        summ.append(f"기준봉 {bc['text']}")
    if any(k in vol for k in ("조정 중", "바닥권", "수급 유입")):
        summ.append(vol)

    # 시나리오
    up = dn = None
    if R:
        nxt = R[1] if len(R) > 1 else None
        up = f"{won(R[0][0])}원({R[0][1]}) 종가 돌파 시 " + (f"{won(nxt[0])}원({nxt[1]})까지 {pc(nxt[0] / c - 1, 0)} 여력" if nxt else "추가 상승 여력 확대")
    if S:
        nxt = S[1] if len(S) > 1 else None
        dn = f"{won(S[0][0])}원({S[0][1]}) 이탈 시 " + (f"{won(nxt[0])}원({nxt[1]})까지 {pc(nxt[0] / c - 1, 0)} 추가 하락 가능" if nxt else "하단 지지 약화")
    if not S:
        dn = "뚜렷한 하단 지지 부재 · 저점 경신 시 하락 가속 가능"
    s1 = f"{won(S[0][0])}원" if S else "하락 진정(거래 감소·아랫꼬리) 확인"
    r1 = f"{won(R[0][0])}원" if R else "저항선"
    plan = {
        "uptrend": f"추세 유효 구간 · {s1} 이탈 전까지 보유 관점",
        "price_corr": (f"{s1} 지지 확인 후 분할 접근 · 이탈 시 비중 축소" if S else f"{s1} 후 분할 접근"),
        "price_corr_rb": f"반등 지속 여부는 {r1} 회복이 관건",
        "time_corr": f"{r1} 거래량 동반 돌파 시 추세 재개 신호",
        "base": f"{r1} 돌파 시 추세 전환 신호 · 그 전까지 관망",
        "downtrend": f"추세 전환 확인 전 관망 · {r1} 회복 시 재검토",
        "box": f"{s1} 지지 · {r1} 저항 박스권 대응",
        "mixed": f"{s1} 지지 여부 확인 · {r1} 돌파 시 상단 확대",
    }[k]
    span = f" · 고점({_date(d, sw['ih'])}) 이후 {ph['days']}영업일" if k in ("price_corr", "price_corr_rb", "time_corr") else ""
    rows = [["추세", ma["text"]], ["국면", f"{ph['name']}{span}"],
            ["지지", " / ".join(f"{won(p)} {nm}" for p, nm in S[:2]) or "-"],
            ["저항", " / ".join(f"{won(p)} {nm}" for p, nm in R[:2]) or "-"]]
    if vp:
        z1, z2 = vp.get("above"), vp.get("below")
        rows.append(["매물대", " / ".join(([f"상단 {won(z1[0])}~{won(z1[1])}"] if z1 else []) + ([f"하단 {won(z2[0])}~{won(z2[1])}"] if z2 else [])) or f"현재가 부근 집중 ({won(vp['poc'])})"])
    if fib:
        base = "상승폭" if fib["kind"] == "retr" else "하락폭"
        rows.append(["피보나치", f"{base} 기준 " + " · ".join(f"{r * 100:g}% {won(p)}" for r, p in fib["levels"] if r in (0.382, 0.5, 0.618))
                     + f" (현재 {fib['now'] * 100:.0f}%)"])
    if dg["text"]:
        rows.append(["빗각", " / ".join(dg["text"])])
    if bc:
        rows.append(["기준봉", bc["text"]])
    rows += ex["rows"]
    if abs(ex["streak"]) >= 3:
        vol += f" · {abs(ex['streak'])}일 연속 {'상승' if ex['streak'] > 0 else '하락'}"
    rows += [["거래량", vol], ["캔들", cdl], ["보조지표", osc + (" · " + ex["ind"] if ex["ind"] else "")]]
    return {"head": head, "phase": ph, "summary": summ[:4], "up": up, "down": dn, "plan": plan, "rows": rows,
            "tl": dg["lines"], "base": bc, "ov": ex["ov"],
            "fib": fib, "vp": {"bins": vp.get("bins"), "above": vp.get("above"), "below": vp.get("below"), "poc": vp.get("poc")} if vp else None,
            "support": S, "resist": R}
