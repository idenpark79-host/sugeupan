"""포트폴리오 백테스트 — 추천 규칙을 과거에 그대로 적용해 실제로 사고팔았다면?

규칙 (실전 추천과 동일)
- 매일 장 마감 후 후보를 기대 수익률 순으로 정렬 → 다음 거래일 시가에 빈 자리만큼 매수
- 한 종목당 투자금 = 그날 평가금액 ÷ 최대 보유 종목 수 (현금이 부족하면 매수 생략)
- 목표가(진입가 × (1 + ATR × 목표 배수)) 도달 시 익절, 손절가 이탈 시 손절,
  20거래일 경과 시 종가 정리. 같은 날 둘 다 닿으면 손절로 처리(보수적). 왕복 비용 0.25% 차감
- 이미 보유 중인 종목은 다시 사지 않음
비교: 같은 기간 코스피, 전 종목 동일 비중
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import strategy


class Mat:
    """패널을 (날짜 × 종목) 행렬로."""

    def __init__(self, pn: strategy.Panel):
        self.dates = np.unique(pn.date)
        self.ti = np.searchsorted(self.dates, pn.date)
        T, S = len(self.dates), len(pn.meta)
        self.O, self.H, self.L, self.C = (np.full((T, S), np.nan) for _ in range(4))
        for M, src in ((self.O, pn.O), (self.H, pn.H), (self.L, pn.L), (self.C, pn.C)):
            M[self.ti, pn.sid] = src
        self.A = np.full((T, S), np.nan)
        self.A[self.ti, pn.sid] = pn.atrp
        # 거래정지·결측일은 직전 종가로 평가
        self.Cf = pd.DataFrame(self.C).ffill().values


def _run(m: Mat, cands: dict, t0: int, t1: int, P: int, H: int, cost: float):
    cash, pos, trades = 1.0, [], []
    eq = np.full(t1 - t0 + 1, np.nan)
    expo = np.zeros_like(eq)
    for t in range(t0, t1 + 1):
        equity_prev = cash + sum(p["sh"] * m.Cf[t - 1, p["sid"]] for p in pos) if t > t0 else cash
        held = {p["sid"] for p in pos}
        for sid, score, kt, ks in cands.get(t - 1, []):
            if len(pos) >= P:
                break
            if sid in held:
                continue
            px = m.O[t, sid]
            a = m.A[t - 1, sid]
            if not np.isfinite(px) or not np.isfinite(a) or px <= 0:
                continue
            alloc = min(cash, equity_prev / P)
            if alloc < equity_prev / P * 0.5:
                break
            cash -= alloc
            pos.append({"sid": sid, "entry": px, "sh": alloc / px, "tgt": px * (1 + kt * a),
                        "stp": px * (1 - ks * a), "day": 0, "t0": t, "score": score})
            held.add(sid)
        keep = []
        for p in pos:
            s = p["sid"]
            o, h, l, c = m.O[t, s], m.H[t, s], m.L[t, s], m.C[t, s]
            if not np.isfinite(c):
                keep.append(p)
                continue
            p["day"] += 1
            ex = None
            if l <= p["stp"]:
                ex = min(o, p["stp"]) if np.isfinite(o) else p["stp"]
            elif h >= p["tgt"]:
                ex = max(o, p["tgt"]) if np.isfinite(o) else p["tgt"]
            elif p["day"] >= H:
                ex = c
            if ex is None:
                keep.append(p)
                continue
            r = ex / p["entry"] - 1 - cost
            cash += p["sh"] * p["entry"] * (1 + r)
            trades.append((p["t0"], t, s, p["entry"], ex, r, p["day"], p["score"]))
        pos = keep
        inv = sum(p["sh"] * m.Cf[t, p["sid"]] for p in pos)
        eq[t - t0] = cash + inv
        expo[t - t0] = inv / eq[t - t0] if eq[t - t0] > 0 else 0
    return eq, expo, trades


def _metrics(eq: np.ndarray, dates: np.ndarray) -> dict:
    r = np.diff(eq) / eq[:-1]
    yrs = max(len(eq) / 252, 1e-9)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    sd = r.std()
    return {"total": float(eq[-1] / eq[0] - 1), "cagr": float((eq[-1] / eq[0]) ** (1 / yrs) - 1),
            "mdd": float(dd.min()), "vol": float(sd * np.sqrt(252)),
            "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0}


def _periodic(eq, dates, freq):
    s = pd.Series(eq, index=pd.to_datetime(dates))
    last = s.resample(freq).last().dropna()
    prev = last.shift(1)
    prev.iloc[0] = s.iloc[0]
    return (last / prev - 1)


def run(pn: strategy.Panel, mr, res, market_close: pd.Series, cfg: dict, log=print) -> dict:
    H = cfg.get("horizon_days", 20)
    cost = cfg.get("cost_pct", 0.25) / 100
    m = Mat(pn)
    T = len(m.dates)

    # ── 후보 생성 ──
    ai, sig = {}, {}
    ai_t0 = None
    if mr is not None and mr.pred is not None:
        ok = np.isfinite(mr.pred) & pn.warm & np.isfinite(pn.atrp)
        idx = np.flatnonzero(ok)
        exp = mr.iso_r.predict(mr.pred[idx])
        df = pd.DataFrame({"t": m.ti[idx], "sid": pn.sid[idx], "s": mr.pred[idx], "e": exp})
        df = df[df.e > 0].sort_values(["t", "s"], ascending=[True, False])
        for t, g in df.groupby("t"):
            ai[int(t)] = [(int(a), float(e), mr.kt, mr.ks) for a, e in zip(g.sid.values[:40], g.e.values[:40])]
        ai_t0 = int(m.ti[idx].min()) if len(idx) else None
    sig_t0 = None
    # 검증 신호 엔진: 학습 구간만 보고 고른 신호(bt_strategies)를 검증 구간에 적용 → 미래 정보 없음
    strats = getattr(res, "bt_strategies", None) or (res.strategies if res is not None else [])
    if strats:
        split = getattr(res, "split", None)
        rows = []
        for s in strats:
            mask = pn.cond[:, [pn.keys.index(k) for k in s.combo]].all(axis=1)
            ev = strategy._events(pn, mask) & np.isfinite(pn.atrp)
            if split is not None:
                ev &= pn.date >= split
            i = np.flatnonzero(ev)
            rows.append(pd.DataFrame({"t": m.ti[i], "sid": pn.sid[i], "e": s.avg_tr, "kt": s.kt, "ks": s.ks}))
        df = pd.concat(rows).sort_values(["t", "e"], ascending=[True, False]).drop_duplicates(["t", "sid"])
        for t, g in df.groupby("t"):
            sig[int(t)] = [(int(a), float(e), float(x), float(y)) for a, e, x, y in zip(g.sid, g.e, g.kt, g.ks)]
        if split is not None:
            sig_t0 = int(np.searchsorted(m.dates, split))
        elif len(df):
            sig_t0 = int(df.t.min())
    comb = {}
    for t in set(ai) | set(sig):
        lst = {}
        for c in ai.get(t, []) + sig.get(t, []):
            if c[0] not in lst or c[1] > lst[c[0]][1]:
                lst[c[0]] = c
        comb[t] = sorted(lst.values(), key=lambda c: -c[1])

    engines = []
    if ai:
        engines.append(("ai", "AI 모델", ai, ai_t0))
    if sig:
        engines.append(("sig", "검증 신호", sig, sig_t0))
    if ai and sig:
        engines.insert(0, ("mix", "통합 추천", comb, max(ai_t0, sig_t0)))
    if not engines:
        return {"ok": False}

    mk = market_close.reindex(pd.to_datetime(m.dates)).ffill().values
    ew_r = np.nanmean(m.Cf[1:] / m.Cf[:-1] - 1, axis=1)
    ew = np.r_[1.0, np.cumprod(1 + np.nan_to_num(ew_r))]

    out = {"ok": True, "horizon": H, "cost": cost * 100, "variants": {}}
    for key, label, cands, t0 in engines:
        t0 = max(t0 + 1, 1)
        t1 = T - 1
        if t1 - t0 < 60:
            continue
        d = m.dates[t0:t1 + 1]
        kospi = mk[t0:t1 + 1] / mk[t0]
        eqw = ew[t0:t1 + 1] / ew[t0]
        for P in (5, 10, 20):
            eq, expo, trades = _run(m, cands, t0, t1, P, H, cost)
            tr = pd.DataFrame(trades, columns=["t0", "t1", "sid", "entry", "exit", "ret", "days", "score"])
            met = _metrics(eq, d)
            if len(tr):
                wins, loss = tr.ret[tr.ret > 0], tr.ret[tr.ret <= 0]
                met.update(trades=int(len(tr)), win=float((tr.ret > 0).mean()), avg=float(tr.ret.mean()),
                           avgWin=float(wins.mean()) if len(wins) else 0.0, avgLoss=float(loss.mean()) if len(loss) else 0.0,
                           pf=float(wins.sum() / -loss.sum()) if loss.sum() < 0 else None,
                           hold=float(tr.days.mean()), best=float(tr.ret.max()), worst=float(tr.ret.min()))
            met["expo"] = float(np.nanmean(expo))
            km = _metrics(kospi, d)
            em = _metrics(eqw, d)
            yr = _periodic(eq, d, "YE")
            yk = _periodic(kospi, d, "YE")
            mo = _periodic(eq, d, "ME")
            peak = np.maximum.accumulate(eq)
            step = max(1, len(d) // 260)
            v = {"label": label, "P": P, "from": str(pd.Timestamp(d[0]).date()), "to": str(pd.Timestamp(d[-1]).date()),
                 "m": {k: (round(x, 5) if isinstance(x, float) else x) for k, x in met.items()},
                 "kospi": {k: round(x, 5) for k, x in km.items()}, "ew": {k: round(x, 5) for k, x in em.items()},
                 "curve": [[str(pd.Timestamp(d[i]).date()), round(float(eq[i]), 4), round(float(kospi[i]), 4),
                            round(float(eqw[i]), 4), round(float(eq[i] / peak[i] - 1), 4)]
                           for i in sorted(set(range(0, len(d), step)) | {len(d) - 1})],
                 "yearly": [[int(k.year), round(float(a), 4), round(float(yk.get(k, np.nan)), 4)] for k, a in yr.items()],
                 "monthly": [[k.strftime("%Y-%m"), round(float(a), 4)] for k, a in mo.items()]}
            if len(tr):
                h = np.clip(tr.ret.values * 100, -25, 40)
                cnt, edges = np.histogram(h, bins=np.arange(-25, 42.5, 2.5))
                v["hist"] = [[float(e), int(c)] for e, c in zip(edges[:-1], cnt)]
                last = tr.sort_values("t1", ascending=False).head(80)
                v["log"] = [[str(pd.Timestamp(m.dates[a]).date()), str(pd.Timestamp(m.dates[b]).date()),
                             pn.meta.iloc[s].Code, pn.meta.iloc[s].Name, round(float(e), 2), round(float(x), 2),
                             round(float(r) * 100, 2), int(dd)] for a, b, s, e, x, r, dd in
                            zip(last.t0, last.t1, last.sid, last.entry, last.exit, last.ret, last.days)]
            out["variants"][f"{key}{P}"] = v
            if "byStock" not in out and P == 10 and len(tr):
                # 종목별 과거 추천 매매 (종목 화면 차트 표시·성적용) — 대표 전략·10종목 기준
                bs = {}
                for a, b, s, e, x, r, dd in zip(tr.t0, tr.t1, tr.sid, tr.entry, tr.exit, tr.ret, tr.days):
                    bs.setdefault(pn.meta.iloc[s].Code, []).append(
                        [str(pd.Timestamp(m.dates[a]).date()), str(pd.Timestamp(m.dates[b]).date()),
                         round(float(e), 2), round(float(x), 2), round(float(r) * 100, 2), int(dd)])
                out["byStockKey"] = f"{key}{P}"
                out["byStock"] = {c: {"n": len(v2), "win": round(sum(t[4] > 0 for t in v2) / len(v2) * 100, 1),
                                      "avg": round(sum(t[4] for t in v2) / len(v2), 2),
                                      "t": sorted(v2)[-12:]} for c, v2 in bs.items()}
            log(f"    백테스트 {label} {P}종목: 총 {met['total']*100:+.1f}% · 연 {met['cagr']*100:+.1f}% · "
                f"MDD {met['mdd']*100:.1f}% (코스피 {km['total']*100:+.1f}%)")
    out["engines"] = [[k, l] for k, l, *_ in engines if any(x.startswith(k) for x in out["variants"])]
    out["sigList"] = [[s.name, int(s.n_tr), round(float(s.avg_tr) * 100, 2), s.kt, s.ks] for s in strats]
    split = getattr(res, "split", None) if res is not None else None
    out["split"] = str(pd.Timestamp(split).date()) if split is not None else None
    out["aiFrom"] = str(pd.Timestamp(m.dates[ai_t0]).date()) if ai_t0 is not None else None
    return out
