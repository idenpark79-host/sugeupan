"""포트폴리오 백테스트 — 추천 규칙을 과거에 그대로 적용해 실제로 사고팔았다면? (1주·1개월 각각)

규칙 (실전 추천과 동일)
- 매일 장 마감 후 모델 점수 상위 종목(기대 수익 플러스)을 순서대로 → 다음 거래일 시가에 빈 자리만큼 매수
- 한 종목당 투자금 = 그날 평가금액 ÷ 최대 보유 종목 수 (현금이 부족하면 매수 생략)
- 목표가(진입가 × (1 + 변동폭 × 목표 배수)) 도달 시 익절, 손절가 이탈 시 손절,
  기간(5 또는 20거래일) 경과 시 종가 정리. 같은 날 둘 다 닿으면 손절로 처리(보수적). 왕복 비용 0.25% 차감
- 점수는 워크포워드(각 시점 이전 데이터로만 학습) 결과만 사용 → 미래 정보 없음
비교: 같은 기간 코스피, 전 종목 동일 비중
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import strategy

HZ = {5: "1주", 20: "1개월"}


class Mat:
    """패널을 (날짜 × 종목) 행렬로."""

    def __init__(self, pn: strategy.Panel, atr_cap: float):
        self.dates = np.unique(pn.date)
        self.ti = np.searchsorted(self.dates, pn.date)
        T, S = len(self.dates), len(pn.meta)
        self.O, self.H, self.L, self.C = (np.full((T, S), np.nan) for _ in range(4))
        for M, src in ((self.O, pn.O), (self.H, pn.H), (self.L, pn.L), (self.C, pn.C)):
            M[self.ti, pn.sid] = src
        self.O[self.O <= 0] = np.nan
        self.A = np.full((T, S), np.nan)
        self.A[self.ti, pn.sid] = np.minimum(pn.atrp, atr_cap)
        self.Cf = pd.DataFrame(self.C).ffill().values         # 거래정지·결측일은 직전 종가로 평가


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


def _metrics(eq: np.ndarray) -> dict:
    r = np.diff(eq) / eq[:-1]
    yrs = max(len(eq) / 252, 1e-9)
    dd = eq / np.maximum.accumulate(eq) - 1
    sd = r.std()
    return {"total": float(eq[-1] / eq[0] - 1), "cagr": float((eq[-1] / eq[0]) ** (1 / yrs) - 1),
            "mdd": float(dd.min()), "vol": float(sd * np.sqrt(252)),
            "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0}


def _periodic(eq, dates, freq):
    s = pd.Series(eq, index=pd.to_datetime(dates))
    last = s.resample(freq).last().dropna()
    prev = last.shift(1)
    prev.iloc[0] = s.iloc[0]
    return last / prev - 1


def run(pn: strategy.Panel, results: dict, market_close: pd.Series, cfg: dict, log=print) -> dict:
    cost = cfg.get("cost_pct", 0.25) / 100
    cap = cfg.get("atr_cap", 0.05)
    m = Mat(pn, cap)
    T = len(m.dates)
    mk = market_close.reindex(pd.to_datetime(m.dates)).ffill().bfill().values
    ew_r = np.nanmean(m.Cf[1:] / m.Cf[:-1] - 1, axis=1)
    ew = np.r_[1.0, np.cumprod(1 + np.nan_to_num(ew_r))]
    out = {"ok": True, "cost": cost * 100, "variants": {}, "horizons": []}
    for H in sorted(results, reverse=True):
        r = results[H]
        ok = np.isfinite(r.pct) & pn.warm & np.isfinite(pn.atrp)
        idx = np.flatnonzero(ok)
        e = r.exp_of(r.pct[idx], np.minimum(pn.atrp[idx], cap))
        df = pd.DataFrame({"t": m.ti[idx], "sid": pn.sid[idx], "s": r.pct[idx], "e": e})
        df = df[df.e > 0].sort_values(["t", "s"], ascending=[True, False])
        if getattr(r, "gate", False) and r.regime_days is not None:      # 시장 국면 나쁜 날은 신규 매수 안 함
            df = df[r.regime_days[df.t.values]]
        cands = {int(t): [(int(a), float(s), r.kt, r.ks) for a, s in zip(g.sid.values[:40], g.s.values[:40])]
                 for t, g in df.groupby("t")}
        if not cands:
            continue
        t0 = max(min(cands) + 1, 1)
        t1 = T - 1
        if t1 - t0 < 60:
            continue
        d = m.dates[t0:t1 + 1]
        kospi = mk[t0:t1 + 1] / mk[t0]
        eqw = ew[t0:t1 + 1] / ew[t0]
        out["horizons"].append([H, HZ.get(H, f"{H}일")])
        runs = [(P, None, r.kt, r.ks) for P in (5, 10, 20)]
        for nm, st in (getattr(r, "styles", None) or {}).items():
            if nm != "base" and (st[0], st[1]) != (r.kt, r.ks):
                runs.append((10, nm, st[0], st[1]))          # 다른 매도 전략은 10종목만
        for P, style, kt, ks in runs:
            cd = cands if style is None else {t: [(a, s, kt, ks) for a, s, _, _ in v] for t, v in cands.items()}
            eq, expo, trades = _run(m, cd, t0, t1, P, H, cost)
            tr = pd.DataFrame(trades, columns=["t0", "t1", "sid", "entry", "exit", "ret", "days", "score"])
            met = _metrics(eq)
            if len(tr):
                wins, loss = tr.ret[tr.ret > 0], tr.ret[tr.ret <= 0]
                met.update(trades=int(len(tr)), win=float((tr.ret > 0).mean()), avg=float(tr.ret.mean()),
                           avgWin=float(wins.mean()) if len(wins) else 0.0, avgLoss=float(loss.mean()) if len(loss) else 0.0,
                           pf=float(wins.sum() / -loss.sum()) if loss.sum() < 0 else None,
                           hold=float(tr.days.mean()), best=float(tr.ret.max()), worst=float(tr.ret.min()))
            met["expo"] = float(np.nanmean(expo))
            km, em = _metrics(kospi), _metrics(eqw)
            yr, yk, mo = _periodic(eq, d, "YE"), _periodic(kospi, d, "YE"), _periodic(eq, d, "ME")
            peak = np.maximum.accumulate(eq)
            step = max(1, len(d) // 260)
            rnd = lambda dct: {k: (round(x, 5) if isinstance(x, float) else x) for k, x in dct.items()}
            v = {"label": HZ.get(H, f"{H}일"), "H": H, "P": P, "kt": kt, "ks": ks, "style": style or "base", "gate": bool(getattr(r, "gate", False)),
                 "from": str(pd.Timestamp(d[0]).date()), "to": str(pd.Timestamp(d[-1]).date()),
                 "m": rnd(met), "kospi": rnd(km), "ew": rnd(em),
                 "curve": [[str(pd.Timestamp(d[i]).date()), round(float(eq[i]), 4), round(float(kospi[i]), 4),
                            round(float(eqw[i]), 4), round(float(eq[i] / peak[i] - 1), 4)]
                           for i in sorted(set(range(0, len(d), step)) | {len(d) - 1})],
                 "yearly": [[int(k.year), round(float(a), 4), round(float(yk.get(k, np.nan)), 4)] for k, a in yr.items()],
                 "monthly": [[k.strftime("%Y-%m"), round(float(a), 4)] for k, a in mo.items()]}
            if len(tr):
                hh = np.clip(tr.ret.values * 100, -25, 40)
                step_h = 1.0 if H <= 5 else 2.5
                cnt, edges = np.histogram(hh, bins=np.arange(-25, 40 + step_h, step_h))
                v["hist"] = [[float(e), int(c)] for e, c in zip(edges[:-1], cnt)]
                last = tr.sort_values("t1", ascending=False).head(80)
                v["log"] = [[str(pd.Timestamp(m.dates[a]).date()), str(pd.Timestamp(m.dates[b]).date()),
                             pn.meta.iloc[s].Code, pn.meta.iloc[s].Name, round(float(e), 2), round(float(x), 2),
                             round(float(rr) * 100, 2), int(dd)] for a, b, s, e, x, rr, dd in
                            zip(last.t0, last.t1, last.sid, last.entry, last.exit, last.ret, last.days)]
                if P == 10 and style is None:                  # 종목별 과거 추천 매매 (종목 화면용)
                    bs = {}
                    for a, b, s, e, x, rr, dd in zip(tr.t0, tr.t1, tr.sid, tr.entry, tr.exit, tr.ret, tr.days):
                        bs.setdefault(pn.meta.iloc[s].Code, []).append(
                            [str(pd.Timestamp(m.dates[a]).date()), str(pd.Timestamp(m.dates[b]).date()),
                             round(float(e), 2), round(float(x), 2), round(float(rr) * 100, 2), int(dd)])
                    out.setdefault("byStock", {})[str(H)] = {
                        c: {"n": len(v2), "win": round(sum(t[4] > 0 for t in v2) / len(v2) * 100, 1),
                            "avg": round(sum(t[4] for t in v2) / len(v2), 2), "t": sorted(v2)[-12:]}
                        for c, v2 in bs.items()}
            out["variants"][f"{H}_{P}" + (f"_{style}" if style else "")] = v
            log(f"    백테스트 {HZ.get(H)} {P}종목{' ' + style if style else ''}: 총 {met['total'] * 100:+.1f}% · 연 {met['cagr'] * 100:+.1f}% · "
                f"MDD {met['mdd'] * 100:.1f}% · 승률 {met.get('win', 0) * 100:.1f}% (코스피 {km['total'] * 100:+.1f}%)")
    if not out["variants"]:
        return {"ok": False}
    return out
