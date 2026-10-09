"""상승 확률 모델 — 전 종목·전 일자의 지표·수급 특징으로 '1개월 내 수익 마감' 확률을 학습

1) 라벨: 다음 날 시가 매수 → 목표가(ATR×kt) 익절 / 손절가(ATR×ks) 손절 / 20거래일 뒤 종가 청산, 비용 차감 후 수익이면 1
2) 검증: 앞 70% 기간 학습 → 20거래일 간격 → 뒤 30% 기간에서 평가 (실제로 미래를 모르는 상황 재현)
3) 보정: 검증 구간에서 '모델 점수 → 실제 승률' 관계(등위 회귀)를 맞춰, 화면에 보이는 확률 = 과거 실제 승률
4) 최종: 전체 기간으로 다시 학습해 오늘 전 종목을 채점, 확률 높은 순으로 추천
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import strategy


@dataclass
class ModelResult:
    picks: list
    base_win: float
    auc: float
    top_win: float                 # 검증 구간에서 매일 상위 N개를 골랐을 때 실제 승률
    top_avg: float
    top_hit: float
    top_n: int
    calib: list                    # [(예측 구간 평균, 실제 승률, 표본 수)]
    monthly: list                  # [(YYYY-MM, 상위 N 승률, 평균 수익, 건수)]
    n_train: int
    n_test: int
    test_period: tuple
    kt: float
    ks: float
    horizon: int
    lifts: dict = field(default_factory=dict)   # 조건별 승률 향상폭 (근거 표시용)
    recent: list = field(default_factory=list)  # 검증 구간 최근 일별 상위 추천


def _auc(y, p):
    order = np.argsort(p)
    y = y[order]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=7)


def run(pn: strategy.Panel, cfg: dict, top_n: int = 20, log=print) -> ModelResult:
    from sklearn.isotonic import IsotonicRegression

    H = cfg.get("horizon_days", 20)
    cost = cfg.get("cost_pct", 0.25) / 100
    kt, ks = cfg.get("model_target_atr", 2.0), cfg.get("model_stop_atr", 1.5)
    ret, hit, valid = strategy.simulate(pn, kt, ks, H, cost)
    X = pn.feat
    lab = valid & pn.warm & np.isfinite(ret)
    y = (ret > 0).astype(np.int8)

    dates = np.unique(pn.date[lab])
    cut = dates[int(len(dates) * 0.7)]
    gap_end = dates[min(int(len(dates) * 0.7) + H, len(dates) - 1)]
    tr = lab & (pn.date < cut)
    te = lab & (pn.date > gap_end)
    rng = np.random.default_rng(0)

    def sample(mask, k):
        idx = np.flatnonzero(mask)
        return idx if len(idx) <= k else rng.choice(idx, k, replace=False)

    itr = sample(tr, 250_000)
    log(f"    모델 학습 {len(itr):,}건 / 검증 {te.sum():,}건")
    m1 = _clf().fit(X[itr], y[itr])
    p_te = m1.predict_proba(X[te])[:, 1]
    y_te, r_te, h_te = y[te], ret[te], hit[te]
    auc = _auc(y_te, p_te)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(p_te, y_te)

    # 보정표 (10분위)
    q = np.quantile(p_te, np.linspace(0, 1, 11))
    calib = []
    for a, b in zip(q[:-1], q[1:]):
        m = (p_te >= a) & (p_te <= b)
        if m.sum():
            calib.append((float(iso.predict([p_te[m].mean()])[0]), float(y_te[m].mean()), int(m.sum())))

    # 매일 상위 N 종목 시뮬레이션 (검증 구간)
    te_idx = np.flatnonzero(te)
    df = pd.DataFrame({"d": pn.date[te_idx], "p": p_te, "y": y_te, "r": r_te, "h": h_te, "i": te_idx})
    top = df.sort_values("p", ascending=False).groupby("d", sort=False).head(top_n)
    top = top.assign(m=pd.to_datetime(top.d).dt.strftime("%Y-%m"))
    monthly = [(k, float(g.y.mean()), float(g.r.mean()), int(len(g))) for k, g in top.groupby("m")]

    # 조건별 승률 향상폭 (학습 구간)
    base = float(y[tr].mean())
    lifts = {}
    for i, k in enumerate(pn.keys):
        m = tr & pn.cond[:, i]
        if m.sum() > 300:
            lifts[k] = float(y[m].mean() - base)

    # 최종 모델: 라벨이 있는 전 기간으로 학습 → 오늘 채점
    iall = sample(lab, 300_000)
    m2 = _clf().fit(X[iall], y[iall])
    last_date = pn.date.max()
    ends = np.flatnonzero(np.r_[pn.sid[1:] != pn.sid[:-1], True])
    today = np.array([i for i in ends if pn.date[i] == last_date and pn.warm[i] and np.isfinite(pn.atrp[i])])
    raw = m2.predict_proba(X[today])[:, 1]
    prob = iso.predict(raw)

    # 같은 확률대의 과거 평균 수익·목표 도달률 (검증 구간 기준)
    def bucket_stats(pv):
        cal = iso.predict(p_te)
        m = np.abs(cal - pv) <= 0.03
        if m.sum() < 50:
            m = np.argsort(np.abs(cal - pv))[:500]
        return float(r_te[m].mean()), float(h_te[m].mean())

    order = np.argsort(-raw)[:top_n]          # 순위는 원점수, 표시는 보정 확률
    picks = []
    for j in order:
        i = today[j]
        row = pn.meta.iloc[pn.sid[i]]
        d = pn.frames[row.Code]
        c, pc = float(d.Close.iloc[-1]), float(d.Close.iloc[-2])
        atrp = float(pn.atrp[i])
        avg, hr = bucket_stats(prob[j])
        active = [k for k in pn.keys if pn.cond[i, pn.keys.index(k)] and lifts.get(k, 0) > 0.005]
        active.sort(key=lambda k: -lifts[k])
        picks.append({"code": row.Code, "name": row.Name, "close": c, "chg": c - pc, "pct": (c / pc - 1) * 100,
                      "prob": float(prob[j]), "exp": avg, "hitp": hr,
                      "target": c * (1 + kt * atrp), "stop": c * (1 - ks * atrp), "atrp": atrp,
                      "why": [(strategy.COND_NAME[k], strategy.COND_DESC[k]) for k in active[:4]], "df": d})

    # 검증 구간 마지막 30거래일의 일별 상위 10 (데모용 누적 성적 예시)
    last30 = np.sort(top.d.unique())[-30:]
    rec = top[top.d.isin(last30)].copy()
    rec["rk"] = rec.groupby("d").p.rank(ascending=False, method="first")
    rec = rec[rec.rk <= 10]
    recent = [{"date": pd.Timestamp(r.d).strftime("%Y-%m-%d"), "code": pn.meta.iloc[pn.sid[r.i]].Code,
               "name": pn.meta.iloc[pn.sid[r.i]].Name, "rank": int(r.rk), "prob": float(iso.predict([r.p])[0]),
               "ref": float(pn.C[r.i]), "target": float(pn.C[r.i] * (1 + kt * pn.atrp[r.i])),
               "stop": float(pn.C[r.i] * (1 - ks * pn.atrp[r.i]))} for r in rec.itertuples()]

    tp = pd.Timestamp(df.d.min()), pd.Timestamp(df.d.max())
    return ModelResult(picks=picks, base_win=float(y_te.mean()), auc=auc, top_win=float(top.y.mean()),
                       top_avg=float(top.r.mean()), top_hit=float(top.h.mean()), top_n=top_n, calib=calib,
                       monthly=monthly, n_train=len(itr), n_test=int(te.sum()), test_period=tp,
                       kt=kt, ks=ks, horizon=H, lifts=lifts, recent=recent)
