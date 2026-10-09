"""기대 수익률 모델 — 10년치 전 종목 데이터로 '지금 사서 1개월 운용했을 때의 수익률'을 예측

설계
1) 라벨: 다음 날 시가 매수 → 목표가 익절 / 손절가 손절 / 20거래일 뒤 종가 청산, 비용 0.25% 차감한 실제 거래 수익률
2) 학습 목표: 같은 날 전 종목 평균을 뺀 '상대 수익률' — 시장 전체 등락(예측이 거의 불가능)을 걷어내고
   종목 간 차이를 배우게 함. 극단값은 1~99% 구간으로 자름
3) 특징: 지표·수급 60여 개 + 주요 특징의 '당일 전 종목 대비 순위'(상대 위치)
4) 검증: 워크포워드 — 최근 3년을 6개월 단위로 나눠, 각 구간 직전까지의 데이터로만 학습한 모델로 예측
   (실제 운영과 같은 조건). 이 예측으로 '매일 상위 N 종목을 샀다면'의 성적을 계산
5) 보정: 워크포워드 예측 점수 → 실제 거래 수익률/승률 관계(등위 회귀)로 화면의 기대 수익·확률을 산출
6) 매도가: 목표·손절 배수 후보 중 워크포워드 상위 N 종목의 평균 수익이 가장 높은 조합 선택
7) 운영: 매일 실행 때 전체 기간으로 다시 학습(자동 재학습) 후 오늘 전 종목 채점
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import strategy

RANK_FEATS = ["ret5", "ret20", "ret60", "ret120", "rsi", "vol_r", "d_ma20", "d_ma60", "hi52", "atrp",
              "bb_pos", "obv_z", "frg5", "inst5", "pen5", "frg20", "inst20"]


@dataclass
class ModelResult:
    picks: list
    base_avg: float                # 검증 기간 전 종목 평균 거래 수익
    base_win: float
    top_avg: float                 # 매일 상위 N 종목 평균 거래 수익 (워크포워드)
    top_win: float
    top_hit: float
    top_n: int
    ic: float                      # 예측과 실제의 일별 순위 상관 평균
    deciles: list                  # [(예상 수익, 실제 수익, 실제 승률, 건수)]
    monthly: list                  # [(YYYY-MM, 상위 N 평균 수익, 승률, 건수)]
    n_train: int
    n_oos: int
    oos_period: tuple
    folds: int
    years: float
    kt: float
    ks: float
    horizon: int
    exit_table: list               # [(kt, ks, 상위N 평균 수익, 승률)]
    lifts: dict = field(default_factory=dict)
    recent: list = field(default_factory=list)


def _reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=400, l2_regularization=1.0, random_state=7)


def _cs_ranks(pn: strategy.Panel) -> np.ndarray:
    """주요 특징의 일자별 전 종목 대비 백분위."""
    names = pn.feat_names
    cols = [names.index(f) for f in RANK_FEATS if f in names]
    df = pd.DataFrame(pn.feat[:, cols], columns=[names[c] for c in cols])
    df["d"] = pn.date
    r = df.groupby("d").rank(pct=True)
    return r.values.astype(np.float32), ["rk_" + c for c in r.columns]


def _daily_top(dates, score, mask, n):
    idx = np.flatnonzero(mask)
    df = pd.DataFrame({"d": dates[idx], "s": score[idx], "i": idx})
    return df.sort_values("s", ascending=False).groupby("d", sort=False).head(n)


def run(pn: strategy.Panel, cfg: dict, top_n: int = 20, log=print) -> ModelResult:
    from sklearn.isotonic import IsotonicRegression

    H = cfg.get("horizon_days", 20)
    cost = cfg.get("cost_pct", 0.25) / 100
    kt0, ks0 = cfg.get("model_target_atr", 2.0), cfg.get("model_stop_atr", 1.5)

    rk, rk_names = _cs_ranks(pn)
    X = np.hstack([pn.feat, rk])
    ret, hit, valid = strategy.simulate(pn, kt0, ks0, H, cost)
    lab = valid & pn.warm & np.isfinite(ret)
    lo, hi = np.nanpercentile(ret[lab], [1, 99])
    y = np.clip(ret, lo, hi)
    dmean = pd.Series(y[lab]).groupby(pn.date[lab]).transform("mean").values
    target = np.full(len(y), np.nan)
    target[lab] = y[lab] - dmean                          # 상대 수익률

    dates = np.unique(pn.date[lab])
    years = len(dates) / 252
    rng = np.random.default_rng(0)

    def sample(mask, k):
        idx = np.flatnonzero(mask)
        return idx if len(idx) <= k else rng.choice(idx, k, replace=False)

    # ── 워크포워드 검증 ──
    oos_days = min(int(252 * 3), int(len(dates) * 0.45))
    fold_len = 126
    starts = list(range(len(dates) - oos_days, len(dates), fold_len))
    pred = np.full(len(y), np.nan)
    n_train = 0
    for k, s0 in enumerate(starts):
        s1 = min(s0 + fold_len, len(dates))
        d_from, d_to = dates[s0], dates[s1 - 1]
        tr_end = dates[max(s0 - H - 1, 0)]                # 라벨이 미래를 보지 않도록 20거래일 간격
        tr = lab & (pn.date <= tr_end)
        te = pn.warm & (pn.date >= d_from) & (pn.date <= d_to)
        it = sample(tr, 350_000)
        n_train = max(n_train, len(it))
        m = _reg().fit(X[it], target[it])
        pred[te] = m.predict(X[te])
        log(f"    워크포워드 {k + 1}/{len(starts)}: {pd.Timestamp(d_from):%Y.%m}~{pd.Timestamp(d_to):%Y.%m} 학습 {len(it):,}건")
    oos = lab & np.isfinite(pred)

    # 일별 순위 상관(IC)
    o = pd.DataFrame({"d": pn.date[oos], "p": pred[oos], "r": ret[oos]})
    ic = float(o.groupby("d").apply(lambda g: g.p.rank().corr(g.r.rank()) if len(g) > 10 else np.nan,
                                    include_groups=False).mean())

    # ── 매도 기준(목표·손절 배수) 선택: 워크포워드 상위 N의 평균 수익 최대 ──
    grid_t = cfg.get("exit_grid", {}).get("target_atr", [1.5, 2.0, 2.5, 3.0, 4.0])
    grid_s = cfg.get("exit_grid", {}).get("stop_atr", [1.0, 1.5, 2.0, 2.5])
    top = _daily_top(pn.date, pred, oos, top_n)
    ti = top.i.values
    exit_table, best = [], None
    for a in grid_t:
        for b in grid_s:
            r2, h2, v2 = strategy.simulate(pn, a, b, H, cost)
            ok = np.isfinite(r2[ti])
            avg, win = float(np.mean(r2[ti][ok])), float(np.mean(r2[ti][ok] > 0))
            exit_table.append((a, b, avg, win))
            if best is None or avg > best[2]:
                best = (a, b, avg, r2, h2, v2)
    kt, ks, _, ret, hit, valid = best
    log(f"    매도 기준 선택: 목표 ATR×{kt}, 손절 ATR×{ks}")
    lab2 = valid & pn.warm & np.isfinite(ret)
    oos = lab2 & np.isfinite(pred)
    top = _daily_top(pn.date, pred, oos, top_n)
    ti = top.i.values

    # 보정: 예측 점수 → 실제 거래 수익 / 승률
    iso_r = IsotonicRegression(out_of_bounds="clip").fit(pred[oos], np.clip(ret[oos], lo * 1.5, hi * 1.5))
    iso_w = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(pred[oos], (ret[oos] > 0).astype(float))
    q = np.quantile(pred[oos], np.linspace(0, 1, 11))
    deciles = []
    for a, b in zip(q[:-1], q[1:]):
        m = oos & (pred >= a) & (pred <= b)
        if m.sum():
            deciles.append((float(iso_r.predict([pred[m].mean()])[0]), float(ret[m].mean()),
                            float((ret[m] > 0).mean()), int(m.sum())))
    tdf = pd.DataFrame({"m": pd.to_datetime(pn.date[ti]).strftime("%Y-%m"), "r": ret[ti], "h": hit[ti]})
    monthly = [(k, float(g.r.mean()), float((g.r > 0).mean()), int(len(g))) for k, g in tdf.groupby("m")]

    base = lab2 & (pn.date >= dates[len(dates) - oos_days])
    lifts = {}
    for i, k in enumerate(pn.keys):
        m = lab2 & pn.cond[:, i]
        if m.sum() > 500:
            lifts[k] = float(ret[m].mean() - ret[lab2].mean())

    # 최근 30거래일 일별 상위 10 (미리보기용 누적 성적 예시)
    last30 = np.sort(top.d.unique())[-30:]
    rec_df = top[top.d.isin(last30)].copy()
    rec_df["rk"] = rec_df.groupby("d").s.rank(ascending=False, method="first")
    rec_df = rec_df[rec_df.rk <= 10]
    recent = [{"date": pd.Timestamp(r.d).strftime("%Y-%m-%d"), "code": pn.meta.iloc[pn.sid[r.i]].Code,
               "name": pn.meta.iloc[pn.sid[r.i]].Name, "rank": int(r.rk),
               "prob": float(iso_w.predict([r.s])[0]), "exp": float(iso_r.predict([r.s])[0]),
               "ref": float(pn.C[r.i]), "target": float(pn.C[r.i] * (1 + kt * pn.atrp[r.i])),
               "stop": float(pn.C[r.i] * (1 - ks * pn.atrp[r.i])), "src": "AI"} for r in rec_df.itertuples()]

    # ── 최종 모델: 전 기간 자동 재학습 → 오늘 채점 ──
    lab_t = lab
    iall = sample(lab_t, 400_000)
    m_final = _reg().fit(X[iall], target[iall])
    last_date = pn.date.max()
    ends = np.flatnonzero(np.r_[pn.sid[1:] != pn.sid[:-1], True])
    today = np.array([i for i in ends if pn.date[i] == last_date and pn.warm[i] and np.isfinite(pn.atrp[i])])
    raw = m_final.predict(X[today])
    order = np.argsort(-raw)[:top_n]
    picks = []
    for j in order:
        i = today[j]
        row = pn.meta.iloc[pn.sid[i]]
        d = pn.frames[row.Code]
        c, pc = float(d.Close.iloc[-1]), float(d.Close.iloc[-2])
        atrp = float(pn.atrp[i])
        active = [k for k in pn.keys if pn.cond[i, pn.keys.index(k)] and lifts.get(k, 0) > 0.001]
        active.sort(key=lambda k: -lifts[k])
        picks.append({"code": row.Code, "name": row.Name, "close": c, "chg": c - pc, "pct": (c / pc - 1) * 100,
                      "exp": float(iso_r.predict([raw[j]])[0]), "prob": float(iso_w.predict([raw[j]])[0]),
                      "target": c * (1 + kt * atrp), "stop": c * (1 - ks * atrp), "atrp": atrp, "src": "AI",
                      "why": [(strategy.COND_NAME[k], strategy.COND_DESC[k]) for k in active[:4]], "df": d})

    od = pn.date[oos]
    return ModelResult(
        picks=picks, base_avg=float(ret[base].mean()), base_win=float((ret[base] > 0).mean()),
        top_avg=float(ret[ti].mean()), top_win=float((ret[ti] > 0).mean()), top_hit=float(hit[ti].mean()),
        top_n=top_n, ic=ic, deciles=deciles, monthly=monthly, n_train=len(iall), n_oos=int(oos.sum()),
        oos_period=(pd.Timestamp(od.min()), pd.Timestamp(od.max())), folds=len(starts), years=years,
        kt=kt, ks=ks, horizon=H, exit_table=exit_table, lifts=lifts, recent=recent)
