"""기대 수익률 모델 v3 — 1주(5거래일)·1개월(20거래일) 두 기간을 따로 학습

설계 (실제 10년 데이터로 검증한 결과를 반영)
1) 라벨: 다음 날 시가에 사서 H거래일 뒤 종가에 판 수익률(비용 차감)에서 같은 날 전 종목 평균을 뺀 '상대 수익률'
   — 시장 전체 등락을 걷어내고 종목 간 차이를 학습 (실험상 매매 규칙 수익률·승패를 직접 학습하는 것보다 우수)
2) 특징: 지표·캔들·수급 60여 개 + 전 종목 대비 순위 + 시장 국면(지수 추세·변동성·시장 폭) + 변동성·베타·최대 일간 상승 등
3) 검증: 최근 3년을 6개월씩 나눈 워크포워드 — 각 구간 직전까지의 데이터로만 학습한 모델로 채점 (실제 운영과 동일 조건)
4) 매도 규칙: 목표가·손절가(일평균 변동폭 배수) 후보 중, 워크포워드 상위 종목의 '수익 마감 확률'이 기준(기본 60%) 이상인
   규칙 가운데 평균 수익이 가장 높은 것을 고름 — 목표가를 가깝게 둘수록 확률은 오르고 1회 수익은 줄어드는 관계를 데이터로 결정
5) 보정: 매일의 점수 순위(백분위) → 실제 거래 수익·승률을 구간별로 집계해 부드럽게 연결 (같은 숫자가 반복되지 않음)
6) 등급: 오늘 점수 상위 2%=S, 5%=A, 10%=B — 등급별 과거 승률·평균 수익을 함께 표시
7) 근거: 모델 점수를 사람이 읽을 수 있는 요인(낙폭·추세·수급 등)으로 근사해 종목마다 '왜'를 설명
8) 운영: 매일 전체 기간으로 다시 학습해 오늘 전 종목 채점
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import strategy

RANK_FEATS = ["ret5", "ret20", "ret60", "ret120", "rsi", "vol_r", "d_ma20", "d_ma60", "hi52", "atrp",
              "bb_pos", "obv_z", "frg5", "inst5", "pen5", "frg20", "inst20"]
HZ = {5: "1주", 20: "1개월"}


@dataclass
class HorizonResult:
    H: int
    picks: list
    kt: float
    ks: float
    min_win: float
    atr_cap: float
    exit_table: list               # [(kt, ks, 상위 평균, 상위 승률, 전체 평균, 전체 승률)]
    calib: list                    # [(백분위 하한, 상한, 평균 수익, 승률, 건수)]
    grades: dict                   # {"S": [평균, 승률, 건수], ...}
    deciles: list                  # [(평균 수익, 승률, 건수)] 점수 10분위
    monthly: list                  # [(YYYY-MM, 상위 N 평균, 승률, 건수)]
    base: tuple                    # (평균, 승률) 검증 기간 전 종목
    top: tuple                     # (평균, 승률, 목표 도달률) 상위 N
    top_n: int
    ic: float
    oos_period: tuple
    folds: int
    years: float
    n_train: int
    n_oos: int
    pct: np.ndarray = field(default=None, repr=False)      # 워크포워드 점수의 일별 백분위 (백테스트용)
    exp_of: object = field(default=None, repr=False)        # (백분위, 변동폭) → 기대 수익
    win_of: object = field(default=None, repr=False)        # (백분위, 변동폭) → 수익 마감 확률
    today: dict = field(default_factory=dict, repr=False)   # 코드 → (백분위, 기대수익, 승률)
    gate: bool = False                                      # 시장 국면 필터 사용 여부 (코스피 20일선 상승 중에만 매수)
    regime_ok: bool = True                                  # 오늘 시장 국면
    regime_days: np.ndarray = field(default=None, repr=False)   # 날짜별 국면 (np.unique(date) 순서)
    off: tuple = None                                       # 국면 나쁠 때 상위 종목 성적 (평균, 승률)
    styles: dict = field(default_factory=dict)              # 매도 전략별 {이름: (목표배수, 손절배수, 상위 평균, 승률, 목표 도달률)}


# ── 특징 ─────────────────────────────────────────────────────
def _ranks(pn, cols_idx, names):
    df = pd.DataFrame(pn.feat[:, cols_idx], columns=[names[c] for c in cols_idx])
    return df.groupby(pn.date).rank(pct=True).values.astype(np.float32)


def build_X(pn: strategy.Panel, market_close: pd.Series):
    names = list(pn.feat_names)
    cols = [names.index(f) for f in RANK_FEATS if f in names]
    rk = _ranks(pn, cols, names)
    rk_names = ["rk_" + names[c] for c in cols]
    # 시장 국면
    dates = np.unique(pn.date)
    k = market_close.reindex(pd.DatetimeIndex(dates)).ffill()
    mk = pd.DataFrame(index=k.index)
    for n in (5, 20, 60):
        mk[f"m_ret{n}"] = k.pct_change(n)
    for n in (20, 60, 120):
        mk[f"m_d{n}"] = k / k.rolling(n).mean() - 1
    mk["m_vol20"] = k.pct_change().rolling(20).std()
    mk["m_dd"] = k / k.rolling(250, min_periods=60).max() - 1
    d20 = pn.feat[:, names.index("d_ma20")]
    r1 = pn.feat[:, names.index("ret1")]
    b = pd.DataFrame({"d": pn.date, "a": (d20 > 0).astype(float), "u": (r1 > 0).astype(float)}).groupby("d").mean()
    mk["m_br20"] = b.a.reindex(mk.index).values
    mk["m_adv"] = b.u.reindex(mk.index).rolling(5).mean().values
    mk["m_br20_chg"] = mk.m_br20.diff(5)
    M = mk.reindex(pd.DatetimeIndex(pn.date)).values.astype(np.float32)
    # 종목 변동성·꼬리·장기 모멘텀·베타
    C, sid = pd.Series(pn.C), pd.Series(pn.sid)
    g = C.groupby(sid)
    r = g.pct_change()
    rg = r.groupby(sid)
    roll = lambda s, n, fn: getattr(s.groupby(sid).rolling(n), fn)().reset_index(level=0, drop=True)
    mret = pd.Series(k.pct_change().reindex(pd.DatetimeIndex(pn.date)).values)
    E = pd.DataFrame({
        "std20": roll(r, 20, "std"), "std60": roll(r, 60, "std"),
        "max20": roll(r, 20, "max"), "min20": roll(r, 20, "min"),
        "mom12_1": g.shift(21) / g.shift(250) - 1, "ret10": g.pct_change(10),
        "up_days20": roll((r > 0).astype(float), 20, "mean"),
        "beta60": roll(r * mret, 60, "mean") / roll(mret ** 2, 60, "mean"),
    }).sort_index()
    del rg
    Ex = E.values.astype(np.float32)
    Er = pd.DataFrame(Ex).groupby(pn.date).rank(pct=True).values.astype(np.float32)
    X = np.hstack([pn.feat, rk, M, Ex, Er])
    X[~np.isfinite(X)] = np.nan
    fnames = names + rk_names + list(mk.columns) + list(E.columns) + ["rk_" + c for c in E.columns]
    return X, fnames


# ── 학습·평가 도구 ─────────────────────────────────────────────
def _reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=600, l2_regularization=1.0, random_state=7)


def fwd_ret(pn, H, cost):
    N = len(pn.C)
    idx = np.arange(N)
    ok = idx + H < N
    ok[ok] &= pn.sid[(idx + H)[ok]] == pn.sid[ok]
    r = np.full(N, np.nan)
    e = pn.O[(idx + 1)[ok]]
    with np.errstate(divide="ignore", invalid="ignore"):
        r[ok] = np.where(e > 0, pn.C[(idx + H)[ok]] / e - 1 - cost, np.nan)
    r[~np.isfinite(r) | (np.abs(r) > 3)] = np.nan
    return r


def daily_pct(date, score, mask):
    out = np.full(len(score), np.nan)
    m = mask & np.isfinite(score)
    out[m] = pd.Series(score[m]).groupby(date[m]).rank(pct=True).values
    return out


def _daily_top(dates, score, mask, n):
    idx = np.flatnonzero(mask & np.isfinite(score))
    df = pd.DataFrame({"d": dates[idx], "s": score[idx], "i": idx})
    return df.sort_values("s", ascending=False).groupby("d", sort=False).head(n)


def _smooth(pct, val, edges):
    """백분위 구간 평균 → 단조 증가로 보정 → 구간 중앙 사이를 선형 보간하는 함수."""
    from sklearn.isotonic import IsotonicRegression
    mids, means, ws, rows = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pct >= a) & (pct < b) if b < 1 else (pct >= a)
        if m.sum() < 30:
            continue
        mids.append((a + b) / 2); means.append(float(np.mean(val[m]))); ws.append(int(m.sum()))
        rows.append((a, b, int(m.sum())))
    iso = IsotonicRegression(out_of_bounds="clip").fit(mids, means, sample_weight=ws)
    fit = iso.predict(mids)
    fn = lambda p: np.interp(p, mids, fit)
    return fn, rows


FACTORS = [  # (특징, 사람이 읽는 이름, 높을 때 설명, 낮을 때 설명)
    ("ret5", "1주 수익률", "1주 상승 탄력", "1주 낙폭 큼 · 단기 반등 여지"),
    ("ret20", "1개월 수익률", "1개월 상승 흐름", "1개월 조정 · 가격 부담 완화"),
    ("ret120", "6개월 수익률", "6개월 상대강도 상위", "6개월 소외 · 순환매 후보"),
    ("hi52", "52주 고점 대비", "52주 고점 부근", "고점 대비 깊은 조정"),
    ("rsi", "RSI", "RSI 강세", "RSI 침체 · 과매도"),
    ("d_ma20", "20일선 이격", "20일선 위 추세 유지", "20일선 아래 · 평균 회귀 여지"),
    ("vol_r", "거래량", "거래량 급증 · 관심 유입", "거래량 감소 · 매도세 진정"),
    ("atrp", "변동성", "변동성 커 목표 도달 여유", "변동성 낮아 안정적"),
    ("frg5", "외국인 5일 순매수", "외국인 5일 순매수 상위", "외국인 매도 진정 국면"),
    ("inst5", "기관 5일 순매수", "기관 5일 순매수 상위", "기관 매도 일단락"),
    ("pen5", "연기금 5일 순매수", "연기금 매수 유입", "연기금 비중 축소 마무리"),
    ("obv_z", "OBV", "거래량 기준 매집 흐름", "거래량 이탈 진정"),
    ("bb_pos", "볼린저 위치", "볼린저 상단 · 강한 추세", "볼린저 하단 · 반등 구간"),
    ("d_ma60", "60일선 이격", "", ""),
    ("ret60", "3개월 수익률", "", ""),
    ("max20", "20일 최대 상승일", "", ""),
    ("beta60", "시장 베타", "", ""),
    ("sto_k", "스토캐스틱", "", ""),
    ("up_days20", "20일 상승일 비율", "", ""),
    ("lo52", "52주 저점 대비", "", ""),
]


def _surrogate(X, fnames, pct, mask, date):
    """모델 점수(백분위)를 주요 요인의 일별 순위로 선형 근사 → 요인별 계수."""
    from sklearn.linear_model import Ridge
    use = [(f, fnames.index(f)) for f, *_ in FACTORS if f in fnames]
    idx = np.flatnonzero(mask & np.isfinite(pct))
    if len(idx) > 200_000:
        idx = np.random.default_rng(1).choice(idx, 200_000, replace=False)
    Z = pd.DataFrame(X[np.ix_(idx, [c for _, c in use])]).groupby(date[idx]).rank(pct=True).values - 0.5
    ok = np.isfinite(Z).all(axis=1)
    m = Ridge(alpha=10).fit(Z[ok], pct[idx][ok] - 0.5)
    return {f: float(c) for (f, _), c in zip(use, m.coef_)}


def _say(f, v, z):
    """요인 값 → 선정 사유 한 줄. 값이 애매하면 None."""
    p = lambda x: f"{x * 100:+.0f}%".replace("-", "−")
    if f == "ret5":
        return (f"1주 {p(v)} 단기 급락 · 기술적 반등 기대" if v <= -0.08 else f"1주 {p(v)} 조정" if v <= -0.03 else
                f"1주 {p(v)} 단기 강세" if v >= 0.08 else f"1주 {p(v)} 상승" if v >= 0.03 else None)
    if f == "ret20":
        return (f"1개월 {p(v)} 낙폭과대 · 가격 메리트 부각" if v <= -0.15 else f"1개월 {p(v)} 조정" if v <= -0.05 else
                f"1개월 {p(v)} 상승 추세" if v >= 0.08 else None)
    if f == "ret120":
        return f"6개월 {p(v)} 소외 · 순환매 기대" if v <= -0.25 else f"6개월 {p(v)} 주도주 흐름" if v >= 0.3 else None
    if f == "hi52":
        return f"52주 고점 대비 {p(v)} · 저가 매력" if v <= -0.4 else f"52주 고점 근접({p(v)})" if v >= -0.05 else None
    if f == "rsi":
        return f"RSI {v:.0f} 과매도권" if v <= 35 else f"RSI {v:.0f} 강세권" if v >= 65 else None
    if f == "d_ma20":
        return f"20일선 이격 {p(v)} · 평균회귀 기대" if v <= -0.05 else f"20일선 이격 {p(v)} · 추세 유지" if v >= 0.05 else None
    if f == "vol_r":
        return f"거래량 평균의 {v:.1f}배 · 매물 소화 진행" if v <= 0.7 else f"거래량 평균의 {v:.1f}배 · 수급 유입" if v >= 1.8 else None
    if f in ("frg5", "inst5", "pen5"):
        who = {"frg5": "외국인", "inst5": "기관", "pen5": "연기금"}[f]
        return f"{who} 5일 순매수 상위" if z >= 0.85 and v > 0 else None
    if f == "obv_z":
        return "OBV 상승 · 매집 흔적" if z >= 0.85 else None
    if f == "d_ma60":
        return f"60일선 이격 {p(v)} · 중기 낙폭과대" if v <= -0.1 else f"60일선 이격 {p(v)} · 중기 추세 우위" if v >= 0.1 else None
    if f == "ret60":
        return f"3개월 {p(v)} 하락 · 가격 메리트" if v <= -0.2 else f"3개월 {p(v)} 상승 · 주도주 흐름" if v >= 0.25 else None
    if f == "max20":
        return f"최근 일간 {p(v)} 급등 이력 · 시장 관심 유효" if v >= 0.12 else None
    if f == "beta60":
        return f"베타 {v:.1f} · 시장 반등 시 탄력" if v >= 1.4 else f"베타 {v:.1f} · 방어적 성격" if v <= 0.6 else None
    if f == "sto_k":
        return f"스토캐스틱 {v:.0f} 바닥권" if v <= 20 else f"스토캐스틱 {v:.0f} 강세권" if v >= 80 else None
    if f == "up_days20":
        k = round(v * 20)
        return f"20일 중 {20 - k}일 하락 · 매도 소진 국면" if v <= 0.35 else f"20일 중 {k}일 상승 · 꾸준한 매수세" if v >= 0.65 else None
    if f == "lo52":
        return f"52주 저점 부근({p(v)})" if v <= 0.1 else None
    if f == "bb_pos":
        return "볼린저밴드 상단 돌파 시도" if v >= 0.95 else "볼린저밴드 하단 · 반등 구간" if v <= 0.1 else None
    return None


def reasons(coef, zrow, vals):
    """모델 기여도 상위 요인 → 짧은 문장 (최대 3개)."""
    out = []
    for f, *_ in FACTORS:
        if f == "atrp" or f not in coef or f not in zrow or not np.isfinite(zrow[f]):   # 변동성은 거의 모든 상위 종목에 공통
            continue
        c = coef[f] * (zrow[f] - 0.5)
        v = vals.get(f)
        if c > 0 and v is not None and np.isfinite(v):
            t = _say(f, v, zrow[f])
            if t:
                out.append((c, t))
    out.sort(reverse=True)
    return [t for _, t in out[:3]]


def tick(p: float) -> int:
    """한국거래소 호가 단위 (2023년 이후 코스피·코스닥 공통)."""
    for lim, t in ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500)):
        if p < lim:
            return t
    return 1_000


def tick_dn(p: float) -> float:
    t = tick(p)
    return float(np.floor(p / t) * t)


def tick_up(p: float) -> float:
    t = tick(p)
    return float(np.ceil(p / t) * t)


# ── 실행 ──────────────────────────────────────────────────────
def run_horizon(pn, X, fnames, H, cfg, regime, top_n=20, log=print) -> HorizonResult:
    hc = cfg.get("horizons", {}).get(str(H), {})
    gate = bool(hc.get("regime_gate", False))
    G = regime if gate else np.ones(len(pn.C), bool)
    cost = cfg.get("cost_pct", 0.25) / 100
    y = fwd_ret(pn, H, cost)
    lab = np.isfinite(y) & pn.warm
    lo, hi = np.nanpercentile(y[lab], [1, 99])
    yc = np.clip(y, lo, hi)
    target = np.full(len(y), np.nan)
    target[lab] = yc[lab] - pd.Series(yc[lab]).groupby(pn.date[lab]).transform("mean").values
    dates = np.unique(pn.date[lab])
    years = len(np.unique(pn.date)) / 252
    rng = np.random.default_rng(0)

    def sample(mask, k):
        idx = np.flatnonzero(mask)
        return idx if len(idx) <= k else rng.choice(idx, k, replace=False)

    oos_days = min(int(252 * 3), int(len(dates) * 0.45))
    fold = 126
    starts = list(range(len(dates) - oos_days, len(dates), fold))
    pred = np.full(len(y), np.nan)
    n_train = 0
    for k, s0 in enumerate(starts):
        s1 = min(s0 + fold, len(dates))
        tr = lab & (pn.date <= dates[max(s0 - H - 1, 0)])
        te = pn.warm & (pn.date >= dates[s0]) & (pn.date <= dates[s1 - 1])
        it = sample(tr, cfg.get("train_rows", 300_000))
        n_train = max(n_train, len(it))
        pred[te] = _reg().fit(X[it], target[it]).predict(X[te])
        log(f"    [{HZ[H]}] 워크포워드 {k + 1}/{len(starts)}: {pd.Timestamp(dates[s0]):%Y.%m}~{pd.Timestamp(dates[s1 - 1]):%Y.%m} 학습 {len(it):,}건")
    oosm = pn.warm & np.isfinite(pred) & (pn.date >= dates[starts[0]])
    pct = daily_pct(pn.date, pred, oosm)
    oos = oosm & lab
    o = pd.DataFrame({"d": pn.date[oos], "p": pct[oos], "r": y[oos]})
    o["rr"] = o.groupby("d").r.rank(pct=True)
    ic = float(o.groupby("d")[["p", "rr"]].corr().unstack().iloc[:, 1].mean())

    # 매도 규칙 선택
    grid_t = hc.get("target_atr", [0.5, 0.75, 1.0, 1.5, 2.0] if H <= 5 else [1.0, 1.5, 2.0, 3.0, 4.0])
    grid_s = hc.get("stop_atr", [1.0, 1.5, 2.0] if H <= 5 else [1.5, 2.0, 3.0])
    min_win = hc.get("min_win", cfg.get("min_win", 0.6))
    cap = cfg.get("atr_cap", 0.05)
    top = _daily_top(pn.date, pct, oos & G, top_n)
    ti = top.i.values
    table, sims = [], {}
    for a in grid_t:
        for b in grid_s:
            r2, h2, v2 = strategy.simulate(pn, a, b, H, cost, atr_cap=cap)
            r2 = np.where(v2, r2, np.nan)
            rt = r2[ti][np.isfinite(r2[ti])]
            ra = r2[oos & G][np.isfinite(r2[oos & G])]
            table.append((a, b, float(rt.mean()), float((rt > 0).mean()), float(ra.mean()), float((ra > 0).mean())))
            sims[(a, b)] = (r2, h2)
    ok = [t for t in table if t[3] >= min_win and t[2] > 0]
    best = max(ok or table, key=lambda t: t[2])
    kt, ks = best[0], best[1]
    ret, hit = sims[(kt, ks)]
    pos = [t for t in table if t[2] > 0]
    safe = max(pos or table, key=lambda t: (t[3], t[2]))          # 확률 우선: 평균이 플러스인 규칙 중 승률 최고
    aggr = max(table, key=lambda t: t[2])                          # 수익 우선: 평균 최고
    style_rules = {"safe": (safe[0], safe[1]), "base": (kt, ks), "aggr": (aggr[0], aggr[1])}
    log(f"    [{HZ[H]}] 매도 규칙: 목표 변동폭×{kt}, 손절 ×{ks} — 상위 {top_n} 평균 {best[2] * 100:+.2f}%, 승률 {best[3] * 100:.1f}%")

    # 백분위·변동폭 → 기대 수익·승률 (변동폭 단위 기대수익 × 종목 변동폭, 승률은 로지스틱 — 종목마다 다른 값)
    from sklearn.linear_model import LogisticRegression
    ac = np.clip(pn.atrp, 0.005, cap)
    edges = np.r_[np.linspace(0, 0.8, 9), [0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]]
    lg = lambda p_: np.log(np.clip(p_, 1e-3, 1 - 1e-3) / (1 - np.clip(p_, 1e-3, 1 - 1e-3)))

    def calib_fns(mask, ret=ret):
        pa, rows_ = _smooth(pct[mask], ret[mask] / ac[mask], edges)
        si = np.flatnonzero(mask)
        if len(si) > 300_000:
            si = rng.choice(si, 300_000, replace=False)
        lr = LogisticRegression(C=1.0).fit(np.c_[lg(pct[si]), ac[si] * 100], ret[si] > 0)
        e_ = lambda p_, a_: pa(p_) * np.clip(a_, 0.005, cap)
        w_ = lambda p_, a_: lr.predict_proba(np.c_[lg(np.atleast_1d(p_)), np.atleast_1d(np.clip(a_, 0.005, cap)) * 100])[:, 1]
        # 종목별 기대 수익 = 승률 × 평균 이익 + (1−승률) × 평균 손실 (상위 10% 실제 매매에서, 변동폭 단위)
        hi = mask & (pct >= 0.9)
        rr = ret[hi] / ac[hi]
        g, l = float(rr[rr > 0].mean()), float(rr[rr <= 0].mean())
        e2 = lambda p_, a_: (lambda w: (w * g + (1 - w) * l) * np.clip(np.atleast_1d(a_), 0.005, cap))(w_(p_, a_))
        return e_, w_, rows_, e2

    om_all = oos & np.isfinite(ret)
    om = om_all & G
    exp_of, win_of, rows, e2_on = calib_fns(om)
    exp_off = win_off = e2_off = None
    off = None
    if gate and (om_all & ~G).sum() > 5000:
        exp_off, win_off, _, e2_off = calib_fns(om_all & ~G)
        toff = _daily_top(pn.date, pct, om_all & ~G, top_n).i.values
        off = (float(ret[toff].mean()), float((ret[toff] > 0).mean()))
    calib = [(a, b, float(np.mean(ret[om][(pct[om] >= a) & ((pct[om] < b) if b < 1 else True)])),
              float(np.mean(ret[om][(pct[om] >= a) & ((pct[om] < b) if b < 1 else True)] > 0)), n) for a, b, n in rows]
    grades = {}
    for g_, a in (("S", 0.98), ("A", 0.95), ("B", 0.90)):
        m = om & (pct >= a)
        grades[g_] = [float(ret[m].mean()), float((ret[m] > 0).mean()), int(m.sum())]
    deciles = []
    for a in np.linspace(0, 0.9, 10):
        m = om & (pct >= a) & (pct < a + 0.1 + (1e-9 if a >= 0.9 else 0))
        deciles.append((float(ret[m].mean()), float((ret[m] > 0).mean()), int(m.sum())))
    # 일별 순위 구간 성적 (1~5위, 6~10위 …) — 상위 몇 종목이 실제로 얼마나 벌었나
    o2 = pd.DataFrame({"d": pn.date[om], "p": pct[om], "r": ret[om], "a": ac[om]})
    o2["k"] = o2.groupby("d").p.rank(ascending=False, method="first")
    ranks = []
    for a, b in ((1, 5), (6, 10), (11, 20), (21, 50)):
        sub = o2[(o2.k >= a) & (o2.k <= b)]
        ra = sub.r / sub.a
        ranks.append((a, b, float(sub.r.mean()), float((sub.r > 0).mean()), int(len(sub)),
                      float(ra[ra > 0].mean()), float(ra[ra <= 0].mean())))
        log(f"    [{HZ[H]}] 일별 {a}~{b}위: 평균 {sub.r.mean() * 100:+.2f}% 승률 {(sub.r > 0).mean() * 100:.1f}% ({len(sub):,}건)")
    top = _daily_top(pn.date, pct, om, top_n)
    ti = top.i.values
    tdf = pd.DataFrame({"m": pd.to_datetime(pn.date[ti]).strftime("%Y-%m"), "r": ret[ti]})
    monthly = [(k, float(g.r.mean()), float((g.r > 0).mean()), int(len(g))) for k, g in tdf.groupby("m")]
    base_m = om
    coef = _surrogate(X, fnames, pct, oosm, pn.date)
    dates_all = np.unique(pn.date)
    regime_days = pd.Series(G, index=pn.date).groupby(level=0).first().reindex(dates_all).fillna(False).values
    reg_today = bool(regime_days[-1])
    use_e, use_w = (e2_on, win_of) if (reg_today or exp_off is None) else (e2_off, win_off)
    on_mask = om if (reg_today or exp_off is None) else (om_all & ~G)
    style_fns, styles = {}, {}
    for nm, (a, b) in style_rules.items():
        rs, hs = sims[(a, b)]
        row = next(t for t in table if t[0] == a and t[1] == b)
        styles[nm] = (a, b, row[2], row[3], float(np.nanmean(hs[ti])))
        if (a, b) == (kt, ks):
            style_fns[nm] = (use_e, use_w)
        else:
            _, w_s, _, e_s = calib_fns(on_mask & np.isfinite(rs), ret=rs)
            style_fns[nm] = (e_s, w_s)
    log(f"    [{HZ[H]}] 매도 전략: " + " / ".join(f"{k} ×{v[0]}·×{v[1]} {v[2] * 100:+.2f}% {v[3] * 100:.1f}%" for k, v in styles.items()))

    # 최종 모델 → 오늘 채점
    iall = sample(lab, cfg.get("final_rows", 400_000))
    m_final = _reg().fit(X[iall], target[iall])
    last = pn.date.max()
    ends = np.flatnonzero(np.r_[pn.sid[1:] != pn.sid[:-1], True])
    today = np.array([i for i in ends if pn.date[i] == last and pn.warm[i] and np.isfinite(pn.atrp[i])])
    raw = m_final.predict(X[today])
    tp = pd.Series(raw).rank(pct=True).values
    fi = {f: fnames.index(f) for f, *_ in FACTORS if f in fnames}
    Zt = {f: pd.Series(X[today, c]).rank(pct=True).values for f, c in fi.items()}
    todays = {}
    picks = []
    for j in np.argsort(-tp):
        i = today[j]
        code = pn.meta.iloc[pn.sid[i]].Code
        atrp = float(min(pn.atrp[i], cap))
        e, w = float(use_e(tp[j], atrp)[0]), float(use_w(tp[j], atrp)[0])
        todays[code] = (float(tp[j]), e, w)
        if len(picks) >= top_n or e <= 0:
            continue
        row = pn.meta.iloc[pn.sid[i]]
        d = pn.frames[row.Code]
        c, pc = float(d.Close.iloc[-1]), float(d.Close.iloc[-2])
        grade = "S" if tp[j] >= 0.98 else "A" if tp[j] >= 0.95 else "B" if tp[j] >= 0.90 else "C"
        zrow = {f: float(Zt[f][j]) for f in fi}
        vals = {f: float(X[i, c2]) for f, c2 in fi.items()}
        picks.append({"code": row.Code, "name": row.Name, "close": c, "chg": c - pc, "pct": (c / pc - 1) * 100,
                      "exp": e, "prob": w, "score": float(tp[j]), "grade": grade,
                      "target": tick_dn(c * (1 + kt * atrp)), "stop": tick_up(c * (1 - ks * atrp)), "atrp": atrp,
                      "why": reasons(coef, zrow, vals), "df": d,
                      "styles": {nm: {"target": tick_dn(c * (1 + a * atrp)), "stop": tick_up(c * (1 - b * atrp)),
                                      "exp": float(style_fns[nm][0](tp[j], atrp)[0]), "prob": float(style_fns[nm][1](tp[j], atrp)[0])}
                                 for nm, (a, b) in style_rules.items()}})
    od = pn.date[oos]
    return HorizonResult(
        H=H, picks=picks, kt=kt, ks=ks, min_win=min_win, atr_cap=cap, exit_table=table, calib=calib, grades=grades,
        deciles=deciles, monthly=monthly,
        base=(float(ret[base_m].mean()), float((ret[base_m] > 0).mean())),
        top=(float(ret[ti].mean()), float((ret[ti] > 0).mean()), float(hit[ti].mean())),
        top_n=top_n, ic=ic, oos_period=(pd.Timestamp(od.min()), pd.Timestamp(od.max())), folds=len(starts),
        years=years, n_train=len(iall), n_oos=int(om.sum()), pct=pct, exp_of=exp_of, win_of=win_of, today=todays,
        gate=gate, regime_ok=reg_today, regime_days=regime_days, off=off, styles=styles)


def run(pn: strategy.Panel, cfg: dict, market_close: pd.Series, top_n: int = 20, log=print) -> dict:
    X, fnames = build_X(pn, market_close)
    log(f"    특징 {X.shape[1]}개 × {X.shape[0]:,}행")
    regime = market_regime(pn, market_close)
    out = {}
    for H in cfg.get("model_horizons", [5, 20]):
        n = cfg.get("horizons", {}).get(str(H), {}).get("picks", top_n)
        out[H] = run_horizon(pn, X, fnames, H, cfg, regime, top_n=n, log=log)
    return out


def market_regime(pn, market_close: pd.Series) -> np.ndarray:
    """코스피 20일 이동평균이 5거래일 전보다 높으면 '상승 국면' (행 단위 bool)."""
    dates = np.unique(pn.date)
    k = market_close.reindex(pd.DatetimeIndex(dates)).ffill()
    ma = k.rolling(20).mean()
    up = (ma / ma.shift(5) - 1 > 0)
    return up.reindex(pd.DatetimeIndex(pn.date)).fillna(False).values.astype(bool)
