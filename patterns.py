"""차트 패턴 탐지 — 캔들, 이동평균 지지·저항, 추세 전환, 가격 구조(헤드앤숄더·이중바닥 등), 다이버전스, 거래량.

모든 신호는 그날 종가까지의 정보만 사용한다 (고점·저점은 좌우 K일로 '확정'된 뒤에만 인식).
통계(stats)는 신호가 처음 나타난 날 다음 날 시가에 사서 5·20거래일 뒤 종가에 판 결과로 계산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (키, 이름, 방향: +1 상승 / -1 하락 / 0 중립, 설명)
PATTERNS = [
    # 캔들
    ("vol_bull", "거래량 실린 장대양봉", 1, "평소 2배 넘는 거래량으로 강하게 올라 마감"),
    ("vol_bear", "거래량 실린 장대음봉", -1, "평소 2배 넘는 거래량으로 강하게 내려 마감"),
    ("hammer", "망치형", 1, "하락 끝에 긴 아래꼬리 — 저가 매수세 유입"),
    ("shooting", "유성형", -1, "상승 끝에 긴 윗꼬리 — 고점 매물 출회"),
    ("bull_engulf", "상승장악형", 1, "전날 음봉을 감싸는 양봉 — 매수 우위 전환"),
    ("bear_engulf", "하락장악형", -1, "전날 양봉을 감싸는 음봉 — 매도 우위 전환"),
    ("morning_star", "샛별형", 1, "음봉·작은 봉·양봉 3일 반전"),
    ("evening_star", "석별형", -1, "양봉·작은 봉·음봉 3일 반전"),
    ("three_white", "적삼병", 1, "3일 연속 몸통 있는 양봉"),
    ("three_black", "흑삼병", -1, "3일 연속 몸통 있는 음봉"),
    ("gap_up", "상승 갭", 1, "전날 고가 위에서 거래"),
    ("gap_down", "하락 갭", -1, "전날 저가 아래에서 거래"),
    ("doji_top", "고점 도지", -1, "상승 뒤 시가·종가가 같은 봉 — 방향 고민"),
    # 이동평균
    ("ma5_sup", "5일선 지지", 1, "5일선까지 밀렸다가 위에서 마감"),
    ("ma5_rej", "5일선 저항", -1, "5일선에 닿고 밀려 아래에서 마감"),
    ("ma20_sup", "20일선 지지", 1, "20일선에서 반등"),
    ("ma20_rej", "20일선 저항", -1, "20일선에 막혀 하락"),
    ("ma60_sup", "60일선 지지", 1, "60일선에서 반등"),
    ("ma60_rej", "60일선 저항", -1, "60일선에 막혀 하락"),
    ("ma20_up", "20일선 돌파", 1, "거래량과 함께 20일선 위로"),
    ("ma20_dn", "20일선 이탈", -1, "20일선 아래로 마감"),
    ("ma60_up", "60일선 돌파", 1, "60일선 위로 올라섬"),
    ("ma60_dn", "60일선 이탈", -1, "60일선 아래로 마감"),
    ("gc", "골든크로스", 1, "5일선이 20일선 위로"),
    ("dc", "데드크로스", -1, "5일선이 20일선 아래로"),
    ("gc_long", "중기 골든크로스", 1, "20일선이 60일선 위로"),
    ("dc_long", "중기 데드크로스", -1, "20일선이 60일선 아래로"),
    ("align_on", "정배열 전환", 1, "5>20>60>120일선 순서 완성"),
    ("align_off", "역배열 전환", -1, "5<20<60<120일선 순서 완성"),
    # 구조
    ("hs_top", "헤드앤숄더 완성", -1, "세 고점 중 가운데가 가장 높고 목선 이탈"),
    ("inv_hs", "역헤드앤숄더 완성", 1, "세 저점 중 가운데가 가장 낮고 목선 돌파"),
    ("dbl_bottom", "이중바닥 돌파", 1, "비슷한 두 저점 뒤 중간 고점 돌파"),
    ("dbl_top", "이중천장 이탈", -1, "비슷한 두 고점 뒤 중간 저점 이탈"),
    ("box_up", "박스권 상단 돌파", 1, "60일 가격대 위로 거래량 돌파"),
    ("box_dn", "박스권 하단 이탈", -1, "60일 가격대 아래로 이탈"),
    ("tri_up", "삼각수렴 상향 이탈", 1, "고점은 낮아지고 저점은 높아지다 위로 이탈"),
    ("tri_dn", "삼각수렴 하향 이탈", -1, "수렴 끝에 아래로 이탈"),
    ("hi52", "52주 신고가", 1, "1년 중 가장 높은 종가"),
    ("lo52", "52주 신저가", -1, "1년 중 가장 낮은 종가"),
    ("pullback", "상승 추세 눌림", 1, "상승 추세에서 거래량 줄며 20일선 부근 조정"),
    ("overheat", "단기 과열", -1, "20일선보다 15% 넘게 위"),
    ("oversold", "낙폭 과대", 1, "20일선보다 12% 넘게 아래"),
    ("squeeze", "변동성 수축", 0, "볼린저 밴드폭이 반년 내 최저권 — 큰 움직임 임박"),
    ("bb_up", "볼린저 상단 돌파", 1, "밴드 위로 마감"),
    ("bb_dn", "볼린저 하단 이탈", -1, "밴드 아래로 마감"),
    # 모멘텀·거래량
    ("rsi_os", "RSI 과매도", 1, "RSI 30 아래"),
    ("rsi_ob", "RSI 과매수", -1, "RSI 70 위"),
    ("bull_div", "RSI 상승 다이버전스", 1, "주가 저점은 낮아졌는데 RSI 저점은 높아짐"),
    ("bear_div", "RSI 하락 다이버전스", -1, "주가 고점은 높아졌는데 RSI 고점은 낮아짐"),
    ("macd_gc", "MACD 골든크로스", 1, "MACD가 시그널 위로"),
    ("macd_dc", "MACD 데드크로스", -1, "MACD가 시그널 아래로"),
    ("vol_dry", "거래량 급감", 0, "5일 평균 거래량이 60일 평균의 절반 이하"),
    ("obv_lead", "OBV 선행 신고가", 1, "주가보다 먼저 거래량 지표가 신고가 — 매집"),
]
P_NAME = {k: n for k, n, _, _ in PATTERNS}
P_BIAS = {k: b for k, _, b, _ in PATTERNS}
P_DESC = {k: d for k, _, _, d in PATTERNS}
KEYS = [k for k, *_ in PATTERNS]
K_PIV = 5                      # 고점·저점 확정에 필요한 좌우 봉 수


def _pivots(h: np.ndarray, l: np.ndarray, k: int = K_PIV):
    """t 시점에 '확정'된 고점·저점 (t-k가 앞뒤 k봉 중 최고/최저)."""
    n = len(h)
    ph = np.zeros(n, bool)
    pl = np.zeros(n, bool)
    for t in range(2 * k, n):
        w = slice(t - 2 * k, t + 1)
        c = t - k
        if h[c] == h[w].max() and h[c] > h[c - 1]:
            ph[t] = True
        if l[c] == l[w].min() and l[c] < l[c - 1]:
            pl[t] = True
    return ph, pl


def _structure(d: pd.DataFrame) -> dict:
    """확정 고점·저점 순서로 헤드앤숄더·이중바닥/천장·삼각수렴·다이버전스 판정."""
    h, l, c, rsi = d.High.values, d.Low.values, d.Close.values, d.RSI.values
    n = len(c)
    ph, pl = _pivots(h, l)
    out = {k: np.zeros(n, bool) for k in ("hs_top", "inv_hs", "dbl_bottom", "dbl_top", "tri_up", "tri_dn", "bull_div", "bear_div")}
    highs, lows = [], []                 # (봉 위치, 가격, RSI)
    hs_neck = ihs_neck = db_neck = dt_neck = None
    tri_hi = tri_lo = None
    for t in range(n):
        if ph[t]:
            i = t - K_PIV
            highs.append((i, h[i], rsi[i]))
            highs = highs[-4:]
            if len(highs) >= 2:
                (i1, p1, r1), (i2, p2, r2) = highs[-2], highs[-1]
                if p2 > p1 * 1.0 and r2 < r1 - 3 and np.isfinite(r1) and np.isfinite(r2) and i2 - i1 <= 40:
                    out["bear_div"][t] = True
                # 이중천장: 비슷한 두 고점, 사이 저점
                if i2 - i1 >= 8 and abs(p2 / p1 - 1) <= 0.03:
                    mid = [p for j, p, _ in lows if i1 < j < i2]
                    if mid and min(mid) < min(p1, p2) * 0.95:
                        dt_neck = (min(mid), t + 30)
            if len(highs) >= 3:
                (i1, p1, _), (i2, p2, _), (i3, p3, _) = highs[-3:]
                if p2 >= max(p1, p3) * 1.03 and abs(p3 / p1 - 1) <= 0.06 and i3 - i1 <= 90:
                    mid = [p for j, p, _ in lows if i1 < j < i3]
                    if mid:
                        hs_neck = (min(mid), t + 30)
        if pl[t]:
            i = t - K_PIV
            lows.append((i, l[i], rsi[i]))
            lows = lows[-4:]
            if len(lows) >= 2:
                (i1, p1, r1), (i2, p2, r2) = lows[-2], lows[-1]
                if p2 < p1 and r2 > r1 + 3 and np.isfinite(r1) and np.isfinite(r2) and i2 - i1 <= 40:
                    out["bull_div"][t] = True
                if i2 - i1 >= 8 and abs(p2 / p1 - 1) <= 0.03:
                    mid = [p for j, p, _ in highs if i1 < j < i2]
                    if mid and max(mid) > max(p1, p2) * 1.05:
                        db_neck = (max(mid), t + 30)
            if len(lows) >= 3:
                (i1, p1, _), (i2, p2, _), (i3, p3, _) = lows[-3:]
                if p2 <= min(p1, p3) * 0.97 and abs(p3 / p1 - 1) <= 0.06 and i3 - i1 <= 90:
                    mid = [p for j, p, _ in highs if i1 < j < i3]
                    if mid:
                        ihs_neck = (max(mid), t + 30)
        # 삼각수렴: 최근 두 고점 하락 + 두 저점 상승
        if (ph[t] or pl[t]) and len(highs) >= 2 and len(lows) >= 2:
            if highs[-1][1] < highs[-2][1] and lows[-1][1] > lows[-2][1] and t - min(highs[-2][0], lows[-2][0]) <= 60:
                tri_hi, tri_lo = (highs[-1][1], t + 20), (lows[-1][1], t + 20)
        if t == 0:
            continue
        # 목선·수렴선 돌파 확인 (유효기간 안에서 처음 넘는 날)
        if hs_neck and t <= hs_neck[1] and c[t] < hs_neck[0] <= c[t - 1]:
            out["hs_top"][t] = True; hs_neck = None
        if ihs_neck and t <= ihs_neck[1] and c[t] > ihs_neck[0] >= c[t - 1]:
            out["inv_hs"][t] = True; ihs_neck = None
        if db_neck and t <= db_neck[1] and c[t] > db_neck[0] >= c[t - 1]:
            out["dbl_bottom"][t] = True; db_neck = None
        if dt_neck and t <= dt_neck[1] and c[t] < dt_neck[0] <= c[t - 1]:
            out["dbl_top"][t] = True; dt_neck = None
        if tri_hi and t <= tri_hi[1]:
            if c[t] > tri_hi[0] >= c[t - 1]:
                out["tri_up"][t] = True; tri_hi = tri_lo = None
            elif tri_lo and c[t] < tri_lo[0] <= c[t - 1]:
                out["tri_dn"][t] = True; tri_hi = tri_lo = None
    return out


def flags(d: pd.DataFrame) -> pd.DataFrame:
    """지표가 붙은 일봉(indicators.add_indicators 결과)에서 패턴 신호표 (행: 날짜, 열: 패턴)."""
    o, h, l, c, v = d.Open, d.High, d.Low, d.Close, d.Volume
    p = d.shift(1)
    rng_ = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    up_t = h - np.maximum(c, o)
    lo_t = np.minimum(c, o) - l
    atr = d.ATR
    vr = v / d.VOL_MA20.replace(0, np.nan)
    ret5 = c / c.shift(5) - 1
    green, red = c > o, c < o
    big = body >= 0.6 * atr
    r = pd.DataFrame(index=d.index)
    r["vol_bull"] = green & big & (vr >= 2) & (c > p.Close) & (up_t <= body * 0.5)
    r["vol_bear"] = red & big & (vr >= 2) & (c < p.Close)
    r["hammer"] = (lo_t >= 2 * body) & (up_t <= 0.3 * rng_) & (ret5 < -0.03) & (l <= l.rolling(10).min())
    r["shooting"] = (up_t >= 2 * body) & (lo_t <= 0.3 * rng_) & (ret5 > 0.03) & (h >= h.rolling(10).max())
    r["bull_engulf"] = (p.Close < p.Open) & green & (o <= p.Close) & (c >= p.Open) & (ret5 < 0)
    r["bear_engulf"] = (p.Close > p.Open) & red & (o >= p.Close) & (c <= p.Open) & (ret5 > 0)
    p2 = d.shift(2)
    small1 = (p.Close - p.Open).abs() <= 0.3 * (p2.Open - p2.Close).abs()
    r["morning_star"] = (p2.Close < p2.Open) & ((p2.Open - p2.Close) >= 0.6 * atr) & small1 & green & (c >= (p2.Open + p2.Close) / 2)
    r["evening_star"] = (p2.Close > p2.Open) & ((p2.Close - p2.Open) >= 0.6 * atr) & small1 & red & (c <= (p2.Open + p2.Close) / 2)
    b3 = lambda s: s.rolling(3).sum() == 3
    r["three_white"] = b3((green & (body >= 0.3 * atr) & (c > p.Close)).astype(int))
    r["three_black"] = b3((red & (body >= 0.3 * atr) & (c < p.Close)).astype(int))
    r["gap_up"] = l > p.High
    r["gap_down"] = h < p.Low
    r["doji_top"] = (body <= 0.1 * rng_) & (ret5 > 0.05) & (h >= h.rolling(20).max())
    # 이동평균 지지·저항
    for n in (5, 20, 60):
        ma, pma = d[f"MA{n}"], p[f"MA{n}"]
        slope = ma - ma.shift(3)
        r[f"ma{n}_sup"] = (p.Close > pma) & (l <= ma * 1.005) & (c >= ma) & (slope > 0)
        r[f"ma{n}_rej"] = (p.Close < pma) & (h >= ma * 0.995) & (c < ma) & (slope < 0)
    r["ma20_up"] = (c > d.MA20) & (p.Close <= p.MA20) & (vr >= 1.5)
    r["ma20_dn"] = (c < d.MA20) & (p.Close >= p.MA20)
    r["ma60_up"] = (c > d.MA60) & (p.Close <= p.MA60)
    r["ma60_dn"] = (c < d.MA60) & (p.Close >= p.MA60)
    r["gc"] = (d.MA5 > d.MA20) & (p.MA5 <= p.MA20)
    r["dc"] = (d.MA5 < d.MA20) & (p.MA5 >= p.MA20)
    r["gc_long"] = (d.MA20 > d.MA60) & (p.MA20 <= p.MA60)
    r["dc_long"] = (d.MA20 < d.MA60) & (p.MA20 >= p.MA60)
    al = (d.MA5 > d.MA20) & (d.MA20 > d.MA60) & (d.MA60 > d.MA120)
    ra = (d.MA5 < d.MA20) & (d.MA20 < d.MA60) & (d.MA60 < d.MA120)
    r["align_on"] = al & ~al.shift(1, fill_value=False)
    r["align_off"] = ra & ~ra.shift(1, fill_value=False)
    # 구조
    hh60, ll60 = h.rolling(60).max().shift(1), l.rolling(60).min().shift(1)
    box = (hh60 / ll60 - 1) <= 0.25
    r["box_up"] = box & (c > hh60) & (p.Close <= hh60) & (vr >= 1.5)
    r["box_dn"] = box & (c < ll60) & (p.Close >= ll60)
    prev_hi = c.rolling(252, min_periods=200).max().shift(1)
    prev_lo = c.rolling(252, min_periods=200).min().shift(1)
    r["hi52"] = c > prev_hi
    r["lo52"] = c < prev_lo
    up_tr = (d.MA20 > d.MA60) & (d.MA60 > d.MA60.shift(5)) & (c / c.shift(40) - 1 > 0.05)
    r["pullback"] = up_tr & ((c / d.MA20 - 1).between(-0.02, 0.025)) & (v.rolling(5).mean() < 0.8 * d.VOL_MA20) & (c < c.rolling(10).max() * 0.97)
    r["overheat"] = c / d.MA20 - 1 > 0.15
    r["oversold"] = c / d.MA20 - 1 < -0.12
    r["squeeze"] = d.BB_width <= d.BB_width.rolling(120, min_periods=60).quantile(0.1)
    r["bb_up"] = (c > d.BB_upper) & (p.Close <= p.BB_upper)
    r["bb_dn"] = (c < d.BB_lower) & (p.Close >= p.BB_lower)
    r["rsi_os"] = d.RSI < 30
    r["rsi_ob"] = d.RSI > 70
    r["macd_gc"] = (d.MACD > d.MACD_signal) & (p.MACD <= p.MACD_signal)
    r["macd_dc"] = (d.MACD < d.MACD_signal) & (p.MACD >= p.MACD_signal)
    r["vol_dry"] = v.rolling(5).mean() < 0.5 * v.rolling(60).mean()
    r["obv_lead"] = (d.OBV >= d.OBV.rolling(60).max()) & (c < c.rolling(60).max() * 0.97)
    st = _structure(d)
    for k, a in st.items():
        r[k] = a
    return r[KEYS].fillna(False).astype(bool)


def onsets(f: pd.DataFrame, cooldown: int = 5) -> pd.DataFrame:
    """상태형 신호(RSI 과매도 등)는 '처음 나타난 날'만 남김."""
    prev = f.astype(int).rolling(cooldown, min_periods=1).max().shift(1).fillna(0).astype(bool)
    return f & ~prev


def stats(frames: dict, horizons=(5, 20), cost: float = 0.0025) -> dict:
    """전 종목·전 기간 패턴 발생 후 성적 — {키: {n, h: [평균, 시장대비, 승률]}}."""
    rows = []
    for code, d in frames.items():
        f = onsets(flags(d))
        o, c = d.Open.values, d.Close.values
        n = len(c)
        fw = {}
        for H in horizons:
            r = np.full(n, np.nan)
            if n > H + 1:
                ent = np.r_[o[1:], np.nan][:n - H]
                with np.errstate(divide="ignore", invalid="ignore"):
                    r[:n - H] = np.where(ent > 0, c[H:] / ent - 1 - cost, np.nan)
            r[~np.isfinite(r) | (np.abs(r) > 3)] = np.nan        # 시가 누락·데이터 오류 제외
            fw[H] = r
        idx = np.arange(n)
        warm = idx >= 130
        df = pd.DataFrame({"date": d.index, **{f"r{H}": fw[H] for H in horizons}})
        for k in KEYS:
            df[k] = f[k].values & warm
        rows.append(df)
    P = pd.concat(rows, ignore_index=True)
    out = {}
    for H in horizons:
        P[f"m{H}"] = P.groupby("date")[f"r{H}"].transform("mean")
    base = {H: [float(P[f"r{H}"].mean()), float((P[f"r{H}"] > 0).mean())] for H in horizons}
    for k in KEYS:
        s = P[P[k]]
        e = {"n": int(len(s))}
        for H in horizons:
            x = s[f"r{H}"].dropna()
            if len(x) < 30:
                continue
            e[str(H)] = [round(float(x.mean()) * 100, 2), round(float((x - s.loc[x.index, f"m{H}"]).mean()) * 100, 2),
                         round(float((x > 0).mean()) * 100, 1), int(len(x))]
        out[k] = e
    out["_base"] = {str(H): [round(b[0] * 100, 2), round(b[1] * 100, 1)] for H, b in base.items()}
    return out


def levels(d: pd.DataFrame, lookback: int = 120) -> dict:
    """오늘 종가 기준 가장 가까운 지지선·저항선 (확정 고점·저점, 이동평균, 52주 고저).
    종가에서 최소 max(1.5%, 일평균 변동폭 절반) 떨어진 가격만 의미 있는 선으로 본다."""
    c = float(d.Close.iloc[-1])
    gap = max(0.015, float(d.ATR_pct.iloc[-1]) * 0.5 if np.isfinite(d.ATR_pct.iloc[-1]) else 0.015)
    x = d.iloc[-lookback:]
    ph, pl = _pivots(x.High.values, x.Low.values)
    cand = []
    for t in np.flatnonzero(ph):
        cand.append((float(x.High.values[t - K_PIV]), "전고점"))
    for t in np.flatnonzero(pl):
        cand.append((float(x.Low.values[t - K_PIV]), "전저점"))
    for n in (5, 20, 60, 120):
        v = d[f"MA{n}"].iloc[-1]
        if np.isfinite(v):
            cand.append((float(v), f"{n}일선"))
    y = d.iloc[-250:]
    cand.append((float(y.High.max()), "52주 최고가"))
    cand.append((float(y.Low.min()), "52주 최저가"))
    below = [z for z in cand if z[0] < c * (1 - gap)]
    above = [z for z in cand if z[0] > c * (1 + gap)]
    sup = max(below, key=lambda z: z[0]) if below else None
    res = min(above, key=lambda z: z[0]) if above else None
    return {"support": sup, "resist": res}
