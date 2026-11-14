"""한국 주식 매매 기법(셋업) 10년 실데이터 검증 — research/setups.json.
'Wake 연구' 워크플로에서 실행: python tools/setups_research.py

셋업마다 '오늘 종가에 신호 → 다음 날 시가 매수'로 사고, 여러 청산 규칙으로 판 결과를 비교한다.
거래비용 0.3%(수수료·세금·슬리피지) 차감. 시장 국면(지수 60·20일선, 상승 종목 비율, 변동성)별로도 나눠 본다."""
from __future__ import annotations
import json, sys, time, os
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import site_build as sb

OUT = Path("research"); OUT.mkdir(exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
COST = 0.003

src = sb.Source(bool(os.environ.get("DEMO")))
lst = src.listing()
lst = lst[lst.Code.str.endswith("0") & ~lst.Name.str.contains("스팩|리츠|ETN|ETF")]
u = lst[lst.Marcap >= 1e11].sort_values("Marcap", ascending=False)
log("시세", len(u))
prices = src.scan_prices(list(u.Code), 10)
log("시세 완료", len(prices))
bigc = [c for c in u[u.Marcap >= 1e12].Code if c in prices]
flows = src.scan_flows({c: prices[c] for c in bigc}, 10)
log("수급 완료", len(flows))

codes = [c for c in u.Code if c in prices]
C = pd.DataFrame({c: prices[c].Close for c in codes}).sort_index()
idx = C.index
O = pd.DataFrame({c: prices[c].Open for c in codes}).reindex(idx)
H = pd.DataFrame({c: prices[c].High for c in codes}).reindex(idx)
L = pd.DataFrame({c: prices[c].Low for c in codes}).reindex(idx)
V = pd.DataFrame({c: prices[c].Volume for c in codes}).reindex(idx)
nd, nc = C.shape
log("패널", C.shape, idx[0].date(), idx[-1].date())
cap_now = u.set_index("Code").Marcap.reindex(codes).values
BIG = np.broadcast_to(cap_now >= 1e12, C.shape)

# ── 지표 ──
ma = {n: C.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60, 120, 200)}
amt = C * V
amt20 = amt.rolling(20, min_periods=15).mean()
vol20 = V.rolling(20, min_periods=15).mean()
tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()]).groupby(level=0).max().reindex(idx)
atr = tr.rolling(14, min_periods=10).mean()
def rsi(n):
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
rsi2, rsi14 = rsi(2), rsi(14)
hi20p, hi60p = H.rolling(20).max().shift(1), H.rolling(60).max().shift(1)
hi250p = H.rolling(250, min_periods=200).max().shift(1)
bb_mid, bb_sd = ma[20], C.rolling(20).std()
ret1 = C.pct_change()
body = (C - O) / O
liq = (amt20 >= 3e9)                                         # 하루 평균 거래대금 30억 이상
align = (ma[5] > ma[20]) & (ma[20] > ma[60]) & (ma[60] > ma[120])
up200 = C > ma[200]

# 시장 국면: 코스피 + 전 종목 폭
try:
    kos = src.kospi(11)["Close"].reindex(idx).ffill()
except Exception:
    kos = (C * cap_now).sum(axis=1)
kma20, kma60 = kos.rolling(20).mean(), kos.rolling(60).mean()
breadth = (C > ma[20]).sum(axis=1) / C.notna().sum(axis=1)
kvol = kos.pct_change().rolling(20).std()
kvolp = kvol.rolling(250, min_periods=120).rank(pct=True)
gate = ((kos > kma60) & (kos > kma20) & (breadth >= .4) & (kvolp < .85)).values
risk_off = ((kos < kma60) & (kos < kma20)).values

# 기준봉: 최근 10일 안에 +7% 이상 장대양봉 + 거래량 3배 이상
kb = (ret1 >= .07) & (body >= .05) & (V >= vol20.shift(1) * 3)
kb_recent = kb.rolling(10, min_periods=1).max().astype(bool) & ~kb
kb_open = O.where(kb).ffill(limit=10)
kb_vol = V.where(kb).ffill(limit=10)

# 수급: 외국인·기관 동반 순매수 일수
fr = pd.DataFrame({c: flows[c]["외국인"] for c in flows if "외국인" in flows[c]}).reindex(idx).reindex(columns=codes)
ins = pd.DataFrame({c: flows[c]["기관합계"] for c in flows if "기관합계" in flows[c]}).reindex(idx).reindex(columns=codes)
both = ((fr > 0) & (ins > 0)).astype(float).where(fr.notna())
both5 = both.rolling(5).sum()
fi20 = (fr + ins).rolling(20).sum() / (amt20 * 20)

S = {}
# 1) 기준봉 눌림목: 정배열 + 최근 10일 내 기준봉 + 거래량 줄며 5일선 아래 눌림 + 기준봉 시가 위 유지
S["눌림목(기준봉 후 5일선 눌림)"] = liq & (ma[20] > ma[60]) & kb_recent & (C < ma[5]) & (C > kb_open) & (V < kb_vol / 3)
# 2) 20일선 눌림목: 정배열 상승 추세에서 20일선 근처(±2%)까지 내려온 날, 거래량 감소
S["20일선 눌림목(정배열)"] = liq & align & ((C / ma[20] - 1).abs() <= .02) & (V < vol20) & (C.shift(5) > C)
# 3) 전고점(60일) 돌파 + 거래대금 2배 + 양봉
S["전고점 돌파(60일 최고가·거래 2배)"] = liq & (C > hi60p) & (amt >= amt20.shift(1) * 2) & (body > .02) & (ma[20] > ma[60])
# 4) 52주 신고가 돌파
S["52주 신고가 돌파"] = liq & (C > hi250p) & (amt >= amt20.shift(1) * 1.5) & (body > 0)
# 5) 볼린저 스퀴즈 후 상단 돌파
bw = (4 * bb_sd / bb_mid)
S["볼린저 수축 후 상단 돌파"] = liq & (bw.shift(1) <= bw.rolling(120, min_periods=60).quantile(.2).shift(1)) & (C > bb_mid + 2 * bb_sd) & (amt >= amt20.shift(1) * 1.5)
# 6) 외국인·기관 쌍끌이: 5일 중 4일 이상 동반 순매수 + 20일선 위
S["외국인·기관 쌍끌이 매수"] = liq & (both5 >= 4) & (C > ma[20]) & (fi20 > 0)
# 7) 과매도 반등(대형·200일선 위): RSI(2) 10 미만 — 단기 급락 후 평균 회귀
S["과매도 반등(200일선 위·RSI2<10)"] = liq & up200 & (rsi2 < 10)
# 8) 낙폭과대 반등: 20일선 이격도 88% 이하 + 대형주
S["낙폭과대 반등(이격도 88%↓·대형주)"] = liq & BIG & (C / ma[20] <= .88)
# 9) 3일 연속 하락 후 반등 양봉(대형주 200일선 위)
S["3일 하락 후 첫 양봉(200일선 위)"] = liq & up200 & (ret1.shift(1) < 0) & (ret1.shift(2) < 0) & (ret1.shift(3) < 0) & (C > O) & (C > C.shift(1))
# 10) 갭상승 유지(시가 +3% 갭, 종가 시가 위, 거래 2배)
S["갭상승 유지"] = liq & (O >= C.shift(1) * 1.03) & (C >= O) & (amt >= amt20.shift(1) * 2)
# 11) RSI14 50 회복 + 20일선 상승
S["RSI 50 회복(상승 추세)"] = liq & (rsi14 > 50) & (rsi14.shift(1) <= 50) & (ma[20] > ma[20].shift(5)) & (C > ma[60])
# 12) 쌍바닥: 60일 내 최저가 부근(3%)에서 다시 지지 + 양봉 + RSI 다이버전스 단순형
lo60 = L.rolling(60).min()
S["쌍바닥(60일 저점 재지지 양봉)"] = liq & (L <= lo60 * 1.03) & (L.shift(10).rolling(40).min() <= lo60 * 1.03) & (C > O) & (rsi14 > rsi14.rolling(30).min() + 8)

# 비교 기준: 같은 종목군에서 아무 날이나 무작위로 산 경우 (같은 청산 규칙) — 셋업의 실제 우위 = 셋업 − 기준
rng = np.random.default_rng(7)
S["(비교) 무작위 진입"] = liq & pd.DataFrame(rng.random(C.shape) < 0.01, index=idx, columns=codes)

Cn, On, Hn, Ln = (x.values.astype(float) for x in (C, O, H, L))
ma5n, ma10n, ma20n = ma[5].values, ma[10].values, ma[20].values
atrn = atr.values


def exits(d, c, rule):
    """다음 날 시가 매수 후 청산 규칙별 수익률·보유일."""
    e = On[d + 1, c]
    a = np.nan_to_num(atrn[d, c], nan=np.nanmedian(atrn))
    out = np.full(len(d), np.nan); days = np.zeros(len(d))
    kind, hmax = rule[0], rule[1]
    w = d[:, None] + np.arange(1, hmax + 1)[None, :]
    w = np.minimum(w, nd - 1)
    cl, hi, lo, op = Cn[w, c[:, None]], Hn[w, c[:, None]], Ln[w, c[:, None]], On[w, c[:, None]]
    last = cl[:, -1]
    if kind == "hold":
        return last / e - 1 - COST, np.full(len(d), hmax)
    if kind == "atr":                                  # 손절 k_s·ATR(종가), 목표 k_t·ATR(장중)
        ks, kt = rule[2], rule[3]
        stop, tgt = e - ks * a, e + kt * a
        hs, ht = cl <= stop[:, None], hi >= tgt[:, None]
    elif kind == "ma5":                                # 평균 회귀: 종가가 5일선 위로 마감하면 다음 날 시가 매도, 손절 3·ATR
        stop = e - 3 * a
        hs = cl <= stop[:, None]
        ht = cl > ma5n[w, c[:, None]]
    elif kind == "ma10":                               # 추세 추종: 종가가 10일선 아래로 마감하면 매도, 손절 2·ATR
        stop = e - 2 * a
        hs = (cl <= stop[:, None]) | (cl < ma10n[w, c[:, None]])
        ht = np.zeros_like(hs)
    elif kind == "pct":                                # 고정 %: 목표 +t, 손절 -s(종가)
        s_, t_ = rule[2], rule[3]
        hs, ht = cl <= (e * (1 - s_))[:, None], hi >= (e * (1 + t_))[:, None]
    B = hmax + 1
    fs = np.where(hs.any(1), hs.argmax(1), B); ft = np.where(ht.any(1), ht.argmax(1), B)
    r = np.arange(len(d))
    nxt = lambda k: np.where(k + 1 < hmax, op[r, np.minimum(k + 1, hmax - 1)], cl[r, np.minimum(k, hmax - 1)])
    if kind in ("ma5",):
        px_t = nxt(ft)                                 # 신호 다음 날 시가
    elif kind == "atr":
        px_t = np.fmax(op[r, np.minimum(ft, hmax - 1)], e + rule[3] * a)
    elif kind == "pct":
        px_t = np.fmax(op[r, np.minimum(ft, hmax - 1)], e * (1 + rule[3]))
    else:
        px_t = last
    px_s = nxt(fs)
    stopped = (fs < B) & (fs <= ft) if kind != "ma5" else (fs < B) & (fs < ft)
    hit = (ft < B) & ~stopped
    px = np.where(stopped, px_s, np.where(hit, px_t, last))
    days = np.where(stopped, fs + 1, np.where(hit, ft + 1, hmax))
    return px / e - 1 - COST, days


RULES = {"10일 보유": ("hold", 10), "20일 보유": ("hold", 20),
         "ATR 손절2·목표3 (20일)": ("atr", 20, 2, 3), "ATR 손절1.5·목표1.5 (10일)": ("atr", 10, 1.5, 1.5),
         "ATR 손절2·목표1 (10일)": ("atr", 10, 2, 1),
         "5일선 회복 시 매도 (최대 10일)": ("ma5", 10), "10일선 이탈 시 매도 (최대 30일)": ("ma10", 30),
         "목표+7%·손절-5% (20일)": ("pct", 20, .05, .07), "목표+5%·손절-7% (20일)": ("pct", 20, .07, .05)}
years = idx.year.values
res = {"period": [str(idx[0].date()), str(idx[-1].date())], "stocks": nc, "gate_share": round(float(np.nanmean(gate[250:])) * 100, 1),
       "today": {"date": str(idx[-1].date()), "gate": bool(gate[-1]), "risk_off": bool(risk_off[-1]),
                 "kos_ma20": round(float(kos.iloc[-1] / kma20.iloc[-1] - 1) * 100, 2), "kos_ma60": round(float(kos.iloc[-1] / kma60.iloc[-1] - 1) * 100, 2),
                 "breadth": round(float(breadth.iloc[-1]) * 100, 1), "kvolp": round(float(kvolp.iloc[-1]) * 100, 1)}, "setups": {}}


def stat(r, dy, m):
    r = r[m]; dy = dy[m]
    if len(r) < 30:
        return None
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": int(len(r)), "win": round(float((r > 0).mean()) * 100, 1), "avg": round(float(r.mean()) * 100, 2),
            "med": round(float(np.median(r)) * 100, 2), "pf": round(float(g / l), 2) if l > 0 else None,
            "days": round(float(dy.mean()), 1), "worst5": round(float(np.percentile(r, 5)) * 100, 1)}


for name, sig in S.items():
    m = sig.fillna(False).values.copy()
    m[:250] = False; m[nd - 31:] = False                # 지표 안정 구간 이후, 청산 여유
    d, c = np.nonzero(m)
    today = int(np.nansum(sig.fillna(False).values[-1]))
    if len(d) < 50:
        res["setups"][name] = {"n": int(len(d)), "today": today}
        log(name, "신호 부족", len(d)); continue
    # 하루에 같은 셋업이 너무 많이 몰리면 거래대금 상위 20개만(실제 추천 형태)
    df = pd.DataFrame({"d": d, "c": c, "a": amt20.values[d, c]}).sort_values(["d", "a"], ascending=[True, False]).groupby("d").head(20)
    d, c = df.d.values, df.c.values
    g, ro, yr, big_ = gate[d], risk_off[d], years[d], BIG[d, c]
    out = {"n": int(len(d)), "today": today, "per_year": round(len(d) / (len(np.unique(yr)) or 1), 0), "rules": {}}
    for rn, rule in RULES.items():
        r, dy = exits(d, c, rule)
        ok = np.isfinite(r)
        rec = {"all": stat(r, dy, ok), "gate_on": stat(r, dy, ok & g), "gate_off": stat(r, dy, ok & ~g),
               "risk_off": stat(r, dy, ok & ro), "big": stat(r, dy, ok & big_), "small": stat(r, dy, ok & ~big_),
               "recent": stat(r, dy, ok & (yr >= idx[-1].year - 1)),
               "years": {str(y): (lambda s: [s["win"], s["avg"]] if s else None)(stat(r, dy, ok & (yr == y))) for y in np.unique(yr)}}
        out["rules"][rn] = rec
    best = max(out["rules"].items(), key=lambda kv: (kv[1]["all"] or {}).get("win", 0))
    log(f"{name}: n={out['n']} 오늘 {today} | 최고 승률 규칙 {best[0]} {best[1]['all']}")
    res["setups"][name] = out
json.dump(res, open(OUT / "setups.json", "w"), ensure_ascii=False, indent=1, default=float)
log("완료")
