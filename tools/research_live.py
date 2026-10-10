"""실데이터(KRX 로그인) 연구용 — 추천 규칙 후보와 투자 주체별 매매 특성을 검증해 research/*.json으로 저장.
GitHub Actions의 'Wake 연구' 워크플로에서만 실행한다(결과는 research-results 브랜치에 저장)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import site_build as sb, rating, picks

OUT = Path("research"); OUT.mkdir(exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
import os
src = sb.Source(bool(os.environ.get("DEMO")))
lst = src.listing()
lst = lst[lst.Code.str.endswith("0") & ~lst.Name.str.contains("스팩|리츠")]
cap = dict(zip(lst.Code, lst.Marcap))
codes = lst[lst.Marcap >= 1e11].Code.tolist()
big = lst[lst.Marcap >= 1e12].Code.tolist()
log("시세", len(codes))
prices = src.scan_prices(codes, 10)
log("수급", len(big))
flows = src.scan_flows({c: prices[c] for c in big if c in prices}, 10)
log("수급 완료", len(flows))

# 1) 투자 주체별 매매 특성 — 20일 순매수 강도 상위 10%일 때 이후 5·20·60일 시장 대비 성과, 매수 지속 기간
G = ["외국인", "기관합계", "연기금", "투신", "사모", "금융투자", "보험", "은행", "기타법인", "개인"]
rows = []
C = pd.DataFrame({c: prices[c].Close for c in flows if c in prices}).sort_index()
O = pd.DataFrame({c: prices[c].Open for c in flows if c in prices}).reindex(C.index)
V = pd.DataFrame({c: prices[c].Volume for c in flows if c in prices}).reindex(C.index)
amt60 = (C * V).rolling(60, min_periods=30).mean()
fw = {h: (C.shift(-h) / O.shift(-1) - 1) for h in (5, 20, 60)}
fx = {h: f.sub(f.mean(axis=1), axis=0) for h, f in fw.items()}
res_g = {}
for g in G:
    net = pd.DataFrame({c: flows[c][g] for c in C.columns if g in flows[c]}).reindex(C.index)
    if net.empty:
        continue
    for w in (5, 20):
        inten = net.rolling(w, min_periods=int(w * .6)).sum() / (amt60 * w)
        rk = inten.rank(axis=1, pct=True)
        top, bot = rk >= .9, rk <= .1
        r = {}
        for h in (5, 20, 60):
            e = fx[h]
            r[str(h)] = [round(float(e[top].stack().mean()) * 100, 2), round(float((fw[h][top].stack() > 0).mean()) * 100, 1),
                         round(float(e[bot].stack().mean()) * 100, 2)]
        # 매수 지속성: 오늘 순매수면 다음 20일 중 순매수 일수 비율
        pos = (net > 0).astype(float).where(net.notna())
        persist = float(pos.shift(-1).rolling(20).mean().shift(-19)[pos == 1].stack().mean())
        res_g[f"{g}_{w}"] = {"after": r, "persist": round(persist * 100, 1)}
    log("주체", g, res_g.get(f"{g}_20"))
json.dump(res_g, open(OUT / "investors.json", "w"), ensure_ascii=False, indent=1)

# 2) 추천 규칙 후보 — 실제 수급 포함 학습
cap_ctx = {}
def grab(c, log=print):
    cap_ctx.update({k: c[k] for k in ("C", "O", "Hh", "L", "nd", "ri", "ci", "d_i", "amt60", "vol60", "midx", "starts", "P", "F", "w", "dates", "codes")})
    return None
picks.build = grab
rating.build(prices, cap, dict(zip(lst.Code, lst.Name)), src.sectors(lst), log=log, flows=flows)
c = cap_ctx
Cn, On = c["C"].values.astype(float), c["O"].values.astype(float)
nd, amt, sig, midx, oos0 = c["nd"], c["amt60"].values, c["vol60"].values, c["midx"].values, c["starts"][0]
P, F = c["P"][["d", "c", "pct"]], c["F"]
dates = c["dates"]
out = {}
for H in (5, 20, 60):
    f = np.full_like(Cn, np.nan); f[:-H - 1] = Cn[H + 1:] / On[1:-H] - 1
    D = P[(P.d >= oos0) & (P.d + H < nd - 1)].copy()
    D = D[amt[D.d.values, D.c.values] >= 2e9]
    D["f"] = f[D.d.values, D.c.values]
    D["big"] = c["w"].values[D.d.values, D.c.values] >= 1e12
    for k in ("f_frg20", "f_inst20", "f_pen20", "f_trust20", "f_pef20", "f_smart", "mm", "amtr", "vol20", "hi52", "t55", "d20"):
        if k in F:
            D[k] = F[k].values[D.d.values, D.c.values]
    def ev(Q, key="pct", n=10):
        Q = Q.sort_values(["d", key], ascending=[True, False]).groupby("d").head(n)
        s = Q.f.dropna()
        mo = pd.Series(Q.f.values).groupby(pd.to_datetime(dates[Q.d.values]).strftime("%Y-%m")).mean()
        return {"n": int(len(s)), "days": int(Q.d.nunique()), "win": round(float((s > 0).mean()) * 100, 1),
                "avg": round(float(s.mean()) * 100, 2), "med": round(float(s.median()) * 100, 2), "mon": round(float((mo > 0).mean()) * 100, 1)}
    base = D[D.pct >= .75]
    V = {"all": {"win": round(float((D.f > 0).mean()) * 100, 1), "avg": round(float(D.f.mean()) * 100, 2)}, "A_base": ev(base)}
    if "f_smart" in D:
        fl = base[base.f_frg20.notna()]
        V["B_flow_cov_base"] = ev(fl)
        V["C_frg_buy"] = ev(fl[fl.f_frg20 > 0])
        V["D_smart2"] = ev(fl[fl.f_smart >= 2])
        V["E_pen_buy"] = ev(fl[fl.f_pen20 > 0])
        V["F_frg_pen"] = ev(fl[(fl.f_frg20 > 0) & (fl.f_pen20 > 0)])
        fl = fl.assign(fr=fl.groupby("d").f_frg20.rank(pct=True) + fl.groupby("d").pct.rank(pct=True))
        V["G_rank_frg+score"] = ev(fl, "fr")
        V["H_smart2_mm6"] = ev(fl[(fl.f_smart >= 2) & (fl.mm >= 6)])
    V["I_mm7"] = ev(base[base.mm >= 7])
    V["J_big"] = ev(base[base.big]); V["K_small"] = ev(base[~base.big])
    out[str(H)] = V
    log("H", H, json.dumps(V, ensure_ascii=False))
json.dump(out, open(OUT / "rules.json", "w"), ensure_ascii=False, indent=1)
log("끝")
