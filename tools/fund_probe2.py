"""site_build.Source.fund_hist + rating 재무 특징이 실제 데이터에서 동작하는지 점검."""
import json, sys, traceback, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
Path("research").mkdir(exist_ok=True)
out = {}
try:
    import pandas as pd, site_build as sb, rating
    src = sb.Source(False)
    lst = src.listing(); out["listing_cols"] = list(map(str, lst.columns))
    codes = lst.sort_values("Marcap", ascending=False).Code.head(60).tolist()
    t = time.time(); pr = src.scan_prices(codes, 3); out["prices"] = [len(pr), round(time.time() - t)]
    idx = next(iter(pr.values())).index; out["idx"] = [str(idx[0]), str(idx[-1]), str(idx.dtype)]
    t = time.time()
    try:
        fund = src.fund_hist(idx); out["fund"] = [len(fund or {}), round(time.time() - t)]
        if fund: k = max(fund); out["fund_last"] = [str(k), list(fund[k].columns), int(len(fund[k]))]
    except Exception:
        out["fund_err"] = traceback.format_exc()[-2000:]; fund = None
    try:
        rt = rating.build(pr, dict(zip(lst.Code, lst.Marcap)), dict(zip(lst.Code, lst.Name)), src.sectors(lst), fund=fund, log=lambda *a: out.setdefault("log", []).append(" ".join(map(str, a))))
        out["value"] = rt["meta"].get("value")
    except Exception:
        out["rating_err"] = traceback.format_exc()[-3000:]
except Exception:
    out["fatal"] = traceback.format_exc()[-3000:]
json.dump(out, open("research/fund_probe2.json", "w"), ensure_ascii=False, indent=1, default=str)
