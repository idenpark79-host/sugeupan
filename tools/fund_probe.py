"""한국거래소 투자지표(BPS·EPS) 조회가 되는지 확인 — research/fund_probe.json."""
import json, traceback
from pathlib import Path
Path("research").mkdir(exist_ok=True)
out = {}
try:
    import pykrx; out["pykrx"] = getattr(pykrx, "__version__", "?")
    from pykrx import stock
    day = stock.get_nearest_business_day_in_a_week(); out["day"] = day
    for name, fn in [("all_today", lambda: stock.get_market_fundamental(day, market="ALL")),
                     ("all_2020", lambda: stock.get_market_fundamental("20200529", market="ALL")),
                     ("kospi_today", lambda: stock.get_market_fundamental(day, market="KOSPI")),
                     ("by_date_005930", lambda: stock.get_market_fundamental("20250101", day, "005930", freq="m"))]:
        try:
            df = fn()
            out[name] = {"rows": int(len(df)), "cols": [str(c) for c in df.columns], "head": json.loads(df.head(3).to_json(force_ascii=False))}
        except Exception as e:
            out[name] = {"err": repr(e), "tb": traceback.format_exc()[-1500:]}
except Exception as e:
    out["fatal"] = traceback.format_exc()[-2000:]
json.dump(out, open("research/fund_probe.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, default=str)[:3000])
