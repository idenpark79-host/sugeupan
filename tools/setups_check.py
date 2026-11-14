"""운영용 setups.build를 실데이터로 돌려 결과를 research/setups_check.json에 저장 (점검용)."""
import json, sys, os, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import site_build as sb, setups
OUT = Path("research"); OUT.mkdir(exist_ok=True)
src = sb.Source(bool(os.environ.get("DEMO")))
lst = src.listing()
u = lst[(lst.Marcap >= 1e11) & lst.Code.str.endswith("0") & ~lst.Name.str.contains("스팩|리츠")].sort_values("Marcap", ascending=False)
pr = src.scan_prices(list(u.Code), 10)
fl = src.scan_flows({c: pr[c] for c in u[u.Marcap >= 1e12].Code if c in pr}, 10)
try:
    r = setups.build({c: d[["Open", "High", "Low", "Close", "Volume"]] for c, d in pr.items()}, dict(zip(lst.Code, lst.Marcap)), dict(zip(lst.Code, lst.Name)), flows=fl)
except Exception:
    r = {"error": traceback.format_exc()}
print(json.dumps(r, ensure_ascii=False, default=str)[:3000])
json.dump(r, open(OUT / "setups_check.json", "w"), ensure_ascii=False, indent=1, default=str)
