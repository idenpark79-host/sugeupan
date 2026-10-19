"""장중 시세 스냅샷 — 한국거래소 전 종목 현재가(지연)를 live/quick.json으로 저장. 'Wake 장중 시세' 워크플로가 약 20분마다 실행."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pykrx import stock
import universe

now = pd.Timestamp.now(tz="Asia/Seoul")
day = now.strftime("%Y%m%d")
if now.weekday() >= 5:
    sys.exit("주말")
u = universe.krx_listing(day)
if u.empty or u.Close.fillna(0).sum() == 0:
    sys.exit("오늘 시세 없음(휴장일)")
out = {"t": now.isoformat(timespec="minutes"), "day": now.strftime("%Y-%m-%d"),
       "close": bool(now.hour * 60 + now.minute >= 15 * 60 + 40),
       "p": {r.Code: [float(r.Close), round(float(r.ChangePct), 2)] for r in u.itertuples() if r.Close > 0}}
idx = {}
for nm, t in (("KOSPI", "1001"), ("KOSDAQ", "2001")):
    try:
        df = stock.get_index_ohlcv(day, day, t)
        if len(df):
            c = float(df["종가"].iloc[-1]); o = df.get("등락률")
            idx[nm] = [c, round(float(o.iloc[-1]), 2) if o is not None else None]
    except Exception as e:
        print("지수 실패", nm, e)
out["idx"] = idx
Path("live").mkdir(exist_ok=True)
Path("live/quick.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(out["t"], len(out["p"]), "종목", idx)
