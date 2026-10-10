"""추천 성적 누적 — 매일 1주·1개월 추천을 기록하고, 이후 실제 시세로 채점

규칙(백테스트와 동일): 추천 다음 거래일 시가에 매수 → 목표가 도달 시 익절, 손절가 이탈 시 손절,
기간(1주=5거래일, 1개월=20거래일)이 지나면 종가에 정리. 비용 0.25% 차감. 같은 날 둘 다 닿으면 손절로 처리.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["date", "code", "name", "rank", "prob", "exp", "src", "ref", "target", "stop",
        "status", "entry", "exit_date", "exit", "ret", "days", "h"]
OBJ = ("exit_date", "status", "src", "name", "date", "code")


def _obj(df):
    for c in OBJ:
        df[c] = df[c].astype(object)
    return df


def load(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype={"code": str, "date": str, "exit_date": str}).reindex(columns=COLS)
    else:
        df = pd.DataFrame(columns=COLS)
    df["h"] = pd.to_numeric(df["h"], errors="coerce").fillna(20).astype(int)     # 예전 기록은 1개월
    return _obj(df)


def add_today(tr: pd.DataFrame, date: str, picks: list, h: int, n: int = 10) -> pd.DataFrame:
    if len(tr) and ((tr.date == date) & (tr.h == h)).any():
        return tr                                        # 같은 날 재실행 시 중복 기록 방지
    rows = [{"date": date, "code": p["code"], "name": p["name"], "rank": i, "prob": round(p["prob"], 4),
             "exp": round(p.get("exp", np.nan), 5), "src": p.get("grade", ""),
             "ref": p["close"], "target": round(p["target"]), "stop": round(p["stop"]),
             "status": "open", "h": h} for i, p in enumerate(picks[:n], 1)]
    if not rows:
        return tr
    return _obj(pd.concat([tr, pd.DataFrame(rows, columns=COLS)], ignore_index=True))


def update(tr: pd.DataFrame, prices: dict, cost: float = 0.0025) -> pd.DataFrame:
    tr = _obj(tr.copy())
    for i, r in tr[tr.status == "open"].iterrows():
        df = prices.get(r.code)
        if df is None:
            continue
        H = int(r.h) if pd.notna(r.h) else 20
        after = df[df.index > pd.Timestamp(r.date)]
        if after.empty:
            continue
        entry = float(after.Open.iloc[0])
        if not entry > 0:
            continue
        tr.at[i, "entry"] = entry
        done = None
        for k, (t, row) in enumerate(after.iloc[:H].iterrows(), 1):
            if row.Low <= r.stop:
                done = (t, min(row.Open, r.stop) if k > 1 else min(entry, r.stop), k)
                break
            if row.High >= r.target:
                done = (t, max(row.Open, r.target) if k > 1 else max(entry, r.target), k)
                break
        if done is None and len(after) >= H:
            done = (after.index[H - 1], float(after.Close.iloc[H - 1]), H)
        if done:
            t, px, k = done
            ret = float(px) / entry - 1 - cost
            tr.at[i, "status"] = "win" if ret > 0 else "loss"
            tr.at[i, "exit_date"] = t.strftime("%Y-%m-%d")
            tr.at[i, "exit"] = float(px)
            tr.at[i, "ret"] = ret
            tr.at[i, "days"] = k
        else:
            tr.at[i, "ret"] = float(after.Close.iloc[-1]) / entry - 1        # 진행 중: 현재 평가 수익
            tr.at[i, "days"] = len(after)
    return tr


def _summary(tr: pd.DataFrame) -> dict:
    closed = tr[tr.status.isin(["win", "loss"])]
    opened = tr[tr.status == "open"]
    s = {"total": int(len(tr)), "closed": int(len(closed)), "open": int(len(opened)),
         "days": int(tr.date.nunique()) if len(tr) else 0, "since": tr.date.min() if len(tr) else None}
    if len(closed):
        s.update(win=float((closed.ret > 0).mean()), avg=float(closed.ret.mean()),
                 expected=float(pd.to_numeric(closed.prob).mean()),
                 expectedAvg=float(pd.to_numeric(closed.exp).mean()) if closed.exp.notna().any() else None,
                 best=float(closed.ret.max()), worst=float(closed.ret.min()))
        m = closed.assign(m=closed.date.str[:7]).groupby("m")
        s["monthly"] = [[k, float(g.ret.mean()), float((g.ret > 0).mean()), int(len(g))] for k, g in m]

    def rows(df):
        return [[r.date, r.code, r.name, int(r.rank), round(float(r.prob) * 100, 1), r.ref, r.target, r.stop,
                 r.status, None if pd.isna(r.entry) else float(r.entry),
                 None if pd.isna(r.ret) else round(float(r.ret) * 100, 2),
                 None if pd.isna(r.days) else int(r.days), None if pd.isna(r.exit_date) else r.exit_date,
                 None if pd.isna(r.exp) else round(float(r.exp) * 100, 2)]
                for r in df.itertuples()]
    s["openRows"] = rows(opened.sort_values(["date", "rank"], ascending=[False, True]).head(40))
    s["closedRows"] = rows(closed.sort_values("exit_date", ascending=False).head(40))
    return s


def summary(tr: pd.DataFrame) -> dict:
    return {"h": {str(h): _summary(tr[tr.h == h]) for h in (5, 20)}}
