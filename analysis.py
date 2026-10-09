"""동향 판정, 매매 신호 탐지, 시황 요약 (규칙 기반)."""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Result:
    ticker: str
    name: str
    close: float
    returns: dict            # {"1D": %, "1W": %, ...}
    trend: str               # 강세 / 약세 / 중립 ...
    score: int               # -6 ~ +6
    alignment: str           # 정배열 / 역배열 / 혼조
    rsi: float
    signals: list = field(default_factory=list)
    comment: str = ""
    df: pd.DataFrame | None = None
    prev: float = float("nan")


def _ret(c: pd.Series, n: int) -> float | None:
    return (c.iloc[-1] / c.iloc[-1 - n] - 1) * 100 if len(c) > n else None


def _crossed(a: pd.Series, b: pd.Series, lookback: int) -> str | None:
    diff = (a - b).dropna().iloc[-(lookback + 1):]
    if len(diff) < 2:
        return None
    sign = diff.gt(0)
    for i in range(len(sign) - 1, 0, -1):
        if sign.iloc[i] != sign.iloc[i - 1]:
            return "up" if sign.iloc[i] else "down"
    return None


def analyze(ticker: str, name: str, df: pd.DataFrame, p: dict) -> Result:
    last = df.iloc[-1]
    c = df["Close"]
    lb = p.get("cross_lookback_days", 5)

    returns = {"1D": _ret(c, 1), "1W": _ret(c, 5), "1M": _ret(c, 21), "3M": _ret(c, 63)}
    ytd = c[c.index.year == c.index[-1].year]
    returns["YTD"] = (c.iloc[-1] / ytd.iloc[0] - 1) * 100 if len(ytd) > 1 else None

    score, signals = 0, []

    # 1) 이동평균 위치·배열
    for n in (20, 60, 120):
        ma = last.get(f"MA{n}")
        if pd.notna(ma):
            score += 1 if last["Close"] > ma else -1
    mas = [last.get(f"MA{n}") for n in (5, 20, 60, 120)]
    if all(pd.notna(m) for m in mas):
        if mas == sorted(mas, reverse=True):
            alignment, score = "정배열", score + 1
        elif mas == sorted(mas):
            alignment, score = "역배열", score - 1
        else:
            alignment = "혼조"
    else:
        alignment = "판단 불가"

    # 2) 20일선 기울기
    ma20 = df["MA20"].dropna()
    if len(ma20) > 5:
        slope = (ma20.iloc[-1] / ma20.iloc[-6] - 1) * 100
        score += 1 if slope > 0.3 else -1 if slope < -0.3 else 0

    # 3) 이벤트성 신호
    x = _crossed(df["MA5"], df["MA20"], lb)
    if x == "up":
        signals.append(("골든크로스", "5일선이 20일선 상향 돌파", +1))
    elif x == "down":
        signals.append(("데드크로스", "5일선이 20일선 하향 이탈", -1))

    x = _crossed(df["MACD"], df["MACD_signal"], lb)
    if x == "up":
        signals.append(("MACD 매수전환", "MACD가 시그널선 상향 돌파", +1))
    elif x == "down":
        signals.append(("MACD 매도전환", "MACD가 시그널선 하향 이탈", -1))

    rsi = float(last["RSI"]) if pd.notna(last["RSI"]) else float("nan")
    if rsi >= p.get("rsi_overbought", 70):
        signals.append(("RSI 과매수", f"RSI {rsi:.0f} — 단기 과열", 0))
    elif rsi <= p.get("rsi_oversold", 30):
        signals.append(("RSI 과매도", f"RSI {rsi:.0f} — 반등 가능 구간", 0))

    if last["Close"] > last["BB_upper"]:
        signals.append(("밴드 상단 돌파", "볼린저밴드 상단 위에서 마감", 0))
    elif last["Close"] < last["BB_lower"]:
        signals.append(("밴드 하단 이탈", "볼린저밴드 하단 아래에서 마감", 0))

    vol_ma = last.get("VOL_MA20")
    if pd.notna(vol_ma) and vol_ma > 0:
        ratio = last["Volume"] / vol_ma
        if ratio >= p.get("volume_surge_ratio", 2.0):
            signals.append(("거래량 급증", f"20일 평균 대비 {ratio:.1f}배", 0))

    hi52, lo52 = c.iloc[-252:].max(), c.iloc[-252:].min()
    if last["Close"] >= hi52 * 0.98:
        signals.append(("52주 신고가 근접", f"고점 {hi52:,.2f} 대비 {(last['Close']/hi52-1)*100:+.1f}%", +1))
    elif last["Close"] <= lo52 * 1.02:
        signals.append(("52주 신저가 근접", f"저점 {lo52:,.2f} 대비 {(last['Close']/lo52-1)*100:+.1f}%", -1))

    score += sum(s[2] for s in signals)
    score = max(-6, min(6, score))
    trend = ("강한 상승" if score >= 4 else "상승" if score >= 2 else
             "강한 하락" if score <= -4 else "하락" if score <= -2 else "중립")

    res = Result(ticker, name, float(last["Close"]), returns, trend, score,
                 alignment, rsi, [(s[0], s[1]) for s in signals], df=df)
    res.prev = float(c.iloc[-2])
    res.comment = _comment(res)
    return res


def _fmt(v):
    return "-" if v is None else f"{v:+.1f}%"


def _comment(r: Result) -> str:
    parts = [f"1개월 {_fmt(r.returns['1M'])}, 3개월 {_fmt(r.returns['3M'])}.",
             f"이동평균 {r.alignment}, RSI {r.rsi:.0f}으로 추세 판정은 '{r.trend}'."]
    if r.signals:
        parts.append("주요 신호: " + ", ".join(s[0] for s in r.signals) + ".")
    return " ".join(parts)


def market_summary(index_results: list[Result], stock_results: list[Result]) -> list[str]:
    lines = []
    if index_results:
        movers = ", ".join(f"{r.name} {_fmt(r.returns['1D'])}" for r in index_results)
        lines.append(f"전일 대비 {movers}.")
        eq = [r for r in index_results if "환율" not in r.name]
        up = sum(1 for r in eq if (r.returns["1W"] or 0) > 0)
        tone = "위험선호" if up >= len(eq) * 0.75 else "위험회피" if up <= len(eq) * 0.25 else "혼조"
        lines.append(f"주간 기준 주요 지수 {len(eq)}개 중 {up}개 상승 — 전반적 분위기는 {tone}.")
        fx = next((r for r in index_results if "환율" in r.name), None)
        if fx and fx.returns["1W"] is not None:
            d = "상승(원화 약세)" if fx.returns["1W"] > 0 else "하락(원화 강세)"
            lines.append(f"원/달러 환율은 주간 {fx.returns['1W']:+.1f}% {d}.")
    if stock_results:
        strong = [r.name for r in stock_results if r.score >= 2]
        weak = [r.name for r in stock_results if r.score <= -2]
        lines.append(f"관심종목 {len(stock_results)}개 중 상승 추세 {len(strong)}개"
                     + (f"({', '.join(strong)})" if strong else "")
                     + f", 하락 추세 {len(weak)}개" + (f"({', '.join(weak)})" if weak else "") + ".")
    return lines
