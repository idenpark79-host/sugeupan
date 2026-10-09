"""차트 생성."""
from __future__ import annotations

import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib import font_manager
import numpy as np
import pandas as pd

NAVY, GREY, LIGHT, GRID = "#1F2A44", "#6B7280", "#C9CDD3", "#ECEEF1"
UP, DOWN = "#F5222D", "#1677FF"           # 주식창 관례: 상승 밝은 빨강, 하락 밝은 파랑
MA_STYLE = {"MA5": ("#A0A7B2", 0.9), "MA20": ("#E08A00", 1.2), "MA60": ("#12A150", 1.1), "MA120": ("#7A5AF8", 1.1)}


def _set_font():
    for f in ("Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic",
              "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9,
                         "axes.edgecolor": LIGHT, "axes.linewidth": 0.6,
                         "xtick.color": GREY, "ytick.color": GREY, "axes.labelcolor": GREY})


_set_font()


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def _price_fmt(v, _=None):
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def _candles(ax, d, x):
    up = d["Close"] >= d["Open"]
    colors = np.where(up, UP, DOWN)
    ax.vlines(x, d["Low"], d["High"], color=colors, linewidth=0.8)
    body_lo = np.minimum(d["Open"], d["Close"])
    body_h = np.maximum((d["Close"] - d["Open"]).abs(), d["Close"] * 1e-4)
    ax.bar(x, body_h, bottom=body_lo, width=0.7, color=colors, linewidth=0)
    return colors


def _xdates(ax, d):
    ticks = np.linspace(0, len(d) - 1, 7, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([d.index[i].strftime("%y.%m.%d") for i in ticks])


def _last_price_tag(ax, d, x):
    c, pc = d["Close"].iloc[-1], d["Close"].iloc[-2]
    col = UP if c >= pc else DOWN
    ax.axhline(c, color=col, linewidth=0.7, linestyle=":")
    ax.annotate(_price_fmt(c), xy=(x[-1], c), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8.5, color="white", fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.25", fc=col, ec=col), annotation_clip=False)


def stock_chart(df: pd.DataFrame, title: str, days: int = 120) -> str:
    d = df.iloc[-days:]
    x = np.arange(len(d))
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [3.4, 1, 1, 1], "hspace": 0.08})
    ax, axv, axr, axm = axes
    ax.fill_between(x, d["BB_lower"], d["BB_upper"], color="#F2F3F5", linewidth=0, label="볼린저밴드")
    colors = _candles(ax, d, x)
    for col, (c, lw) in MA_STYLE.items():
        if d[col].notna().any():
            ax.plot(x, d[col], color=c, linewidth=lw, label=col)
    _last_price_tag(ax, d, x)
    ax.set_title(title, loc="left", fontsize=12, color=NAVY, fontweight="bold", pad=8)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=5, fontsize=8, frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_price_fmt))

    if d["Volume"].sum() > 0:
        axv.bar(x, d["Volume"], width=0.7, color=colors, alpha=0.6, linewidth=0)
        axv.plot(x, d["VOL_MA20"], color=GREY, linewidth=0.9)
    axv.set_ylabel("거래량")
    axv.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))

    axr.plot(x, d["RSI"], color=NAVY, linewidth=1)
    axr.axhline(70, color=UP, linewidth=0.7, linestyle="--", alpha=0.6)
    axr.axhline(30, color=DOWN, linewidth=0.7, linestyle="--", alpha=0.6)
    axr.set_ylim(0, 100)
    axr.set_ylabel("RSI")

    axm.bar(x, d["MACD_hist"], width=0.7, color=np.where(d["MACD_hist"] >= 0, UP, DOWN), alpha=0.45, linewidth=0)
    axm.plot(x, d["MACD"], color=NAVY, linewidth=1, label="MACD")
    axm.plot(x, d["MACD_signal"], color="#E08A00", linewidth=1, label="Signal")
    axm.set_ylabel("MACD")
    axm.legend(loc="upper left", ncol=2, fontsize=7, frameon=False)
    for a in axes:
        _style(a)
    _xdates(axm, d)
    return _to_b64(fig)


def pick_chart(df: pd.DataFrame, title: str, target: float, stop: float, days: int = 120) -> str:
    """추천 종목: 캔들 + 이동평균 + 목표가/손절가 라인 + 거래량."""
    d = df.iloc[-days:]
    x = np.arange(len(d))
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06})
    colors = _candles(ax, d, x)
    for col, (c, lw) in MA_STYLE.items():
        if col != "MA5" and d[col].notna().any():
            ax.plot(x, d[col], color=c, linewidth=lw, label=col)
    xr = len(d) - 1
    for lvl, col, lab in ((target, UP, "목표가"), (stop, DOWN, "손절가")):
        ax.axhline(lvl, color=col, linewidth=1.1, linestyle="--")
        ax.annotate(f"{lab} {_price_fmt(lvl)}", xy=(xr, lvl), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=col, fontweight="bold", annotation_clip=False)
    ax.axhspan(stop, target, color="#F7F8FA", zorder=0)
    _last_price_tag(ax, d, x)
    lo, hi = min(d["Low"].min(), stop), max(d["High"].max(), target)
    ax.set_ylim(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.06)
    ax.set_title(title, loc="left", fontsize=11.5, color=NAVY, fontweight="bold", pad=6)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=8, frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_price_fmt))
    axv.bar(x, d["Volume"], width=0.7, color=colors, alpha=0.6, linewidth=0)
    axv.set_ylabel("거래량")
    axv.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))
    for a in (ax, axv):
        _style(a)
    _xdates(axv, d)
    return _to_b64(fig)


def return_hist(rets: np.ndarray, title: str) -> str:
    """기법 신호 후 거래 수익률 분포 — 이익 빨강, 손실 파랑."""
    r = rets * 100
    fig, ax = plt.subplots(figsize=(10, 2.6))
    lo, hi = np.percentile(r, 1), np.percentile(r, 99)
    bins = np.linspace(min(lo, -1), max(hi, 1), 41)
    n, edges, patches = ax.hist(np.clip(r, bins[0], bins[-1]), bins=bins, linewidth=0)
    for p, e in zip(patches, edges[:-1]):
        p.set_facecolor(UP if e >= 0 else DOWN)
        p.set_alpha(0.85)
    ax.axvline(0, color=NAVY, linewidth=0.8)
    ax.axvline(r.mean(), color=NAVY, linewidth=1, linestyle="--")
    ax.annotate(f"평균 {r.mean():+.2f}%", xy=(r.mean(), n.max()), xytext=(6, -4),
                textcoords="offset points", fontsize=8.5, color=NAVY, va="top")
    ax.set_title(title, loc="left", fontsize=10.5, color=NAVY, fontweight="bold")
    ax.set_xlabel("거래 1회당 수익률 (%, 비용 차감)")
    ax.set_ylabel("횟수")
    _style(ax)
    return _to_b64(fig)


def index_compare_chart(series: dict[str, pd.Series], days: int = 63) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    palette = [NAVY, "#E08A00", "#12A150", "#7A5AF8", GREY]
    for i, (name, s) in enumerate(series.items()):
        s = s.iloc[-days:]
        ax.plot(s.index, (s / s.iloc[0] - 1) * 100, color=palette[i % len(palette)],
                linewidth=1.5 if i == 0 else 1.1, label=name)
    ax.axhline(0, color=LIGHT, linewidth=0.8)
    ax.set_ylabel("누적 수익률 (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series), fontsize=8, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%d"))
    _style(ax)
    return _to_b64(fig)
