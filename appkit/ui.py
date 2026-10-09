"""공통 화면 요소 — 스타일, 숫자 표기, 시세 헤더, 차트."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import indicators

UP, DOWN, FLAT = "#F5222D", "#1677FF", "#6B7280"
NAVY, GREY, GRID = "#1F2A44", "#6B7280", "#ECEEF1"
FONT = "'Malgun Gothic','맑은 고딕','Apple SD Gothic Neo',sans-serif"
INV_COLOR = {"개인": "#9CA3AF", "외국인": "#E08A00", "기관합계": NAVY, "연기금": "#12A150",
             "금융투자": "#7A5AF8", "투신": "#0EA5A4", "사모": "#B45309", "보험": "#64748B",
             "은행": "#A16207", "기타법인": "#D1D5DB"}

CSS = f"""
<style>
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, textarea, select {{font-family:{FONT} !important}}
.block-container {{padding-top:2.2rem; max-width:1280px}}
h1,h2,h3 {{color:{NAVY}; letter-spacing:-0.3px}}
.up {{color:{UP}}} .down {{color:{DOWN}}} .flat {{color:{FLAT}}}
.tile {{border:1px solid #E5E7EB; padding:10px 14px; background:#fff}}
.tile .n {{font-size:12.5px; color:#4B5563; font-weight:600}}
.tile .v {{font-size:23px; font-weight:800; letter-spacing:-0.5px; line-height:1.25}}
.tile .c {{font-size:12.5px; font-weight:700}}
.ph .nm {{font-size:24px; font-weight:800; color:{NAVY}}}
.ph .sub {{font-size:13px; color:#6B7280}}
.ph .px {{font-size:40px; font-weight:800; letter-spacing:-1px; line-height:1.1}}
.ph .ch {{font-size:16px; font-weight:700}}
.kv {{display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); border:1px solid #E5E7EB; background:#fff}}
.kv>div {{padding:8px 12px; border-right:1px solid #F0F1F3; border-bottom:1px solid #F0F1F3}}
.kv .k {{font-size:12px; color:#6B7280}} .kv .v {{font-size:15px; font-weight:700}}
.demo {{background:#F3F4F6; border:1px solid #E5E7EB; padding:6px 12px; font-size:13px; color:#374151; margin-bottom:10px}}
.newsitem {{padding:8px 0; border-bottom:1px solid #F0F1F3}}
.newsitem a {{color:{NAVY}; font-weight:600; text-decoration:none}}
.newsitem .m {{font-size:12px; color:#6B7280}}
.bar {{position:relative; height:14px; background:#ECEEF1}}
.bar .f {{position:absolute; left:0; top:0; bottom:0; background:{NAVY}}}
.bar .b {{position:absolute; top:-3px; bottom:-3px; width:2px; background:{UP}}}
</style>
"""


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)


def demo_banner(demo: bool):
    if demo:
        st.markdown("<div class='demo'><b>데모 모드</b> — 가상 시세입니다. 설정에서 끄면 실제 데이터를 사용합니다.</div>",
                    unsafe_allow_html=True)


def cls(v) -> str:
    return "up" if v > 0 else "down" if v < 0 else "flat"


def num(v, dec=None) -> str:
    if v is None or v != v:
        return "-"
    if dec is None:
        dec = 0 if abs(v) >= 1000 else 2
    return f"{v:,.{dec}f}"


def won(v) -> str:
    """원 단위 금액을 억/조로."""
    if v is None or v != v:
        return "-"
    a = abs(v)
    s = "-" if v < 0 else ""
    if a >= 1e12:
        return f"{s}{a/1e12:,.2f}조"
    if a >= 1e8:
        return f"{s}{a/1e8:,.0f}억"
    if a >= 1e4:
        return f"{s}{a/1e4:,.0f}만"
    return f"{v:,.0f}"


def signed_won(v) -> str:
    return ("+" if v > 0 else "") + won(v)


def chg_html(diff, pct, dec=None) -> str:
    d = num(diff, dec)
    return f"<span class='{cls(pct)}'>{'+' if diff > 0 else ''}{d} ({pct:+.2f}%)</span>"


def tile(name, value, diff, pct, dec=None) -> str:
    if dec is None and value == value:
        dec = 0 if abs(value) >= 1000 else 2
    return (f"<div class='tile'><div class='n'>{html.escape(name)}</div>"
            f"<div class='v {cls(pct)}'>{num(value, dec)}</div><div class='c'>{chg_html(diff, pct, dec)}</div></div>")


def price_header(name, code, market, close, diff, pct, extra=""):
    st.markdown(
        f"<div class='ph'><div class='nm'>{html.escape(name)} <span class='sub'>{code} · {market}{extra}</span></div>"
        f"<div class='px {cls(pct)}'>{num(close)}</div><div class='ch'>{chg_html(diff, pct, 0 if close >= 1000 else 2)}</div></div>",
        unsafe_allow_html=True)


def kv_grid(items: list[tuple[str, str]]):
    st.markdown("<div class='kv'>" + "".join(f"<div><div class='k'>{k}</div><div class='v'>{v}</div></div>"
                                              for k, v in items) + "</div>", unsafe_allow_html=True)


def prob_bar(p, base=None) -> str:
    b = f"<span class='b' style='left:{base*100:.1f}%'></span>" if base is not None else ""
    return f"<div class='bar'><span class='f' style='width:{p*100:.1f}%'></span>{b}</div>"


def _layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=60, t=30, b=36), paper_bgcolor="white",
                      plot_bgcolor="white", font=dict(family=FONT, size=12, color="#374151"),
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", font=dict(size=11)),
                      hovermode="x unified", dragmode="pan")
    fig.update_xaxes(gridcolor=GRID, showline=False, rangeslider_visible=False)
    fig.update_yaxes(gridcolor=GRID, side="right", tickformat=",")
    return fig


def candle_chart(df: pd.DataFrame, show: list[str], levels: dict | None = None, height=620):
    """df: OHLCV. show: ['MA','BB','거래량','RSI','MACD'] 중 선택."""
    d = indicators.add_indicators(df)
    sub = [s for s in ("거래량", "RSI", "MACD") if s in show]
    rows = 1 + len(sub)
    heights = [0.58] + [0.42 / max(len(sub), 1)] * len(sub) if sub else [1]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    x = d.index.strftime("%Y-%m-%d")
    fig.add_trace(go.Candlestick(x=x, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="시세",
                                 increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN)), 1, 1)
    if "MA" in show:
        for n, c in ((5, "#A0A7B2"), (20, "#E08A00"), (60, "#12A150"), (120, "#7A5AF8")):
            fig.add_trace(go.Scatter(x=x, y=d[f"MA{n}"], name=f"MA{n}", line=dict(color=c, width=1.2)), 1, 1)
    if "BB" in show:
        fig.add_trace(go.Scatter(x=x, y=d.BB_upper, name="BB 상단", line=dict(color="#C9CDD3", width=1)), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=d.BB_lower, name="BB 하단", line=dict(color="#C9CDD3", width=1),
                                 fill="tonexty", fillcolor="rgba(201,205,211,0.15)"), 1, 1)
    for lab, (val, col) in (levels or {}).items():
        fig.add_hline(y=val, line=dict(color=col, dash="dash", width=1.2), row=1, col=1,
                      annotation_text=f"{lab} {num(val)}", annotation_position="right",
                      annotation_font=dict(color=col, size=11))
    r = 2
    colors = np.where(d.Close >= d.Open, UP, DOWN)
    if "거래량" in sub:
        fig.add_trace(go.Bar(x=x, y=d.Volume, name="거래량", marker_color=colors, opacity=0.6, showlegend=False), r, 1)
        r += 1
    if "RSI" in sub:
        fig.add_trace(go.Scatter(x=x, y=d.RSI, name="RSI", line=dict(color=NAVY, width=1.2), showlegend=False), r, 1)
        fig.add_hline(y=70, line=dict(color=UP, dash="dot", width=1), row=r, col=1)
        fig.add_hline(y=30, line=dict(color=DOWN, dash="dot", width=1), row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1, title_text="RSI")
        r += 1
    if "MACD" in sub:
        fig.add_trace(go.Bar(x=x, y=d.MACD_hist, marker_color=np.where(d.MACD_hist >= 0, UP, DOWN),
                             opacity=0.45, showlegend=False, name="히스토그램"), r, 1)
        fig.add_trace(go.Scatter(x=x, y=d.MACD, name="MACD", line=dict(color=NAVY, width=1.1), showlegend=False), r, 1)
        fig.add_trace(go.Scatter(x=x, y=d.MACD_signal, name="Signal", line=dict(color="#E08A00", width=1.1),
                                 showlegend=False), r, 1)
        fig.update_yaxes(title_text="MACD", row=r, col=1)
    fig.update_xaxes(type="category", nticks=8)
    return _layout(fig, height)


def flow_chart(flow: pd.DataFrame, investors: list[str], cumulative: bool, height=380):
    fig = go.Figure()
    x = flow.index.strftime("%m.%d")
    for inv in investors:
        y = flow[inv].cumsum() if cumulative else flow[inv]
        if cumulative:
            fig.add_trace(go.Scatter(x=x, y=y / 1e8, name=inv, line=dict(color=INV_COLOR[inv], width=1.8)))
        else:
            fig.add_trace(go.Bar(x=x, y=y / 1e8, name=inv, marker_color=INV_COLOR[inv]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category", nticks=10)
    fig.update_yaxes(ticksuffix="억")
    fig.add_hline(y=0, line=dict(color="#9CA3AF", width=1))
    fig = _layout(fig, height)
    fig.update_layout(dragmode=False)
    return fig


def hbar(labels, values, height=420, unit="억"):
    colors = [UP if v > 0 else DOWN for v in values]
    fig = go.Figure(go.Bar(y=labels, x=values, orientation="h", marker_color=colors, cliponaxis=False,
                           text=[f"{v:+,.0f}" for v in values], textposition="outside"))
    fig = _layout(fig, height)
    fig.update_layout(dragmode=False, hovermode="closest", margin=dict(l=10, r=50, t=10, b=30))
    fig.update_yaxes(autorange="reversed", side="left", automargin=True, tickformat=None)
    lo, hi = min(min(values), 0), max(max(values), 0)
    pad = (hi - lo) * 0.18 or 1
    fig.update_xaxes(ticksuffix=unit, automargin=True, range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)])
    return fig


def line_chart(series: dict, height=300, pct=False, title_y=None):
    pal = [NAVY, "#E08A00", "#12A150", "#7A5AF8", GREY]
    fig = go.Figure()
    for i, (n, s) in enumerate(series.items()):
        y = (s / s.iloc[0] - 1) * 100 if pct else s
        fig.add_trace(go.Scatter(x=s.index, y=y, name=n, line=dict(color=pal[i % len(pal)], width=1.6)))
    fig = _layout(fig, height)
    fig.update_layout(dragmode=False, yaxis_title=title_y)
    fig.update_xaxes(tickformat="%y.%m.%d")
    if pct:
        fig.update_yaxes(ticksuffix="%", tickformat=".1f")
    return fig


def color_change(v):
    """st.dataframe Styler용 등락 색상."""
    if isinstance(v, (int, float)) and v == v:
        return f"color:{UP}; font-weight:600" if v > 0 else f"color:{DOWN}; font-weight:600" if v < 0 else ""
    return ""
