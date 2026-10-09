"""시장 현황."""
import pandas as pd
import streamlit as st

from appkit import market, ui
from appkit.views import common as c


def _tiles():
    cols = st.columns(len(market.INDEX_SYMBOLS))
    for col, name in zip(cols, market.INDEX_SYMBOLS):
        try:
            h = market.index_history(name, 120, c.demo())
            last, prev = h.Close.iloc[-1], h.Close.iloc[-2]
            col.markdown(ui.tile(name, last, last - prev, (last / prev - 1) * 100, dec=2), unsafe_allow_html=True)
        except Exception:
            col.markdown(ui.tile(name, float("nan"), 0, 0), unsafe_allow_html=True)


def _breadth(lst):
    rows = []
    for mk in ("KOSPI", "KOSDAQ"):
        m = lst[lst.Market == mk]
        rows.append((mk, (m.ChangePct > 0).sum(), (m.ChangePct == 0).sum(), (m.ChangePct < 0).sum(),
                     m.Amount.sum()))
    cols = st.columns(2)
    for col, (mk, up, fl, dn, amt) in zip(cols, rows):
        tot = max(up + fl + dn, 1)
        col.markdown(
            f"<div class='tile'><div class='n'>{mk} 종목 등락</div>"
            f"<div style='display:flex;height:10px;margin:8px 0 6px'>"
            f"<div style='width:{up/tot*100:.1f}%;background:{ui.UP}'></div>"
            f"<div style='width:{fl/tot*100:.1f}%;background:#D1D5DB'></div>"
            f"<div style='width:{dn/tot*100:.1f}%;background:{ui.DOWN}'></div></div>"
            f"<div class='c'><span class='up'>상승 {up:,}</span> · <span class='flat'>보합 {fl:,}</span> · "
            f"<span class='down'>하락 {dn:,}</span> · <span class='flat'>거래대금 {ui.won(amt)}</span></div></div>",
            unsafe_allow_html=True)


def _market_flow():
    st.subheader("투자자별 순매수")
    a, b = st.columns([1, 3])
    mk = a.radio("시장", ["KOSPI", "KOSDAQ"], horizontal=True, key="home_mk")
    days = {"1주": 7, "1개월": 31, "3개월": 92}[b.radio("기간", ["1주", "1개월", "3개월"], horizontal=True, key="home_fd")]
    flow = c.safe(market.investor_flow, mk, days, c.demo())
    if flow is None:
        return
    inv = ["개인", "외국인", "기관합계", "연기금"]
    today, week = flow.iloc[-1], flow.iloc[-5:].sum()
    total = flow.sum()
    cols = st.columns(4)
    for col, i in zip(cols, inv):
        col.markdown(
            f"<div class='tile'><div class='n'>{i}</div>"
            f"<div class='v {ui.cls(today[i])}'>{ui.signed_won(today[i])}</div>"
            f"<div class='c flat'>최근 5일 <span class='{ui.cls(week[i])}'>{ui.signed_won(week[i])}</span> · "
            f"기간 <span class='{ui.cls(total[i])}'>{ui.signed_won(total[i])}</span></div></div>",
            unsafe_allow_html=True)
    st.plotly_chart(ui.flow_chart(flow, inv, cumulative=True, height=320), width="stretch",
                    theme=None, config={"displayModeBar": False})
    st.caption("누적 순매수 대금 (억 원) — 선이 올라가면 그 주체가 사고 있다는 뜻")


def _rankings(lst):
    st.subheader("순위")
    mk = st.radio("시장", ["전체", "KOSPI", "KOSDAQ"], horizontal=True, key="rank_mk")
    d = lst if mk == "전체" else lst[lst.Market == mk]
    d = d[d.Volume > 0]
    tabs = st.tabs(["상승률", "하락률", "거래대금", "시가총액", "거래량"])
    specs = [("ChangePct", False), ("ChangePct", True), ("Amount", False), ("Marcap", False), ("Volume", False)]
    for tab, (col, asc), key in zip(tabs, specs, ["up", "dn", "amt", "cap", "vol"]):
        with tab:
            t = d.sort_values(col, ascending=asc).head(30)
            view = pd.DataFrame({"Code": t.Code, "종목": t.Name, "시장": t.Market, "현재가": t.Close,
                                 "등락률(%)": t.ChangePct, "거래대금(억)": t.Amount / 1e8, "시총(억)": t.Marcap / 1e8})
            sty = view.style.map(ui.color_change, subset=["등락률(%)"]).format(
                {"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}", "거래대금(억)": "{:,.0f}", "시총(억)": "{:,.0f}"})
            c.selectable_table(view, f"rank_{key}", styler=sty, height=420,
                               column_config={"Code": None})
    st.caption("행을 누르면 종목 상세로 이동합니다.")


def render():
    st.title("시장 현황")
    ui.demo_banner(c.demo())
    _tiles()
    lst = c.listing()
    st.write("")
    _breadth(lst)
    st.write("")
    _market_flow()
    _rankings(lst)
