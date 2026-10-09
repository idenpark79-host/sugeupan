"""투자자 매매동향 — 시장 전체 추이와 투자자별 순매수·순매도 상위 종목."""
import pandas as pd
import streamlit as st

from appkit import market, ui
from appkit.views import common as c


def render():
    st.title("투자자 매매동향")
    ui.demo_banner(c.demo())
    if not c.has_krx():
        c.krx_notice()

    a, b, d = st.columns([1.2, 2.6, 1.6])
    mk = a.radio("시장", ["KOSPI", "KOSDAQ"], horizontal=True)
    inv = b.radio("투자자", ["외국인", "기관합계", "연기금", "개인", "금융투자", "투신", "사모"], horizontal=True)
    days = {"1일": 1, "1주": 7, "1개월": 31, "3개월": 92}[d.radio("기간", ["1일", "1주", "1개월", "3개월"],
                                                               index=1, horizontal=True)]

    flow = c.safe(market.investor_flow, mk, max(days, 31), c.demo())
    if flow is not None:
        st.subheader(f"{mk} 시장 — 투자자별 일별 순매수")
        st.plotly_chart(ui.flow_chart(flow.iloc[-max(days, 20):], ["개인", "외국인", "기관합계", "연기금"],
                                      cumulative=False, height=300),
                        width="stretch", theme=None, config={"displayModeBar": False})

    rank = c.safe(market.net_purchase_rank, mk, inv, days, c.demo())
    if rank is None:
        return
    st.subheader(f"{inv} 순매수 · 순매도 상위 ({mk}, 최근 {days}일)")
    top, bot = rank.head(20), rank.tail(20).iloc[::-1]
    l, r = st.columns(2)
    for col, t, title in ((l, top, "순매수 상위"), (r, bot, "순매도 상위")):
        with col:
            st.markdown(f"**{title}**")
            st.plotly_chart(ui.hbar(t.Name.tolist()[:10], (t.순매수대금 / 1e8).tolist()[:10], height=330),
                            width="stretch", theme=None, config={"displayModeBar": False})
            lst = c.listing().set_index("Code")
            view = pd.DataFrame({"Code": t.Code.values, "종목": t.Name.values,
                                 "순매수(억)": (t.순매수대금 / 1e8).values,
                                 "현재가": t.Code.map(lst.Close).values,
                                 "등락률(%)": t.Code.map(lst.ChangePct).values})
            sty = view.style.map(ui.color_change, subset=["순매수(억)", "등락률(%)"]).format(
                {"순매수(억)": "{:+,.0f}", "현재가": "{:,.0f}", "등락률(%)": "{:+.2f}"}, na_rep="-")
            c.selectable_table(view, f"fr_{title}", styler=sty, height=420, column_config={"Code": None})
    st.caption("행을 누르면 종목 상세로 이동합니다. 순매수 = 매수 대금 - 매도 대금.")
