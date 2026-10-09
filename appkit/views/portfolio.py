"""보유종목 — 평가금액·손익·비중."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from appkit import ui
from appkit.views import common as c


def render():
    st.title("보유종목")
    ui.demo_banner(c.demo())
    u = c.user()
    lst = c.listing()
    label = dict(zip(lst.Code, lst.Name + " (" + lst.Code + ")"))
    opts = sorted(label.values())

    st.caption("보유 종목·수량·평균 매수가를 입력하면 현재가 기준으로 손익을 계산합니다. 행 추가는 표 아래 + 버튼.")
    base = pd.DataFrame([{"종목": label.get(p["code"], p["code"]), "수량": p["qty"], "평균단가": p["avg"]}
                         for p in u["portfolio"]], columns=["종목", "수량", "평균단가"])
    ed = st.data_editor(base, num_rows="dynamic", width="stretch", key="pf_editor", column_config={
        "종목": st.column_config.SelectboxColumn("종목", options=opts, required=True, width="large"),
        "수량": st.column_config.NumberColumn("수량", min_value=0, step=1, format="%d"),
        "평균단가": st.column_config.NumberColumn("평균단가(원)", min_value=0, step=10, format="%,.0f")})
    if st.button("저장"):
        u["portfolio"] = [{"code": r.종목[-7:-1], "qty": float(r.수량 or 0), "avg": float(r.평균단가 or 0)}
                          for r in ed.dropna(subset=["종목"]).itertuples()]
        c.save()
        st.toast("보유종목을 저장했습니다.")
        st.rerun()

    if not u["portfolio"]:
        return
    px = lst.set_index("Code")
    rows = []
    for p in u["portfolio"]:
        if p["code"] not in px.index or not p["qty"]:
            continue
        cur, chg = px.at[p["code"], "Close"], px.at[p["code"], "Change"]
        buy, val = p["qty"] * p["avg"], p["qty"] * cur
        rows.append({"Code": p["code"], "종목": px.at[p["code"], "Name"], "수량": p["qty"], "평균단가": p["avg"],
                     "현재가": cur, "등락률(%)": px.at[p["code"], "ChangePct"], "매입금액": buy, "평가금액": val,
                     "평가손익": val - buy, "수익률(%)": (val / buy - 1) * 100 if buy else 0,
                     "오늘 손익": p["qty"] * chg})
    if not rows:
        return
    t = pd.DataFrame(rows)
    buy, val, today = t.매입금액.sum(), t.평가금액.sum(), t["오늘 손익"].sum()
    pnl = val - buy
    pct = pnl / buy * 100 if buy else 0
    st.write("")
    cols = st.columns(4)
    cols[0].markdown(f"<div class='tile'><div class='n'>총 평가금액</div><div class='v'>{val:,.0f}</div>"
                     f"<div class='c flat'>매입 {buy:,.0f}</div></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='tile'><div class='n'>평가손익</div><div class='v {ui.cls(pnl)}'>{pnl:+,.0f}</div>"
                     f"<div class='c {ui.cls(pct)}'>{pct:+.2f}%</div></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='tile'><div class='n'>오늘 손익</div><div class='v {ui.cls(today)}'>{today:+,.0f}</div>"
                     f"<div class='c {ui.cls(today)}'>{today / (val - today) * 100 if val - today else 0:+.2f}%</div></div>",
                     unsafe_allow_html=True)
    cols[3].markdown(f"<div class='tile'><div class='n'>보유 종목 수</div><div class='v'>{len(t)}</div>"
                     f"<div class='c flat'>수익 {int((t.평가손익 > 0).sum())} · 손실 {int((t.평가손익 < 0).sum())}</div></div>",
                     unsafe_allow_html=True)
    st.write("")
    sty = t.style.map(ui.color_change, subset=["등락률(%)", "평가손익", "수익률(%)", "오늘 손익"]).format(
        {"수량": "{:,.0f}", "평균단가": "{:,.0f}", "현재가": "{:,.0f}", "등락률(%)": "{:+.2f}", "매입금액": "{:,.0f}",
         "평가금액": "{:,.0f}", "평가손익": "{:+,.0f}", "수익률(%)": "{:+.2f}", "오늘 손익": "{:+,.0f}"})
    c.selectable_table(t, "pf_tbl", styler=sty, column_config={"Code": None})

    a, b = st.columns(2)
    t = t.sort_values("평가금액", ascending=False)
    w = go.Figure(go.Bar(x=t.종목, y=t.평가금액 / val * 100, marker_color=ui.NAVY,
                         text=[f"{v:.1f}%" for v in t.평가금액 / val * 100], textposition="outside"))
    w.update_layout(title="종목별 비중", yaxis_ticksuffix="%")
    a.plotly_chart(ui._layout(w, 320).update_layout(dragmode=False), width="stretch",
                   theme=None, config={"displayModeBar": False})
    a.write("")
    b.plotly_chart(ui.hbar(t.종목.tolist(), (t.평가손익 / 1e4).tolist(), height=320, unit="만"),
                   width="stretch", theme=None, config={"displayModeBar": False})
