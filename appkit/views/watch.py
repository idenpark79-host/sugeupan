"""관심종목 · 가격 알림."""
import pandas as pd
import streamlit as st

from appkit import market, ui
from appkit.views import common as c


def _triggered():
    u = c.user()
    try:
        lst = market.listing(c.demo()).set_index("Code")
    except Exception:
        return []
    out = []
    for code, al in u["alerts"].items():
        if code not in lst.index:
            continue
        px = lst.at[code, "Close"]
        if al.get("above") and px >= al["above"]:
            out.append((code, lst.at[code, "Name"], px, "이상", al["above"]))
        if al.get("below") and px <= al["below"]:
            out.append((code, lst.at[code, "Name"], px, "이하", al["below"]))
    return out


def check_alerts_toast():
    """앱을 열 때 한 번, 도달한 알림을 알려 준다."""
    if st.session_state.get("_alert_checked"):
        return
    st.session_state._alert_checked = True
    for code, name, px, kind, lv in _triggered():
        st.toast(f"{name} {ui.num(px)} — 알림가 {ui.num(lv)} {kind} 도달")


def render():
    st.title("관심종목 · 알림")
    ui.demo_banner(c.demo())
    u = c.user()
    lst = c.listing()

    add = st.selectbox("관심종목 추가", (lst.Name + " (" + lst.Code + ")").tolist(), index=None,
                       placeholder="종목명 또는 코드 검색")
    if add:
        code = add[-7:-1]
        if code not in u["watchlist"]:
            u["watchlist"].append(code)
            c.save()
            st.rerun()

    hits = _triggered()
    if hits:
        st.markdown("**도달한 알림**")
        for code, name, px, kind, lv in hits:
            st.markdown(f"- **{name}** 현재 {ui.num(px)} — 설정가 {ui.num(lv)} {kind}")

    w = lst[lst.Code.isin(u["watchlist"])].set_index("Code").reindex(u["watchlist"]).dropna(subset=["Name"])
    if w.empty:
        st.caption("관심종목이 없습니다. 위에서 추가하거나 종목 상세 화면에서 [관심 추가]를 누르세요.")
        return
    trend = {}
    for code in w.index:
        try:
            trend[code] = market.history(code, 45, c.demo()).Close.iloc[-22:].tolist()
        except Exception:
            trend[code] = []
    view = pd.DataFrame({
        "Code": w.index, "종목": w.Name.values, "현재가": w.Close.values, "전일 대비": w.Change.values,
        "등락률(%)": w.ChangePct.values, "1개월 추이": [trend[c_] for c_ in w.index],
        "거래대금(억)": (w.Amount / 1e8).values,
        "알림 이상": [u["alerts"].get(c_, {}).get("above") or float("nan") for c_ in w.index],
        "알림 이하": [u["alerts"].get(c_, {}).get("below") or float("nan") for c_ in w.index]})
    sty = view.style.map(ui.color_change, subset=["전일 대비", "등락률(%)"]).format(
        {"현재가": "{:,.0f}", "전일 대비": "{:+,.0f}", "등락률(%)": "{:+.2f}", "거래대금(억)": "{:,.0f}",
         "알림 이상": "{:,.0f}", "알림 이하": "{:,.0f}"}, na_rep="-")
    c.selectable_table(view, "watch_tbl", styler=sty, column_config={
        "Code": None, "1개월 추이": st.column_config.LineChartColumn("1개월 추이", width="small")})
    st.caption("행을 누르면 종목 상세로 이동합니다. 알림 가격은 종목 상세 화면에서 설정합니다.")

    with st.expander("관심종목 삭제"):
        rm = st.multiselect("삭제할 종목", w.index.tolist(), format_func=lambda x: f"{w.at[x, 'Name']} ({x})")
        if st.button("삭제") and rm:
            u["watchlist"] = [x for x in u["watchlist"] if x not in rm]
            for x in rm:
                u["alerts"].pop(x, None)
            c.save()
            st.rerun()
