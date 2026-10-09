"""종목 상세 — 시세·차트·투자자 동향·투자지표·기술적 분석·뉴스."""
import pandas as pd
import streamlit as st

import analysis
import indicators
from appkit import market, ui
from appkit.views import common as c

PERIODS = {"3개월": 95, "6개월": 185, "1년": 370, "3년": 1100}


def _search(lst):
    opts = (lst.Name + " (" + lst.Code + ")").tolist()
    code = st.session_state.get("code", "005930")
    idx = next((i for i, o in enumerate(opts) if o.endswith(f"({code})")), 0)
    sel = st.selectbox("종목 검색", opts, index=idx, placeholder="종목명 또는 코드 검색")
    code = sel[-7:-1]
    st.session_state.code = code
    return lst[lst.Code == code].iloc[0]


def _actions(row):
    u = c.user()
    code = row.Code
    a, b, d, e = st.columns([1.1, 1.3, 1.3, 2])
    inw = code in u["watchlist"]
    if a.button("관심 해제" if inw else "관심 추가", width="stretch"):
        u["watchlist"].remove(code) if inw else u["watchlist"].append(code)
        c.save()
        st.rerun()
    al = u["alerts"].get(code, {})
    above = b.number_input("이 가격 이상 알림", value=float(al.get("above", 0)), step=100.0, format="%.0f")
    below = d.number_input("이 가격 이하 알림", value=float(al.get("below", 0)), step=100.0, format="%.0f")
    e.write("")
    if e.button("알림 저장", use_container_width=False):
        u["alerts"][code] = {"above": above, "below": below}
        if code not in u["watchlist"]:
            u["watchlist"].append(code)
        c.save()
        st.toast("알림을 저장했습니다. 관심종목 화면에서 확인할 수 있습니다.")


def _chart(code):
    a, b, d = st.columns([1.4, 1, 2.6])
    per = a.radio("기간", list(PERIODS), index=2, horizontal=True)
    freq = {"일봉": "D", "주봉": "W", "월봉": "M"}[b.radio("주기", ["일봉", "주봉", "월봉"], horizontal=True)]
    show = d.multiselect("보조지표", ["MA", "BB", "거래량", "RSI", "MACD"], default=["MA", "거래량", "RSI"])
    days = max(PERIODS[per], 1100 if freq == "M" else 370 if freq == "W" else 0)
    h = market.history(code, days + 200, c.demo())        # 이동평균 계산용 여유분
    h = market.resample(h, freq)
    n = {"D": int(PERIODS[per] / 365 * 252), "W": int(PERIODS[per] / 7), "M": int(PERIODS[per] / 30)}[freq]
    full = indicators.add_indicators(h)
    fig = ui.candle_chart(h, show)
    fig.update_xaxes(range=[max(len(h) - n, 0) - 0.5, len(h) - 0.5])
    st.plotly_chart(fig, width="stretch", theme=None, config={"scrollZoom": True, "displaylogo": False})
    return full


def _flows(code):
    days = {"1개월": 31, "3개월": 92, "6개월": 185, "1년": 370}[
        st.radio("조회 기간", ["1개월", "3개월", "6개월", "1년"], index=1, horizontal=True, key="sf_days")]
    flow = c.safe(market.investor_flow, code, days, c.demo())
    if flow is None:
        return
    inv = ["외국인", "기관합계", "연기금", "개인"]
    cols = st.columns(4)
    for col, i in zip(cols, inv):
        d1, d5, dn = flow[i].iloc[-1], flow[i].iloc[-5:].sum(), flow[i].sum()
        streak = 0
        for v in flow[i].iloc[::-1]:
            if v == 0 or (streak and (v > 0) != (streak > 0)):
                break
            streak += 1 if v > 0 else -1
        st_txt = f"{abs(streak)}일 연속 {'순매수' if streak > 0 else '순매도'}" if streak else "-"
        col.markdown(f"<div class='tile'><div class='n'>{i}</div>"
                     f"<div class='v {ui.cls(d1)}'>{ui.signed_won(d1)}</div>"
                     f"<div class='c flat'>5일 <span class='{ui.cls(d5)}'>{ui.signed_won(d5)}</span> · "
                     f"기간 <span class='{ui.cls(dn)}'>{ui.signed_won(dn)}</span></div>"
                     f"<div class='c flat'>{st_txt}</div></div>", unsafe_allow_html=True)
    sel = st.multiselect("표시할 투자자", market.INVESTORS, default=inv, key="sf_inv")
    cum = st.toggle("누적으로 보기", value=True, key="sf_cum")
    st.plotly_chart(ui.flow_chart(flow, sel, cumulative=cum), width="stretch",
                    theme=None, config={"displayModeBar": False})
    fh = c.safe(market.foreign_holding, code, days, c.demo())
    if fh is not None and len(fh):
        st.markdown(f"**외국인 지분율** {fh.지분율.iloc[-1]:.2f}% "
                    f"(기간 중 {fh.지분율.iloc[-1] - fh.지분율.iloc[0]:+.2f}%p)")
        st.plotly_chart(ui.line_chart({"외국인 지분율": fh.지분율}, height=220, title_y="%"),
                        width="stretch", theme=None, config={"displayModeBar": False})
    with st.expander("일별 상세 (억 원)"):
        t = (flow[sel].iloc[::-1] / 1e8).round(1)
        t.index = t.index.strftime("%Y-%m-%d")
        st.dataframe(t.style.map(ui.color_change).format("{:+,.1f}"), width="stretch")


def _fundamentals(code, row):
    f = c.safe(market.fundamentals, code, c.demo())
    if f is None or f.empty:
        return
    last = f.iloc[-1]
    ui.kv_grid([("PER", f"{last.PER:,.2f}배" if last.PER else "-"), ("PBR", f"{last.PBR:,.2f}배"),
                ("EPS", f"{last.EPS:,.0f}원"), ("BPS", f"{last.BPS:,.0f}원"),
                ("배당수익률", f"{last.DIV:,.2f}%"), ("주당배당금", f"{last.DPS:,.0f}원"),
                ("시가총액", ui.won(row.Marcap))])
    st.write("")
    a, b = st.columns(2)
    a.plotly_chart(ui.line_chart({"PER": f.PER.replace(0, float("nan"))}, height=240, title_y="배"),
                   width="stretch", theme=None, config={"displayModeBar": False})
    b.plotly_chart(ui.line_chart({"PBR": f.PBR}, height=240, title_y="배"),
                   width="stretch", theme=None, config={"displayModeBar": False})
    st.caption("KRX 기준 최근 실적 반영값. 적자 기업은 PER이 0으로 표시됩니다.")


def _technical(code, row, full):
    d = full.dropna(subset=["MA20"])
    res = analysis.analyze(code, row.Name, d, {})
    ui.kv_grid([("추세 판정", f"{res.trend} ({res.score:+d})"), ("이평 배열", res.alignment),
                ("RSI", f"{res.rsi:.0f}"), ("1개월", f"{res.returns['1M']:+.1f}%" if res.returns['1M'] else "-"),
                ("3개월", f"{res.returns['3M']:+.1f}%" if res.returns['3M'] else "-"),
                ("ATR(변동폭)", f"{d.ATR_pct.iloc[-1]*100:.1f}%")])
    st.write("")
    if res.signals:
        st.markdown("**발생 신호**")
        for n, desc in res.signals:
            st.markdown(f"- **{n}** — {desc}")
    else:
        st.markdown("특이 신호 없음")
    st.caption("추세 점수는 이동평균 위치·배열·기울기와 신호를 합산한 값(-6 ~ +6)입니다. "
               "과거 승률 기반 추천은 [AI 추천] 메뉴에서 확인하세요.")


def _news(name):
    items = c.safe(market.news, name, c.demo())
    if not items:
        st.caption("뉴스가 없습니다.")
        return
    for it in items:
        t = it["time"].strftime("%m.%d %H:%M") if it["time"] is not None else ""
        st.markdown(f"<div class='newsitem'><a href='{it['link']}' target='_blank'>{it['title']}</a>"
                    f"<div class='m'>{it['source']} · {t}</div></div>", unsafe_allow_html=True)


def render():
    ui.demo_banner(c.demo())
    lst = c.listing()
    row = _search(lst)
    code = row.Code
    h = market.history(code, 400, c.demo())
    last = h.iloc[-1]
    prev = h.Close.iloc[-2]
    hi52, lo52 = h.High.iloc[-252:].max(), h.Low.iloc[-252:].min()
    ui.price_header(row.Name, code, row.Market, last.Close, last.Close - prev, (last.Close / prev - 1) * 100,
                    extra=f" · 기준일 {h.index[-1]:%Y-%m-%d}")
    st.write("")
    ui.kv_grid([("시가", ui.num(last.Open)), ("고가", ui.num(last.High)), ("저가", ui.num(last.Low)),
                ("거래량", f"{last.Volume:,.0f}"), ("거래대금", ui.won(last.Volume * last.Close)),
                ("52주 최고", ui.num(hi52)), ("52주 최저", ui.num(lo52)), ("시가총액", ui.won(row.Marcap))])
    st.write("")
    _actions(row)
    _chart(code)
    tabs = st.tabs(["투자자 동향", "투자지표", "기술적 분석", "뉴스"])
    with tabs[0]:
        _flows(code)
    with tabs[1]:
        _fundamentals(code, row)
    with tabs[2]:
        _technical(code, row, indicators.add_indicators(h))
    with tabs[3]:
        _news(row.Name)
