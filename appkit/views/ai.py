"""AI 추천 — 과거 승률이 가장 높은 기법으로 고른 1개월 추천 종목과 매도가."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import scanner
import strategy as stg
from appkit import ui
from appkit.views import common as c
from pathlib import Path

CFG = json.loads((Path(__file__).resolve().parents[2] / "config.json").read_text(encoding="utf-8")).get("scan", {})


def _run():
    bar = st.progress(0.0, text="준비 중")
    stage_w = {"시세 수집": (0.0, 0.55), "매매동향 수집": (0.55, 0.85), "기법 조합 백테스트": (0.85, 1.0)}

    def prog(stage, done, total):
        a, b = stage_w[stage]
        bar.progress(min(a + (b - a) * done / max(total, 1), 1.0), text=f"{stage} {done}/{total}")

    logs = st.empty()
    lines = []

    def log(msg):
        lines.append(msg.strip())
        logs.caption(" · ".join(lines))

    res = scanner.run(CFG, c.demo(), use_flows=c.has_krx(), log=log, progress=prog)
    bar.empty()
    return res


def _lead(res):
    s, b = res.strategies[0], res.baseline
    l, r = st.columns([1.1, 1])
    with l:
        st.caption(f"현재 가장 확률이 높은 기법 · {res.n_stocks}개 종목 · {res.period[0]:%Y.%m}~{res.period[1]:%Y.%m} 검증")
        st.markdown(f"**{s.name}**")
        st.markdown(f"<div style='font-size:46px;font-weight:800;color:{ui.NAVY};line-height:1.1'>{s.win*100:.1f}%</div>",
                    unsafe_allow_html=True)
        st.caption(f"신호 뒤 1개월({res.horizon}거래일) 안에 수익으로 끝난 비율 — 과거 {s.n:,}번 중 "
                   f"{round(s.win*s.n):,}번 수익, 거래당 평균 {s.avg*100:+.2f}%")
    with r:
        st.caption("이 기법")
        st.markdown(ui.prob_bar(s.win, b["win"]), unsafe_allow_html=True)
        st.caption(f"아무 날 아무 종목을 같은 방식으로 샀을 때 — {b['win']*100:.1f}%")
        st.markdown(ui.prob_bar(b["win"]).replace(ui.NAVY, "#9CA3AF"), unsafe_allow_html=True)
        st.caption(f"목표가 도달률 {s.hit*100:.0f}% · 학습 구간 {s.win_tr*100:.1f}% / 검증 구간 {s.win_te*100:.1f}%")


def _picks(res):
    if not res.picks:
        st.info("오늘은 검증된 상위 기법의 신호가 새로 발생한 종목이 없습니다. 확률이 낮은 신호로 채우지 않았습니다.")
        return
    t = pd.DataFrame([{
        "Code": p.code, "종목": p.name, "현재가": p.close, "등락률(%)": p.chg_pct,
        "과거 승률": p.strategy.win, "평균 수익(%)": p.strategy.avg * 100,
        "목표가": p.target, "목표(%)": (p.target / p.close - 1) * 100,
        "손절가": p.stop, "손절(%)": (p.stop / p.close - 1) * 100,
        "신호": "오늘" if p.days_ago == 0 else f"{p.days_ago}일 전", "기법": p.strategy.name} for p in res.picks])
    sty = (t.style.map(ui.color_change, subset=["등락률(%)", "평균 수익(%)", "목표(%)", "손절(%)"])
           .map(lambda v: f"color:{ui.UP};font-weight:700", subset=["목표가"])
           .map(lambda v: f"color:{ui.DOWN};font-weight:700", subset=["손절가"])
           .format({"현재가": "{:,.0f}", "등락률(%)": "{:+.2f}", "평균 수익(%)": "{:+.2f}", "목표가": "{:,.0f}",
                    "목표(%)": "{:+.1f}", "손절가": "{:,.0f}", "손절(%)": "{:+.1f}"}))
    c.selectable_table(t, "ai_picks", styler=sty, column_config={
        "Code": None,
        "과거 승률": st.column_config.ProgressColumn("과거 승률", min_value=0, max_value=1, format="percent")})
    st.caption("행을 누르면 종목 상세로 이동 · 목표가·손절가는 오늘 종가 기준이며, 실제 매수가에 맞춰 같은 비율로 조정 · "
               f"{res.horizon}거래일 안에 둘 다 닿지 않으면 기한 종가에 매도")

    for i, p in enumerate(res.picks, 1):
        s = p.strategy
        with st.expander(f"{i}. {p.name}  {ui.num(p.close)} ({p.chg_pct:+.2f}%)  —  목표 {ui.num(p.target)} / "
                         f"손절 {ui.num(p.stop)}"):
            ui.kv_grid([("매수 기준", ui.num(p.close)),
                        ("목표가", f"<span class='up'>{ui.num(p.target)} ({(p.target/p.close-1)*100:+.1f}%)</span>"),
                        ("손절가", f"<span class='down'>{ui.num(p.stop)} ({(p.stop/p.close-1)*100:+.1f}%)</span>"),
                        ("보유 기한", f"{res.horizon}거래일"), ("과거 승률", f"{s.win*100:.1f}% ({s.n:,}회)"),
                        ("이 종목 과거", f"{round(p.own_win*p.own_n)}/{p.own_n}회 수익" if p.own_n else "사례 없음")])
            st.markdown("**신호 구성** — " + " · ".join(f"{stg.COND_NAME[k]}({stg.COND_DESC[k]})" for k in s.combo))
            d = p.df.iloc[-120:][["Open", "High", "Low", "Close", "Volume"]]
            fig = ui.candle_chart(p.df[["Open", "High", "Low", "Close", "Volume"]], ["MA", "거래량"],
                                  levels={"목표가": (p.target, ui.UP), "손절가": (p.stop, ui.DOWN)}, height=420)
            fig.update_xaxes(range=[len(p.df) - len(d) - 0.5, len(p.df) - 0.5])
            st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False}, key=f"ai_ch_{p.code}")


def _validation(res):
    b = res.baseline
    rows = [{"기법": "기준 — 아무 날 매수", "신호 수": b["n"], "승률": b["win"], "학습": np.nan, "검증": np.nan,
             "평균 수익(%)": b["avg"] * 100, "목표 도달": b["hit"], "익절·손절(ATR)": "-"}]
    rows += [{"기법": s.name, "신호 수": s.n, "승률": s.win, "학습": s.win_tr, "검증": s.win_te,
              "평균 수익(%)": s.avg * 100, "목표 도달": s.hit, "익절·손절(ATR)": f"{s.kt:g} · {s.ks:g}"}
             for s in res.strategies]
    t = pd.DataFrame(rows)
    st.dataframe(t.style.map(ui.color_change, subset=["평균 수익(%)"]).format(
        {"신호 수": "{:,}", "학습": "{:.1%}", "검증": "{:.1%}", "평균 수익(%)": "{:+.2f}", "목표 도달": "{:.0%}"},
        na_rep="-"), hide_index=True, width="stretch", column_config={
        "승률": st.column_config.ProgressColumn("승률", min_value=0, max_value=1, format="percent")})

    s = res.strategies[0]
    r = s.rets * 100
    bins = np.linspace(np.percentile(r, 1), np.percentile(r, 99), 40)
    cnt, edges = np.histogram(np.clip(r, bins[0], bins[-1]), bins=bins)
    mid = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=mid, y=cnt, marker_color=[ui.UP if m >= 0 else ui.DOWN for m in mid],
                           width=(edges[1] - edges[0]) * 0.95))
    fig.add_vline(x=0, line=dict(color=ui.NAVY, width=1))
    fig.add_vline(x=r.mean(), line=dict(color=ui.NAVY, width=1, dash="dash"),
                  annotation_text=f"평균 {r.mean():+.2f}%", annotation_position="top right")
    fig.update_layout(title=f"1위 기법 거래 결과 분포 ({s.n:,}회, 비용 차감)", xaxis_title="거래 1회 수익률 (%)",
                      yaxis_title="횟수", bargap=0)
    st.plotly_chart(ui._layout(fig, 300).update_layout(dragmode=False, hovermode="closest"),
                    width="stretch", theme=None, config={"displayModeBar": False})

    flow_note = ("외국인·기관·연기금 수급 조건 4개 포함" if getattr(res, "used_flows", False)
                 else "수급 조건은 KRX 로그인 정보가 없어 제외됨")
    with st.expander("검증 방법과 유의 사항"):
        st.markdown(
            f"- 추세·모멘텀·돌파·눌림·반전·거래량·캔들·시장국면·수급 조건을 1~3개씩 조합한 {res.n_tested:,}가지 기법을 시험 ({flow_note})\n"
            f"- 신호 다음 날 시가 매수, 목표가 익절 / 손절가 손절 / {res.horizon}거래일 경과 시 종가 매도, 왕복 비용 0.25% 차감. "
            "같은 날 목표가·손절가가 모두 닿으면 손절로 처리\n"
            "- 앞 65% 기간(학습)과 뒤 35% 기간(검증) 모두에서 수익이 유지된 기법만 채택, 신호 수가 적을수록 불리하게 보정\n"
            "- 상위 기법마다 목표가·손절가를 ATR(평균 변동폭) 배수로 다시 최적화\n\n"
            "**유의** — 현재 시총 1조 이상 종목만으로 과거를 검증하므로 생존 편향이 있고, 수천 개 조합 중 최고를 고르는 과정도 "
            "과대평가 요인입니다. 실제 승률은 표보다 낮을 수 있으며, 실적·뉴스 등 차트 외 요인은 반영되지 않습니다. "
            "투자 권유가 아닙니다.")


def render():
    st.title("AI 추천")
    ui.demo_banner(c.demo())
    res = scanner.load_cached(c.demo())
    a, b = st.columns([3, 1])
    if res is None:
        a.caption("오늘 스캔 결과가 없습니다. 코스피·코스닥 시총 1조 이상 종목 전체를 분석합니다 "
                  "(첫 실행 5~10분, 수급 포함 시 추가 수 분).")
    else:
        a.caption(f"오늘 스캔 결과 · {res.n_stocks}개 종목 · 과거 검증 {res.period[0]:%Y.%m}~{res.period[1]:%Y.%m}")
    if b.button("스캔 실행" if res is None else "다시 스캔", type="primary", width="stretch"):
        res = _run()
    if res is None:
        return
    if not res.strategies:
        st.warning("조건을 통과한 기법이 없습니다. config.json 의 min_train_signals 등 기준을 완화해 보세요.")
        return
    _lead(res)
    st.subheader("1개월 추천 종목")
    _picks(res)
    st.subheader("기법 검증 결과")
    _validation(res)
