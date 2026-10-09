"""HTML 리포트 생성 (차트 이미지 내장, 단일 파일)."""
from __future__ import annotations

import html
from datetime import datetime

import strategy as st

E = html.escape

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#F4F5F7;color:#1F2937;font-family:'Malgun Gothic','맑은 고딕','Apple SD Gothic Neo',sans-serif;font-size:14px;line-height:1.55}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 60px}
header{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid #1F2A44;padding-bottom:10px;margin-bottom:18px;flex-wrap:wrap;gap:6px}
h1{margin:0;font-size:22px;color:#1F2A44;letter-spacing:-0.4px}
.meta{color:#6B7280;font-size:12.5px}
h2{font-size:17px;color:#1F2A44;margin:38px 0 12px;padding-bottom:6px;border-bottom:1px solid #D1D5DB}
h2 small{font-weight:400;color:#6B7280;font-size:12.5px;margin-left:8px}
.up{color:#F5222D}.down{color:#1677FF}.flat{color:#6B7280}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.tile{background:#fff;border:1px solid #E5E7EB;padding:12px 14px}
.tile .n{font-size:12.5px;color:#4B5563;font-weight:600}
.tile .v{font-size:24px;font-weight:700;letter-spacing:-0.5px;margin-top:2px}
.tile .c{font-size:13px;font-weight:600}
.card{background:#fff;border:1px solid #E5E7EB;padding:18px 20px;margin-bottom:14px}
.lead{display:grid;grid-template-columns:1.1fr 1fr;gap:24px;align-items:center}
.lead .big{font-size:44px;font-weight:800;color:#1F2A44;letter-spacing:-1.5px;line-height:1.1}
.lead .lbl{font-size:12.5px;color:#6B7280}
.lead .tech{font-size:15px;font-weight:700;color:#1F2A44;margin:4px 0 6px}
.bar{position:relative;height:16px;background:#ECEEF1;margin:4px 0}
.bar .f{position:absolute;left:0;top:0;bottom:0;background:#1F2A44}
.bar .f.alt{background:#9CA3AF}
.bar .b{position:absolute;top:-4px;bottom:-4px;width:2px;background:#F5222D}
.bar .t{position:absolute;right:6px;top:-1px;font-size:11.5px;font-weight:700;color:#1F2A44}
.legend{font-size:12px;color:#6B7280;margin-top:4px}
.legend i{display:inline-block;width:10px;height:10px;vertical-align:-1px;margin:0 4px 0 10px}
table{width:100%;border-collapse:collapse;font-size:13px;background:#fff}
th,td{border:1px solid #E5E7EB;padding:7px 9px;text-align:right;vertical-align:middle}
th{background:#F3F4F6;color:#374151;font-weight:600;white-space:nowrap}
td.l,th.l{text-align:left}
td.px{font-size:15px;font-weight:700}
td .bw{display:flex;align-items:center;gap:8px}td .bw .bar{flex:1;height:12px;min-width:80px;margin:0}td .bw b{min-width:44px;text-align:right}
.pick{padding:0;overflow:hidden}
.pick .top{display:flex;justify-content:space-between;align-items:flex-start;padding:16px 20px 10px;gap:12px;flex-wrap:wrap}
.pick .nm{font-size:19px;font-weight:800;color:#1F2A44}
.pick .nm .rk{display:inline-block;background:#1F2A44;color:#fff;font-size:12px;padding:1px 7px;margin-right:8px;vertical-align:3px}
.pick .sub{color:#6B7280;font-size:12.5px;margin-top:2px}
.pick .price{text-align:right}
.pick .price .v{font-size:30px;font-weight:800;letter-spacing:-0.8px;line-height:1.1}
.pick .price .c{font-size:14px;font-weight:700}
.levels{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #E5E7EB;border-bottom:1px solid #E5E7EB}
.levels>div{padding:10px 14px;border-right:1px solid #E5E7EB}
.levels>div:last-child{border-right:0}
.levels .k{font-size:12px;color:#6B7280}
.levels .v{font-size:18px;font-weight:800}
.levels .p{font-size:12.5px;font-weight:600}
.why{padding:12px 20px;display:grid;grid-template-columns:1fr 1fr;gap:20px}
.why ul{margin:4px 0 0;padding-left:18px}
.why li{margin:2px 0}
.pick img{width:100%;display:block;padding:0 10px 10px}
.card img{width:100%;display:block}
.tag{display:inline-block;border:1px solid #D1D5DB;padding:0 7px;margin:2px 4px 2px 0;font-size:12px;color:#374151}
.muted{color:#6B7280;font-size:12.5px}
.method ol{margin:6px 0;padding-left:20px}.method li{margin:3px 0}
.note{color:#6B7280;font-size:12px;margin-top:10px;line-height:1.6}
@media (max-width:720px){.lead,.why{grid-template-columns:1fr}.levels{grid-template-columns:1fr 1fr}
 .levels>div:nth-child(2){border-right:0}.lead .big{font-size:36px}}
"""


def _cls(v):
    return "up" if v > 0 else "down" if v < 0 else "flat"


def _num(v, idx=False):
    if v != v:
        return "-"
    return f"{v:,.2f}" if idx or abs(v) < 1000 else f"{v:,.0f}"


def _chg(diff, pct, idx=False):
    return f'<span class="{_cls(pct)}">{diff:+,.2f} ({pct:+.2f}%)</span>' if idx or abs(diff) < 100 \
        else f'<span class="{_cls(pct)}">{diff:+,.0f} ({pct:+.2f}%)</span>'


def _pct(v):
    return "-" if v is None or v != v else f'<span class="{_cls(v)}">{v:+.2f}%</span>'


def _cap(v):
    return f"{v/1e12:,.1f}조" if v >= 1e12 else f"{v/1e8:,.0f}억"


def _bar(p, base=None, alt=False, label=True):
    b = f"<span class='b' style='left:{base*100:.1f}%' title='기준 {base*100:.1f}%'></span>" if base is not None else ""
    t = f"<span class='t'>{p*100:.1f}%</span>" if label else ""
    return f"<div class='bar'><span class='f{' alt' if alt else ''}' style='width:{p*100:.1f}%'></span>{b}{t}</div>"


def _tbar(p, base=None, alt=False):
    return f"<div class='bw'>{_bar(p, base, alt, label=False)}<b>{p*100:.1f}%</b></div>"


def _tiles(index_results):
    out = []
    for r in index_results:
        pct = (r.close / r.prev - 1) * 100
        out.append(f"<div class='tile'><div class='n'>{E(r.name)}</div>"
                   f"<div class='v {_cls(pct)}'>{_num(r.close, True)}</div>"
                   f"<div class='c'>{_chg(r.close - r.prev, pct, True)}</div></div>")
    return f"<div class='tiles'>{''.join(out)}</div>"


def _lead(scan):
    s, b = scan.strategies[0], scan.baseline
    wins = round(s.win * s.n)
    return (f"<div class='card lead'><div>"
            f"<div class='lbl'>현재 가장 확률이 높은 기법 (과거 {scan.n_stocks}개 종목 · {scan.period[0]:%Y.%m}~{scan.period[1]:%Y.%m} 검증)</div>"
            f"<div class='tech'>{E(s.name)}</div>"
            f"<div class='big'>{s.win*100:.1f}%</div>"
            f"<div class='muted'>이 신호가 뜬 뒤 1개월(20거래일) 안에 수익으로 끝난 비율 — "
            f"과거 {s.n:,}번 중 {wins:,}번 수익, 거래당 평균 {s.avg*100:+.2f}%</div></div>"
            f"<div><div class='lbl'>이 기법</div>{_bar(s.win, b['win'])}"
            f"<div class='lbl' style='margin-top:10px'>아무 날 아무 종목을 같은 방식으로 샀을 때</div>{_bar(b['win'], alt=True)}"
            f"<div class='lbl' style='margin-top:10px'>목표가 도달률 {s.hit*100:.1f}% (기준 {b['hit']*100:.1f}%) · "
            f"학습 구간 {s.win_tr*100:.1f}% / 검증 구간 {s.win_te*100:.1f}%</div></div></div>")


def _pick_table(picks, base):
    rows = []
    for i, p in enumerate(picks, 1):
        s = p.strategy
        rows.append(
            f"<tr><td>{i}</td>"
            f"<td class='l'><b>{E(p.name)}</b> <span class='muted'>{p.code}</span></td>"
            f"<td class='px {_cls(p.chg_pct)}'>{_num(p.close)}</td><td style='white-space:nowrap'>{_chg(p.chg, p.chg_pct)}</td>"
            f"<td style='min-width:150px'>{_tbar(s.win, base)}</td>"
            f"<td>{_pct(s.avg*100)}</td>"
            f"<td class='up'><b>{_num(p.target)}</b><br><span class='muted'>{(p.target/p.close-1)*100:+.1f}%</span></td>"
            f"<td class='down'><b>{_num(p.stop)}</b><br><span class='muted'>{(p.stop/p.close-1)*100:+.1f}%</span></td>"
            f"<td class='l' style='font-size:12px'>{E(s.name)}</td></tr>")
    return ("<table><tr><th>순위</th><th class='l'>종목</th><th>현재가</th><th>전일 대비</th>"
            "<th>과거 승률</th><th>평균 수익</th><th>목표가(익절)</th><th>손절가</th><th class='l'>신호 기법</th></tr>"
            + "".join(rows) + "</table>")


def _pick_card(i, p, base, chart, horizon):
    s = p.strategy
    own = (f"이 종목에서 같은 신호 {p.own_n}회 중 {round(p.own_win*p.own_n)}회 수익"
           if p.own_n else "이 종목에서는 같은 신호의 과거 사례 없음 (전체 종목 통계 기준)")
    when = "오늘" if p.days_ago == 0 else f"{p.days_ago}거래일 전"
    conds = "".join(f"<li><b>{E(st.COND_NAME[k])}</b> <span class='muted'>— {E(st.COND_DESC[k])}</span></li>"
                    for k in s.combo)
    return (f"<div class='card pick'><div class='top'><div>"
            f"<div class='nm'><span class='rk'>{i}</span>{E(p.name)}</div>"
            f"<div class='sub'>{p.code} · {E(p.market)} · 시가총액 {_cap(p.marcap)} · 신호 발생 {when}</div></div>"
            f"<div class='price'><div class='v {_cls(p.chg_pct)}'>{_num(p.close)}</div>"
            f"<div class='c'>{_chg(p.chg, p.chg_pct)}</div></div></div>"
            f"<div class='levels'>"
            f"<div><div class='k'>매수 기준 (다음 거래일 시가 부근)</div><div class='v'>{_num(p.close)}</div></div>"
            f"<div><div class='k'>목표가 · 익절</div><div class='v up'>{_num(p.target)}</div>"
            f"<div class='p up'>{(p.target/p.close-1)*100:+.1f}% (ATR {s.kt:g}배)</div></div>"
            f"<div><div class='k'>손절가</div><div class='v down'>{_num(p.stop)}</div>"
            f"<div class='p down'>{(p.stop/p.close-1)*100:+.1f}% (ATR {s.ks:g}배)</div></div>"
            f"<div><div class='k'>보유 기한</div><div class='v'>{horizon}거래일</div>"
            f"<div class='p muted'>미도달 시 기한 종가 매도</div></div></div>"
            f"<div class='why'><div><div class='lbl muted'>과거 승률 ({s.n:,}회 신호)</div>{_bar(s.win, base)}"
            f"<div class='muted'>평균 수익 {s.avg*100:+.2f}% · 목표가 도달 {s.hit*100:.0f}% · {own}</div></div>"
            f"<div><div class='lbl muted'>신호 구성</div><ul>{conds}</ul></div></div>"
            f"<img src='data:image/png;base64,{chart}'></div>")


def _strategy_section(scan, hist_chart):
    b = scan.baseline
    rows = [f"<tr><td class='l'><b>기준</b> <span class='muted'>아무 날 매수</span></td><td>{b['n']:,}</td>"
            f"<td style='min-width:170px'>{_tbar(b['win'], alt=True)}</td><td>-</td><td>-</td>"
            f"<td>{_pct(b['avg']*100)}</td><td>{b['hit']*100:.0f}%</td><td>-</td></tr>"]
    for i, s in enumerate(scan.strategies, 1):
        rows.append(f"<tr><td class='l'><b>{i}.</b> {E(s.name)}</td><td>{s.n:,}</td>"
                    f"<td>{_tbar(s.win, b['win'])}</td><td>{s.win_tr*100:.1f}%</td><td>{s.win_te*100:.1f}%</td>"
                    f"<td>{_pct(s.avg*100)}</td><td>{s.hit*100:.0f}%</td>"
                    f"<td>익절 {s.kt:g} · 손절 {s.ks:g}</td></tr>")
    tbl = ("<table><tr><th class='l'>기법 (조건 조합)</th><th>신호 수</th><th>승률 (붉은 선 = 기준)</th>"
           "<th>학습 구간</th><th>검증 구간</th><th>평균 수익</th><th>목표 도달</th><th>매도 기준(ATR 배수)</th></tr>"
           + "".join(rows) + "</table>")
    method = (
        "<div class='card method'><b>검증 방법</b><ol>"
        f"<li>추세·모멘텀·돌파·눌림·반전·거래량·캔들·시장국면 등 기본 조건 {len(st.CONDITIONS)}개를 계산하고, "
        f"1~3개를 조합한 {scan.n_tested:,}가지 기법을 모두 시험했습니다.</li>"
        f"<li>신호 다음 날 시가에 매수하고, 목표가 도달 시 익절, 손절가 이탈 시 손절, {scan.horizon}거래일 경과 시 종가 매도. "
        "왕복 거래비용 0.25%를 차감했습니다. 같은 날 목표가·손절가가 모두 닿으면 손절로 처리했습니다.</li>"
        "<li>앞 65% 기간(학습)과 뒤 35% 기간(검증) 모두에서 수익이 유지된 기법만 남기고, "
        "신호 수가 적을수록 불리하게 보정한 승률 하한값으로 순위를 매겼습니다.</li>"
        "<li>상위 기법마다 목표가·손절가를 ATR(평균 변동폭) 배수로 다시 최적화했습니다.</li></ol>"
        "<div class='note'>유의 사항 — 현재 시가총액 1조 이상 종목만으로 과거를 검증하므로, 과거에 커진 종목이 포함되는 "
        "생존 편향이 있어 실제 승률은 표보다 낮을 수 있습니다. 많은 조합 중 최고를 고르는 과정 자체도 과대평가 요인입니다. "
        "과거 확률은 미래 수익을 보장하지 않으며, 뉴스·실적 등 차트 외 요인은 반영되지 않습니다.</div></div>")
    return tbl + f"<div class='card' style='margin-top:12px'><img src='data:image/png;base64,{hist_chart}'></div>" + method


def _table(results):
    head = ("<tr><th class='l'>종목</th><th>현재가</th><th>전일 대비</th><th>1주</th><th>1개월</th><th>3개월</th>"
            "<th>YTD</th><th>RSI</th><th class='l'>추세</th></tr>")
    rows = []
    for r in results:
        pct = (r.close / r.prev - 1) * 100
        rows.append(f"<tr><td class='l'><b>{E(r.name)}</b></td><td class='px {_cls(pct)}'>{_num(r.close)}</td>"
                    f"<td>{_chg(r.close - r.prev, pct)}</td>"
                    + "".join(f"<td>{_pct(r.returns[k])}</td>" for k in ("1W", "1M", "3M", "YTD"))
                    + f"<td>{r.rsi:.0f}</td><td class='l'>{r.trend} ({r.score:+d})</td></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def build(index_results, stock_results, summary_lines, index_chart, stock_charts,
          scan=None, pick_charts=None, hist_chart=None, ai_comment=None, demo=False) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    dates = [r.df.index[-1] for r in index_results + stock_results]
    asof = max(dates).strftime("%Y-%m-%d") if dates else "-"
    p = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>주식 리포트 {asof}</title><style>{CSS}</style></head><body><div class='wrap'>",
         f"<header><h1>주식 시황 · 추천 리포트</h1><div class='meta'>기준일 {asof} · 생성 {now}"
         f"{' · 데모 데이터(가상 시세)' if demo else ''}</div></header>"]
    if index_results:
        p.append(_tiles(index_results))

    sec = 1
    if scan is not None:
        base = scan.baseline["win"]
        p.append(f"<h2>{sec}. 1개월 추천 종목<small>코스피·코스닥 시가총액 1조 원 이상 {scan.n_stocks}개 종목 스캔</small></h2>")
        sec += 1
        if scan.strategies:
            p.append(_lead(scan))
        if scan.picks:
            p.append(_pick_table(scan.picks, base))
            p.append("<div class='legend'>승률 막대의 붉은 선 = 아무 날 매수했을 때의 기준 승률"
                     " · 목표가·손절가는 오늘 종가 기준이며 실제 매수가에 맞춰 같은 비율로 조정하세요</div>")
            for i, pk in enumerate(scan.picks, 1):
                p.append(_pick_card(i, pk, base, pick_charts[pk.code], scan.horizon))
        else:
            p.append("<div class='card muted'>오늘은 검증된 상위 기법의 신호가 새로 발생한 종목이 없습니다. "
                     "확률이 낮은 신호로 무리하게 채우지 않았습니다.</div>")
        if scan.strategies:
            p.append(f"<h2>{sec}. 기법 검증 결과<small>과거 데이터에 적용했을 때의 실제 성적</small></h2>")
            sec += 1
            p.append(_strategy_section(scan, hist_chart))

    p.append(f"<h2>{sec}. 시황 요약</h2><div class='card'>")
    sec += 1
    p += [f"<p style='margin:4px 0'>{E(l)}</p>" for l in summary_lines]
    if ai_comment:
        p.append(f"<div style='border-top:1px solid #E5E7EB;margin-top:10px;padding-top:10px;white-space:pre-wrap'>"
                 f"<b>AI 코멘트</b>\n{E(ai_comment)}</div>")
    p.append("</div>")
    if index_chart:
        p.append(f"<div class='card'><img src='data:image/png;base64,{index_chart}'></div>")

    if stock_results:
        ranked = sorted(stock_results, key=lambda r: r.score, reverse=True)
        p.append(f"<h2>{sec}. 관심종목</h2>")
        p.append(_table(ranked))
        for r in ranked:
            tags = "".join(f"<span class='tag' title='{E(d)}'>{E(n)}</span>" for n, d in r.signals) \
                or "<span class='muted'>특이 신호 없음</span>"
            p.append(f"<div class='card' style='margin-top:12px'><div class='muted'>{E(r.comment)}</div>"
                     f"<div style='margin:4px 0 8px'>{tags}</div>"
                     f"<img src='data:image/png;base64,{stock_charts[r.ticker]}'></div>")

    p.append("<p class='note'>본 리포트는 과거 시세와 기술적 지표에 기반한 자동 분석 결과이며 투자 권유가 아닙니다. "
             "최종 투자 판단과 책임은 본인에게 있습니다.</p></div></body></html>")
    return "".join(p)
