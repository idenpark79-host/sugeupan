"""종목별 차트 견해 — 추세·최근 패턴·과열/침체·수급·지지/저항을 짧은 문장으로.

결과 예)
  tone: "강세" / "약세" / "중립"
  head: "거래량 실린 장대양봉으로 20일선 회복"
  points: ["5>20>60일선 정배열, 60일선 상승 — 중기 상승 추세", "외국인 4일 연속 순매수", ...]
  plan: "지지 289,000(20일선) · 저항 312,000(직전 고점)"
  pats: [[키, 이름, 며칠 전, 방향, 1주 [평균, 시장대비, 승률, n], 1개월 [...]], ...]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import patterns as P

# 견해 문장에서 우선순위 (앞일수록 먼저)
ORDER = ["hs_top", "inv_hs", "dbl_bottom", "dbl_top", "box_up", "box_dn", "tri_up", "tri_dn", "hi52", "lo52",
         "vol_bull", "vol_bear", "morning_star", "evening_star", "bull_engulf", "bear_engulf", "hammer", "shooting",
         "three_white", "three_black", "gap_up", "gap_down", "align_on", "align_off", "gc_long", "dc_long",
         "ma60_up", "ma60_dn", "ma20_up", "ma20_dn", "ma60_sup", "ma60_rej", "ma20_sup", "ma20_rej", "ma5_sup", "ma5_rej",
         "bull_div", "bear_div", "gc", "dc", "macd_gc", "macd_dc", "doji_top", "pullback", "overheat", "oversold",
         "rsi_os", "rsi_ob", "bb_up", "bb_dn", "squeeze", "vol_dry", "obv_lead"]
STATE = {"overheat", "oversold", "rsi_os", "rsi_ob", "squeeze", "vol_dry", "pullback", "obv_lead"}   # 상태형: 오늘 값만
RECENT = {"vol_bull": 2, "vol_bear": 2, "hammer": 2, "shooting": 2, "bull_engulf": 2, "bear_engulf": 2,
          "morning_star": 2, "evening_star": 2, "three_white": 1, "three_black": 1, "gap_up": 1, "gap_down": 1,
          "doji_top": 1, "ma5_sup": 1, "ma5_rej": 1}                 # 그 외 사건형은 최근 5거래일


def won(v):
    if v >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _phrase(k, d, ago):
    """패턴별 짧은 서술 (숫자 포함)."""
    c = d.Close.iloc[-1]
    when = "오늘" if ago == 0 else "어제" if ago == 1 else f"{ago}일 전"
    vr = d.Volume.iloc[-1 - ago] / d.VOL_MA20.iloc[-1 - ago] if d.VOL_MA20.iloc[-1 - ago] else np.nan
    ma = lambda n: won(d[f"MA{n}"].iloc[-1])
    t = {
        "vol_bull": f"{when} 거래량 {vr:.1f}배 실린 장대양봉",
        "vol_bear": f"{when} 거래량 {vr:.1f}배 실린 장대음봉",
        "hammer": f"{when} 망치형",
        "shooting": f"{when} 유성형",
        "bull_engulf": f"{when} 상승장악형",
        "bear_engulf": f"{when} 하락장악형",
        "morning_star": "샛별형",
        "evening_star": "석별형",
        "three_white": "적삼병",
        "three_black": "흑삼병",
        "gap_up": "상승 갭",
        "gap_down": "하락 갭",
        "doji_top": "고점 도지",
        "ma5_sup": f"5일선({ma(5)}) 지지",
        "ma5_rej": f"5일선({ma(5)}) 맞고 밀림",
        "ma20_sup": f"20일선({ma(20)})에서 반등",
        "ma20_rej": f"20일선({ma(20)})에 막혀 하락",
        "ma60_sup": f"60일선({ma(60)})에서 반등",
        "ma60_rej": f"60일선({ma(60)})에 막혀 하락",
        "ma20_up": f"거래량 동반 20일선({ma(20)}) 돌파",
        "ma20_dn": f"20일선({ma(20)}) 이탈",
        "ma60_up": f"60일선({ma(60)}) 회복",
        "ma60_dn": f"60일선({ma(60)}) 이탈",
        "gc": "5일선·20일선 골든크로스",
        "dc": "5일선·20일선 데드크로스",
        "gc_long": "20일선·60일선 골든크로스",
        "dc_long": "20일선·60일선 데드크로스",
        "align_on": "이평선 정배열 완성",
        "align_off": "이평선 역배열 완성",
        "hs_top": "헤드앤숄더 목선 이탈",
        "inv_hs": "역헤드앤숄더 목선 돌파",
        "dbl_bottom": "이중바닥 돌파",
        "dbl_top": "이중천장 이탈",
        "box_up": "60일 박스권 상단 돌파",
        "box_dn": "60일 박스권 하단 이탈",
        "tri_up": "삼각수렴 위로 이탈",
        "tri_dn": "삼각수렴 아래로 이탈",
        "hi52": "52주 신고가 경신",
        "lo52": "52주 신저가",
        "pullback": "상승 추세 중 눌림",
        "overheat": f"20일선 +{(c / d.MA20.iloc[-1] - 1) * 100:.0f}% 과열",
        "oversold": f"20일선 −{(1 - c / d.MA20.iloc[-1]) * 100:.0f}% 낙폭 과대",
        "squeeze": "변동성 수축",
        "bb_up": "볼린저 상단 돌파",
        "bb_dn": "볼린저 하단 이탈",
        "rsi_os": f"RSI {d.RSI.iloc[-1]:.0f} 과매도",
        "rsi_ob": f"RSI {d.RSI.iloc[-1]:.0f} 과매수",
        "bull_div": "RSI 상승 다이버전스",
        "bear_div": "RSI 하락 다이버전스",
        "macd_gc": "MACD 골든크로스",
        "macd_dc": "MACD 데드크로스",
        "vol_dry": "거래량 급감",
        "obv_lead": "OBV 선행 신고가",
    }
    return t.get(k, P.P_NAME[k])


GROUPS = [{"ma20_up", "ma20_dn", "ma20_sup", "ma20_rej"}, {"ma60_up", "ma60_dn", "ma60_sup", "ma60_rej"},
          {"ma5_sup", "ma5_rej"}, {"gc", "dc"}, {"gc_long", "dc_long"}, {"macd_gc", "macd_dc"}, {"align_on", "align_off"},
          {"gap_up", "gap_down"}, {"three_white", "three_black"}, {"vol_bull", "vol_bear"}, {"tri_up", "tri_dn"},
          {"box_up", "box_dn"}, {"dbl_bottom", "dbl_top"}, {"hs_top", "inv_hs"}, {"hi52", "lo52"},
          {"bull_engulf", "bear_engulf", "hammer", "shooting", "morning_star", "evening_star"}, {"bb_up", "bb_dn"},
          {"rsi_os", "rsi_ob"}, {"overheat", "oversold"}, {"bull_div", "bear_div"}]


def recent(f: pd.DataFrame, window: int = 5):
    """최근 발생 패턴 [(키, 며칠 전)] — 우선순위 순, 서로 반대인 신호는 가장 최근 것만."""
    out = []
    n = len(f)
    for k in ORDER:
        if k not in f:
            continue
        col = f[k].values
        if k in STATE:
            if col[-1]:
                out.append((k, 0))
            continue
        w = RECENT.get(k, window)
        hits = np.flatnonzero(col[max(0, n - w):])
        if len(hits):
            out.append((k, int(w - 1 - hits[-1]) if n >= w else int(n - 1 - hits[-1])))
    keep = []
    for k, ago in out:
        g = next((g for g in GROUPS if k in g), None)
        if g and any(k2 in g and (a2 < ago or (a2 == ago)) for k2, a2 in keep):
            continue
        if g:
            keep = [(k2, a2) for k2, a2 in keep if not (k2 in g and a2 > ago)]
        keep.append((k, ago))
    keep.sort(key=lambda z: ORDER.index(z[0]))
    return keep


def view(d: pd.DataFrame, pstats: dict | None = None, flow: pd.DataFrame | None = None,
         f: pd.DataFrame | None = None) -> dict:
    if f is None:
        f = P.flags(d)
    x = d.iloc[-1]
    c = float(x.Close)
    rec = recent(f)
    # 추세
    ma5, ma20, ma60, ma120 = (float(x[f"MA{n}"]) if pd.notna(x[f"MA{n}"]) else np.nan for n in (5, 20, 60, 120))
    s60 = d.MA60.iloc[-1] / d.MA60.iloc[-6] - 1 if len(d) > 70 else 0
    al = ma5 > ma20 > ma60 > ma120
    ra = ma5 < ma20 < ma60 < ma120
    score = 0.0
    if al:
        trend = "정배열 상승 추세"; score += 2
    elif ra:
        trend = "역배열 하락 추세"; score -= 2
    elif c > ma60 and s60 > 0:
        trend = "60일선 위 중기 상승"; score += 1
    elif c < ma60 and s60 < 0:
        trend = "60일선 아래 중기 하락"; score -= 1
    else:
        trend = "이평선 혼조"
    for k, ago in rec[:6]:
        b = P.P_BIAS[k]
        score += b * (1.0 if ago <= 1 else 0.6)
    rsi = float(x.RSI) if pd.notna(x.RSI) else 50
    hi = d.High.iloc[-250:].max()
    from_hi = c / hi - 1
    # 수급
    flow_txt = None
    if flow is not None and len(flow):
        for col, nm in (("외국인", "외국인"), ("기관합계", "기관"), ("연기금", "연기금")):
            if col not in flow:
                continue
            v = flow[col].fillna(0).values
            k = 0
            for z in v[::-1]:
                if not z or (k and (z > 0) != (k > 0)):
                    break
                k += 1 if z > 0 else -1
            if abs(k) >= 3:
                flow_txt = f"{nm} {abs(k)}일 연속 순{'매수' if k > 0 else '매도'}"
                score += 0.8 if k > 0 else -0.8
                break
    tone = "강세" if score >= 2 else "약세" if score <= -2 else "중립"
    # 헤드라인: 가장 중요한 패턴 1~2개
    heads = [_phrase(k, d, ago) for k, ago in rec[:2]]
    if not heads:
        heads = [trend]
    head = " · ".join(heads)
    pts = [f"{trend}"]
    if len(rec) > 2:
        pts.append(" · ".join(_phrase(k, d, ago) for k, ago in rec[2:4]))
    if rsi >= 70 and not any(k in ("rsi_ob", "overheat") for k, _ in rec):
        pts.append(f"RSI {rsi:.0f} 과열권")
    elif rsi <= 30 and not any(k in ("rsi_os", "oversold") for k, _ in rec):
        pts.append(f"RSI {rsi:.0f} 침체권")
    if from_hi > -0.03:
        pts.append("52주 고점 부근")
    elif from_hi < -0.35:
        pts.append(f"52주 고점 대비 {from_hi * 100:.0f}%")
    if flow_txt:
        pts.append(flow_txt)
    lv = P.levels(d)
    sup, res = lv["support"], lv["resist"]
    plan = " · ".join(([f"지지 {won(sup[0])}({sup[1]})"] if sup else []) + ([f"저항 {won(res[0])}({res[1]})"] if res else []))
    ks_ = {k for k, _ in rec}
    if tone == "강세":
        act = "강하지만 과열 · 눌림 확인 후 접근" if ks_ & {"overheat", "rsi_ob"} or rsi >= 72 else "상승 흐름 유효 · 지지선 이탈 시 정리"
    elif tone == "약세":
        act = "하락 중 과매도 · 반등은 저항선에서 확인" if ks_ & {"oversold", "rsi_os"} or rsi <= 32 else "추세 회복 전까지 관망"
    else:
        act = "돌파 방향 확인 후 대응" if "squeeze" in ks_ else "지지·저항 사이에서 대응"
    pats = []
    for k, ago in rec[:6]:
        st = (pstats or {}).get(k, {})
        pats.append([k, P.P_NAME[k], ago, P.P_BIAS[k], st.get("5"), st.get("20")])
    return {"tone": tone, "score": round(score, 1), "head": head, "points": pts[:4], "plan": plan, "act": act,
            "support": [round(sup[0], 2), sup[1]] if sup else None, "resist": [round(res[0], 2), res[1]] if res else None,
            "pats": pats}


def perspective(d: pd.DataFrame, view: dict | None) -> dict:
    """추천 종목의 매매 관점 — 반등형 / 돌파형 / 추세형 / 전환형 / 눌림형 + 한 줄 설명."""
    x = d.iloc[-1]
    c = float(x.Close)
    ma20, ma60 = float(x.MA20), float(x.MA60)
    d20 = c / ma20 - 1 if ma20 else 0
    r20 = c / float(d.Close.iloc[-21]) - 1 if len(d) > 21 else 0
    rsi = float(x.RSI) if pd.notna(x.RSI) else 50
    s60 = float(d.MA60.iloc[-1] / d.MA60.iloc[-6] - 1) if len(d) > 70 else 0
    keys = {p[0] for p in (view or {}).get("pats", [])}
    first = lambda ks: P.P_NAME[sorted(ks, key=ORDER.index)[0]]
    brk = keys & {"inv_hs", "dbl_bottom", "box_up", "tri_up", "hi52", "bb_up", "ma60_up", "gc_long", "align_on", "vol_bull", "obv_lead"}
    rev = keys & {"morning_star", "hammer", "bull_engulf", "bull_div", "ma20_up", "gc", "macd_gc", "three_white"}
    if c < ma60 and (d20 < -0.05 or rsi < 40 or r20 < -0.1):
        lead = (f"1개월 {r20 * 100:+.0f}% {'급락' if r20 <= -0.15 else '조정'}" if r20 <= -0.1 else
                f"20일선 아래 {d20 * 100:+.0f}%" if d20 <= -0.05 else f"RSI {rsi:.0f} 침체")
        return {"type": "반등형", "text": f"{lead} · 과매도 반등 노림"}
    brk = {k for k in brk if c > ma20 and (k not in {"ma60_up", "gc_long", "align_on"} or c > ma60)}   # 이미 무너진 돌파는 제외
    if brk:
        tail = "추세 가속" if c > ma60 and s60 > 0 else "추세 전환 시도"
        return {"type": "돌파형", "text": f"{first(brk)} · {tail}"}
    if c > ma60 and s60 > 0 and d20 > -0.03:
        tail = "눌림 구간" if d20 <= 0.03 else "탄력 유지" if d20 < 0.12 else "과열 주의"
        return {"type": "추세형", "text": f"60일선 위 상승 추세 · {tail}"}
    rev = {k for k in rev if c > ma20 or k not in {"ma20_up", "gc"}}
    if rev:
        return {"type": "전환형", "text": f"{first(rev)} · 바닥 다지기"}
    if c > ma60:
        return {"type": "눌림형", "text": f"60일선 위 조정 · 20일선 {d20 * 100:+.0f}%"}
    return {"type": "전환형", "text": "하락 둔화 · 바닥 다지기"}
