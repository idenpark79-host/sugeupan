"""종목별 기술적 의견 — 최근 신호, 국면(가격조정·기간조정 등), 지지·저항, 매물대, 피보나치, 대응 전략.

결과 예)
  tone: "상승" / "중립" / "하락"   (기술적 추세 판단)
  head: "가격조정 진행, 피보나치 61.8% 되돌림 구간"   (technical.report 헤드라인)
  sig:  "헤드앤숄더 넥라인 이탈 · 52주 신저가 경신"     (최근 신호 1~2개)
  ta:   technical.report 전체 (항목표·종합 의견·시나리오·피보나치·매물대)
  pats: [[키, 이름, 며칠 전, 방향, 1주 [평균, 시장대비, 승률, n], 1개월 [...]], ...]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import patterns as P
import technical as T

# 문장에서 우선순위 (앞일수록 먼저)
ORDER = ["hs_top", "inv_hs", "dbl_bottom", "dbl_top", "box_up", "box_dn", "tri_up", "tri_dn", "hi52", "lo52",
         "tl_dn_brk", "tl_up_brk", "base_hi", "base_lo", "tl_up_sup", "tl_dn_rej", "base_mid",
         "cup_handle", "flag_bull", "n_wave", "triple_bottom", "wedge_fall", "wedge_rise", "island_bot", "island_top",
         "period_brk", "prev_high_brk", "prev_low_brk", "retest", "v_rebound", "ath", "limit_up", "limit_dn", "limit_up_pull",
         "ichi_cloud_up", "ichi_cloud_dn", "ma240_up", "ma240_dn", "ma120_up", "ma120_dn", "gc_120", "dc_120",
         "vol_max_up", "vol_bottom_brk", "ma_conv_up", "squeeze_brk", "fib_bounce", "prev_low_sup", "gap_sup", "gap_fill_up",
         "vwap_up", "vwap_dn", "ma240_sup", "ma120_sup",
         "vol_bull", "vol_bear", "morning_star", "evening_star", "bull_engulf", "bear_engulf", "hammer", "shooting",
         "three_white", "three_black", "gap_up", "gap_down", "align_on", "align_off", "gc_long", "dc_long",
         "ma60_up", "ma60_dn", "ma20_up", "ma20_dn", "ma60_sup", "ma60_rej", "ma20_sup", "ma20_rej", "ma5_sup", "ma5_rej",
         "near_high", "fib_zone", "bull_div", "bear_div", "gc", "dc", "macd_gc", "macd_dc", "upper_tail", "lower_tail",
         "doji_top", "pullback", "price_corr", "time_corr", "overheat", "oversold", "rsi_os", "rsi_ob", "bb_up", "bb_dn",
         "ma_conv", "squeeze", "vol_dry", "obv_lead",
         "rising3", "falling3", "harami_bull", "harami_bear", "piercing", "dark_cloud", "inv_hammer", "hanging", "bear_recover",
         "ichi_tk_gc", "ichi_tk_dc", "ichi_chikou", "ichi_twist", "ma20_turn", "ma10_sup", "dmi_gc", "dmi_dc", "adx_up", "adx_dn",
         "sar_buy", "sar_sell", "macd_zero_up", "macd_zero_dn", "rsi_os_exit", "rsi_ob_exit", "rsi50_up", "sto_gc", "sto_dc",
         "cci_up", "cci_dn", "wr_os_exit", "bb_rebound", "env_low", "env_high", "mfi_os", "mfi_ob", "half_off",
         "up_streak", "dn_streak"]
STATE = {"overheat", "oversold", "rsi_os", "rsi_ob", "squeeze", "vol_dry", "pullback", "obv_lead",
         "price_corr", "time_corr", "fib_zone", "ma_conv", "near_high", "half_off", "mfi_os", "mfi_ob", "env_low",
         "up_streak", "dn_streak"}          # 상태형: 오늘 값만
RECENT = {"vol_bull": 2, "vol_bear": 2, "hammer": 2, "shooting": 2, "bull_engulf": 2, "bear_engulf": 2,
          "morning_star": 2, "evening_star": 2, "three_white": 1, "three_black": 1, "gap_up": 1, "gap_down": 1,
          "doji_top": 1, "ma5_sup": 1, "ma5_rej": 1, "upper_tail": 1, "lower_tail": 1,
          "harami_bull": 2, "harami_bear": 2, "piercing": 2, "dark_cloud": 2, "inv_hammer": 2, "hanging": 2, "ma10_sup": 1}   # 그 외 사건형은 최근 5거래일
TONE = {1: "상승", 0: "중립", -1: "하락"}

won = T.won


def _phrase(k, d, ago):
    """신호별 짧은 서술 (가격 포함)."""
    c = d.Close.iloc[-1]
    when = "당일" if ago == 0 else "전일" if ago == 1 else f"{ago}일 전"
    vr = d.Volume.iloc[-1 - ago] / d.VOL_MA20.iloc[-1 - ago] if d.VOL_MA20.iloc[-1 - ago] else np.nan
    ma = lambda n: won(T.rt(float(d[f"MA{n}"].iloc[-1])))
    t = {
        "vol_bull": f"{when} 거래량 {vr:.1f}배 동반 장대양봉",
        "vol_bear": f"{when} 거래량 {vr:.1f}배 동반 장대음봉",
        "hammer": f"{when} 망치형 출현",
        "shooting": f"{when} 유성형 출현",
        "bull_engulf": f"{when} 상승장악형 출현",
        "bear_engulf": f"{when} 하락장악형 출현",
        "morning_star": "샛별형 출현",
        "evening_star": "석별형 출현",
        "three_white": "적삼병 출현",
        "three_black": "흑삼병 출현",
        "gap_up": "상승 갭 발생",
        "gap_down": "하락 갭 발생",
        "doji_top": "고점권 도지",
        "ma5_sup": f"5일선({ma(5)}) 지지",
        "ma5_rej": f"5일선({ma(5)}) 저항에 밀림",
        "ma20_sup": f"20일선({ma(20)}) 지지 반등",
        "ma20_rej": f"20일선({ma(20)}) 저항 확인",
        "ma60_sup": f"60일선({ma(60)}) 지지 반등",
        "ma60_rej": f"60일선({ma(60)}) 저항 확인",
        "ma20_up": f"거래량 동반 20일선({ma(20)}) 돌파",
        "ma20_dn": f"20일선({ma(20)}) 하향 이탈",
        "ma60_up": f"60일선({ma(60)}) 회복",
        "ma60_dn": f"60일선({ma(60)}) 하향 이탈",
        "gc": "단기 골든크로스(5·20일)",
        "dc": "단기 데드크로스(5·20일)",
        "gc_long": "중기 골든크로스(20·60일)",
        "dc_long": "중기 데드크로스(20·60일)",
        "align_on": "이평선 정배열 전환",
        "align_off": "이평선 역배열 전환",
        "hs_top": "헤드앤숄더 넥라인 이탈",
        "inv_hs": "역헤드앤숄더 넥라인 돌파",
        "dbl_bottom": "쌍바닥 넥라인 돌파",
        "dbl_top": "쌍봉 넥라인 이탈",
        "box_up": "60일 박스권 상단 돌파",
        "box_dn": "60일 박스권 하단 이탈",
        "tri_up": "삼각수렴 상방 이탈",
        "tri_dn": "삼각수렴 하방 이탈",
        "hi52": "52주 신고가 경신",
        "lo52": "52주 신저가 경신",
        "pullback": "상승 추세 내 눌림목",
        "overheat": f"20일선 이격 +{(c / d.MA20.iloc[-1] - 1) * 100:.0f}% 단기 과열",
        "oversold": f"20일선 이격 −{(1 - c / d.MA20.iloc[-1]) * 100:.0f}% 낙폭과대",
        "squeeze": "볼린저밴드 수축 · 변동성 확대 임박",
        "bb_up": "볼린저밴드 상단 돌파",
        "bb_dn": "볼린저밴드 하단 이탈",
        "rsi_os": f"RSI {d.RSI.iloc[-1]:.0f} 과매도",
        "rsi_ob": f"RSI {d.RSI.iloc[-1]:.0f} 과매수",
        "bull_div": "RSI 상승 다이버전스",
        "bear_div": "RSI 하락 다이버전스",
        "macd_gc": "MACD 골든크로스",
        "macd_dc": "MACD 데드크로스",
        "vol_dry": "거래량 급감 · 매물 소진",
        "obv_lead": "OBV 선행 신고가 · 매집 신호",
        "price_corr": "가격조정 국면",
        "time_corr": "기간조정 국면",
        "fib_zone": "피보나치 50~61.8% 되돌림",
        "ma_conv": "이평선 수렴",
        "near_high": "전고점 재도전",
        "tl_dn_brk": f"{when} 하락 빗각 돌파",
        "tl_dn_rej": f"{when} 하락 빗각 저항 확인",
        "tl_up_sup": f"{when} 상승 빗각 지지",
        "tl_up_brk": f"{when} 상승 빗각 이탈",
        "base_mid": f"{when} 기준봉 중심값 지지",
        "base_hi": f"{when} 기준봉 고가 돌파",
        "base_lo": f"{when} 기준봉 시가 이탈",
        "prev_high_brk": f"{when} 전고점 돌파", "prev_low_brk": f"{when} 전저점 이탈", "prev_low_sup": f"{when} 전저점 지지",
        "retest": f"{when} 돌파 후 되돌림 지지 · 저항의 지지 전환", "limit_up": f"{when} 상한가", "limit_dn": f"{when} 하한가",
        "limit_up_pull": f"{when} 상한가 봉 중심값 지지", "ma120_up": f"120일선({ma(120)}) 돌파", "ma120_dn": f"120일선({ma(120)}) 이탈",
        "ma120_sup": f"120일선({ma(120)}) 지지", "vol_max_up": f"{when} 1년 최대 거래량 양봉", "vol_bottom_brk": f"{when} 거래량 바닥 후 첫 급증",
        "fib_bounce": f"{when} 피보나치 되돌림 구간 반등", "ath": "상장 후 최고가 경신", "half_off": "52주 고점 대비 반토막",
        "env_low": "엔벨로프(20, −20%) 하단 도달", "up_streak": "5일 연속 상승", "dn_streak": "5일 연속 하락",
        "upper_tail": f"{when} 윗꼬리 매물 출회",
        "lower_tail": f"{when} 아랫꼬리 지지",
    }
    return t.get(k, P.P_NAME[k])


GROUPS = [{"ma20_up", "ma20_dn", "ma20_sup", "ma20_rej"}, {"ma60_up", "ma60_dn", "ma60_sup", "ma60_rej"},
          {"ma5_sup", "ma5_rej"}, {"gc", "dc"}, {"gc_long", "dc_long"}, {"macd_gc", "macd_dc"}, {"align_on", "align_off"},
          {"gap_up", "gap_down"}, {"three_white", "three_black"}, {"vol_bull", "vol_bear"}, {"tri_up", "tri_dn"},
          {"box_up", "box_dn"}, {"dbl_bottom", "dbl_top"}, {"hs_top", "inv_hs"}, {"hi52", "lo52"},
          {"bull_engulf", "bear_engulf", "hammer", "shooting", "morning_star", "evening_star", "upper_tail", "lower_tail"},
          {"bb_up", "bb_dn"}, {"rsi_os", "rsi_ob"}, {"overheat", "oversold"}, {"bull_div", "bear_div"},
          {"price_corr", "time_corr"}, {"tl_dn_brk", "tl_dn_rej"}, {"tl_up_sup", "tl_up_brk"}, {"base_mid", "base_hi", "base_lo"},
          {"harami_bull", "harami_bear", "piercing", "dark_cloud", "inv_hammer", "hanging"}, {"rising3", "falling3"},
          {"limit_up", "limit_dn"}, {"up_streak", "dn_streak"}, {"ma120_up", "ma120_dn", "ma120_sup"}, {"ma240_up", "ma240_dn", "ma240_sup"},
          {"gc_120", "dc_120"}, {"prev_high_brk", "prev_low_brk"}, {"wedge_fall", "wedge_rise"}, {"island_bot", "island_top"},
          {"env_low", "env_high"}, {"vwap_up", "vwap_dn"}, {"sto_gc", "sto_dc"}, {"macd_zero_up", "macd_zero_dn"},
          {"rsi_os_exit", "rsi_ob_exit"}, {"cci_up", "cci_dn"}, {"mfi_os", "mfi_ob"}, {"dmi_gc", "dmi_dc"}, {"adx_up", "adx_dn"},
          {"sar_buy", "sar_sell"}, {"ichi_cloud_up", "ichi_cloud_dn"}, {"ichi_tk_gc", "ichi_tk_dc"}]


def recent(f: pd.DataFrame, window: int = 5):
    """최근 발생 신호 [(키, 며칠 전)] — 우선순위 순, 서로 반대인 신호는 가장 최근 것만."""
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
    ta = T.report(d)
    ma5, ma20, ma60, ma120 = (float(x[f"MA{n}"]) if pd.notna(x[f"MA{n}"]) else np.nan for n in (5, 20, 60, 120))
    s60 = d.MA60.iloc[-1] / d.MA60.iloc[-6] - 1 if len(d) > 70 else 0
    score = 0.0
    if ma5 > ma20 > ma60 > ma120:
        score += 2
    elif ma5 < ma20 < ma60 < ma120:
        score -= 2
    elif c > ma60 and s60 > 0:
        score += 1
    elif c < ma60 and s60 < 0:
        score -= 1
    for k, ago in rec[:8]:
        score += P.P_BIAS[k] * (1.0 if ago <= 1 else 0.6) * (0.8 if P.CAT.get(k) in ("보조지표", "일목균형표") else 1.0)
    if ta:
        score += ta["phase"]["bias"]
    # 수급 연속성
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
    tone = TONE[1 if score >= 2 else -1 if score <= -2 else 0]
    SKIP = {"price_corr", "time_corr", "fib_zone", "ma_conv"}          # 국면·가격대는 ta 헤드라인에서 다룸
    rec_s = [(k, a) for k, a in rec if k not in SKIP]
    sig = " · ".join(_phrase(k, d, ago) for k, ago in rec_s[:2])
    head = ta.get("head") if ta else (sig or "-")
    pts = []
    if ta:
        pts.append(ta["rows"][0][1])
    if sig:
        pts.append(sig)
    more = [_phrase(k, d, ago) for k, ago in rec_s[2:4]]
    if more:
        pts.append(" · ".join(more))
    if flow_txt:
        pts.append(flow_txt)
    S = (ta or {}).get("support") or []
    R = (ta or {}).get("resist") or []
    plan = " · ".join(([f"지지 {won(S[0][0])}({S[0][1]})"] if S else []) + ([f"저항 {won(R[0][0])}({R[0][1]})"] if R else []))
    pats = []
    for k, ago in rec[:12]:
        st = (pstats or {}).get(k, {})
        pats.append([k, P.P_NAME[k], ago, P.P_BIAS[k], st.get("5"), st.get("20")])
    return {"tone": tone, "score": round(score, 1), "head": head, "sig": sig, "points": pts[:4], "plan": plan,
            "act": (ta or {}).get("plan", ""), "flow": flow_txt,
            "support": S[0] if S else None, "resist": R[0] if R else None, "pats": pats, "ta": ta}


def lite(v: dict | None) -> dict | None:
    """추천 목록용 — 차트 오버레이(매물대 구간·피보나치)는 빼고 문장만."""
    if not v:
        return v
    v = dict(v)
    ta = v.get("ta")
    if ta:
        v["ta"] = {k: ta[k] for k in ("head", "summary", "up", "down", "plan") if k in ta}
        v["ta"]["phase"] = {k: ta["phase"][k] for k in ("key", "name", "desc")}
    return v


TYPES = ("낙폭과대 반등", "추세 돌파", "추세 추종", "눌림목", "바닥권 전환")


def perspective(d: pd.DataFrame, view: dict | None) -> dict:
    """추천 종목의 투자 포인트 유형 + 한 줄 설명 (개조식)."""
    x = d.iloc[-1]
    c = float(x.Close)
    ma20, ma60, ma120 = float(x.MA20), float(x.MA60), float(x.MA120) if pd.notna(x.MA120) else np.nan
    d20 = c / ma20 - 1 if ma20 else 0
    r20 = c / float(d.Close.iloc[-21]) - 1 if len(d) > 21 else 0
    rsi = float(x.RSI) if pd.notna(x.RSI) else 50
    s60 = float(d.MA60.iloc[-1] / d.MA60.iloc[-6] - 1) if len(d) > 70 else 0
    keys = {p[0] for p in (view or {}).get("pats", [])}
    ph = ((view or {}).get("ta") or {}).get("phase", {})
    first = lambda ks: _phrase(sorted(ks, key=ORDER.index)[0], d, 0).replace("당일 ", "")
    brk = keys & {"inv_hs", "dbl_bottom", "box_up", "tri_up", "hi52", "bb_up", "ma60_up", "gc_long", "align_on", "vol_bull", "obv_lead", "near_high", "tl_dn_brk", "base_hi",
                   "prev_high_brk", "n_wave", "cup_handle", "flag_bull", "wedge_fall", "triple_bottom", "period_brk", "ichi_cloud_up",
                   "ma120_up", "ma240_up", "squeeze_brk", "ma_conv_up", "ath", "vol_bottom_brk"}
    rev = keys & {"morning_star", "hammer", "bull_engulf", "bull_div", "ma20_up", "gc", "macd_gc", "three_white", "lower_tail", "tl_up_sup", "base_mid",
                   "harami_bull", "piercing", "inv_hammer", "bear_recover", "v_rebound", "island_bot", "prev_low_sup", "fib_bounce",
                   "sto_gc", "rsi_os_exit", "bb_rebound", "sar_buy", "gap_fill_up", "ichi_tk_gc", "dmi_gc", "limit_up_pull"}
    if ph.get("key") == "time_corr":
        return {"type": "눌림목", "text": f"기간조정 {ph.get('days', 0)}영업일째 · 박스 상단 돌파 대기"}
    if c < ma60 and (d20 < -0.05 or rsi < 40 or r20 < -0.1):
        lead = (f"1개월 {T.pc(r20)} {'급락' if r20 <= -0.15 else '조정'}" if r20 <= -0.1 else
                f"20일선 이격 {T.pc(d20)}" if d20 <= -0.05 else f"RSI {rsi:.0f} 과매도권")
        return {"type": "낙폭과대 반등", "text": f"{lead} · 기술적 반등 기대"}
    brk = {k for k in brk if c > ma20 and (k not in {"ma60_up", "gc_long", "align_on"} or c > ma60)}   # 이미 무너진 돌파는 제외
    if brk:
        tail = "상승 추세 가속 기대" if c > ma60 and s60 > 0 else "추세 전환 기대"
        return {"type": "추세 돌파", "text": f"{first(brk)} · {tail}"}
    if c > ma60 and s60 > 0 and d20 > -0.03:
        tail = "눌림 구간 진입" if d20 <= 0.03 else "상승 탄력 유지" if d20 < 0.12 else "단기 과열 유의"
        return {"type": "추세 추종", "text": f"60일선 상회 상승 추세 · {tail}"}
    if ph.get("key") == "price_corr" and np.isfinite(ma120) and c > ma120:
        return {"type": "눌림목", "text": f"{ph.get('desc', '').split(' · ')[0]} · 중기 추세 내 조정"}
    rev = {k for k in rev if c > ma20 or k not in {"ma20_up", "gc"}}
    if rev:
        return {"type": "바닥권 전환", "text": f"{first(rev)} · 바닥권 매수세 유입"}
    if c > ma60:
        return {"type": "눌림목", "text": f"60일선 상회 조정 · 20일선 이격 {T.pc(d20)}"}
    return {"type": "바닥권 전환", "text": "하락 둔화 · 바닥 형성 시도"}
