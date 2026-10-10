"""차트 패턴 탐지 — 캔들, 이동평균 지지·저항, 추세 전환, 가격 구조(헤드앤숄더·이중바닥 등), 다이버전스, 거래량.

모든 신호는 그날 종가까지의 정보만 사용한다 (고점·저점은 좌우 K일로 '확정'된 뒤에만 인식).
통계(stats)는 신호가 처음 나타난 날 다음 날 시가에 사서 5·20거래일 뒤 종가에 판 결과로 계산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (키, 이름, 방향: +1 상승 / -1 하락 / 0 중립, 설명)
PATTERNS = [
    # 캔들
    ("vol_bull", "거래량 실린 장대양봉", 1, "평소 2배 넘는 거래량으로 강하게 올라 마감"),
    ("vol_bear", "거래량 실린 장대음봉", -1, "평소 2배 넘는 거래량으로 강하게 내려 마감"),
    ("hammer", "망치형", 1, "하락 끝 긴 아랫꼬리 — 저가 매수세 유입"),
    ("shooting", "유성형", -1, "상승 끝 긴 윗꼬리 — 고점 매물 출회"),
    ("bull_engulf", "상승장악형", 1, "전날 음봉을 감싸는 양봉 — 매수 우위 전환"),
    ("bear_engulf", "하락장악형", -1, "전날 양봉을 감싸는 음봉 — 매도 우위 전환"),
    ("morning_star", "샛별형", 1, "음봉·작은 봉·양봉 3일 반전"),
    ("evening_star", "석별형", -1, "양봉·작은 봉·음봉 3일 반전"),
    ("three_white", "적삼병", 1, "3일 연속 몸통 있는 양봉"),
    ("three_black", "흑삼병", -1, "3일 연속 몸통 있는 음봉"),
    ("gap_up", "상승 갭", 1, "전날 고가 위에서 거래"),
    ("gap_down", "하락 갭", -1, "전날 저가 아래에서 거래"),
    ("doji_top", "고점 도지", -1, "상승 뒤 시가·종가가 같은 봉 — 방향 고민"),
    # 이동평균
    ("ma5_sup", "5일선 지지", 1, "5일선까지 밀렸다가 위에서 마감"),
    ("ma5_rej", "5일선 저항", -1, "5일선 터치 후 밀려 아래에서 마감"),
    ("ma20_sup", "20일선 지지", 1, "20일선에서 반등"),
    ("ma20_rej", "20일선 저항", -1, "20일선에 막혀 하락"),
    ("ma60_sup", "60일선 지지", 1, "60일선에서 반등"),
    ("ma60_rej", "60일선 저항", -1, "60일선에 막혀 하락"),
    ("ma20_up", "20일선 돌파", 1, "거래량 동반 20일선 상향 돌파"),
    ("ma20_dn", "20일선 이탈", -1, "20일선 하향 이탈"),
    ("ma60_up", "60일선 회복", 1, "60일선 상향 돌파"),
    ("ma60_dn", "60일선 이탈", -1, "60일선 하향 이탈"),
    ("gc", "단기 골든크로스", 1, "5일선이 20일선 상향 돌파"),
    ("dc", "단기 데드크로스", -1, "5일선이 20일선 하향 돌파"),
    ("gc_long", "중기 골든크로스", 1, "20일선이 60일선 상향 돌파"),
    ("dc_long", "중기 데드크로스", -1, "20일선이 60일선 하향 돌파"),
    ("align_on", "정배열 전환", 1, "5>20>60>120일선 순서 완성"),
    ("align_off", "역배열 전환", -1, "5<20<60<120일선 순서 완성"),
    # 구조
    ("hs_top", "헤드앤숄더", -1, "세 고점 중 가운데가 가장 높고 넥라인 이탈"),
    ("inv_hs", "역헤드앤숄더", 1, "세 저점 중 가운데가 가장 낮고 넥라인 돌파"),
    ("dbl_bottom", "쌍바닥 돌파", 1, "비슷한 두 저점 뒤 넥라인(중간 고점) 돌파"),
    ("dbl_top", "쌍봉 이탈", -1, "비슷한 두 고점 뒤 넥라인(중간 저점) 이탈"),
    ("box_up", "박스권 상단 돌파", 1, "60일 가격대 위로 거래량 돌파"),
    ("box_dn", "박스권 하단 이탈", -1, "60일 가격대 아래로 이탈"),
    ("tri_up", "삼각수렴 상방 이탈", 1, "고점은 낮아지고 저점은 높아지다 위로 이탈"),
    ("tri_dn", "삼각수렴 하방 이탈", -1, "수렴 끝에 아래로 이탈"),
    ("hi52", "52주 신고가", 1, "1년 중 가장 높은 종가"),
    ("lo52", "52주 신저가", -1, "1년 중 가장 낮은 종가"),
    ("pullback", "눌림목", 1, "상승 추세에서 거래량 줄며 20일선 부근 조정"),
    ("overheat", "단기 과열", -1, "20일선 대비 +15% 넘게 이격"),
    ("oversold", "낙폭과대", 1, "20일선 대비 −12% 넘게 이격"),
    ("squeeze", "볼린저밴드 수축", 0, "밴드폭이 반년 내 최저권 — 변동성 확대 임박"),
    ("bb_up", "볼린저밴드 상단 돌파", 1, "밴드 위로 마감"),
    ("bb_dn", "볼린저밴드 하단 이탈", -1, "밴드 아래로 마감"),
    # 모멘텀·거래량
    ("rsi_os", "RSI 과매도", 1, "RSI 30 아래"),
    ("rsi_ob", "RSI 과매수", -1, "RSI 70 위"),
    ("bull_div", "RSI 상승 다이버전스", 1, "주가 저점은 낮아졌는데 RSI 저점은 높아짐"),
    ("bear_div", "RSI 하락 다이버전스", -1, "주가 고점은 높아졌는데 RSI 고점은 낮아짐"),
    ("macd_gc", "MACD 골든크로스", 1, "MACD가 시그널 위로"),
    ("macd_dc", "MACD 데드크로스", -1, "MACD가 시그널 아래로"),
    ("vol_dry", "거래량 급감", 0, "5일 평균 거래량이 60일 평균의 절반 이하 — 매물 소진"),
    ("obv_lead", "OBV 선행 신고가", 1, "주가보다 OBV가 먼저 신고가 — 매집 신호"),
    # 국면·가격대
    ("price_corr", "가격조정 진입", 0, "20% 넘게 오른 뒤 고점 대비 15% 이상 단기 하락"),
    ("time_corr", "기간조정", 0, "20% 넘게 오른 뒤 고점 대비 15% 범위에서 15일 이상 횡보"),
    ("fib_zone", "피보나치 50~61.8% 되돌림", 1, "직전 상승폭의 절반가량을 되돌린 자리"),
    ("ma_conv", "이평선 수렴", 0, "5·20·60일선이 3% 안에 모임 — 방향성 결정 구간"),
    ("near_high", "전고점 재도전", 1, "조정 뒤 120일 고점 3% 이내로 복귀"),
    ("upper_tail", "윗꼬리 매물", -1, "고가 대비 크게 밀린 긴 윗꼬리"),
    ("lower_tail", "아랫꼬리 지지", 1, "저가 대비 크게 올라온 긴 아랫꼬리"),
    # 빗각(추세선)·기준봉
    ("tl_dn_brk", "하락 빗각 돌파", 1, "주요 고점과 이후 낮아진 고점을 이은 빗각을 종가로 돌파"),
    ("tl_dn_rej", "하락 빗각 저항", -1, "하락 빗각에 닿은 뒤 밀려 마감"),
    ("tl_up_sup", "상승 빗각 지지", 1, "주요 저점과 이후 높아진 저점을 이은 빗각에서 지지"),
    ("tl_up_brk", "상승 빗각 이탈", -1, "상승 빗각을 종가로 2% 넘게 이탈"),
    ("base_mid", "기준봉 중심값 지지", 1, "거래량 3배·+8% 이상 장대양봉(기준봉)의 절반 가격에서 첫 지지"),
    ("base_hi", "기준봉 고가 돌파", 1, "기준봉 고가를 종가로 돌파"),
    ("base_lo", "기준봉 시가 이탈", -1, "기준봉 시가를 종가로 이탈 — 기준봉 매수세 무력화"),
    # 캔들 (추가)
    ("harami_bull", "상승잉태형", 1, "하락 중 전일 장대음봉 몸통 안에 작은 양봉"),
    ("harami_bear", "하락잉태형", -1, "상승 중 전일 장대양봉 몸통 안에 작은 음봉"),
    ("piercing", "관통형", 1, "전일 음봉 몸통 절반 위까지 회복하는 양봉"),
    ("dark_cloud", "먹구름형", -1, "전일 양봉 몸통 절반 아래까지 밀리는 음봉"),
    ("inv_hammer", "역망치형", 1, "하락 끝 긴 윗꼬리·작은 몸통 — 매수세 시험"),
    ("hanging", "교수형", -1, "상승 끝 긴 아랫꼬리·작은 몸통 — 매도세 출현"),
    ("rising3", "상승삼법", 1, "장대양봉 뒤 작은 조정 봉 3개, 다시 장대양봉으로 고점 경신"),
    ("falling3", "하락삼법", -1, "장대음봉 뒤 작은 반등 봉 3개, 다시 장대음봉으로 저점 경신"),
    ("bear_recover", "장대음봉 시가 회복", 1, "5일 내 장대음봉의 시가를 종가로 회복 — 하락 무력화"),
    ("limit_up", "상한가", 1, "가격제한폭(+30%)까지 상승 마감"),
    ("limit_up_pull", "상한가 후 눌림 지지", 1, "10일 내 상한가 봉의 중심값까지 눌린 뒤 지지"),
    ("limit_dn", "하한가", -1, "가격제한폭(−30%)까지 하락 마감"),
    ("up_streak", "5일 연속 상승", 0, "5거래일 연속 종가 상승"),
    ("dn_streak", "5일 연속 하락", 0, "5거래일 연속 종가 하락"),
    # 이동평균 (추가)
    ("ma10_sup", "10일선 지지", 1, "10일선까지 밀렸다가 위에서 마감"),
    ("ma120_up", "120일선 돌파", 1, "6개월선(120일) 상향 돌파"),
    ("ma120_dn", "120일선 이탈", -1, "6개월선(120일) 하향 이탈"),
    ("ma120_sup", "120일선 지지", 1, "120일선에서 반등"),
    ("ma240_up", "240일선 돌파", 1, "1년선(240일) 상향 돌파"),
    ("ma240_dn", "240일선 이탈", -1, "1년선(240일) 하향 이탈"),
    ("ma240_sup", "240일선 지지", 1, "240일선에서 반등"),
    ("gc_120", "장기 골든크로스", 1, "60일선이 120일선 상향 돌파"),
    ("dc_120", "장기 데드크로스", -1, "60일선이 120일선 하향 돌파"),
    ("ma20_turn", "20일선 상승 전환", 1, "10일 넘게 내리던 20일선이 상승으로 전환"),
    ("ma_conv_up", "이평선 수렴 후 상방 발산", 1, "5·20·60일선 3% 이내 수렴 뒤 거래량 동반 상승 이탈"),
    # 차트 패턴 (추가)
    ("prev_high_brk", "전고점 돌파", 1, "직전 고점(스윙 하이)을 종가로 돌파"),
    ("prev_low_brk", "전저점 이탈", -1, "직전 저점(스윙 로우)을 종가로 이탈"),
    ("prev_low_sup", "전저점 지지", 1, "직전 저점 ±2%에서 양봉 반등"),
    ("retest", "돌파 후 되돌림 지지", 1, "전고점 돌파 뒤 돌파 가격까지 눌렸다 지지 — 저항이 지지로 전환"),
    ("n_wave", "N자형 상승", 1, "상승 → 30~70% 되돌림(더 높은 저점) → 직전 고점 돌파"),
    ("cup_handle", "컵앤핸들", 1, "U자형 바닥 뒤 짧은 손잡이 조정, 컵 고점 돌파"),
    ("flag_bull", "상승 깃발형", 1, "10일 내 +15% 급등 후 4~15일 좁은 조정, 깃대 고점 돌파"),
    ("wedge_fall", "하락 쐐기 상방 이탈", 1, "고점·저점이 함께 낮아지며 좁아지다 위로 이탈"),
    ("wedge_rise", "상승 쐐기 하방 이탈", -1, "고점·저점이 함께 높아지며 좁아지다 아래로 이탈"),
    ("triple_bottom", "삼중바닥", 1, "비슷한 세 저점 뒤 넥라인 돌파"),
    ("v_rebound", "V자 반등", 1, "10일 내 −20% 급락 후 5일 안에 낙폭 절반 회복"),
    ("island_bot", "섬꼴 반전(바닥)", 1, "하락 갭과 상승 갭 사이에 고립된 저점 구간"),
    ("island_top", "섬꼴 반전(천장)", -1, "상승 갭과 하락 갭 사이에 고립된 고점 구간"),
    ("gap_sup", "상승 갭 지지", 1, "20일 내 미체결 상승 갭까지 눌렸다 지지"),
    ("gap_fill_up", "하락 갭 메움", 1, "20일 내 하락 갭을 상승으로 메움"),
    ("ath", "상장 후 최고가", 1, "보유 데이터 전 기간 최고 종가 경신"),
    # 국면·가격대 (추가)
    ("period_brk", "기간조정 후 돌파", 1, "기간조정 횡보 뒤 직전 고점 돌파"),
    ("fib_bounce", "피보나치 되돌림 반등", 1, "직전 상승폭 38.2~61.8% 되돌림 구간에서 양봉 반등"),
    ("half_off", "고점 대비 반토막", 0, "52주 고점 대비 −50% 이하"),
    ("env_low", "엔벨로프 하단 터치", 1, "20일선 −20% 엔벨로프 하단 도달"),
    ("env_high", "엔벨로프 상단 돌파", -1, "20일선 +20% 엔벨로프 상단 돌파 — 단기 과열"),
    ("vwap_up", "120일 평단가 돌파", 1, "최근 120일 거래량가중평균가(시장 평단가) 상향 돌파"),
    ("vwap_dn", "120일 평단가 이탈", -1, "최근 120일 거래량가중평균가 하향 이탈"),
    ("vol_max_up", "1년 최대 거래량 양봉", 1, "1년 중 가장 많은 거래량을 실은 양봉"),
    ("vol_bottom_brk", "거래량 바닥 후 첫 급증", 1, "60일 최저 수준 거래량 이후 평소 3배 거래량 양봉"),
    # 보조지표 (추가)
    ("sto_gc", "스토캐스틱 침체권 골든크로스", 1, "%K가 %D를 20 이하에서 상향 돌파"),
    ("sto_dc", "스토캐스틱 과열권 데드크로스", -1, "%K가 %D를 80 이상에서 하향 돌파"),
    ("macd_zero_up", "MACD 0선 상향 돌파", 1, "MACD가 0 위로"),
    ("macd_zero_dn", "MACD 0선 하향 이탈", -1, "MACD가 0 아래로"),
    ("rsi_os_exit", "RSI 과매도 탈출", 1, "RSI가 30을 상향 돌파"),
    ("rsi_ob_exit", "RSI 과매수 이탈", -1, "RSI가 70을 하향 이탈"),
    ("rsi50_up", "RSI 50 돌파", 1, "RSI가 50을 상향 돌파 — 매수 우위 전환"),
    ("cci_up", "CCI −100 상향 돌파", 1, "CCI(20)가 −100 위로"),
    ("cci_dn", "CCI +100 하향 이탈", -1, "CCI(20)가 +100 아래로"),
    ("mfi_os", "MFI 과매도", 1, "MFI(14) 20 이하 — 자금 유출 과다"),
    ("mfi_ob", "MFI 과매수", -1, "MFI(14) 80 이상 — 자금 유입 과열"),
    ("wr_os_exit", "Williams %R 과매도 탈출", 1, "%R이 −80을 상향 돌파"),
    ("dmi_gc", "DMI 골든크로스", 1, "+DI가 −DI 상향 돌파 (ADX 20 이상)"),
    ("dmi_dc", "DMI 데드크로스", -1, "+DI가 −DI 하향 돌파 (ADX 20 이상)"),
    ("adx_up", "ADX 상승 추세 강화", 1, "ADX 25 상향 돌파 · +DI 우위"),
    ("adx_dn", "ADX 하락 추세 강화", -1, "ADX 25 상향 돌파 · −DI 우위"),
    ("sar_buy", "파라볼릭 SAR 매수 전환", 1, "SAR이 주가 아래로 전환"),
    ("sar_sell", "파라볼릭 SAR 매도 전환", -1, "SAR이 주가 위로 전환"),
    ("bb_rebound", "볼린저밴드 하단 복귀", 1, "하단 이탈 뒤 밴드 안으로 복귀"),
    ("squeeze_brk", "볼린저 수축 후 상단 돌파", 1, "밴드폭 반년 최저권에서 상단 돌파"),
    # 일목균형표
    ("ichi_cloud_up", "구름대 상향 돌파", 1, "주가가 일목 구름대 위로 돌파"),
    ("ichi_cloud_dn", "구름대 하향 이탈", -1, "주가가 일목 구름대 아래로 이탈"),
    ("ichi_tk_gc", "전환선·기준선 호전", 1, "전환선(9)이 기준선(26) 상향 돌파"),
    ("ichi_tk_dc", "전환선·기준선 역전", -1, "전환선이 기준선 하향 돌파"),
    ("ichi_chikou", "후행스팬 주가 돌파", 1, "종가가 26일 전 종가를 상향 돌파"),
    ("ichi_twist", "양운 전환", 1, "선행스팬1이 선행스팬2 상향 돌파 — 구름 색 전환"),
]
_CAT_START = [("vol_bull", "캔들"), ("ma5_sup", "이동평균"), ("hs_top", "차트 패턴"), ("rsi_os", "보조지표"),
              ("price_corr", "국면·가격대"), ("upper_tail", "캔들"), ("tl_dn_brk", "빗각·기준봉"),
              ("harami_bull", "캔들"), ("ma10_sup", "이동평균"), ("prev_high_brk", "차트 패턴"), ("period_brk", "국면·가격대"),
              ("sto_gc", "보조지표"), ("ichi_cloud_up", "일목균형표")]
CAT = {}
_c = None
for _k, *_ in PATTERNS:
    _c = next((g for k0, g in _CAT_START if k0 == _k), _c)
    CAT[_k] = _c
CAT.update({"squeeze": "보조지표", "bb_up": "보조지표", "bb_dn": "보조지표", "pullback": "국면·가격대",
            "overheat": "국면·가격대", "oversold": "국면·가격대", "hi52": "국면·가격대", "lo52": "국면·가격대"})
P_NAME = {k: n for k, n, _, _ in PATTERNS}
P_BIAS = {k: b for k, _, b, _ in PATTERNS}
P_DESC = {k: d for k, _, _, d in PATTERNS}
KEYS = [k for k, *_ in PATTERNS]
K_PIV = 5                      # 고점·저점 확정에 필요한 좌우 봉 수


def _pivots(h: np.ndarray, l: np.ndarray, k: int = K_PIV):
    """t 시점에 '확정'된 고점·저점 (t-k가 앞뒤 k봉 중 최고/최저)."""
    n = len(h)
    ph = np.zeros(n, bool)
    pl = np.zeros(n, bool)
    for t in range(2 * k, n):
        w = slice(t - 2 * k, t + 1)
        c = t - k
        if h[c] == h[w].max() and h[c] > h[c - 1]:
            ph[t] = True
        if l[c] == l[w].min() and l[c] < l[c - 1]:
            pl[t] = True
    return ph, pl


def _structure(d: pd.DataFrame) -> dict:
    """확정 고점·저점 순서로 헤드앤숄더·이중바닥/천장·삼각수렴·다이버전스 판정."""
    h, l, c, rsi = d.High.values, d.Low.values, d.Close.values, d.RSI.values
    n = len(c)
    ph, pl = _pivots(h, l)
    out = {k: np.zeros(n, bool) for k in ("hs_top", "inv_hs", "dbl_bottom", "dbl_top", "tri_up", "tri_dn", "bull_div", "bear_div")}
    highs, lows = [], []                 # (봉 위치, 가격, RSI)
    hs_neck = ihs_neck = db_neck = dt_neck = None
    tri_hi = tri_lo = None
    for t in range(n):
        if ph[t]:
            i = t - K_PIV
            highs.append((i, h[i], rsi[i]))
            highs = highs[-4:]
            if len(highs) >= 2:
                (i1, p1, r1), (i2, p2, r2) = highs[-2], highs[-1]
                if p2 > p1 * 1.0 and r2 < r1 - 3 and np.isfinite(r1) and np.isfinite(r2) and i2 - i1 <= 40:
                    out["bear_div"][t] = True
                # 이중천장: 비슷한 두 고점, 사이 저점
                if i2 - i1 >= 8 and abs(p2 / p1 - 1) <= 0.03:
                    mid = [p for j, p, _ in lows if i1 < j < i2]
                    if mid and min(mid) < min(p1, p2) * 0.95:
                        dt_neck = (min(mid), t + 30)
            if len(highs) >= 3:
                (i1, p1, _), (i2, p2, _), (i3, p3, _) = highs[-3:]
                if p2 >= max(p1, p3) * 1.03 and abs(p3 / p1 - 1) <= 0.06 and i3 - i1 <= 90:
                    mid = [p for j, p, _ in lows if i1 < j < i3]
                    if mid:
                        hs_neck = (min(mid), t + 30)
        if pl[t]:
            i = t - K_PIV
            lows.append((i, l[i], rsi[i]))
            lows = lows[-4:]
            if len(lows) >= 2:
                (i1, p1, r1), (i2, p2, r2) = lows[-2], lows[-1]
                if p2 < p1 and r2 > r1 + 3 and np.isfinite(r1) and np.isfinite(r2) and i2 - i1 <= 40:
                    out["bull_div"][t] = True
                if i2 - i1 >= 8 and abs(p2 / p1 - 1) <= 0.03:
                    mid = [p for j, p, _ in highs if i1 < j < i2]
                    if mid and max(mid) > max(p1, p2) * 1.05:
                        db_neck = (max(mid), t + 30)
            if len(lows) >= 3:
                (i1, p1, _), (i2, p2, _), (i3, p3, _) = lows[-3:]
                if p2 <= min(p1, p3) * 0.97 and abs(p3 / p1 - 1) <= 0.06 and i3 - i1 <= 90:
                    mid = [p for j, p, _ in highs if i1 < j < i3]
                    if mid:
                        ihs_neck = (max(mid), t + 30)
        # 삼각수렴: 최근 두 고점 하락 + 두 저점 상승
        if (ph[t] or pl[t]) and len(highs) >= 2 and len(lows) >= 2:
            if highs[-1][1] < highs[-2][1] and lows[-1][1] > lows[-2][1] and t - min(highs[-2][0], lows[-2][0]) <= 60:
                tri_hi, tri_lo = (highs[-1][1], t + 20), (lows[-1][1], t + 20)
        if t == 0:
            continue
        # 목선·수렴선 돌파 확인 (유효기간 안에서 처음 넘는 날)
        if hs_neck and t <= hs_neck[1] and c[t] < hs_neck[0] <= c[t - 1]:
            out["hs_top"][t] = True; hs_neck = None
        if ihs_neck and t <= ihs_neck[1] and c[t] > ihs_neck[0] >= c[t - 1]:
            out["inv_hs"][t] = True; ihs_neck = None
        if db_neck and t <= db_neck[1] and c[t] > db_neck[0] >= c[t - 1]:
            out["dbl_bottom"][t] = True; db_neck = None
        if dt_neck and t <= dt_neck[1] and c[t] < dt_neck[0] <= c[t - 1]:
            out["dbl_top"][t] = True; dt_neck = None
        if tri_hi and t <= tri_hi[1]:
            if c[t] > tri_hi[0] >= c[t - 1]:
                out["tri_up"][t] = True; tri_hi = tri_lo = None
            elif tri_lo and c[t] < tri_lo[0] <= c[t - 1]:
                out["tri_dn"][t] = True; tri_hi = tri_lo = None
    return out


def flags(d: pd.DataFrame) -> pd.DataFrame:
    """지표가 붙은 일봉(indicators.add_indicators 결과)에서 패턴 신호표 (행: 날짜, 열: 패턴)."""
    o, h, l, c, v = d.Open, d.High, d.Low, d.Close, d.Volume
    p = d.shift(1)
    rng_ = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    up_t = h - np.maximum(c, o)
    lo_t = np.minimum(c, o) - l
    atr = d.ATR
    vr = v / d.VOL_MA20.replace(0, np.nan)
    ret5 = c / c.shift(5) - 1
    green, red = c > o, c < o
    big = body >= 0.6 * atr
    r = {}
    r["vol_bull"] = green & big & (vr >= 2) & (c > p.Close) & (up_t <= body * 0.5)
    r["vol_bear"] = red & big & (vr >= 2) & (c < p.Close)
    r["hammer"] = (lo_t >= 2 * body) & (up_t <= 0.3 * rng_) & (ret5 < -0.03) & (l <= l.rolling(10).min())
    r["shooting"] = (up_t >= 2 * body) & (lo_t <= 0.3 * rng_) & (ret5 > 0.03) & (h >= h.rolling(10).max())
    r["bull_engulf"] = (p.Close < p.Open) & green & (o <= p.Close) & (c >= p.Open) & (ret5 < 0)
    r["bear_engulf"] = (p.Close > p.Open) & red & (o >= p.Close) & (c <= p.Open) & (ret5 > 0)
    p2 = d.shift(2)
    small1 = (p.Close - p.Open).abs() <= 0.3 * (p2.Open - p2.Close).abs()
    r["morning_star"] = (p2.Close < p2.Open) & ((p2.Open - p2.Close) >= 0.6 * atr) & small1 & green & (c >= (p2.Open + p2.Close) / 2)
    r["evening_star"] = (p2.Close > p2.Open) & ((p2.Close - p2.Open) >= 0.6 * atr) & small1 & red & (c <= (p2.Open + p2.Close) / 2)
    b3 = lambda s: s.rolling(3).sum() == 3
    r["three_white"] = b3((green & (body >= 0.3 * atr) & (c > p.Close)).astype(int))
    r["three_black"] = b3((red & (body >= 0.3 * atr) & (c < p.Close)).astype(int))
    r["gap_up"] = l > p.High
    r["gap_down"] = h < p.Low
    r["doji_top"] = (body <= 0.1 * rng_) & (ret5 > 0.05) & (h >= h.rolling(20).max())
    # 이동평균 지지·저항
    for n in (5, 20, 60):
        ma, pma = d[f"MA{n}"], p[f"MA{n}"]
        slope = ma - ma.shift(3)
        r[f"ma{n}_sup"] = (p.Close > pma) & (l <= ma * 1.005) & (c >= ma) & (slope > 0)
        r[f"ma{n}_rej"] = (p.Close < pma) & (h >= ma * 0.995) & (c < ma) & (slope < 0)
    r["ma20_up"] = (c > d.MA20) & (p.Close <= p.MA20) & (vr >= 1.5)
    r["ma20_dn"] = (c < d.MA20) & (p.Close >= p.MA20)
    r["ma60_up"] = (c > d.MA60) & (p.Close <= p.MA60)
    r["ma60_dn"] = (c < d.MA60) & (p.Close >= p.MA60)
    r["gc"] = (d.MA5 > d.MA20) & (p.MA5 <= p.MA20)
    r["dc"] = (d.MA5 < d.MA20) & (p.MA5 >= p.MA20)
    r["gc_long"] = (d.MA20 > d.MA60) & (p.MA20 <= p.MA60)
    r["dc_long"] = (d.MA20 < d.MA60) & (p.MA20 >= p.MA60)
    al = (d.MA5 > d.MA20) & (d.MA20 > d.MA60) & (d.MA60 > d.MA120)
    ra = (d.MA5 < d.MA20) & (d.MA20 < d.MA60) & (d.MA60 < d.MA120)
    r["align_on"] = al & ~al.shift(1, fill_value=False)
    r["align_off"] = ra & ~ra.shift(1, fill_value=False)
    # 구조
    hh60, ll60 = h.rolling(60).max().shift(1), l.rolling(60).min().shift(1)
    box = (hh60 / ll60 - 1) <= 0.25
    r["box_up"] = box & (c > hh60) & (p.Close <= hh60) & (vr >= 1.5)
    r["box_dn"] = box & (c < ll60) & (p.Close >= ll60)
    prev_hi = c.rolling(252, min_periods=200).max().shift(1)
    prev_lo = c.rolling(252, min_periods=200).min().shift(1)
    r["hi52"] = c > prev_hi
    r["lo52"] = c < prev_lo
    up_tr = (d.MA20 > d.MA60) & (d.MA60 > d.MA60.shift(5)) & (c / c.shift(40) - 1 > 0.05)
    r["pullback"] = up_tr & ((c / d.MA20 - 1).between(-0.02, 0.025)) & (v.rolling(5).mean() < 0.8 * d.VOL_MA20) & (c < c.rolling(10).max() * 0.97)
    r["overheat"] = c / d.MA20 - 1 > 0.15
    r["oversold"] = c / d.MA20 - 1 < -0.12
    r["squeeze"] = d.BB_width <= d.BB_width.rolling(120, min_periods=60).quantile(0.1)
    r["bb_up"] = (c > d.BB_upper) & (p.Close <= p.BB_upper)
    r["bb_dn"] = (c < d.BB_lower) & (p.Close >= p.BB_lower)
    r["rsi_os"] = d.RSI < 30
    r["rsi_ob"] = d.RSI > 70
    r["macd_gc"] = (d.MACD > d.MACD_signal) & (p.MACD <= p.MACD_signal)
    r["macd_dc"] = (d.MACD < d.MACD_signal) & (p.MACD >= p.MACD_signal)
    r["vol_dry"] = v.rolling(5).mean() < 0.5 * v.rolling(60).mean()
    r["obv_lead"] = (d.OBV >= d.OBV.rolling(60).max()) & (c < c.rolling(60).max() * 0.97)
    st = _structure(d)
    for k, a in st.items():
        r[k] = a
    for k, a in _swing_flags(d).items():
        r[k] = a
    r["ma_conv"] = (pd.concat([d.MA5, d.MA20, d.MA60], axis=1).max(axis=1) - pd.concat([d.MA5, d.MA20, d.MA60], axis=1).min(axis=1)) / c <= 0.03
    r["upper_tail"] = (up_t >= 0.55 * rng_) & (rng_ >= 1.2 * atr)
    r["lower_tail"] = (lo_t >= 0.55 * rng_) & (rng_ >= 1.2 * atr)
    for k, a in diag(d)[0].items():
        r[k] = a
    for k, a in base_candle(d)[0].items():
        r[k] = a
    for k, a in ext_flags(d).items():
        r[k] = a
    r = pd.DataFrame({k: np.asarray(r[k]) for k in KEYS}, index=d.index)
    return r.fillna(False).astype(bool)


def _swing_flags(d: pd.DataFrame, look: int = 120) -> dict:
    """120일 고점 기준 가격조정·기간조정·피보나치 되돌림·전고점 재도전 (그날까지의 정보만)."""
    from numpy.lib.stride_tricks import sliding_window_view as sw
    h, l, c = d.High.values.astype(float), d.Low.values.astype(float), d.Close.values.astype(float)
    n = len(c)
    out = {k: np.zeros(n, bool) for k in ("price_corr", "time_corr", "fib_zone", "near_high", "fib_bounce")}
    if n < 2 * look:
        return out
    W = sw(h, look)                                   # W[j] = h[j : j+look], 끝 = t = j+look-1
    Wl = sw(l, look)
    t = np.arange(look - 1, n)
    last = look - 1 - np.argmax(W[:, ::-1], axis=1)   # 창 안에서 가장 최근 최고가 위치
    ih = t - (look - 1) + last
    H = h[ih]
    lmin = pd.Series(l).rolling(look, min_periods=20).min().values
    L = lmin[ih]                                      # 고점 이전 120일 최저가
    pos = np.arange(look)[None, :]
    after = np.where(pos > last[:, None], Wl, np.inf)
    LL = np.where(np.isfinite(after.min(axis=1)), after.min(axis=1), H)
    days = t - ih
    C = c[t]
    with np.errstate(divide="ignore", invalid="ignore"):
        dd, up, rng_, retr = C / H - 1, H / L - 1, H / LL - 1, (H - C) / (H - L)
    pcorr = (up >= 0.2) & (dd <= -0.15) & (days >= 3) & (days <= 40)
    tcorr = (up >= 0.2) & (days >= 15) & (days <= 80) & (rng_ <= 0.15)
    fz = (up >= 0.25) & (retr >= 0.45) & (retr <= 0.66) & (days >= 3)
    nh = (days >= 10) & (dd >= -0.03) & (dd < 0) & (LL / H - 1 <= -0.08)
    o = d.Open.values.astype(float)
    pcl = np.r_[np.nan, c[:-1]][t]
    fb = (up >= 0.2) & (retr >= 0.382) & (retr <= 0.618) & (days >= 5) & (C > o[t]) & (C > pcl) & ((C - o[t]) >= 0.5 * (h[t] - l[t]))
    for k, a in (("price_corr", pcorr), ("time_corr", tcorr), ("fib_zone", fz), ("near_high", nh), ("fib_bounce", fb)):
        out[k][t] = np.nan_to_num(a, nan=0).astype(bool)
    return out


# ── 빗각(추세선) ────────────────────────────────────────────────
def _fit_line(pts, side, c, t, min_gap=10):
    """side=-1: 가장 높은 고점 → 이후 가장 최근의 더 낮은 고점 (하락 빗각)
       side=+1: 가장 낮은 저점 → 이후 가장 최근의 더 높은 저점 (상승 빗각)
       고점(저점)부터 t까지 종가가 선을 1% 넘게 넘어선(밑돈) 적이 없어야 유효."""
    if len(pts) < 2:
        return None
    a = max(pts, key=lambda z: (z[1], -z[0])) if side < 0 else min(pts, key=lambda z: (z[1], -z[0]))
    i1, p1 = a
    for i2, p2 in sorted([z for z in pts if z[0] - i1 >= min_gap], reverse=True):
        if (side < 0 and p2 >= p1 * 0.98) or (side > 0 and p2 <= p1 * 1.02):
            continue
        s = (p2 - p1) / (i2 - i1)
        xs = np.arange(i1, t + 1)
        ln = p1 + s * (xs - i1)
        seg = c[i1:t + 1]
        if (side < 0 and np.all(seg <= ln * 1.01)) or (side > 0 and np.all(seg >= ln * 0.99)):
            return {"i1": int(i1), "p1": float(p1), "i2": int(i2), "p2": float(p2), "s": float(s), "t0": int(t),
                    "brk": None, "evt": -99}
    return None


def diag(d: pd.DataFrame, k: int = K_PIV, span: int = 160, fresh: int = 90):
    """빗각 신호(그날까지의 정보만) + 마지막 날 기준 하락·상승 빗각."""
    h, l, c = d.High.values.astype(float), d.Low.values.astype(float), d.Close.values.astype(float)
    n = len(c)
    out = {key: np.zeros(n, bool) for key in ("tl_dn_brk", "tl_dn_rej", "tl_up_sup", "tl_up_brk")}
    ph, pl = _pivots(h, l, k)
    HI, LO = [], []
    dn = up = None
    for t in range(1, n):
        if ph[t]:
            HI = [z for z in HI if t - z[0] <= span] + [(t - k, h[t - k])]
            new = _fit_line(HI, -1, c, t)
            if new and (dn is None or (new["i1"], new["i2"]) != (dn["i1"], dn["i2"])):
                dn = new
        if pl[t]:
            LO = [z for z in LO if t - z[0] <= span] + [(t - k, l[t - k])]
            new = _fit_line(LO, 1, c, t)
            if new and (up is None or (new["i1"], new["i2"]) != (up["i1"], up["i2"])):
                up = new
        if dn is not None:
            if t - dn["i2"] > fresh and dn["brk"] is None:
                dn = None
            elif dn["brk"] is None and t > dn["t0"]:
                v, v0 = dn["p1"] + dn["s"] * (t - dn["i1"]), dn["p1"] + dn["s"] * (t - 1 - dn["i1"])
                if c[t] > v * 1.005 and c[t - 1] <= v0 * 1.005:
                    out["tl_dn_brk"][t] = True
                    dn["brk"] = t
                elif h[t] >= v * 0.995 and c[t] < v and t - dn["evt"] >= 5:
                    out["tl_dn_rej"][t] = True
                    dn["evt"] = t
        if up is not None:
            if t - up["i2"] > fresh and up["brk"] is None:
                up = None
            elif up["brk"] is None and t > up["t0"]:
                v, v0 = up["p1"] + up["s"] * (t - up["i1"]), up["p1"] + up["s"] * (t - 1 - up["i1"])
                if c[t] < v * 0.98 and c[t - 1] >= v0 * 0.98:
                    out["tl_up_brk"][t] = True
                    up["brk"] = t
                elif l[t] <= v * 1.01 and c[t] >= v and c[t - 1] > v0 and t - up["evt"] >= 5:
                    out["tl_up_sup"][t] = True
                    up["evt"] = t
    return out, dn, up


# ── 기준봉 ──────────────────────────────────────────────────────
def base_candle(d: pd.DataFrame, life: int = 60):
    """기준봉(거래량 20일 평균 3배 이상 · 전일 대비 +8% 이상 · 몸통 +5% 이상 양봉) 이후 신호 + 마지막 기준봉."""
    o, h, l, c, v = (d[x].values.astype(float) for x in ("Open", "High", "Low", "Close", "Volume"))
    vm = d.VOL_MA20.shift(1).values.astype(float)
    n = len(c)
    out = {key: np.zeros(n, bool) for key in ("base_mid", "base_hi", "base_lo")}
    cur = None
    for t in range(1, n):
        if vm[t] > 0 and v[t] >= 3 * vm[t] and c[t - 1] > 0 and c[t] / c[t - 1] - 1 >= 0.08 and o[t] > 0 and c[t] / o[t] - 1 >= 0.05:
            cur = {"i": t, "o": o[t], "h": h[t], "l": l[t], "c": c[t], "m": (o[t] + c[t]) / 2, "vr": v[t] / vm[t],
                   "mid": None, "hi": None, "lo": None, "low_after": np.inf}
            continue
        if cur is None:
            continue
        if t - cur["i"] > life:
            cur = None
            continue
        if t - cur["i"] >= 2:
            if cur["lo"] is None and c[t] < cur["o"] and c[t - 1] >= cur["o"]:
                out["base_lo"][t] = True
                cur["lo"] = t
            elif cur["mid"] is None and cur["lo"] is None and l[t] <= cur["m"] * 1.01 and c[t] >= cur["m"] and cur["low_after"] > cur["m"] * 1.01:
                out["base_mid"][t] = True
                cur["mid"] = t
            if cur["hi"] is None and t - cur["i"] >= 5 and c[t] > cur["h"] and c[t - 1] <= cur["h"]:
                out["base_hi"][t] = True
                cur["hi"] = t
        cur["low_after"] = min(cur["low_after"], l[t])
    return out, cur


# ── 확장 신호: 캔들·장기선·지표·일목·가격 구조 ───────────────────────
def ext_indicators(d: pd.DataFrame) -> pd.DataFrame:
    """추가 지표 (일목균형표·CCI·MFI·Williams %R·SAR·평단가·장기선) — 모두 그날까지의 정보만."""
    h, l, c, v = d.High, d.Low, d.Close, d.Volume.astype(float)
    x = pd.DataFrame(index=d.index)
    x["MA10"] = c.rolling(10).mean()
    x["MA240"] = c.rolling(240).mean()
    tk = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kj = (h.rolling(26).max() + l.rolling(26).min()) / 2
    sa = (tk + kj) / 2
    sb = (h.rolling(52).max() + l.rolling(52).min()) / 2
    x["TENKAN"], x["KIJUN"], x["SPAN_A"], x["SPAN_B"] = tk, kj, sa, sb          # 선행스팬은 원래 26일 앞에 그림
    x["CLOUD_TOP"] = np.maximum(sa.shift(26), sb.shift(26))                        # 오늘 위치의 구름 = 26일 전에 계산된 값
    x["CLOUD_BOT"] = np.minimum(sa.shift(26), sb.shift(26))
    tp = (h + l + c) / 3
    sma = tp.rolling(20).mean()
    mad = (tp - sma).abs().rolling(20).mean()
    x["CCI"] = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    x["MFI"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    x["WR"] = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    x["VWAP120"] = (c * v).rolling(120).sum() / v.rolling(120).sum().replace(0, np.nan)
    x["SAR"] = _sar(h.values.astype(float), l.values.astype(float))
    return x


def _sar(h, l, step=0.02, mx=0.2):
    n = len(h)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    up, af, ep, sar = True, step, h[0], l[0]
    for t in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[t - 1], l[t - 2] if t >= 2 else l[t - 1])
            if l[t] < sar:
                up, sar, ep, af = False, ep, l[t], step
            elif h[t] > ep:
                ep, af = h[t], min(af + step, mx)
        else:
            sar = max(sar, h[t - 1], h[t - 2] if t >= 2 else h[t - 1])
            if h[t] > sar:
                up, sar, ep, af = True, ep, h[t], step
            elif l[t] < ep:
                ep, af = l[t], min(af + step, mx)
        out[t] = sar
    return out


def _cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def ext_flags(d: pd.DataFrame, x: pd.DataFrame | None = None) -> dict:
    if x is None:
        x = ext_indicators(d)
    o, h, l, c, v = d.Open, d.High, d.Low, d.Close, d.Volume.astype(float)
    p = d.shift(1)
    atr = d.ATR
    body = (c - o).abs()
    pbody = (p.Close - p.Open).abs()
    rng_ = (h - l).replace(0, np.nan)
    up_t = h - np.maximum(c, o)
    lo_t = np.minimum(c, o) - l
    ret5 = c / c.shift(5) - 1
    green, red = c > o, c < o
    vr = v / d.VOL_MA20.shift(1).replace(0, np.nan)
    chg = c / p.Close - 1
    r = {}
    # 캔들
    big_p = pbody >= 0.8 * atr
    inside = (np.maximum(c, o) <= np.maximum(p.Close, p.Open)) & (np.minimum(c, o) >= np.minimum(p.Close, p.Open)) & (body <= 0.5 * pbody)
    r["harami_bull"] = (p.Close < p.Open) & big_p & inside & green & (ret5 < -0.03)
    r["harami_bear"] = (p.Close > p.Open) & big_p & inside & red & (ret5 > 0.03)
    pmid = (p.Open + p.Close) / 2
    r["piercing"] = (p.Close < p.Open) & big_p & green & (o < p.Close) & (c > pmid) & (c < p.Open)
    r["dark_cloud"] = (p.Close > p.Open) & big_p & red & (o > p.Close) & (c < pmid) & (c > p.Open)
    small = body <= 0.35 * rng_
    r["inv_hammer"] = small & (up_t >= 2 * body) & (lo_t <= 0.15 * rng_) & (ret5 < -0.04) & (l <= l.rolling(10).min())
    r["hanging"] = small & (lo_t >= 2 * body) & (up_t <= 0.15 * rng_) & (ret5 > 0.05) & (h >= h.rolling(10).max())
    b4 = d.shift(4)
    bigg = lambda oo, cc: (cc - oo) >= 0.8 * atr.shift(4)
    mid3_in = lambda: (h.shift(1).rolling(3).max() <= b4.High) & (l.shift(1).rolling(3).min() >= b4.Low) & \
        ((c.shift(1) - o.shift(1)).abs().rolling(3).max() <= 0.6 * (b4.Close - b4.Open).abs())
    r["rising3"] = bigg(b4.Open, b4.Close) & mid3_in() & green & (body >= 0.8 * atr) & (c > b4.Close)
    r["falling3"] = bigg(b4.Close, b4.Open) & mid3_in() & red & (body >= 0.8 * atr) & (c < b4.Close)
    bear_open = pd.Series(np.where(red & (body >= 1.2 * atr), o, np.nan), index=d.index).ffill(limit=5).shift(1)
    r["bear_recover"] = (c > bear_open) & (p.Close <= bear_open)
    lim = np.where(d.index < pd.Timestamp("2015-06-15"), 0.145, 0.295)
    r["limit_up"] = chg >= lim
    r["limit_dn"] = chg <= -lim
    lu_mid = pd.Series(np.where(r["limit_up"], (o + c) / 2, np.nan), index=d.index).ffill(limit=10).shift(1)
    r["limit_up_pull"] = (l <= lu_mid * 1.01) & (c >= lu_mid) & (l.shift(1) > lu_mid * 1.01)
    upd = (c > p.Close).astype(int)
    dnd = (c < p.Close).astype(int)
    r["up_streak"] = upd.rolling(5).sum() == 5
    r["dn_streak"] = dnd.rolling(5).sum() == 5
    # 이동평균
    ma10, ma120, ma240 = x.MA10, d.MA120, x.MA240
    r["ma10_sup"] = (p.Close > ma10.shift(1)) & (l <= ma10 * 1.005) & (c >= ma10) & (ma10 > ma10.shift(3))
    for nm, ma in (("120", ma120), ("240", ma240)):
        r[f"ma{nm}_up"] = _cross_up(c, ma)
        r[f"ma{nm}_dn"] = _cross_dn(c, ma)
        r[f"ma{nm}_sup"] = (p.Close > ma.shift(1)) & (l <= ma * 1.01) & (c >= ma) & (ma > ma.shift(5))
    r["gc_120"] = _cross_up(d.MA60, ma120)
    r["dc_120"] = _cross_dn(d.MA60, ma120)
    s20 = d.MA20.diff()
    r["ma20_turn"] = (s20 > 0) & ((s20.shift(1) < 0).astype(int).rolling(10).sum() == 10)
    m3 = pd.concat([d.MA5, d.MA20, d.MA60], axis=1)
    conv = (m3.max(axis=1) - m3.min(axis=1)) / c <= 0.03
    hi3 = m3.max(axis=1)
    r["ma_conv_up"] = conv.shift(1, fill_value=False).rolling(5).max().astype(bool) & (c > hi3 * 1.02) & (p.Close <= hi3.shift(1) * 1.02) & (vr >= 1.5)
    # 국면·가격대
    hh52 = h.rolling(250, min_periods=120).max()
    r["half_off"] = c <= hh52 * 0.5
    r["env_low"] = l <= d.MA20 * 0.8
    r["env_high"] = _cross_up(c, d.MA20 * 1.2)
    r["vwap_up"] = _cross_up(c, x.VWAP120)
    r["vwap_dn"] = _cross_dn(c, x.VWAP120)
    r["vol_max_up"] = (v >= v.rolling(250, min_periods=120).max()) & green
    vlow = v.rolling(5).mean().shift(1) <= v.rolling(60).quantile(0.15).shift(1)
    r["vol_bottom_brk"] = vlow.rolling(10).max().astype(bool) & (vr >= 3) & green & (chg > 0.03)
    # 보조지표
    r["sto_gc"] = _cross_up(d.STO_K, d.STO_D) & (d.STO_K.shift(1) <= 20)
    r["sto_dc"] = _cross_dn(d.STO_K, d.STO_D) & (d.STO_K.shift(1) >= 80)
    zero = pd.Series(0.0, index=d.index)
    r["macd_zero_up"] = _cross_up(d.MACD, zero)
    r["macd_zero_dn"] = _cross_dn(d.MACD, zero)
    r["rsi_os_exit"] = _cross_up(d.RSI, zero + 30)
    r["rsi_ob_exit"] = _cross_dn(d.RSI, zero + 70)
    r["rsi50_up"] = _cross_up(d.RSI, zero + 50)
    r["cci_up"] = _cross_up(x.CCI, zero - 100)
    r["cci_dn"] = _cross_dn(x.CCI, zero + 100)
    r["mfi_os"] = x.MFI <= 20
    r["mfi_ob"] = x.MFI >= 80
    r["wr_os_exit"] = _cross_up(x.WR, zero - 80)
    r["dmi_gc"] = _cross_up(d.PDI, d.MDI) & (d.ADX >= 20)
    r["dmi_dc"] = _cross_dn(d.PDI, d.MDI) & (d.ADX >= 20)
    adx25 = _cross_up(d.ADX, zero + 25)
    r["adx_up"] = adx25 & (d.PDI > d.MDI)
    r["adx_dn"] = adx25 & (d.PDI < d.MDI)
    sar = pd.Series(x.SAR.values, index=d.index)
    r["sar_buy"] = (sar < c) & (sar.shift(1) > p.Close)
    r["sar_sell"] = (sar > c) & (sar.shift(1) < p.Close)
    r["bb_rebound"] = (p.Close < p.BB_lower) & (c > d.BB_lower)
    sq = d.BB_width <= d.BB_width.rolling(120, min_periods=60).quantile(0.1)
    r["squeeze_brk"] = sq.shift(1, fill_value=False).rolling(5).max().astype(bool) & _cross_up(c, d.BB_upper)
    # 일목균형표
    r["ichi_cloud_up"] = _cross_up(c, x.CLOUD_TOP)
    r["ichi_cloud_dn"] = _cross_dn(c, x.CLOUD_BOT)
    r["ichi_tk_gc"] = _cross_up(x.TENKAN, x.KIJUN)
    r["ichi_tk_dc"] = _cross_dn(x.TENKAN, x.KIJUN)
    r["ichi_chikou"] = _cross_up(c, c.shift(26))
    r["ichi_twist"] = _cross_up(x.SPAN_A, x.SPAN_B)
    out = {k: np.nan_to_num(np.asarray(a, dtype=float), nan=0).astype(bool) for k, a in r.items()}
    out.update(_swing2(d))
    return out


def _swing2(d: pd.DataFrame) -> dict:
    """확정 고점·저점 기반 가격 구조: 전고점 돌파·전저점·리테스트·N자·컵앤핸들·쐐기·삼중바닥·깃발·V자·섬꼴·갭."""
    o, h, l, c, v = (d[k].values.astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    keys = ("prev_high_brk", "prev_low_brk", "prev_low_sup", "retest", "n_wave", "cup_handle", "flag_bull",
            "wedge_fall", "wedge_rise", "triple_bottom", "v_rebound", "island_bot", "island_top", "gap_sup",
            "gap_fill_up", "ath", "period_brk")
    out = {k: np.zeros(n, bool) for k in keys}
    if n < 60:
        return out
    ph, pl = _pivots(h, l)
    H, L = [], []
    ph_lvl = pl_lvl = None          # (가격, 위치, 사용됨)
    brk = None                      # 최근 전고점 돌파 (가격, 날)
    cups = []                       # (림 가격, 오른쪽 림 위치, 오른쪽 림 가격)
    tb_neck = None
    cmax = np.maximum.accumulate(np.where(np.isfinite(c), c, -np.inf))
    vavg = pd.Series(v).rolling(20).mean().values
    gaps_up, gaps_dn = [], []       # (생성일, 갭 하단, 갭 상단)
    last_sup = -99
    wf = wr = None
    for t in range(1, n):
        if ph[t]:
            i = t - K_PIV
            H.append((i, h[i]))
            H = H[-6:]
            ph_lvl = [h[i], i, False]
            # 컵: 이전 고점 R1과 비슷한 오른쪽 림
            for i1, r1 in H[:-1]:
                if 30 <= i - i1 <= 150 and abs(h[i] / r1 - 1) <= 0.05:
                    bot = l[i1:i + 1].min()
                    if r1 * 0.55 <= bot <= r1 * 0.85:
                        cups.append((max(r1, h[i]), i, h[i]))
                        break
            cups = [z for z in cups if t - z[1] <= 25]
        if pl[t]:
            i = t - K_PIV
            L.append((i, l[i]))
            L = L[-6:]
            pl_lvl = [l[i], i, False]
            if len(L) >= 3:
                (i1, p1), (i2, p2), (i3, p3) = L[-3:]
                lo3 = min(p1, p2, p3)
                if max(p1, p2, p3) <= lo3 * 1.04 and 20 <= i3 - i1 <= 120:
                    mid = [p for j, p in H if i1 < j < i3]
                    if mid:
                        tb_neck = (max(mid), t + 30)
        # 전고점 돌파 / N자
        if ph_lvl and not ph_lvl[2] and t - ph_lvl[1] >= 10 and c[t] > ph_lvl[0] >= c[t - 1]:
            out["prev_high_brk"][t] = True
            ph_lvl[2] = True
            brk = (ph_lvl[0], t)
            if len(L) >= 2:
                (ia, pa), (ic, pc_) = L[-2], L[-1]
                ib = ph_lvl[1]
                if ia < ib < ic and pc_ > pa and ph_lvl[0] / pa - 1 >= 0.15:
                    rt_ = (ph_lvl[0] - pc_) / (ph_lvl[0] - pa)
                    if 0.3 <= rt_ <= 0.7:
                        out["n_wave"][t] = True
                # 기간조정 후 돌파: 고점 이후 12일 이상, 범위 16% 이내 횡보
                if t - ib >= 12 and ph_lvl[0] / l[ib:t].min() - 1 <= 0.16 and ph_lvl[0] / l[max(0, ib - 60):ib + 1].min() - 1 >= 0.15:
                    out["period_brk"][t] = True
        if pl_lvl and not pl_lvl[2] and t - pl_lvl[1] >= 10 and c[t] < pl_lvl[0] <= c[t - 1]:
            out["prev_low_brk"][t] = True
            pl_lvl[2] = True
        if pl_lvl and not pl_lvl[2] and t - pl_lvl[1] >= 10 and t - last_sup >= 5 and \
                pl_lvl[0] * 0.98 <= l[t] <= pl_lvl[0] * 1.02 and c[t] > o[t] and c[t] > pl_lvl[0]:
            out["prev_low_sup"][t] = True
            last_sup = t
        if brk and 2 <= t - brk[1] <= 12 and l[t] <= brk[0] * 1.015 and c[t] >= brk[0]:
            out["retest"][t] = True
            brk = None
        # 컵앤핸들
        for z in cups:
            rim, i2, r2 = z
            if 5 <= t - i2 <= 25 and l[i2:t].min() >= r2 * 0.88 and c[t] > rim >= c[t - 1]:
                out["cup_handle"][t] = True
                cups = []
                break
        # 삼중바닥
        if tb_neck and t <= tb_neck[1] and c[t] > tb_neck[0] >= c[t - 1]:
            out["triple_bottom"][t] = True
            tb_neck = None
        # 쐐기: 최근 두 고점·두 저점이 같은 방향으로 좁아짐
        if (ph[t] or pl[t]) and len(H) >= 2 and len(L) >= 2:
            (a1, ah1), (a2, ah2) = H[-2], H[-1]
            (b1, bl1), (b2, bl2) = L[-2], L[-1]
            wf = wr = None
            if t - min(a1, b1) <= 70:
                sh, sl = (ah2 - ah1) / (a2 - a1), (bl2 - bl1) / (b2 - b1)
                if sh < 0 and sl < 0 and sh < sl:
                    wf = (ah2, sh, a2, t + 20)
                if sh > 0 and sl > 0 and sl > sh:
                    wr = (bl2, sl, b2, t + 20)
        if wf and t <= wf[3]:
            if c[t] > wf[0] + wf[1] * (t - wf[2]) and c[t - 1] <= wf[0] + wf[1] * (t - 1 - wf[2]):
                out["wedge_fall"][t] = True
                wf = None
        if wr and t <= wr[3]:
            if c[t] < wr[0] + wr[1] * (t - wr[2]) and c[t - 1] >= wr[0] + wr[1] * (t - 1 - wr[2]):
                out["wedge_rise"][t] = True
                wr = None
        # 깃발형: 깃대(10일 내 +15%) 뒤 4~15일 좁은 조정, 깃대 고점 돌파
        if t >= 30:
            j = t - 1 - int(np.argmax(h[t - 16:t][::-1]))          # 최근 15일 최고가 위치(가장 최근)
            if 4 <= t - 1 - j <= 15:
                pole_lo = l[max(0, j - 10):j + 1].min()
                if h[j] / pole_lo - 1 >= 0.15 and l[j + 1:t].min() >= h[j] - 0.5 * (h[j] - pole_lo) and c[t] > h[j] >= c[t - 1]:
                    if np.nanmean(v[j + 1:t]) < np.nanmean(v[max(0, j - 10):j + 1]):
                        out["flag_bull"][t] = True
        # V자 반등
        if t >= 20:
            j = t - 5 + int(np.argmin(l[t - 5:t]))                 # 최근 5일 안의 저점
            pk = h[max(0, j - 10):j + 1].max()
            if pk > 0 and l[j] / pk - 1 <= -0.2:
                half = l[j] + 0.5 * (pk - l[j])
                if c[t] >= half > c[t - 1]:
                    out["v_rebound"][t] = True
        # 갭
        if l[t] > h[t - 1]:
            gaps_up.append((t, h[t - 1], l[t]))
        if h[t] < l[t - 1]:
            gaps_dn.append((t, h[t], l[t - 1]))
        gaps_up = [g for g in gaps_up if t - g[0] <= 20 and l[t] >= g[1]]          # 완전히 메워지면 제거
        gaps_dn = [g for g in gaps_dn if t - g[0] <= 20]
        for g in gaps_up:
            if t - g[0] >= 2 and l[t] <= g[2] and c[t] >= g[1] and l[t - 1] > g[2]:
                out["gap_sup"][t] = True
                break
        for g in list(gaps_dn):
            if t - g[0] >= 1 and c[t] >= g[2] > c[t - 1]:
                out["gap_fill_up"][t] = True
                gaps_dn.remove(g)
                break
        # 섬꼴 반전: 1~10일 전 갭과 오늘 반대 방향 갭 사이 고립
        if l[t] > h[t - 1]:                                          # 오늘 상승 갭
            for a in range(max(1, t - 10), t):
                if h[a] < l[a - 1] and h[a:t].max() < min(l[a - 1], l[t]):
                    out["island_bot"][t] = True
                    break
        if h[t] < l[t - 1]:
            for a in range(max(1, t - 10), t):
                if l[a] > h[a - 1] and l[a:t].min() > max(h[a - 1], h[t]):
                    out["island_top"][t] = True
                    break
        # 상장 후 최고가 (1년 이상 데이터)
        if t >= 250 and c[t] > cmax[t - 1]:
            out["ath"][t] = True
    return out


def onsets(f: pd.DataFrame, cooldown: int = 5) -> pd.DataFrame:
    """상태형 신호(RSI 과매도 등)는 '처음 나타난 날'만 남김."""
    prev = f.astype(int).rolling(cooldown, min_periods=1).max().shift(1).fillna(0).astype(bool)
    return f & ~prev


SEG = {"L": "대형주", "M": "중형주", "S": "중소형주"}


def _events(job):
    """종목 하나의 신호 발생 위치와 이후 수익률 (병렬 처리용)."""
    import indicators
    d, g, horizons, cost = job
    if d is None or len(d) < 160:
        return None
    if "MA20" not in d:
        d = indicators.add_indicators(d)
    try:
        f = onsets(flags(d)).values
    except Exception:
        return None
    o, c = d.Open.values.astype(float), d.Close.values.astype(float)
    n = len(c)
    day = (d.index.values.astype("datetime64[D]") - np.datetime64("1990-01-01", "D")).astype(int)
    warm = np.arange(n) >= 130
    rH = {}
    for H in horizons:
        r = np.full(n, np.nan)
        if n > H + 1:
            ent = np.r_[o[1:], np.nan][:n - H]
            with np.errstate(divide="ignore", invalid="ignore"):
                r[:n - H] = np.where(ent > 0, c[H:] / ent - 1 - cost, np.nan)
        r[~np.isfinite(r) | (np.abs(r) > 3)] = np.nan
        r[~warm] = np.nan
        rH[H] = r
    rows, ks = np.nonzero(f & warm[:, None])
    return g, day, rH, rows, ks, d.index[130], d.index[-1]


def stats(frames, horizons=(5, 20), cost: float = 0.0025, seg: dict | None = None, log=None, workers: int = 0) -> dict:
    """전 종목·전 기간 신호 발생 후 성과 — 신호 다음 날 시가 매수, 5·20거래일 뒤 종가 매도.
    frames: {코드: 일봉} 또는 (코드, 일봉) 반복자 (지표가 없으면 계산). seg: {코드: "L"/"M"/"S"} 시가총액 구간.
    반환: {키: {n, "5": [평균, 시장 대비, 승률, 표본], "20": [...], "seg": {구간: {...}}}, "_base": ..., "_meta": ...}"""
    it = frames.items() if isinstance(frames, dict) else frames
    ND = 20000
    sums = {H: np.zeros(ND) for H in horizons}
    cnts = {H: np.zeros(ND) for H in horizons}
    bsum = {H: {g: [0.0, 0, 0] for g in ("ALL", *SEG)} for H in horizons}       # [합, 건수, 상승 건수]
    ev = {"k": [], "d": [], "g": [], **{f"r{H}": [] for H in horizons}}
    gi = {g: i for i, g in enumerate(SEG)}
    nst, segn, dmin, dmax = 0, {g: 0 for g in SEG}, None, None
    def consume(res):
        nonlocal nst, dmin, dmax
        if res is None:
            return
        g, day, rH, rows, ks, d0, d1 = res
        for H in horizons:
            r = rH[H]
            ok = np.isfinite(r)
            np.add.at(sums[H], day[ok], r[ok])
            np.add.at(cnts[H], day[ok], 1)
            for gg in ("ALL", g):
                bsum[H][gg][0] += float(r[ok].sum())
                bsum[H][gg][1] += int(ok.sum())
                bsum[H][gg][2] += int((r[ok] > 0).sum())
            ev[f"r{H}"].append(r[rows].astype(np.float32))
        ev["k"].append(ks.astype(np.int16))
        ev["d"].append(day[rows].astype(np.int32))
        ev["g"].append(np.full(len(rows), gi.get(g, 0), np.int8))
        nst += 1
        segn[g] = segn.get(g, 0) + 1
        dmin = d0 if dmin is None or d0 < dmin else dmin
        dmax = d1 if dmax is None or d1 > dmax else dmax
        if log and nst % 200 == 0:
            log(f"    신호 통계 {nst}종목")

    jobs = ((d, (seg or {}).get(code, "L"), horizons, cost) for code, d in it)
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_events, jobs, chunksize=8):
                consume(res)
    else:
        for j in jobs:
            consume(_events(j))
    if not nst:
        return {}
    E = pd.DataFrame({k: np.concatenate(v) for k, v in ev.items()})
    for H in horizons:
        m = np.where(cnts[H] > 0, sums[H] / np.maximum(cnts[H], 1), np.nan)
        E[f"x{H}"] = E[f"r{H}"] - m[E.d.values]
    out = {k: {"n": 0, "seg": {}} for k in KEYS}
    segs = list(SEG)
    for by in (["k"], ["k", "g"]):
        cnt = E.groupby(by).size()
        for key, nn in cnt.items():
            key = key if isinstance(key, tuple) else (key,)
            e = out[KEYS[int(key[0])]] if len(key) == 1 else out[KEYS[int(key[0])]]["seg"].setdefault(segs[int(key[1])], {})
            e["n"] = int(nn)
        for H in horizons:
            x = E[by + [f"r{H}", f"x{H}"]].dropna()
            x = x.assign(w=(x[f"r{H}"] > 0).astype(float))
            gb = x.groupby(by).agg(r=(f"r{H}", "mean"), xm=(f"x{H}", "mean"), w=("w", "mean"), n=("w", "size"))
            for key, row in gb.iterrows():
                if row.n < 30:
                    continue
                key = key if isinstance(key, tuple) else (key,)
                e = out[KEYS[int(key[0])]] if len(key) == 1 else out[KEYS[int(key[0])]]["seg"].setdefault(segs[int(key[1])], {})
                e[str(H)] = [round(float(row.r) * 100, 2), round(float(row.xm) * 100, 2), round(float(row.w) * 100, 1), int(row.n)]
    base = {}
    for H in horizons:
        for gg, (sm, nn, wn) in bsum[H].items():
            if nn:
                base.setdefault(gg, {})[str(H)] = [round(sm / nn * 100, 2), round(wn / nn * 100, 1)]
    out["_base"] = {**base.get("ALL", {}), "seg": {g: base[g] for g in SEG if g in base}}
    out["_meta"] = {"stocks": nst, "seg": segn, "events": int(len(E)), "signals": len(KEYS),
                    "from": str(pd.Timestamp(dmin).date()), "to": str(pd.Timestamp(dmax).date()),
                    "days": int(sum(bsum[horizons[-1]]["ALL"][1:2]))}
    return out


def levels(d: pd.DataFrame, lookback: int = 120) -> dict:
    """오늘 종가 기준 가장 가까운 지지선·저항선 (확정 고점·저점, 이동평균, 52주 고저).
    종가에서 최소 max(1.5%, 일평균 변동폭 절반) 떨어진 가격만 의미 있는 선으로 본다."""
    c = float(d.Close.iloc[-1])
    gap = max(0.015, float(d.ATR_pct.iloc[-1]) * 0.5 if np.isfinite(d.ATR_pct.iloc[-1]) else 0.015)
    x = d.iloc[-lookback:]
    ph, pl = _pivots(x.High.values, x.Low.values)
    cand = []
    for t in np.flatnonzero(ph):
        cand.append((float(x.High.values[t - K_PIV]), "전고점"))
    for t in np.flatnonzero(pl):
        cand.append((float(x.Low.values[t - K_PIV]), "전저점"))
    for n in (5, 20, 60, 120):
        v = d[f"MA{n}"].iloc[-1]
        if np.isfinite(v):
            cand.append((float(v), f"{n}일선"))
    y = d.iloc[-250:]
    cand.append((float(y.High.max()), "52주 최고가"))
    cand.append((float(y.Low.min()), "52주 최저가"))
    below = [z for z in cand if z[0] < c * (1 - gap)]
    above = [z for z in cand if z[0] > c * (1 + gap)]
    sup = max(below, key=lambda z: z[0]) if below else None
    res = min(above, key=lambda z: z[0]) if above else None
    return {"support": sup, "resist": res}
