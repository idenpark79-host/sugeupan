"""투자 주체별 매매 특성 — 성향 설명 + 실제 데이터로 본 '이 주체가 많이 산 뒤 주가' 통계."""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE = {
    "외국인": ("글로벌 운용사·연기금·투자은행", "대형주 중심으로 수주~수개월 추세를 따라가는 편. 환율과 글로벌 자금 흐름의 영향을 크게 받음"),
    "연기금": ("국민연금 등 공적 연금", "장기 보유 목적. 하락 때 나눠 사고 비중을 맞추는 매매(리밸런싱)가 많아 꾸준한 매수는 중장기 하방을 받쳐 주는 역할"),
    "투신": ("자산운용사 공모펀드", "펀드로 들어오고 나가는 돈에 따라 매매하는 중기 관점. 분기 말에 수익률 관리 매매(윈도드레싱)가 나타나기도 함"),
    "사모": ("소수 고액 자금 펀드", "단기·이벤트 중심으로 매매 회전이 빠름. 매수가 며칠 이어지지 않는 경우가 많음"),
    "금융투자": ("증권사 자기매매·ETF 유동성 공급", "차익거래·ETF 설정 등 기계적인 초단기 매매 비중이 커서 방향성 신호로는 약함"),
    "보험": ("보험사 자산운용", "배당·안정성을 중시하는 장기 보유. 매매 빈도는 낮음"),
    "은행": ("은행 고유자산", "비중이 작고 매매가 드묾"),
    "기타법인": ("일반 법인", "자사주·계열사 지분 매매 등 경영 목적 거래가 섞여 있음"),
    "개인": ("개인 투자자", "하락 때 사고 상승 때 파는 역추세 성향. 개인만 많이 사는 종목은 오히려 약세인 경우가 많음"),
    "기관합계": ("국내 기관 전체", "연기금·투신·사모·금융투자 등의 합. 성격이 다른 주체가 섞여 있어 세부 주체를 함께 봐야 함"),
}
GROUPS = list(PROFILE)


def study(prices: dict, flows: dict, log=print) -> dict:
    """주체별로 '20일 순매수 강도(순매수 ÷ 거래대금)가 같은 날 상위 10%'였던 종목의 이후 5·20·60일 성과(같은 날 전체 평균 대비)."""
    cols = [c for c in flows if c in prices]
    if len(cols) < 30:
        return {"ok": False}
    C = pd.DataFrame({c: prices[c].Close for c in cols}).sort_index()
    O = pd.DataFrame({c: prices[c].Open for c in cols}).reindex(C.index)
    V = pd.DataFrame({c: prices[c].Volume for c in cols}).reindex(C.index)
    amt = (C * V).rolling(60, min_periods=30).mean()
    fw = {h: C.shift(-h) / O.shift(-1) - 1 for h in (5, 20, 60)}
    fx = {h: f.sub(f.mean(axis=1), axis=0) for h, f in fw.items()}
    out = {}
    for g in GROUPS:
        net = pd.DataFrame({c: flows[c][g] for c in cols if g in flows[c]}).reindex(C.index)
        if net.empty or net.abs().sum().sum() == 0:
            continue
        inten = net.rolling(20, min_periods=12).sum() / (amt * 20)
        top = inten.rank(axis=1, pct=True) >= 0.9
        r = {}
        for h in (5, 20, 60):
            e = fx[h].where(top).stack()
            a = fw[h].where(top).stack()
            r[str(h)] = [round(float(e.mean()) * 100, 2), round(float((a > 0).mean()) * 100, 1)]
        pos = (net > 0).astype(float).where(net.notna())
        nxt = pos.shift(-1).rolling(20, min_periods=10).mean().shift(-19)
        persist = float(nxt.where(pos == 1).stack().mean())
        out[g] = {"after": r, "persist": round(persist * 100, 1), "who": PROFILE[g][0], "style": PROFILE[g][1]}
        log(f"    투자 주체 {g}: 순매수 상위 10% 이후 시장 대비 5일 {r['5'][0]:+.2f}%p · 20일 {r['20'][0]:+.2f}%p · 60일 {r['60'][0]:+.2f}%p · 매수 지속 {persist * 100:.0f}%")
    d0 = C.index[60] if len(C.index) > 60 else C.index[0]
    return {"ok": True, "stocks": len(cols), "from": str(d0.date()), "to": str(C.index[-61].date()) if len(C) > 61 else None, "g": out}
