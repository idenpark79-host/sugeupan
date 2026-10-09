"""추천 스캔 실행 (CLI·앱 공용) — 결과는 하루 단위로 data/ 에 저장해 재사용."""
from __future__ import annotations

import os
import pickle
from datetime import date
from pathlib import Path

import data
import strategy
import universe

DATA = Path(__file__).resolve().parent / "data"


def _cache_path(demo: bool) -> Path:
    return DATA / f"scan_{date.today():%Y%m%d}{'_demo' if demo else ''}.pkl"


def load_cached(demo: bool):
    p = _cache_path(demo)
    if p.exists():
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def run(sc: dict, demo: bool, use_flows: bool | None = None, log=print, progress=None):
    """progress(단계명, 완료, 전체) 콜백 선택."""
    if use_flows is None:
        use_flows = demo or bool(os.environ.get("KRX_ID") and os.environ.get("KRX_PW"))
    u = universe.get_universe(sc.get("min_marcap", 1e12), sc.get("markets", ["KOSPI", "KOSDAQ"]), demo)
    log(f"  대상 {len(u)}개 종목 (시총 {sc.get('min_marcap', 1e12) / 1e12:g}조 이상)")
    years = sc.get("history_years", 3)
    cb = (lambda d, t: progress("시세 수집", d, t)) if progress else None
    prices = universe.fetch_all(list(u["Code"]), years, demo, progress=cb)
    flows = None
    if use_flows:
        cb = (lambda d, t: progress("매매동향 수집", d, t)) if progress else None
        flows = universe.fetch_flows(prices, years, demo, progress=cb)
        log(f"  매매동향 {len(flows)}개 종목 확보")
    else:
        log("  KRX 로그인 정보가 없어 수급(외국인·기관·연기금) 조건은 제외")
    kospi = data.demo_series("^KS11", 252 * years) if demo else data.fetch("^KS11", f"{years}y")
    if progress:
        progress("기법 조합 백테스트", 0, 1)
    log("  기법 조합 백테스트 중")
    pn = strategy.build_panel(prices, u, kospi["Close"], flows)
    res = strategy.mine(pn, sc)
    res.used_flows = bool(flows)
    if progress:
        progress("기법 조합 백테스트", 1, 1)
    DATA.mkdir(exist_ok=True)
    try:
        with open(_cache_path(demo), "wb") as f:
            pickle.dump(res, f)
    except Exception:
        pass
    return res
