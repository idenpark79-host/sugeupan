"""주식 차트·시황·동향 자동 분석

사용법
  python main.py                 # config.json 기준 실행
  python main.py --demo          # 가상 시세로 동작 확인 (인터넷 불필요)
  python main.py --tickers 005380.KS NVDA --period 6mo
  python main.py --open          # 생성 후 브라우저로 열기
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

# 맥 python.org 설치본은 인증서가 비어 있어 SSL 오류가 날 수 있음 → certifi 인증서로 대체
try:
    import ssl
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

import data
import indicators
import analysis
import charts
import report
import strategy
import universe
import scanner

BASE = Path(__file__).resolve().parent

# 앱 설정에 저장된 KRX 계정이 있으면 사용 (수급 조건용)
try:
    from appkit import store as _store
    _store.apply_krx_env(_store.load())
except Exception:
    pass


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_group(tickers: dict, period: str, demo: bool, params: dict, label: str):
    results = []
    for tk, name in tickers.items():
        try:
            df = indicators.add_indicators(data.fetch(tk, period, demo))
            results.append(analysis.analyze(tk, name, df, params))
            print(f"  [{label}] {name:<12} OK")
        except Exception as e:
            print(f"  [{label}] {name:<12} 실패: {e}", file=sys.stderr)
    return results


def llm_comment(cfg: dict, summary: list[str], stocks) -> str | None:
    """선택 기능: ANTHROPIC_API_KEY가 있으면 Claude로 시황 코멘트 생성."""
    if not cfg.get("llm_commentary", {}).get("enabled") or not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        brief = "\n".join(summary) + "\n\n" + "\n".join(
            f"- {r.name}: {r.comment} 신호={[s[0] for s in r.signals]}" for r in stocks)
        msg = anthropic.Anthropic().messages.create(
            model=cfg["llm_commentary"].get("model", "claude-sonnet-5-5"),
            max_tokens=600,
            messages=[{"role": "user", "content":
                       "다음은 기술적 지표 기반 시황 데이터입니다. 투자 권유 없이, 오늘 시장 흐름과 "
                       "관심종목 동향을 5문장 이내 한국어로 간결하게 정리해 주세요.\n\n" + brief}])
        return msg.content[0].text.strip()
    except Exception as e:
        print(f"  AI 코멘트 생략: {e}", file=sys.stderr)
        return None


def run_scan(sc: dict, idx_res, demo: bool, days: int):
    """코스피·코스닥 시총 기준 종목 전체를 스캔해 최고 확률 기법과 추천 종목 산출."""
    print("추천 종목 스캔")
    scan = scanner.load_cached(demo) or scanner.run(sc, demo)
    if not scan.strategies:
        print("  조건을 통과한 기법이 없습니다 (min_train_signals 등 기준 완화 검토).")
        return scan, {}, None
    best = scan.strategies[0]
    print(f"  최고 기법: {best.name} — 승률 {best.win*100:.1f}% (기준 {scan.baseline['win']*100:.1f}%), "
          f"{best.n}회")
    pcs = {p.code: charts.pick_chart(p.df, f"{p.name}  {p.code}", p.target, p.stop, days) for p in scan.picks}
    hist = charts.return_hist(best.rets, f"1위 기법 거래 결과 분포 — {best.n:,}회")
    return scan, pcs, hist


def main():
    ap = argparse.ArgumentParser(description="주식 차트·시황·동향 자동 분석")
    ap.add_argument("--config", default=str(BASE / "config.json"))
    ap.add_argument("--tickers", nargs="*", help="관심종목 직접 지정 (config 대신)")
    ap.add_argument("--period", help="조회 기간: 3mo / 6mo / 1y / 2y")
    ap.add_argument("--demo", action="store_true", help="가상 시세로 실행")
    ap.add_argument("--open", action="store_true", help="완료 후 리포트 열기")
    ap.add_argument("--no-scan", action="store_true", help="추천 종목 스캔 생략 (빠른 실행)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    period = args.period or cfg.get("period", "1y")
    params = cfg.get("signal_params", {})
    watch = {t: t for t in args.tickers} if args.tickers else cfg.get("watchlist", {})

    print(f"데이터 수집 ({period}{', 데모' if args.demo else ''})")
    idx_res = run_group(cfg.get("indices", {}), period, args.demo, params, "지수")
    stk_res = run_group(watch, period, args.demo, params, "종목")
    if not idx_res and not stk_res:
        sys.exit("수집된 데이터가 없습니다. 네트워크 또는 티커를 확인하세요.")

    print("차트 생성")
    days = cfg.get("chart_days", 120)
    idx_chart = charts.index_compare_chart(
        {r.name: r.df["Close"] for r in idx_res if "환율" not in r.name}) if idx_res else ""
    stk_charts = {r.ticker: charts.stock_chart(r.df, f"{r.name}  {r.ticker}", days) for r in stk_res}

    scan, pick_charts, hist_chart = None, {}, None
    sc = cfg.get("scan", {})
    if sc.get("enabled", True) and not args.no_scan:
        scan, pick_charts, hist_chart = run_scan(sc, idx_res, args.demo, days)

    summary = analysis.market_summary(idx_res, stk_res)
    ai = llm_comment(cfg, summary, stk_res)

    out_dir = BASE / cfg.get("output_dir", "reports")
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    html_path = out_dir / f"report_{stamp}.html"
    html_path.write_text(report.build(idx_res, stk_res, summary, idx_chart, stk_charts,
                                      scan, pick_charts, hist_chart, ai, args.demo),
                         encoding="utf-8")

    # 엑셀/추가 분석용 요약 CSV
    rows = [{"구분": "지수" if r in idx_res else "종목", "티커": r.ticker, "이름": r.name,
             "종가": round(r.close, 2), **{k: None if v is None else round(v, 2) for k, v in r.returns.items()},
             "RSI": round(r.rsi, 1), "추세": r.trend, "점수": r.score, "배열": r.alignment,
             "신호": " / ".join(s[0] for s in r.signals)} for r in idx_res + stk_res]
    pd.DataFrame(rows).to_csv(out_dir / f"summary_{stamp}.csv", index=False, encoding="utf-8-sig")

    if scan is not None and scan.picks:
        pd.DataFrame([{"순위": i, "종목코드": p.code, "종목명": p.name, "시장": p.market,
                       "현재가": round(p.close), "목표가": round(p.target), "손절가": round(p.stop),
                       "목표수익률%": round((p.target / p.close - 1) * 100, 1),
                       "손절률%": round((p.stop / p.close - 1) * 100, 1),
                       "과거승률%": round(p.strategy.win * 100, 1), "평균수익%": round(p.strategy.avg * 100, 2),
                       "신호수": p.strategy.n, "기법": p.strategy.name}
                      for i, p in enumerate(scan.picks, 1)]).to_csv(
            out_dir / f"picks_{stamp}.csv", index=False, encoding="utf-8-sig")
        print("\n[추천 종목]")
        for i, p in enumerate(scan.picks, 1):
            print(f"  {i:>2}. {p.name:<10} 현재 {p.close:>10,.0f}  목표 {p.target:>10,.0f}  "
                  f"손절 {p.stop:>10,.0f}  승률 {p.strategy.win*100:.1f}%")

    print("\n" + "\n".join(summary))
    print(f"\n리포트: {html_path}")
    if args.open:
        webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
