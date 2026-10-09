"""주식 앱 — 실행: streamlit run app.py  (맥: app.command 더블클릭)"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# SSL 인증서 보정 (맥 python.org 설치본)
try:
    import ssl
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

import streamlit as st

from appkit import store, ui

st.set_page_config(page_title="주식", layout="wide", initial_sidebar_state="expanded")

if "user" not in st.session_state:
    st.session_state.user = store.load()
store.apply_krx_env(st.session_state.user)
ui.setup_page()

from appkit.views import home, stock, flows, watch, portfolio, ai, report, settings  # noqa: E402

PAGES = {
    "home": st.Page(home.render, title="시장 현황", url_path="home", default=True),
    "stock": st.Page(stock.render, title="종목 상세", url_path="stock"),
    "flows": st.Page(flows.render, title="투자자 매매동향", url_path="flows"),
    "watch": st.Page(watch.render, title="관심종목 · 알림", url_path="watch"),
    "portfolio": st.Page(portfolio.render, title="보유종목", url_path="portfolio"),
    "ai": st.Page(ai.render, title="AI 추천", url_path="ai"),
    "report": st.Page(report.render, title="시황 리포트", url_path="report"),
    "settings": st.Page(settings.render, title="설정", url_path="settings"),
}
st.session_state.pages = PAGES

nav = st.navigation({"시세": [PAGES["home"], PAGES["stock"], PAGES["flows"]],
                     "내 투자": [PAGES["watch"], PAGES["portfolio"]],
                     "분석": [PAGES["ai"], PAGES["report"]],
                     "기타": [PAGES["settings"]]})
watch.check_alerts_toast()
nav.run()
