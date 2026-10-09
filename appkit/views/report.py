"""시황 리포트 — 기존 HTML 리포트 생성·보기·내려받기."""
import subprocess
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from appkit import ui
from appkit.views import common as c

ROOT = Path(__file__).resolve().parents[2]


def render():
    st.title("시황 리포트")
    ui.demo_banner(c.demo())
    st.caption("지수·관심종목·AI 추천을 한 장으로 묶은 HTML 리포트를 만듭니다. "
               "AI 추천은 오늘 스캔 결과가 있으면 그대로 쓰고, 없으면 새로 스캔합니다.")
    if st.button("리포트 생성", type="primary"):
        args = [sys.executable, str(ROOT / "main.py")] + (["--demo"] if c.demo() else [])
        with st.spinner("리포트 생성 중 (스캔이 필요하면 수 분 소요)"):
            r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            st.error("리포트 생성 실패")
            st.code((r.stderr or r.stdout)[-2000:])
    files = sorted((ROOT / "reports").glob("report_*.html"), reverse=True)
    if not files:
        st.caption("아직 생성된 리포트가 없습니다.")
        return
    sel = st.selectbox("리포트", files, format_func=lambda p: p.stem.replace("report_", ""))
    html = sel.read_text(encoding="utf-8")
    st.download_button("내려받기 (HTML)", html, file_name=sel.name, mime="text/html")
    components.html(html, height=1400, scrolling=True)
