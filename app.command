#!/bin/bash
# 주식 앱 실행 — 브라우저가 자동으로 열립니다. 끄려면 이 터미널 창을 닫으세요.
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "먼저 install.command 를 실행하세요."; read -n1 -r; exit 1; }
if ! .venv/bin/python -c "import streamlit, plotly, pykrx" 2>/dev/null; then
  echo "앱에 필요한 부품을 설치합니다 (최초 1회)..."
  .venv/bin/python -m pip install -r requirements.txt -q || { echo "설치 실패"; read -n1 -r; exit 1; }
fi
echo "주식 앱을 시작합니다. 브라우저에서 http://localhost:8501 이 열립니다."
echo "앱을 끄려면 이 창을 닫으세요."
.venv/bin/python -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
