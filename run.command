#!/bin/bash
# 실제 시세로 리포트 생성 후 브라우저로 열기
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "먼저 install.command 를 실행하세요."; read -n1 -r; exit 1; }
.venv/bin/python main.py --open
echo; read -n1 -r -p "아무 키나 누르면 닫힙니다..."
