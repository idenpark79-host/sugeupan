#!/bin/bash
# 가상 시세로 연습 실행 (인터넷 불필요)
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "먼저 install.command 를 실행하세요."; read -n1 -r; exit 1; }
.venv/bin/python main.py --demo --open
echo; read -n1 -r -p "아무 키나 누르면 닫힙니다..."
