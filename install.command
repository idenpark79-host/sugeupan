#!/bin/bash
# 최초 1회: 프로그램 전용 파이썬 환경(.venv)을 만들고 필요한 부품을 설치합니다.
cd "$(dirname "$0")"
echo "=== 주식 분석 프로그램 설치 ==="
if ! command -v python3 >/dev/null 2>&1; then
  echo "파이썬이 없습니다. python.org 에서 설치한 뒤 다시 실행하세요."
  read -n1 -r -p "아무 키나 누르면 닫힙니다..."; exit 1
fi
echo "사용 파이썬: $(python3 --version)"
python3 -m venv .venv || { echo "가상환경 생성 실패"; read -n1 -r; exit 1; }
.venv/bin/python -m pip install --upgrade pip -q
if .venv/bin/python -m pip install -r requirements.txt; then
  echo; echo "설치 완료. 이제 demo.command 를 더블클릭해 보세요."
else
  echo; echo "설치 중 오류가 났습니다. 위 메시지를 복사해 Claude에게 보내 주세요."
fi
read -n1 -r -p "아무 키나 누르면 닫힙니다..."
