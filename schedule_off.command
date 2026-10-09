#!/bin/bash
# 자동 실행 해제
PLIST="$HOME/Library/LaunchAgents/com.stockanalyzer.daily.plist"
launchctl unload "$PLIST" 2>/dev/null; rm -f "$PLIST"
echo "자동 실행을 해제했습니다."
read -n1 -r -p "아무 키나 누르면 닫힙니다..."
