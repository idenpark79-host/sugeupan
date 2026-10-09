#!/bin/bash
# 평일 매일 16:10 자동 실행 등록 (맥이 잠자기 중이었다면 깨어날 때 실행)
cd "$(dirname "$0")"
DIR="$(pwd)"
[ -x .venv/bin/python ] || { echo "먼저 install.command 를 실행하세요."; read -n1 -r; exit 1; }
HOUR=${1:-16}; MIN=${2:-10}
PLIST="$HOME/Library/LaunchAgents/com.stockanalyzer.daily.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$DIR/reports"
{
cat <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.stockanalyzer.daily</string>
  <key>ProgramArguments</key><array>
    <string>$DIR/.venv/bin/python</string><string>$DIR/main.py</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StandardOutPath</key><string>$DIR/reports/auto_run.log</string>
  <key>StandardErrorPath</key><string>$DIR/reports/auto_run.log</string>
  <key>StartCalendarInterval</key><array>
XML
for d in 1 2 3 4 5; do
  echo "    <dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MIN</integer></dict>"
done
echo "  </array>"
echo "</dict></plist>"
} > "$PLIST"
launchctl unload "$PLIST" 2>/dev/null
launchctl load -w "$PLIST" && echo "등록 완료: 평일 ${HOUR}시 ${MIN}분마다 reports 폴더에 리포트가 생성됩니다." \
                           || echo "등록 실패"
echo "※ 이 폴더를 다른 곳으로 옮기면 schedule_on.command 를 다시 실행해야 합니다."
read -n1 -r -p "아무 키나 누르면 닫힙니다..."
