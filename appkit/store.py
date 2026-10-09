"""사용자 데이터(설정·관심종목·알림·보유종목) 로컬 저장 — data/user.json"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FILE = DATA / "user.json"

DEFAULT = {
    "settings": {"demo": False, "krx_id": "", "krx_pw": ""},
    "watchlist": ["005380", "000270", "005930", "000660"],
    "alerts": {},          # code → {"above": 가격, "below": 가격}
    "portfolio": [],       # [{"code":..., "qty":..., "avg":...}]
}


def load() -> dict:
    if FILE.exists():
        try:
            d = json.loads(FILE.read_text(encoding="utf-8"))
            for k, v in DEFAULT.items():
                d.setdefault(k, v)
            for k, v in DEFAULT["settings"].items():
                d["settings"].setdefault(k, v)
            return d
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT))


def save(d: dict):
    DATA.mkdir(exist_ok=True)
    FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(FILE, 0o600)          # 본인 계정만 읽기 가능 (KRX 비밀번호 보관)
    except OSError:
        pass


def apply_krx_env(d: dict):
    """pykrx는 import 시점에 KRX_ID/KRX_PW 환경변수로 로그인한다."""
    s = d["settings"]
    if s.get("krx_id") and s.get("krx_pw"):
        os.environ["KRX_ID"], os.environ["KRX_PW"] = s["krx_id"], s["krx_pw"]
