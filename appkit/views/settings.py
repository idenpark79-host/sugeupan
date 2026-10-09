"""설정."""
import streamlit as st

from appkit import store, ui
from appkit.views import common as c


def render():
    st.title("설정")
    u = c.user()
    s = u["settings"]

    st.subheader("데이터")
    demo = st.toggle("데모 모드 (가상 시세로 화면 확인)", value=bool(s.get("demo")))

    st.subheader("한국거래소(KRX) 계정")
    st.caption("투자자별 매매동향(외국인·기관·연기금), 외국인 지분율, PER·PBR 등은 KRX 데이터로, 로그인이 필요합니다. "
               "data.krx.co.kr 에서 무료 회원가입 후 입력하세요. 이 컴퓨터의 data/user.json 에만 저장됩니다.")
    kid = st.text_input("KRX 아이디", value=s.get("krx_id", ""))
    kpw = st.text_input("KRX 비밀번호", value=s.get("krx_pw", ""), type="password")

    if st.button("저장", type="primary"):
        changed = (kid != s.get("krx_id") or kpw != s.get("krx_pw"))
        s.update({"demo": demo, "krx_id": kid.strip(), "krx_pw": kpw})
        c.save()
        st.cache_data.clear()
        if changed:
            st.success("저장했습니다. KRX 계정은 앱을 껐다 켜야 적용됩니다 (터미널 창을 닫고 app.command 다시 실행).")
        else:
            st.success("저장했습니다.")

    st.subheader("기타")
    if st.button("데이터 새로고침 (캐시 비우기)"):
        st.cache_data.clear()
        st.toast("다음 화면부터 최신 데이터를 다시 받습니다.")
    st.caption(f"사용자 데이터 위치: {store.FILE}")
    st.caption("실시간 호가·체결과 주문은 증권사 Open API(예: 한국투자증권 KIS) 연동이 필요해 이 버전에는 포함하지 않았습니다. "
               "시세는 장중 지연 또는 전일 종가 기준일 수 있습니다.")
