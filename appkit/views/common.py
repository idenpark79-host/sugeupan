"""화면 공통 도우미."""
from __future__ import annotations

import streamlit as st

from appkit import market, store


def user() -> dict:
    return st.session_state.user


def demo() -> bool:
    return bool(user()["settings"].get("demo"))


def save():
    store.save(user())


def has_krx() -> bool:
    s = user()["settings"]
    return demo() or bool(s.get("krx_id") and s.get("krx_pw"))


def listing():
    try:
        return market.listing(demo())
    except Exception as e:
        st.error(f"종목 목록을 불러오지 못했습니다: {e}")
        st.stop()


def name_of(code: str) -> str:
    lst = listing()
    m = lst[lst.Code == code]
    return m.Name.iloc[0] if len(m) else code


def go_stock(code: str):
    st.session_state.code = code
    st.switch_page(st.session_state.pages["stock"])


def krx_notice():
    st.info("투자자별 매매동향·외국인 지분율·투자지표는 한국거래소(KRX) 데이터라 로그인이 필요합니다. "
            "data.krx.co.kr 에서 무료 회원가입 후 [설정] 메뉴에 아이디·비밀번호를 입력하세요.")


def safe(fn, *args, **kw):
    """KRX 로그인 오류 등은 안내 문구로, 그 외 오류는 메시지로 표시하고 None 반환."""
    try:
        return fn(*args, **kw)
    except market.KrxLoginError as e:
        if not has_krx():
            krx_notice()
        else:
            st.warning(str(e))
    except Exception as e:
        st.warning(f"데이터를 불러오지 못했습니다: {e}")
    return None


def selectable_table(df, key: str, code_col="Code", height=None, column_config=None, styler=None):
    """행을 클릭하면 종목 상세로 이동하는 표."""
    ev = st.dataframe(styler if styler is not None else df, key=key, hide_index=True, width="stretch",
                      on_select="rerun", selection_mode="single-row", height=height or "auto",
                      column_config=column_config)
    rows = ev.selection.rows if ev and ev.selection else []
    if rows:
        go_stock(df.iloc[rows[0]][code_col])
