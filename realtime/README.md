# Wake 실시간 시세 중계 (본인 전용)

한국투자증권 Open API 키를 숨겨 두고, 비밀번호를 아는 사람(본인)에게만 현재가를 전달하는 작은 서버입니다.
Vercel에 이 `realtime` 폴더만 올리면 됩니다.

환경변수(Vercel → Settings → Environment Variables)
- `KIS_APP_KEY`, `KIS_APP_SECRET` : 한국투자증권 KIS Developers에서 발급받은 앱키·시크릿
- `WAKE_PASS` : 사이트 '실시간 설정'에 입력할 비밀번호(아무 문자열)
- `ALLOW_ORIGIN` : `https://idenpark79-host.github.io`

주소: `https://<프로젝트 이름>.vercel.app/api/quote`
