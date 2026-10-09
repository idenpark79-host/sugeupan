# 수급판 웹사이트 배포 (GitHub, 무료)

매일 장 마감 후 GitHub가 KRX 데이터를 받아 웹사이트를 갱신합니다.
방문자는 링크만 열면 되고 로그인이 필요 없습니다. 터미널 없이 웹 화면에서만 진행합니다.

## 1. GitHub 가입
github.com 에서 회원가입 (아이디가 사이트 주소에 들어갑니다).

## 2. 저장소 만들기
오른쪽 위 + → New repository → Repository name: `sugeupan` → **Public** 선택 → Create repository

## 3. 파일 올리기
1. 만들어진 저장소 화면에서 **uploading an existing file** 클릭
2. Finder에서 stock_analyzer 폴더를 열고 `⌘ + Shift + .` 를 눌러 숨김 폴더(.github)가 보이게 함
3. 아래를 **제외한** 모든 파일·폴더를 선택해 브라우저 창에 끌어다 놓기
   `.venv` · `cache` · `data` · `reports`
4. 아래쪽 **Commit changes** 클릭
5. 저장소에 `.github/workflows/update.yml` 이 보이는지 확인
   (안 보이면 Add file → Create new file → 이름 칸에 `.github/workflows/update.yml` 입력 →
   이 폴더의 같은 파일 내용을 복사해 붙여넣고 Commit)

## 4. KRX 계정 등록 (비공개로 안전하게 저장됨)
Settings → Secrets and variables → Actions → New repository secret
- Name `KRX_ID` / Secret: data.krx.co.kr 아이디
- Name `KRX_PW` / Secret: 비밀번호

등록하지 않으면 가상 데이터로 사이트가 만들어집니다.

## 5. 웹사이트 켜기
Settings → Pages → Build and deployment → Source: **GitHub Actions**

## 6. 첫 실행
Actions 탭 → (안내가 나오면 워크플로 사용 허용) → 왼쪽 **수급판 데이터 갱신·배포** →
오른쪽 **Run workflow** → 초록색 버튼
- 첫 실행은 30~60분 걸립니다. 초록 체크가 뜨면 완료.
- 이후에는 평일 16시 41분에 자동 갱신됩니다.

## 7. 주소
`https://아이디.github.io/sugeupan/`
휴대폰에서는 브라우저 공유 메뉴의 **홈 화면에 추가**를 누르면 앱처럼 쓸 수 있습니다.

## 문제가 생기면
Actions 탭에서 빨간 X가 뜬 실행을 열어 **데이터 생성** 단계의 마지막 메시지를 캡처해 보내 주세요.
