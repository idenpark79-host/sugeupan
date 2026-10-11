// Wake 실시간 시세 중계 (Vercel 서버리스 함수) — 본인 전용
// · 한국투자증권 Open API 키는 Vercel 환경변수에만 둔다(사이트·GitHub에 절대 넣지 않음).
// · WAKE_PASS(비밀번호)를 아는 사람만 조회 가능 → 시세를 다른 사람에게 제공하지 않는다(본인 계좌 API의 개인 이용 범위).
// 환경변수: KIS_APP_KEY, KIS_APP_SECRET, WAKE_PASS, ALLOW_ORIGIN(예: https://idenpark79-host.github.io)
const BASE = "https://openapi.koreainvestment.com:9443";
let TOKEN = null, EXP = 0, PENDING = null;

async function token() {
  if (TOKEN && Date.now() < EXP - 10 * 60 * 1000) return TOKEN;      // 접근토큰은 24시간 유효 — 재사용
  if (PENDING) return PENDING;
  PENDING = (async () => {
    const r = await fetch(`${BASE}/oauth2/tokenP`, {
      method: "POST", headers: { "content-type": "application/json; charset=utf-8" },
      body: JSON.stringify({ grant_type: "client_credentials", appkey: process.env.KIS_APP_KEY, appsecret: process.env.KIS_APP_SECRET }),
    });
    const j = await r.json();
    if (!j.access_token) throw new Error(j.error_description || j.msg1 || "토큰 발급 실패");
    TOKEN = j.access_token; EXP = Date.now() + (Number(j.expires_in) || 86400) * 1000;
    return TOKEN;
  })();
  try { return await PENDING; } finally { PENDING = null; }
}

async function price(code, t) {
  const u = `${BASE}/uapi/domestic-stock/v1/quotations/inquire-price?FID_COND_MRKT_DIV_CODE=J&FID_INPUT_ISCD=${code}`;
  const r = await fetch(u, { headers: {
    "content-type": "application/json; charset=utf-8", authorization: `Bearer ${t}`,
    appkey: process.env.KIS_APP_KEY, appsecret: process.env.KIS_APP_SECRET, tr_id: "FHKST01010100", custtype: "P" } });
  const j = await r.json();
  const o = j.output;
  if (!o || j.rt_cd !== "0") return null;
  const f = k => Number(o[k]);
  return { p: f("stck_prpr"), chg: f("prdy_vrss"), pct: f("prdy_ctrt"), o: f("stck_oprc"), h: f("stck_hgpr"), l: f("stck_lwpr"), v: f("acml_vol") };
}

export default async function handler(req, res) {
  const allow = process.env.ALLOW_ORIGIN || "https://idenpark79-host.github.io";
  res.setHeader("Access-Control-Allow-Origin", allow);
  res.setHeader("Access-Control-Allow-Headers", "x-wake-key");
  res.setHeader("Vary", "Origin");
  res.setHeader("Cache-Control", "no-store");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (!process.env.WAKE_PASS || req.headers["x-wake-key"] !== process.env.WAKE_PASS) return res.status(401).json({ error: "비밀번호가 맞지 않습니다" });
  const codes = String(req.query.codes || "").split(",").filter(c => /^[0-9A-Z]{6}$/.test(c)).slice(0, 15);
  try {
    const t = await token(), q = {};
    for (const c of codes) {                                         // 초당 호출 한도를 넘지 않게 천천히
      const x = await price(c, t);
      if (x) q[c] = x;
      await new Promise(r => setTimeout(r, 80));
    }
    res.status(200).json({ t: Date.now(), q });
  } catch (e) {
    res.status(502).json({ error: String(e.message || e) });
  }
}
