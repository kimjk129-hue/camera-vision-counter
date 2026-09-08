/* 스마트 카메라 카운터
 * TensorFlow.js + COCO-SSD 로 카메라 영상에서 사람/차량을 실시간 인식하고 개수를 표시합니다.
 */

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const personCountEl = document.getElementById("personCount");
const vehicleCountEl = document.getElementById("vehicleCount");
const statusEl = document.getElementById("status");
const controlsEl = document.getElementById("controls");

const startScreen = document.getElementById("start");
const startBtn = document.getElementById("startBtn");
const startSpinner = document.getElementById("startSpinner");
const startErr = document.getElementById("startErr");

const flipBtn = document.getElementById("flipBtn");
const boxBtn = document.getElementById("boxBtn");
const stopBtn = document.getElementById("stopBtn");

// COCO-SSD 클래스 → 분류
const VEHICLE_CLASSES = new Set(["bicycle", "car", "motorcycle", "bus", "train", "truck"]);
const KO_LABEL = {
  person: "사람",
  bicycle: "자전거", car: "자동차", motorcycle: "오토바이",
  bus: "버스", train: "기차", truck: "트럭",
};

const MIN_SCORE = 0.55;   // 이 확률 이상만 카운트
const DETECT_EVERY_MS = 120; // 인식 주기 (너무 자주 하면 느려짐)

let model = null;
let stream = null;
let running = false;
let showBoxes = true;
let facingMode = "environment"; // 후면 카메라부터
let lastDetect = 0;
let lastPredictions = [];

// ---- 화면 진단 로그 ----
const dbgEl = document.getElementById("debug");
function dbg(line) {
  const t = new Date().toLocaleTimeString();
  if (dbgEl) dbgEl.textContent += `[${t}] ${line}\n`;
  console.log("[DBG]", line);
}
window.addEventListener("error", (e) =>
  dbg("❌ JS 오류: " + (e.message || e.error) + (e.filename ? " @ " + e.filename : ""))
);
window.addEventListener("unhandledrejection", (e) =>
  dbg("❌ 처리안된 거부: " + (e.reason && e.reason.message ? e.reason.message : e.reason))
);
dbg("페이지 로드됨. UA=" + navigator.userAgent);
dbg("보안컨텍스트(HTTPS): " + window.isSecureContext);
dbg("tf 로드됨: " + (typeof tf !== "undefined") + " / coco-ssd 로드됨: " + (typeof cocoSsd !== "undefined"));
dbg("getUserMedia 지원: " + !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia));

function setStatus(text) {
  if (!text) { statusEl.classList.add("hidden"); return; }
  statusEl.textContent = text;
  statusEl.classList.remove("hidden");
}

async function startCamera() {
  if (stream) stopStream();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
  } catch (e) {
    // 해상도/카메라 지정이 안 먹는 기기 → 가장 단순한 요청으로 재시도
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
  }
  video.srcObject = stream;
  await video.play();
  await new Promise((res) => {
    if (video.videoWidth) return res();
    video.onloadedmetadata = () => res();
  });
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
}

function stopStream() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("TIMEOUT:" + label)), ms)
    ),
  ]);
}

function setStartMsg(text) {
  startErr.hidden = !text;
  startErr.style.color = "#9aa7b8";
  startErr.textContent = text || "";
}

async function begin() {
  startErr.hidden = true;
  startBtn.disabled = true;
  startSpinner.hidden = false;

  // 인앱 브라우저(카카오톡/인스타 등) 경고
  const ua = navigator.userAgent || "";
  if (/KAKAOTALK|Instagram|FBAN|FBAV|Line\//i.test(ua)) {
    setStartMsg("⚠️ 카카오톡/인스타 등 앱 안의 브라우저에서는 카메라가 막힐 수 있어요. Chrome 또는 삼성인터넷으로 열어 주세요.");
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    fail(new Error("NO_GETUSERMEDIA"));
    return;
  }

  try {
    dbg("▶ 시작 버튼 클릭됨");
    setStatus("AI 모델 불러오는 중… (최초 1회, 최대 30초)");
    setStartMsg("AI 모델을 내려받는 중입니다…");
    if (typeof cocoSsd === "undefined" || typeof tf === "undefined") {
      throw new Error("SCRIPT_LOAD");
    }
    dbg("tf 백엔드 준비 대기…");
    await withTimeout(tf.ready(), 15000, "tfready");
    dbg("tf 백엔드 = " + tf.getBackend());
    if (!model) {
      dbg("모델 다운로드 시작 (lite_mobilenet_v2)…");
      model = await withTimeout(
        cocoSsd.load({ base: "lite_mobilenet_v2" }),
        30000,
        "model"
      );
      dbg("✅ 모델 로드 완료");
    }

    setStatus("카메라 켜는 중…");
    setStartMsg("카메라 권한을 허용해 주세요…");
    dbg("카메라 요청 (getUserMedia)…");
    await startCamera();
    dbg("✅ 카메라 스트림 시작: " + overlay.width + "x" + overlay.height);

    startScreen.hidden = true;
    controlsEl.hidden = false;
    running = true;
    setStatus(null);
    loop();
  } catch (err) {
    fail(err);
  }
}

function fail(err) {
  console.error(err);
  dbg("❌ 실패: " + (err && (err.message || err.name) ? err.message || err.name : err));
  startSpinner.hidden = true;
  startBtn.disabled = false;
  startBtn.textContent = "다시 시도";
  startErr.hidden = false;
  startErr.style.color = "#ff6b6b";
  startErr.textContent = errMessage(err);
  setStatus(null);
}

function errMessage(err) {
  const n = err && err.name;
  const m = (err && err.message) || "";
  if (m === "NO_GETUSERMEDIA")
    return "이 브라우저에서는 카메라를 쓸 수 없습니다. Chrome 또는 삼성인터넷으로 열어 주세요.";
  if (m === "SCRIPT_LOAD")
    return "AI 라이브러리를 불러오지 못했습니다. 네트워크(광고 차단/사내 와이파이)를 확인하고 새로고침해 주세요.";
  if (m === "TIMEOUT:model")
    return "AI 모델 다운로드가 너무 오래 걸립니다. 와이파이/LTE 상태를 확인하고 '다시 시도'를 눌러 주세요.";
  if (m === "TIMEOUT:tfready")
    return "그래픽 가속(WebGL) 초기화에 실패했습니다. 브라우저를 완전히 종료 후 다시 열거나 다른 브라우저로 시도해 주세요.";
  if (n === "NotAllowedError" || n === "SecurityError")
    return "카메라 권한이 거부되었습니다. 주소창 왼쪽 자물쇠 → 카메라 → 허용으로 바꾼 뒤 다시 시도해 주세요.";
  if (n === "NotFoundError" || n === "OverconstrainedError")
    return "사용 가능한 카메라를 찾지 못했습니다.";
  if (n === "NotReadableError")
    return "다른 앱이 카메라를 사용 중입니다. 카메라 앱을 모두 닫고 다시 시도해 주세요.";
  if (location.protocol !== "https:" && location.hostname !== "localhost")
    return "카메라는 HTTPS 주소에서만 동작합니다. (Vercel 주소로 접속하세요)";
  return "오류가 발생했습니다: " + (m || err);
}

async function loop() {
  if (!running) return;

  const now = performance.now();
  if (model && video.readyState >= 2 && now - lastDetect > DETECT_EVERY_MS) {
    lastDetect = now;
    try {
      const preds = await model.detect(video, 20);
      lastPredictions = preds.filter((p) => p.score >= MIN_SCORE);
      updateCounts(lastPredictions);
    } catch (e) {
      // 프레임 하나 실패는 무시
    }
  }

  draw(lastPredictions);
  requestAnimationFrame(loop);
}

function updateCounts(preds) {
  let people = 0, vehicles = 0;
  for (const p of preds) {
    if (p.class === "person") people++;
    else if (VEHICLE_CLASSES.has(p.class)) vehicles++;
  }
  personCountEl.textContent = people;
  vehicleCountEl.textContent = vehicles;
}

function draw(preds) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!showBoxes) return;

  ctx.lineWidth = Math.max(2, overlay.width / 400);
  ctx.font = `${Math.max(14, overlay.width / 45)}px -apple-system, "Noto Sans KR", sans-serif`;
  ctx.textBaseline = "top";

  for (const p of preds) {
    const [x, y, w, h] = p.bbox;
    const isPerson = p.class === "person";
    const isVehicle = VEHICLE_CLASSES.has(p.class);
    if (!isPerson && !isVehicle) continue;

    const color = isPerson ? "#22e06b" : "#ffb02e";
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const label = `${KO_LABEL[p.class] || p.class} ${Math.round(p.score * 100)}%`;
    const tw = ctx.measureText(label).width;
    const th = parseInt(ctx.font, 10) * 1.3;
    ctx.fillStyle = color;
    ctx.fillRect(x - ctx.lineWidth / 2, Math.max(0, y - th), tw + 10, th);
    ctx.fillStyle = "#0b0f17";
    ctx.fillText(label, x + 5, Math.max(0, y - th) + 2);
  }
}

// ---- 컨트롤 ----
startBtn.addEventListener("click", begin);

flipBtn.addEventListener("click", async () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  try {
    setStatus("카메라 전환 중…");
    await startCamera();
    setStatus(null);
  } catch (e) {
    setStatus("카메라 전환 실패");
    setTimeout(() => setStatus(null), 1500);
  }
});

boxBtn.addEventListener("click", () => {
  showBoxes = !showBoxes;
  boxBtn.textContent = showBoxes ? "박스 끄기" : "박스 켜기";
  if (!showBoxes) ctx.clearRect(0, 0, overlay.width, overlay.height);
});

stopBtn.addEventListener("click", () => {
  running = false;
  stopStream();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  controlsEl.hidden = true;
  startScreen.hidden = false;
  startBtn.disabled = false;
  startSpinner.hidden = true;
});

// 탭 전환 시 리소스 절약
document.addEventListener("visibilitychange", () => {
  if (document.hidden && stream) {
    stream.getVideoTracks().forEach((t) => (t.enabled = false));
  } else if (stream) {
    stream.getVideoTracks().forEach((t) => (t.enabled = true));
  }
});
