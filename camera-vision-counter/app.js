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

function setStatus(text) {
  if (!text) { statusEl.classList.add("hidden"); return; }
  statusEl.textContent = text;
  statusEl.classList.remove("hidden");
}

async function startCamera() {
  if (stream) stopStream();
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
  await new Promise((res) => {
    if (video.videoWidth) return res();
    video.onloadedmetadata = () => res();
  });
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

function stopStream() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

async function begin() {
  startErr.hidden = true;
  startBtn.disabled = true;
  startSpinner.hidden = false;

  try {
    setStatus("AI 모델 불러오는 중…");
    if (!model) {
      // lite_mobilenet_v2: 가장 가벼운 모델 (휴대폰에서 빠름)
      model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
    }

    setStatus("카메라 켜는 중…");
    await startCamera();

    startScreen.hidden = true;
    controlsEl.hidden = false;
    running = true;
    setStatus(null);
    loop();
  } catch (err) {
    console.error(err);
    startSpinner.hidden = true;
    startBtn.disabled = false;
    startErr.hidden = false;
    startErr.textContent = errMessage(err);
    setStatus(null);
  }
}

function errMessage(err) {
  const n = err && err.name;
  if (n === "NotAllowedError" || n === "SecurityError")
    return "카메라 권한이 거부되었습니다. 브라우저 설정에서 카메라를 허용해 주세요.";
  if (n === "NotFoundError" || n === "OverconstrainedError")
    return "사용 가능한 카메라를 찾지 못했습니다.";
  if (location.protocol !== "https:" && location.hostname !== "localhost")
    return "카메라는 HTTPS 주소에서만 동작합니다. (Vercel 주소로 접속하세요)";
  return "오류가 발생했습니다: " + (err && err.message ? err.message : err);
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
