// main.js - robust UI client for Smart Pen
const WS_HOST = window.location.hostname; // same host as Flask
const WS_PORT = 8765;
const WS_PATH = "/ui";
let ws = null;
let wsBackoff = 1000;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const stateEl = document.getElementById("state");
const lastPacketEl = document.getElementById("lastPacket");

function connectWS() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  const url = `ws://${WS_HOST}:${WS_PORT}${WS_PATH}`;
  console.log("[UI] connecting to", url);
  ws = new WebSocket(url);

  ws.onopen = () => {
    console.log("[UI] ws open");
    wsBackoff = 1000;
  };

  ws.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      renderLastPacket(data);
    } catch (err) {
      // If it's not JSON, show raw
      lastPacketEl.textContent = evt.data;
    }
  };

  ws.onclose = () => {
    console.log("[UI] ws closed, will reconnect in", wsBackoff);
    setTimeout(connectWS, wsBackoff);
    wsBackoff = Math.min(16000, wsBackoff * 2);
  };

  ws.onerror = (e) => {
    console.error("[UI] ws error", e);
    ws.close();
  };
}

function renderLastPacket(data) {
  // Expect fields: timestamp_ms, ax, ay, az, gx, gy, gz
  const ts = data.timestamp_ms ? new Date(data.timestamp_ms).toISOString() : new Date().toISOString();
  lastPacketEl.textContent = `time: ${ts}
ax: ${data.ax ?? "-"}
ay: ${data.ay ?? "-"}
az: ${data.az ?? "-"}
gx: ${data.gx ?? "-"}
gy: ${data.gy ?? "-"}
gz: ${data.gz ?? "-"}`;
}

async function postJson(path, obj) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(obj)
  });
  return res.json();
}

startBtn.addEventListener("click", async () => {
  const student = document.getElementById("student").value.trim();
  const sid = document.getElementById("sid").value.trim();
  const hand = document.getElementById("hand").value;
  const sample = document.getElementById("sample").value.trim();
  const quality = document.getElementById("quality").value;

  if (!student || !sid || !sample) {
    alert("Please fill student, ID and sample fields.");
    return;
  }

  try {
    const resp = await postJson("/start", { student, id: sid, hand, sample, quality });
    console.log("start resp", resp);
    stateEl.textContent = "Recording";
  } catch (e) {
    console.error("Start failed", e);
    alert("Start failed — see console");
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    const resp = await postJson("/stop", {});
    console.log("stop resp", resp);
    stateEl.textContent = "Idle";
  } catch (e) {
    console.error("Stop failed", e);
    alert("Stop failed — see console");
  }
});

// poll server status every second
async function pollStatus() {
  try {
    const r = await fetch("/status");
    const j = await r.json();
    stateEl.textContent = j.record ? "Recording" : "Idle";
  } catch (e) {
    stateEl.textContent = "Error";
  }
  setTimeout(pollStatus, 1000);
}

// init
connectWS();
pollStatus();
