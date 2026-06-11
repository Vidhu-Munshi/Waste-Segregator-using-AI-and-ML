// ── WasteVision Backend Integration ──────────────────────────────────────────
// Add this <script> block just before </body> in ai-waste-platform.html
// Set API_BASE to wherever your FastAPI server is running.

const API_BASE = "http://localhost:8000";

// ── Image Upload → /detect ────────────────────────────────────────────────────
async function detectWithBackend(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/detect`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Backend error ${res.status}`);
  return await res.json();
  // Returns: { class, confidence, recyclable, hazard_level, all_scores, bboxes }
}

// ── OCR → /ocr ────────────────────────────────────────────────────────────────
async function ocrWithBackend(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/ocr`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`OCR error ${res.status}`);
  return await res.json();
  // Returns: { text, details: [{text, confidence}] }
}

// ── History → /history ───────────────────────────────────────────────────────
async function loadHistory(limit = 50) {
  const res = await fetch(`${API_BASE}/history?limit=${limit}`);
  return await res.json();
}

// ── PDF Report → /report/pdf ─────────────────────────────────────────────────
function downloadPdfReport() {
  window.open(`${API_BASE}/report/pdf`, "_blank");
}

// ── WebSocket Live Webcam ────────────────────────────────────────────────────
let _ws = null;

function startWebcamBackend(onResult) {
  _ws = new WebSocket(`${API_BASE.replace("http", "ws")}/ws/webcam`);
  _ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (!data.error) onResult(data);
  };
  return _ws;
}

function sendFrameToBackend(base64DataUrl) {
  if (_ws && _ws.readyState === WebSocket.OPEN) {
    _ws.send(JSON.stringify({ image: base64DataUrl }));
  }
}

function stopWebcamBackend() {
  if (_ws) { _ws.close(); _ws = null; }
}

// ── Hook into existing upload handler ────────────────────────────────────────
// Replace the existing handleUpload / processImage functions with these:

async function handleUpload(e) {
  const file = e.dataTransfer?.files[0] || e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = document.getElementById("uploadedImage") || document.querySelector(".upload-preview img");
    if (img) img.src = ev.target.result;
  };
  reader.readAsDataURL(file);

  addLog("INFO", `Analyzing ${file.name}…`);
  notify("Analyzing", "Sending to AI backend…", "b");

  try {
    const result = await detectWithBackend(file);

    // Map backend response to the frontend WASTE_DB shape
    const det = {
      id: generateId(),
      name: result.class.charAt(0).toUpperCase() + result.class.slice(1),
      category: result.class,
      hazard: result.hazard_level,
      confidence: Math.round(result.confidence * 100),
      recyclable: result.recyclable,
      recycleMethod: result.recyclable ? "Send to recycling facility" : "Dispose at hazardous waste site",
      disposal: `Hazard level: ${result.hazard_level}`,
      bboxes: result.bboxes,
      timestamp: new Date().toISOString(),
      icon: "🗑️",
      imageData: null,
    };

    STATE.currentDetection = det;
    showDetectionResult(det.name, det);
    addLog("INFO", `Detected: ${det.name} (${det.confidence}%)`);
    notify("Detection Complete", `${det.name} — ${det.confidence}% confidence`, "g");

    // Optionally run OCR too
    const ocrResult = await ocrWithBackend(file);
    if (ocrResult.text) {
      addLog("INFO", `OCR: ${ocrResult.text.slice(0, 80)}`);
    }
  } catch (err) {
    addLog("ERR", "Backend detection failed: " + err.message);
    notify("Error", "Backend offline — using demo mode", "r");
  }
}

// Override PDF export to use real backend
function exportData(format) {
  document.getElementById("exportMenu")?.classList.remove("open");
  if (format === "json") {
    loadHistory().then((data) => {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const a = Object.assign(document.createElement("a"), {
        href: URL.createObjectURL(blob),
        download: "wastevision-export.json",
      });
      a.click();
      notify("Export", "JSON downloaded from backend", "g");
    });
  } else if (format === "pdf") {
    downloadPdfReport();
    notify("Export", "PDF report opening…", "g");
  } else if (format === "csv") {
    loadHistory().then((data) => {
      const rows = data.detections.map((d) =>
        `${d.id},${d.waste_class},${d.confidence},${d.recyclable ? "Yes" : "No"},${d.hazard_level},${d.timestamp}`
      );
      const csv = "ID,Class,Confidence,Recyclable,Hazard,Timestamp\n" + rows.join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const a = Object.assign(document.createElement("a"), {
        href: URL.createObjectURL(blob),
        download: "wastevision-export.csv",
      });
      a.click();
      notify("Export", "CSV downloaded from backend", "g");
    });
  }
}

// Intercept webcam frames and send every N-th to backend
let _frameCount = 0;
const FRAME_INTERVAL = 30; // send every 30 frames (~1/sec at 30fps)

function patchWebcamLoop() {
  const originalDraw = window.drawLiveCanvas;
  const canvas = document.createElement("canvas");

  window.drawLiveCanvas = function () {
    originalDraw?.();
    _frameCount++;
    if (_frameCount % FRAME_INTERVAL !== 0) return;
    const vid = document.getElementById("liveVideo");
    if (!vid || !vid.videoWidth || !STATE.liveActive) return;
    canvas.width = vid.videoWidth;
    canvas.height = vid.videoHeight;
    canvas.getContext("2d").drawImage(vid, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
    sendFrameToBackend(dataUrl);
  };
}

// Start WS when live camera starts, patch frame loop
const _origStartLive = window.startLive;
window.startLive = function () {
  _origStartLive?.();
  startWebcamBackend((result) => {
    addLog("INFO", `[WS] ${result.class} ${Math.round(result.confidence * 100)}%`);
  });
  patchWebcamLoop();
};

const _origStopLive = window.stopLive;
window.stopLive = function () {
  _origStopLive?.();
  stopWebcamBackend();
};
