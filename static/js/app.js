/* ═══════════════════════════════════════════════════════════
   WasteVision AI - Frontend Application Logic
   ═══════════════════════════════════════════════════════════
   Handles:
   - Page navigation
   - Image upload + preview
   - Webcam single capture
   - API calls to Flask backend
   - Detection result rendering
   - Gemini-powered chat
   - History/memory management
   - Dashboard charts
   ═══════════════════════════════════════════════════════════ */

// ─── GLOBAL STATE ───────────────────────────────────────────
const STATE = {
  currentDetection: null,
  currentSession: null,
  sessions: [],
  memory: [],
  camStream: null,
  capturedDataUrl: null,
  uploadedFile: null,
  charts: { bar: null, pie: null },
  logs: [],
  apiBase: '',  // same origin
};

const PAGE_TITLES = {
  dashboard: { t: 'Dashboard', s: 'Overview · Detection analytics' },
  detect:    { t: 'Detect Waste', s: 'Upload or capture an image for AI classification' },
  chat:      { t: 'AI Assistant', s: 'Context-aware conversation with Gemini' },
  memory:    { t: 'Detection History', s: 'Browse saved detections' },
  admin:     { t: 'Admin Panel', s: 'System configuration and health' },
};

// ─── ICON MAPPING by class name ─────────────────────────────
const CLASS_ICONS = {
  plastic: '🍶', glass: '🍾', metal: '🥫', paper: '📄', cardboard: '📦',
  trash: '🗑️', organic: '🍎', hazardous: '⚠️', recyclable: '♻️',
  'non-recyclable': '🚫', battery: '🔋', keyboard: '⌨️', microwave: '🔌',
  mobile: '📱', mouse: '🖱️', pcb: '💾', player: '📻', printer: '🖨️',
  television: '📺', 'washing machine': '🧺',
};
function iconFor(name) {
  if (!name) return '🗑️';
  const key = name.toLowerCase();
  for (const k of Object.keys(CLASS_ICONS)) {
    if (key.includes(k)) return CLASS_ICONS[k];
  }
  return '🗑️';
}

// ═══════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════
function navTo(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sb-item').forEach(b => b.classList.remove('active'));
  const pageEl = document.getElementById('page-' + page);
  if (pageEl) pageEl.classList.add('active');
  const btn = document.querySelector(`.sb-item[data-page="${page}"]`);
  if (btn) btn.classList.add('active');
  const meta = PAGE_TITLES[page];
  if (meta) {
    document.getElementById('tbTitle').textContent = meta.t;
    document.getElementById('tbSub').textContent = meta.s;
  }
  addLog('INFO', `Navigated to ${meta?.t || page}`);
  if (page === 'memory') loadHistory();
  if (page === 'dashboard') refreshDashboard();
}

// ═══════════════════════════════════════════════════════════
// NOTIFICATIONS
// ═══════════════════════════════════════════════════════════
function notify(title, msg, type = 'g') {
  const icons = { g: 'check-circle', r: 'exclamation-circle', a: 'exclamation-triangle', b: 'info-circle' };
  const colors = { g: 'var(--g)', r: 'var(--r)', a: 'var(--a)', b: 'var(--b)' };
  const stack = document.getElementById('notif-stack');
  const n = document.createElement('div');
  n.className = 'notif';
  n.style.borderColor = colors[type] + '55';
  n.innerHTML = `
    <i class="fas fa-${icons[type]} notif-icon" style="color:${colors[type]}"></i>
    <div class="notif-text">
      <div class="notif-title">${title}</div>${msg}
    </div>
    <button class="notif-close"><i class="fas fa-times"></i></button>`;
  n.querySelector('.notif-close').addEventListener('click', () => removeNotif(n));
  stack.appendChild(n);
  setTimeout(() => removeNotif(n), 4500);
}
function removeNotif(n) {
  n.classList.add('removing');
  setTimeout(() => n.remove(), 300);
}

// ═══════════════════════════════════════════════════════════
// SYSTEM LOGS
// ═══════════════════════════════════════════════════════════
function addLog(level, msg) {
  STATE.logs.unshift({ time: new Date().toLocaleTimeString(), level, msg });
  if (STATE.logs.length > 100) STATE.logs.pop();
  renderLogs();
}
function renderLogs() {
  const el = document.getElementById('sysLogs');
  if (!el) return;
  el.innerHTML = STATE.logs.slice(0, 40).map(l =>
    `<div class="log-row"><span class="log-time">${l.time}</span><span class="log-level ${l.level.toLowerCase()}">${l.level}</span><span class="log-msg">${l.msg}</span></div>`
  ).join('');
}

// ═══════════════════════════════════════════════════════════
// FILE UPLOAD / PREVIEW
// ═══════════════════════════════════════════════════════════
function setupUploadZone() {
  const zone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');

  zone.addEventListener('click', () => fileInput.click());
  document.getElementById('browseBtn').addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) loadFile(file);
  });

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) loadFile(file);
  });
}

function loadFile(file) {
  STATE.uploadedFile = file;
  STATE.capturedDataUrl = null;
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById('imgPreview');
    img.src = e.target.result;
    img.style.display = 'block';
    document.getElementById('previewCanvas').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = false;
    hideResult();
    notify('Image Loaded', file.name, 'b');
    addLog('INFO', `Image loaded: ${file.name}`);
  };
  reader.readAsDataURL(file);
}

function clearDetection() {
  STATE.uploadedFile = null;
  STATE.capturedDataUrl = null;
  document.getElementById('imgPreview').style.display = 'none';
  document.getElementById('previewCanvas').style.display = 'none';
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('fileInput').value = '';
  hideResult();
}

function hideResult() {
  document.getElementById('resultPanel').classList.remove('show');
}

// ═══════════════════════════════════════════════════════════
// WEBCAM (single capture)
// ═══════════════════════════════════════════════════════════
async function startCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    STATE.camStream = stream;
    const vid = document.getElementById('camVideo');
    vid.srcObject = stream;
    vid.style.display = 'block';
    document.getElementById('camPh').style.display = 'none';
    notify('Camera', 'Webcam started', 'b');
    addLog('INFO', 'Webcam started');
  } catch (err) {
    notify('Camera Error', 'Could not access webcam: ' + err.message, 'r');
    addLog('ERR', 'Webcam access denied');
  }
}

function captureFromCam() {
  const vid = document.getElementById('camVideo');
  if (!vid.srcObject) {
    notify('Camera', 'Start the camera first', 'a');
    return;
  }
  const canvas = document.getElementById('previewCanvas');
  canvas.width = vid.videoWidth;
  canvas.height = vid.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(vid, 0, 0);
  canvas.style.display = 'block';
  document.getElementById('imgPreview').style.display = 'none';
  STATE.capturedDataUrl = canvas.toDataURL('image/jpeg', 0.92);
  STATE.uploadedFile = null;
  document.getElementById('analyzeBtn').disabled = false;
  notify('Captured', 'Image captured — ready to analyze', 'g');
  addLog('INFO', 'Image captured from webcam');
}

function stopCam() {
  if (STATE.camStream) {
    STATE.camStream.getTracks().forEach(t => t.stop());
    STATE.camStream = null;
  }
  document.getElementById('camVideo').style.display = 'none';
  document.getElementById('camPh').style.display = 'flex';
  addLog('INFO', 'Webcam stopped');
}

// ═══════════════════════════════════════════════════════════
// ANALYZE — calls Flask backend
// ═══════════════════════════════════════════════════════════
async function analyzeImage() {
  if (!STATE.uploadedFile && !STATE.capturedDataUrl) {
    notify('No Image', 'Upload or capture an image first', 'a');
    return;
  }
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  showAnalyzing(true, 'Sending image to AI model...');
  addLog('INFO', 'Starting analysis');

  try {
    let response;
    if (STATE.uploadedFile) {
      const formData = new FormData();
      formData.append('image', STATE.uploadedFile);
      showAnalyzing(true, 'Running TensorFlow inference...');
      response = await fetch('/api/predict', { method: 'POST', body: formData });
    } else {
      showAnalyzing(true, 'Running TensorFlow inference...');
      response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: STATE.capturedDataUrl }),
      });
    }

    showAnalyzing(true, 'Querying Gemini for explanation...');
    const data = await response.json();
    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Prediction failed');
    }
    showAnalyzing(false);
    renderDetectionResult(data);
    notify('Detection Complete', `${data.prediction.class_name} (${data.prediction.confidence}%)`, 'g');
    addLog('INFO', `Detected: ${data.prediction.class_name} @ ${data.prediction.confidence}%`);
    // Refresh dashboard counts in background
    refreshDashboard();
  } catch (err) {
    showAnalyzing(false);
    notify('Analysis Failed', err.message, 'r');
    addLog('ERR', 'Analysis error: ' + err.message);
  } finally {
    btn.disabled = false;
  }
}

function showAnalyzing(show, msg) {
  const panel = document.getElementById('analyzingPanel');
  if (show) {
    panel.classList.add('show');
    if (msg) document.getElementById('analyzingStep').textContent = msg;
  } else {
    panel.classList.remove('show');
  }
}

function renderDetectionResult(data) {
  const pred = data.prediction;
  const guidance = data.guidance || {};
  const gemini = data.gemini || {};

  STATE.currentDetection = {
    id: data.id,
    class_name: pred.class_name,
    confidence: pred.confidence,
    hazard: guidance.hazard || 'low',
    recyclable: guidance.recyclable,
    icon: guidance.icon || iconFor(pred.class_name),
    image_url: data.image_url,
    explanation: gemini.explanation,
    disposal: gemini.disposal || guidance.tips,
    hazards: gemini.hazards,
    reuse: gemini.reuse,
    sustainability: gemini.sustainability,
    carbon: gemini.carbon_impact,
    all_probs: pred.all_probabilities || [],
    timestamp: data.timestamp,
  };

  const hClass = guidance.hazard === 'high' ? 'hz-high' :
                 guidance.hazard === 'medium' ? 'hz-med' : 'hz-low';
  document.getElementById('resultInner').className = 'result-inner ' + hClass;
  document.getElementById('resultIcon').textContent = guidance.icon || iconFor(pred.class_name);
  document.getElementById('resultTitle').textContent = pred.class_name + ' Detected';
  document.getElementById('resultDesc').textContent = gemini.explanation || 'Waste item classified by AI model.';

  // Badges
  const badges = [
    { t: pred.class_name, c: 'badge-b' },
    { t: (guidance.hazard || 'low') + ' hazard', c: guidance.hazard === 'high' ? 'badge-r' : guidance.hazard === 'medium' ? 'badge-a' : 'badge-g' },
    { t: guidance.recyclable ? 'Recyclable' : 'Non-Recyclable', c: guidance.recyclable ? 'badge-g' : 'badge-r' },
    { t: `${pred.confidence}% conf`, c: 'badge-p' },
  ];
  document.getElementById('resultBadges').innerHTML = badges.map(b => `<span class="badge ${b.c}">${b.t}</span>`).join('');

  // Confidence ring
  const conf = pred.confidence || 0;
  const offset = Math.round(251 * (1 - conf / 100));
  document.getElementById('confCircle').style.strokeDashoffset = offset;
  document.getElementById('confPct').textContent = Math.round(conf) + '%';

  // Details
  document.getElementById('rd-disposal').textContent = gemini.disposal || guidance.tips || '—';
  document.getElementById('rd-hazards').textContent = gemini.hazards || '—';
  document.getElementById('rd-reuse').textContent = gemini.reuse || '—';
  document.getElementById('rd-sustain').textContent = gemini.sustainability || '—';
  document.getElementById('rd-carbon').textContent = gemini.carbon_impact || '—';

  // Top-K probabilities
  const topk = (pred.all_probabilities || []).slice(0, 5).map(p =>
    `<div style="display:flex;justify-content:space-between;padding:2px 0;"><span>${p.name}</span><span style="color:var(--g);">${p.prob}%</span></div>`
  ).join('');
  document.getElementById('rd-topk').innerHTML = topk || '—';

  // Show panel
  const panel = document.getElementById('resultPanel');
  panel.classList.add('show');
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Activity feed
  addActivityFeed(STATE.currentDetection);
}

function addActivityFeed(det) {
  const feed = document.getElementById('activityFeed');
  if (!feed) return;
  const div = document.createElement('div');
  div.className = 'feed-item';
  const color = det.hazard === 'high' ? 'var(--r)' : det.hazard === 'medium' ? 'var(--a)' : 'var(--g)';
  div.innerHTML = `
    <div class="feed-icon" style="background:${color}18;border:1px solid ${color}33;color:${color};">${det.icon}</div>
    <div>
      <div style="font-weight:700;font-size:.75rem;">${det.class_name}</div>
      <div class="feed-meta">${new Date(det.timestamp || Date.now()).toLocaleTimeString()} · ${det.confidence}%</div>
    </div>`;
  feed.insertBefore(div, feed.firstChild);
  while (feed.children.length > 12) feed.lastChild.remove();
}

// ═══════════════════════════════════════════════════════════
// CHAT — calls /api/chat
// ═══════════════════════════════════════════════════════════
function startChatFromDetection(save) {
  if (!STATE.currentDetection) return;
  if (save) {
    const sess = {
      id: 's_' + Date.now(),
      detectionId: STATE.currentDetection.id,
      name: STATE.currentDetection.class_name,
      icon: STATE.currentDetection.icon,
      waste: STATE.currentDetection,
      messages: [],
      created: new Date().toISOString(),
    };
    STATE.sessions.unshift(sess);
    STATE.currentSession = sess;
    updateSessionList();
    document.getElementById('sessCount').textContent = STATE.sessions.length;
    navTo('chat');
    renderChatContext(STATE.currentDetection);
    renderChatMessages([]);
  } else {
    notify('Acknowledged', 'Detection saved to history without chat', 'b');
  }
}

function startNewChat() {
  STATE.currentSession = null;
  document.getElementById('chatMessages').innerHTML = `
    <div class="msg msg-ai">
      <div class="msg-header"><i class="fas fa-robot"></i>WasteAI</div>
      Hello! Detect a waste item first, or ask me general waste management questions.
    </div>`;
  document.getElementById('chatCtxName').textContent = 'WasteAI Assistant';
  document.getElementById('chatCtxDetail').textContent = 'General waste management';
  document.getElementById('chatCtxIcon').textContent = '🤖';
  document.querySelectorAll('.sess-item').forEach(s => s.classList.remove('active-sess'));
}

function renderChatContext(det) {
  document.getElementById('chatCtxIcon').textContent = det.icon || '🗑️';
  document.getElementById('chatCtxName').textContent = det.class_name || 'Unknown';
  document.getElementById('chatCtxDetail').textContent =
    `Hazard: ${det.hazard || '—'} · Confidence: ${det.confidence || 0}%`;
}

function renderChatMessages(messages) {
  const el = document.getElementById('chatMessages');
  if (!messages || messages.length === 0) {
    el.innerHTML = `
      <div class="msg msg-ai">
        <div class="msg-header"><i class="fas fa-robot"></i>WasteAI</div>
        I have full context about this ${STATE.currentSession?.name || 'item'}.
        Ask me anything — recycling, disposal, hazards, environmental impact, reuse ideas!
      </div>`;
    return;
  }
  el.innerHTML = messages.map(m => `
    <div class="msg ${m.role === 'user' ? 'msg-user' : 'msg-ai'}">
      ${m.role === 'assistant' ? '<div class="msg-header"><i class="fas fa-robot"></i>WasteAI</div>' : ''}
      ${escapeHtml(m.content)}
    </div>`).join('');
  el.scrollTop = el.scrollHeight;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function updateSessionList() {
  const el = document.getElementById('sessionList');
  if (STATE.sessions.length === 0) {
    el.innerHTML = '<div class="empty-state"><i class="fas fa-comments"></i><p>No sessions yet.</p></div>';
    return;
  }
  el.innerHTML = STATE.sessions.map(s => `
    <div class="sess-item ${STATE.currentSession?.id === s.id ? 'active-sess' : ''}" data-sess-id="${s.id}">
      <div class="sess-name">${s.icon} ${s.name}</div>
      <div class="sess-preview">${s.messages.length > 0 ? escapeHtml(s.messages[s.messages.length - 1].content.substring(0, 50)) : 'Start chatting...'}</div>
      <div class="sess-time">${new Date(s.created).toLocaleString()}</div>
    </div>`).join('');
  el.querySelectorAll('.sess-item').forEach(item => {
    item.addEventListener('click', () => openSession(item.dataset.sessId));
  });
}

function openSession(id) {
  const sess = STATE.sessions.find(s => s.id === id);
  if (!sess) return;
  STATE.currentSession = sess;
  renderChatContext(sess.waste);
  renderChatMessages(sess.messages);
  updateSessionList();
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  autoResize(input);

  const el = document.getElementById('chatMessages');
  // Append user message
  const userDiv = document.createElement('div');
  userDiv.className = 'msg msg-user';
  userDiv.textContent = msg;
  el.appendChild(userDiv);

  // Append typing indicator
  const typing = document.createElement('div');
  typing.className = 'msg msg-ai';
  typing.innerHTML = `<div class="msg-header"><i class="fas fa-robot"></i>WasteAI</div><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
  el.appendChild(typing);
  el.scrollTop = el.scrollHeight;

  if (STATE.currentSession) STATE.currentSession.messages.push({ role: 'user', content: msg });

  try {
    const context = STATE.currentSession?.waste || {};
    const history = (STATE.currentSession?.messages || []).slice(0, -1).slice(-10);
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, context, history }),
    });
    const data = await res.json();
    typing.innerHTML = `<div class="msg-header"><i class="fas fa-robot"></i>WasteAI</div>${escapeHtml(data.reply || data.error || 'No reply')}`;
    if (STATE.currentSession && data.reply) {
      STATE.currentSession.messages.push({ role: 'assistant', content: data.reply });
      updateSessionList();
    }
    el.scrollTop = el.scrollHeight;
    addLog('INFO', 'Chat reply received from Gemini');
  } catch (err) {
    typing.innerHTML = `<div class="msg-header"><i class="fas fa-robot"></i>WasteAI</div>Error: ${escapeHtml(err.message)}`;
    addLog('ERR', 'Chat error: ' + err.message);
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// ═══════════════════════════════════════════════════════════
// HISTORY / MEMORY
// ═══════════════════════════════════════════════════════════
async function loadHistory() {
  try {
    const res = await fetch('/api/history?limit=200');
    const data = await res.json();
    STATE.memory = data.history || [];
    document.getElementById('memCount').textContent = STATE.memory.length;
    renderMemory(STATE.memory);
  } catch (err) {
    notify('Error', 'Could not load history: ' + err.message, 'r');
  }
}

function renderMemory(items) {
  const grid = document.getElementById('memoryGrid');
  if (!items || items.length === 0) {
    grid.innerHTML = `
      <div class="empty-state" style="grid-column:1/-1;">
        <i class="fas fa-database"></i>
        <h3>No detections yet</h3>
        <p>Upload or capture a waste image to begin</p>
      </div>`;
    return;
  }
  grid.innerHTML = items.map(m => {
    const color = m.hazard === 'high' ? 'var(--r)' : m.hazard === 'medium' ? 'var(--a)' : 'var(--g)';
    const icon = iconFor(m.class_name);
    return `
      <div class="card mem-card" data-id="${m.id}">
        <div style="padding:1rem;">
          <div class="mem-card-img">${m.image_path ? `<img src="${m.image_path}" alt="${escapeHtml(m.class_name)}">` : icon}</div>
          <div class="mem-id">#${m.id} · ${new Date(m.timestamp).toLocaleDateString()}</div>
          <div class="mem-name">${escapeHtml(m.class_name)}</div>
          <div class="mem-meta">
            <span class="badge" style="background:${color}18;border:1px solid ${color}44;color:${color};">${m.hazard || 'low'}</span>
            <span class="badge badge-b">${m.recyclable ? 'Recyclable' : 'Non-Rec.'}</span>
          </div>
          <div class="mem-footer">
            <div class="mem-footer-time">${(m.confidence || 0).toFixed(1)}% conf</div>
            <button class="btn btn-xs btn-outline-r mem-del-btn" data-id="${m.id}"><i class="fas fa-trash"></i></button>
          </div>
        </div>
      </div>`;
  }).join('');

  grid.querySelectorAll('.mem-card').forEach(card => {
    card.addEventListener('click', (e) => {
      if (e.target.closest('.mem-del-btn')) return;
      openMemoryDetail(card.dataset.id);
    });
  });
  grid.querySelectorAll('.mem-del-btn').forEach(b => {
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteHistory(b.dataset.id);
    });
  });
}

function filterMemory() {
  const search = document.getElementById('memSearch').value.toLowerCase();
  const haz = document.getElementById('memHazFilter').value;
  const filtered = STATE.memory.filter(m => {
    const matchSearch = !search || (m.class_name || '').toLowerCase().includes(search);
    const matchHaz = !haz || m.hazard === haz;
    return matchSearch && matchHaz;
  });
  renderMemory(filtered);
}

async function openMemoryDetail(id) {
  try {
    const res = await fetch('/api/history/' + id);
    const m = await res.json();
    if (m.error) return notify('Error', m.error, 'r');
    document.getElementById('memModalTitle').textContent = `${iconFor(m.class_name)} ${m.class_name} #${m.id}`;
    document.getElementById('memModalBody').innerHTML = `
      ${m.image_path ? `<img src="${m.image_path}" style="width:100%;max-height:240px;object-fit:cover;border-radius:10px;margin-bottom:1rem;">` : ''}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;">
        <div><div style="font-size:.65rem;color:var(--t3);font-family:var(--mono);margin-bottom:.3rem;">HAZARD</div><div style="font-weight:700;color:${m.hazard === 'high' ? 'var(--r)' : m.hazard === 'medium' ? 'var(--a)' : 'var(--g)'};">${m.hazard || 'low'}</div></div>
        <div><div style="font-size:.65rem;color:var(--t3);font-family:var(--mono);margin-bottom:.3rem;">CONFIDENCE</div><div style="font-weight:700;color:var(--g);">${(m.confidence || 0).toFixed(2)}%</div></div>
        <div><div style="font-size:.65rem;color:var(--t3);font-family:var(--mono);margin-bottom:.3rem;">RECYCLABLE</div><div style="font-weight:700;">${m.recyclable ? '✓ Yes' : '✗ No'}</div></div>
        <div><div style="font-size:.65rem;color:var(--t3);font-family:var(--mono);margin-bottom:.3rem;">TIMESTAMP</div><div style="font-weight:700;font-size:.8rem;">${new Date(m.timestamp).toLocaleString()}</div></div>
      </div>
      <div style="padding:.75rem;background:rgba(0,255,136,0.04);border-radius:8px;border:1px solid rgba(0,255,136,0.08);margin-bottom:1rem;">
        <div style="font-size:.65rem;color:var(--t3);font-family:var(--mono);margin-bottom:.5rem;">AI EXPLANATION</div>
        <div style="font-size:.82rem;color:var(--t2);">${escapeHtml(m.gemini_explanation || '—')}</div>
      </div>
      <div style="padding:.75rem;background:rgba(0,212,255,0.04);border-radius:8px;border:1px solid rgba(0,212,255,0.08);">
        <div style="font-size:.65rem;color:var(--t3);font-family:var(--mono);margin-bottom:.5rem;">DISPOSAL METHOD</div>
        <div style="font-size:.82rem;color:var(--t2);">${escapeHtml(m.disposal_method || '—')}</div>
      </div>`;
    document.getElementById('memModal').classList.add('open');
    document.getElementById('memModalDelete').onclick = () => {
      deleteHistory(id);
      closeModal('memModal');
    };
    document.getElementById('memModalChat').onclick = () => {
      // Open a chat session from this saved item
      STATE.currentDetection = {
        id: m.id,
        class_name: m.class_name,
        confidence: m.confidence,
        hazard: m.hazard,
        icon: iconFor(m.class_name),
        explanation: m.gemini_explanation,
        disposal: m.disposal_method,
      };
      closeModal('memModal');
      startChatFromDetection(true);
    };
  } catch (err) {
    notify('Error', err.message, 'r');
  }
}

async function deleteHistory(id) {
  try {
    await fetch('/api/history/' + id, { method: 'DELETE' });
    STATE.memory = STATE.memory.filter(m => String(m.id) !== String(id));
    document.getElementById('memCount').textContent = STATE.memory.length;
    renderMemory(STATE.memory);
    notify('Deleted', `Record #${id} removed`, 'a');
    refreshDashboard();
  } catch (err) {
    notify('Error', err.message, 'r');
  }
}

async function clearAllMemory() {
  if (!confirm('Delete ALL detections? This cannot be undone.')) return;
  await fetch('/api/history', { method: 'DELETE' });
  STATE.memory = [];
  document.getElementById('memCount').textContent = 0;
  renderMemory([]);
  notify('Cleared', 'All history deleted', 'r');
  refreshDashboard();
}

function closeModal(id) {
  document.getElementById(id).classList.remove('open');
}

// ═══════════════════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════════════════
async function refreshDashboard() {
  try {
    const res = await fetch('/api/stats');
    const stats = await res.json();
    document.getElementById('sc-total').textContent = stats.total || 0;
    document.getElementById('sc-recyc').textContent = (stats.recyclable_pct || 0) + '%';
    document.getElementById('sc-haz').textContent = stats.hazardous || 0;
    document.getElementById('sc-carbon').textContent = (stats.co2_saved_tons || 0) + 't';
    document.getElementById('sc-total-delta').textContent = `${stats.total || 0} total`;

    document.getElementById('confBar').style.width = (stats.avg_confidence || 0) + '%';
    document.getElementById('confBarText').textContent = (stats.avg_confidence || 0).toFixed(1) + '%';
    document.getElementById('memCount').textContent = stats.total || 0;

    updateCharts(stats.by_class || []);
  } catch (err) {
    console.warn('Stats fetch failed:', err);
  }
}

function initCharts() {
  const chartFont = { family: "'JetBrains Mono',monospace", size: 11 };
  const barCtx = document.getElementById('barChart').getContext('2d');
  STATE.charts.bar = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: ['No data'],
      datasets: [{
        label: 'Detections',
        data: [0],
        backgroundColor: '#00ff8833',
        borderColor: '#00ff88',
        borderWidth: 2, borderRadius: 7,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#7ab8a0', font: chartFont }, grid: { color: 'rgba(0,255,136,0.04)' } },
        y: { ticks: { color: '#7ab8a0', font: chartFont }, grid: { color: 'rgba(0,255,136,0.04)' }, beginAtZero: true },
      },
    },
  });

  const pieCtx = document.getElementById('pieChart').getContext('2d');
  STATE.charts.pie = new Chart(pieCtx, {
    type: 'doughnut',
    data: {
      labels: ['No data'],
      datasets: [{
        data: [1],
        backgroundColor: ['#00d4ff99'],
        borderColor: ['#00d4ff'],
        borderWidth: 2, hoverOffset: 8,
      }],
    },
    options: { responsive: true, maintainAspectRatio: false, cutout: '65%', plugins: { legend: { display: false } } },
  });
}

const CHART_COLORS = [
  '#00d4ff', '#00ff88', '#ffd264', '#c864ff',
  '#ff3366', '#ff9966', '#96dcff', '#b0c4de', '#ffaa00', '#ff6699',
];

function updateCharts(byClass) {
  if (!STATE.charts.bar || !STATE.charts.pie) return;
  const labels = byClass.length ? byClass.map(c => c.class_name) : ['No data'];
  const data = byClass.length ? byClass.map(c => c.c) : [1];
  const colors = labels.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]);
  const colorsFill = colors.map(c => c + '55');

  STATE.charts.bar.data.labels = labels;
  STATE.charts.bar.data.datasets[0].data = data;
  STATE.charts.bar.data.datasets[0].backgroundColor = colorsFill;
  STATE.charts.bar.data.datasets[0].borderColor = colors;
  STATE.charts.bar.update();

  STATE.charts.pie.data.labels = labels;
  STATE.charts.pie.data.datasets[0].data = data;
  STATE.charts.pie.data.datasets[0].backgroundColor = colors.map(c => c + 'aa');
  STATE.charts.pie.data.datasets[0].borderColor = colors;
  STATE.charts.pie.update();
}

// ═══════════════════════════════════════════════════════════
// HEALTH CHECK
// ═══════════════════════════════════════════════════════════
async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const h = await res.json();
    // Sidebar status
    const modelDot = document.getElementById('modelDot');
    const modelStatus = document.getElementById('modelStatus');
    const geminiDot = document.getElementById('geminiDot');
    const geminiStatus = document.getElementById('geminiStatus');

    if (h.model_loaded) {
      modelDot.style.background = 'var(--g)';
      modelDot.style.boxShadow = '0 0 8px var(--g)';
      modelStatus.textContent = 'AI MODEL: ONLINE';
    } else {
      modelDot.style.background = 'var(--a)';
      modelDot.style.boxShadow = '0 0 8px var(--a)';
      modelStatus.textContent = 'AI MODEL: FALLBACK';
    }

    if (h.gemini_configured) {
      geminiDot.style.background = 'var(--b)';
      geminiStatus.textContent = 'GEMINI: CONNECTED';
    } else {
      geminiDot.style.background = 'var(--a)';
      geminiStatus.textContent = 'GEMINI: NOT CONFIGURED';
    }

    // Admin panel health badges
    const hm = document.getElementById('healthModel');
    if (hm) {
      hm.textContent = h.model_loaded ? 'LOADED' : 'FALLBACK';
      hm.className = 'badge ' + (h.model_loaded ? 'badge-g' : 'badge-a');
    }
    const hg = document.getElementById('healthGemini');
    if (hg) {
      hg.textContent = h.gemini_configured ? 'READY' : 'OFFLINE';
      hg.className = 'badge ' + (h.gemini_configured ? 'badge-g' : 'badge-a');
    }

    // Model name/classes
    document.getElementById('modelClasses').textContent =
      `${(h.classes || []).length} classes loaded`;

    addLog('INFO', `Health: model=${h.model_loaded}, gemini=${h.gemini_configured}`);
  } catch (err) {
    addLog('ERR', 'Health check failed: ' + err.message);
  }
}

// ═══════════════════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════════════════
function toggleExportMenu() {
  document.getElementById('exportMenu').classList.toggle('open');
}

function exportData(format) {
  document.getElementById('exportMenu').classList.remove('open');
  if (format === 'json') {
    download('wastevision-export.json',
      JSON.stringify({ history: STATE.memory, sessions: STATE.sessions, exported: new Date().toISOString() }, null, 2),
      'application/json');
    notify('Export', 'JSON downloaded', 'g');
  } else if (format === 'csv') {
    const headers = 'ID,Class,Confidence,Hazard,Recyclable,Timestamp,Disposal\n';
    const rows = STATE.memory.map(m =>
      `${m.id},"${m.class_name}",${m.confidence},${m.hazard},${m.recyclable},${m.timestamp},"${(m.disposal_method || '').replace(/"/g, "'")}"`
    ).join('\n');
    download('wastevision-export.csv', headers + rows, 'text/csv');
    notify('Export', 'CSV downloaded', 'g');
  }
}

function download(name, content, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type }));
  a.download = name;
  a.click();
}

// ═══════════════════════════════════════════════════════════
// EVENT BINDINGS
// ═══════════════════════════════════════════════════════════
function bindEvents() {
  // Sidebar navigation
  document.querySelectorAll('.sb-item').forEach(btn => {
    btn.addEventListener('click', () => navTo(btn.dataset.page));
  });

  // Top bar buttons
  document.getElementById('newDetectionBtn').addEventListener('click', () => navTo('detect'));
  document.getElementById('exportBtn').addEventListener('click', toggleExportMenu);
  document.querySelectorAll('.export-option').forEach(opt => {
    opt.addEventListener('click', () => exportData(opt.dataset.format));
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#exportMenu')) document.getElementById('exportMenu').classList.remove('open');
  });

  // Detection page
  document.getElementById('analyzeBtn').addEventListener('click', analyzeImage);
  document.getElementById('clearBtn').addEventListener('click', clearDetection);
  document.getElementById('camStartBtn').addEventListener('click', startCam);
  document.getElementById('camCaptureBtn').addEventListener('click', captureFromCam);
  document.getElementById('camStopBtn').addEventListener('click', stopCam);
  document.getElementById('chatYesBtn').addEventListener('click', () => startChatFromDetection(true));
  document.getElementById('chatNoBtn').addEventListener('click', () => startChatFromDetection(false));

  // Chat
  document.getElementById('newChatBtn').addEventListener('click', startNewChat);
  document.getElementById('chatSendBtn').addEventListener('click', sendChat);
  const chatInput = document.getElementById('chatInput');
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });
  chatInput.addEventListener('input', () => autoResize(chatInput));
  document.querySelectorAll('.hint-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      chatInput.value = chip.textContent;
      sendChat();
    });
  });

  // Memory
  document.getElementById('memSearch').addEventListener('input', filterMemory);
  document.getElementById('memHazFilter').addEventListener('change', filterMemory);
  document.getElementById('refreshHistoryBtn').addEventListener('click', loadHistory);
  document.getElementById('clearAllBtn').addEventListener('click', clearAllMemory);

  // Modal
  document.getElementById('memModalCloseX').addEventListener('click', () => closeModal('memModal'));
  document.getElementById('memModalClose').addEventListener('click', () => closeModal('memModal'));
  document.getElementById('memModal').addEventListener('click', (e) => {
    if (e.target.id === 'memModal') closeModal('memModal');
  });

  // Admin
  document.getElementById('clearLogsBtn').addEventListener('click', () => {
    STATE.logs = [];
    renderLogs();
  });

  // Upload zone
  setupUploadZone();
}

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
function init() {
  bindEvents();
  initCharts();
  checkHealth();
  refreshDashboard();
  loadHistory();
  addLog('INFO', 'WasteVision AI frontend initialized');
  addLog('INFO', 'Connected to Flask backend');
}

document.addEventListener('DOMContentLoaded', init);
