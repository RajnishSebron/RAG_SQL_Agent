/**
 * Enterprise RAG & SQL Copilot — Frontend Application
 *
 * Vanilla JS client that communicates with the FastAPI backend.
 */

const API = "http://localhost:8000/api";

// ── State ────────────────────────────────────────────────────────────────────
let currentSessionId = null;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $sessionList = document.getElementById("session-list");
const $btnNewSession = document.getElementById("btn-new-session");
const $uploadForm = document.getElementById("upload-form");
const $fileInput = document.getElementById("file-input");
const $uploadStatus = document.getElementById("upload-status");
const $chatMessages = document.getElementById("chat-messages");
const $chatForm = document.getElementById("chat-form");
const $chatInput = document.getElementById("chat-input");
const $statDocs = document.getElementById("stat-docs");
const $statChunks = document.getElementById("stat-chunks");
const $statSessions = document.getElementById("stat-sessions");
const $statMessages = document.getElementById("stat-messages");

// ── Helpers ──────────────────────────────────────────────────────────────────

async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function appendMessage(role, content, sources = []) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = content;

  if (sources && sources.length > 0) {
    const srcDiv = document.createElement("div");
    srcDiv.className = "msg-sources";
    const srcTags = sources.map(s => `${s.filename} (chunk #${s.chunk_index})`).join(", ");
    srcDiv.textContent = `📎 Sources: ${srcTags}`;
    div.appendChild(srcDiv);
  }

  $chatMessages.appendChild(div);
  $chatMessages.scrollTop = $chatMessages.scrollHeight;
}

// ── Analytics ────────────────────────────────────────────────────────────────

async function refreshAnalytics() {
  try {
    const data = await api("/analytics/");
    $statDocs.textContent = data.total_documents;
    $statChunks.textContent = data.total_chunks;
    $statSessions.textContent = data.total_sessions;
    $statMessages.textContent = data.total_messages;
  } catch {
    /* analytics panel is best-effort */
  }
}

// ── Sessions ─────────────────────────────────────────────────────────────────

async function loadSessions() {
  const sessions = await api("/chat/sessions");
  $sessionList.innerHTML = "";
  sessions.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s.title;
    li.dataset.id = s.id;
    if (s.id === currentSessionId) li.classList.add("active");
    li.addEventListener("click", () => selectSession(s.id));
    $sessionList.appendChild(li);
  });
}

async function selectSession(id) {
  currentSessionId = id;
  const session = await api(`/chat/sessions/${id}`);
  $chatMessages.innerHTML = "";
  session.messages.forEach((m) => appendMessage(m.role, m.content));
  await loadSessions(); // refresh active highlight
}

$btnNewSession.addEventListener("click", async () => {
  const session = await api("/chat/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "New Chat" }),
  });
  currentSessionId = session.id;
  $chatMessages.innerHTML = "";
  await loadSessions();
  refreshAnalytics();
});

// ── Upload ───────────────────────────────────────────────────────────────────

$uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $fileInput.files[0];
  if (!file) return;

  $uploadStatus.textContent = "Uploading…";
  const fd = new FormData();
  fd.append("file", file);

  try {
    const doc = await api("/upload/", { method: "POST", body: fd });
    $uploadStatus.textContent = `✅ Uploaded "${doc.filename}"`;
    $fileInput.value = "";
    refreshAnalytics();
  } catch (err) {
    $uploadStatus.textContent = `❌ ${err.message}`;
  }
});

// ── Chat ─────────────────────────────────────────────────────────────────────

$chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $chatInput.value.trim();
  if (!text || !currentSessionId) return;

  appendMessage("user", text);
  $chatInput.value = "";

  try {
    const msg = await api(`/chat/sessions/${currentSessionId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: text, query: text }),
    });
    appendMessage("assistant", msg.answer || msg.content, msg.sources);
    refreshAnalytics();
  } catch (err) {
    appendMessage("assistant", `⚠️ Error: ${err.message}`);
  }
});

// ── Init ─────────────────────────────────────────────────────────────────────

(async () => {
  await loadSessions();
  await refreshAnalytics();
})();

