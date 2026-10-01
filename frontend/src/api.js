// Talks to the FastAPI backend (Phase 6). Every call carries the login token (user accounts).
export const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

// --- session token ---
// ponytail: localStorage is readable by any script on this page, so an XSS bug could steal the token.
// We never render HTML from answers (see <Rich>), which keeps that risk low; an httpOnly cookie is the stronger option.
const TOKEN_KEY = "orchestrator_token";
const readToken = () => { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; } };
let token = readToken();

function setToken(value) {
  token = value;
  try { value ? localStorage.setItem(TOKEN_KEY, value) : localStorage.removeItem(TOKEN_KEY); } catch { /* private mode */ }
}

export class LoggedOut extends Error {} // thrown on 401: the UI shows the login screen again
const auth = () => (token ? { Authorization: `Bearer ${token}` } : {});

async function failure(res, fallback, sessionCall = true) {
  // A 401 on a normal call = the session ended. A 401 from /auth/login = wrong credentials: show the server's message.
  if (res.status === 401 && sessionCall) { setToken(""); return new LoggedOut("Please log in."); }
  const body = await res.json().catch(() => ({}));
  const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
  return new Error(detail || `${fallback} (${res.status})`);
}

async function json(path, options = {}) {
  const res = await fetch(`${API}${path}`, { ...options, headers: { ...options.headers, ...auth() } });
  if (!res.ok) throw await failure(res, `${path} failed`, !path.startsWith("/auth/"));
  return res.status === 204 ? null : res.json();
}

const post = (path, body) =>
  json(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// --- accounts ---
export const me = () => (token ? json("/auth/me") : Promise.reject(new LoggedOut("Please log in.")));
export const register = (username, password) => post("/auth/register", { username, password });
export async function login(username, password) {
  const { token: t } = await post("/auth/login", { username, password });
  setToken(t);
}
export async function logout() {
  await json("/auth/logout", { method: "POST" }).catch(() => {});
  setToken("");
}

// --- data ---
export const getAgents = () => json("/agents");
export const fileUrl = (name) => `/files/${encodeURIComponent(name)}`;
export const getPreview = (name) => json(`${fileUrl(name)}/preview`);

// A plain <a href> can't send the Authorization header, so fetch the file and hand it to the browser as a blob.
export async function downloadFile(name) {
  const res = await fetch(`${API}${fileUrl(name)}`, { headers: auth() });
  if (!res.ok) throw await failure(res, "Download failed");
  const url = URL.createObjectURL(await res.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Upload a document: raw bytes as the body. The server saves it in the workspace (a taken name gets _2) and,
// for text documents, adds it to this user's RAG knowledge base. Returns {name, size, rag_chunks, rag_error}.
export async function uploadFile(file) {
  const res = await fetch(`${API}${fileUrl(file.name)}`, {
    method: "PUT", headers: { "Content-Type": "application/octet-stream", ...auth() }, body: file,
  });
  if (!res.ok) throw await failure(res, "Upload failed");
  return res.json();
}

// Phase 8: memory
export const getThreads = () => json("/threads");
export const getThread = (id) => json(`/threads/${encodeURIComponent(id)}`);
export const getProfile = () => json("/profile");
export const saveProfile = (profile) =>
  json("/profile", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(profile) });

export const streamTask = (message, threadId, onEvent, signal) =>
  streamSSE("/tasks/stream", threadId ? { message, thread_id: threadId } : { message }, onEvent, signal);

// Phase 9: send the human's decisions for a paused run; the rest of the run streams back like a normal task.
export const resumeTask = (threadId, decisions, onEvent, signal) =>
  streamSSE(`/tasks/${encodeURIComponent(threadId)}/resume`, { decisions }, onEvent, signal);

// The browser's EventSource only supports GET, and our stream is a POST with a JSON body,
// so we read the response body ourselves and parse the SSE format: "event: X\ndata: {json}\n\n".
async function streamSSE(path, payload, onEvent, signal) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...auth() },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) throw await failure(res, "Request failed");

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {  // a blank line ends one event
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}
