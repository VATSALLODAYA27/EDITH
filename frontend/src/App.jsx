import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { downloadFile, getAgents, getPreview, getProfile, getThread, getThreads, LoggedOut, login, logout, me, register,
  resumeTask, saveProfile, streamTask } from "./api.js";

// Fixed ring order so the constellation layout never jumps; descriptions come from GET /agents.
const RING = [
  ["research_agent", "📚", "Research"],
  ["browser_agent", "🌐", "Browser"],
  ["rag_agent", "🗂️", "Knowledge"],
  ["document_agent", "📄", "Docs"],
  ["excel_agent", "📊", "Excel"],
  ["ppt_agent", "📽️", "Slides"],
  ["email_agent", "✉️", "Email"],
  ["calendar_agent", "📅", "Calendar"],
];
const META = Object.fromEntries(RING.map(([name, icon, label]) => [name, { icon, label }]));

const EXAMPLES = [
  "Read survey_report.pdf and create survey_deck.pptx summarizing it in 3 slides.",
  "What's our meal allowance when travelling, and what's the capital of Australia?",
  "Check if I'm free tomorrow at 3 PM; if so, draft an email to john.miller@nimbuslabs.com proposing a sync then.",
  "What is the latest stable version of Python?",
];

// Agents answer in markdown (lists, tables, bold...). react-markdown builds React elements - never innerHTML -
// and skipHtml DROPS any raw HTML, because LLM output is untrusted (it may quote a web page or a phishing email).
// It also neutralises javascript: links. Links open in a new tab without giving that page access to ours.
const MD_COMPONENTS = { a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" /> };

function Rich({ text }) {
  return (
    <div className="md">
      <Markdown remarkPlugins={[remarkGfm]} skipHtml components={MD_COMPONENTS}>{text}</Markdown>
    </div>
  );
}

const C = 200, R = 145; // SVG centre and ring radius
const pos = (i) => {
  const a = (i / RING.length) * 2 * Math.PI - Math.PI / 2;
  return [C + R * Math.cos(a), C + R * Math.sin(a)];
};

function Constellation({ nodes, phase, turn, descriptions }) {
  const busy = phase === "running";
  return (
    <svg className="constellation" viewBox="0 0 400 400" role="img"
         aria-label="Orchestrator in the centre, connected to eight agents">
      {RING.map(([name], i) => {
        const [x, y] = pos(i);
        return <line key={name} x1={C} y1={C} x2={x} y2={y} className={`link ${nodes[name] ?? "idle"}`} />;
      })}
      <g className={`hub ${busy ? "busy" : ""}`}>
        <circle cx={C} cy={C} r="46" className="hub-glow" />
        <circle cx={C} cy={C} r="38" className="hub-core" />
        <text x={C} y={C - 4} className="hub-title">ORCHESTRATOR</text>
        <text x={C} y={C + 14} className="hub-sub">{busy ? (turn ? `turn ${turn}` : "planning") : phase}</text>
      </g>
      {RING.map(([name, icon, label], i) => {
        const [x, y] = pos(i);
        const state = nodes[name] ?? "idle";
        return (
          <g key={name} className={`node ${state}`}>
            <title>{`${label}: ${descriptions[name] ?? ""}`}</title>
            <circle cx={x} cy={y} r="27" className="node-ring" />
            <text x={x} y={y + 7} className="node-icon">{icon}</text>
            <text x={x} y={y + 44} className="node-label">{label}</text>
            {state === "done" && <text x={x + 20} y={y - 18} className="node-check">✓</text>}
          </g>
        );
      })}
    </svg>
  );
}

// --- Output file previews: drawn from JSON the API extracts (content + structure, not PowerPoint's exact look) ---
function SlidePreview({ slides }) {
  return (
    <div className="slides">
      {slides.map((s) => (
        <div key={s.number} className={`slide ${s.number === 1 ? "title-slide" : ""}`}>
          <span className="slide-no">{s.number}</span>
          <h3>{s.title || "(untitled)"}</h3>
          {s.number === 1
            ? s.bullets.map((b, i) => <p key={i} className="subtitle">{b}</p>)
            : <ul>{s.bullets.map((b, i) => <li key={i}>{b}</li>)}</ul>}
          {s.notes && <p className="notes" title={s.notes}>🗒 {s.notes}</p>}
        </div>
      ))}
    </div>
  );
}

function DocPreview({ blocks }) {
  return (
    <div className="paper">
      {blocks.map((b, i) => b.kind === "h1" ? <h3 key={i}>{b.text}</h3>
        : b.kind === "h2" ? <h4 key={i}>{b.text}</h4>
        : b.kind === "bullet" ? <p key={i} className="bullet">• {b.text}</p>
        : <p key={i}>{b.text}</p>)}
    </div>
  );
}

function SheetPreview({ sheets }) {
  return sheets.map((sh) => (
    <div key={sh.name} className="sheet">
      <p className="sheet-name">▦ {sh.name}{sh.charts ? ` · 📈 ${sh.charts} chart${sh.charts > 1 ? "s" : ""}` : ""}
        {sh.total_rows > sh.rows.length && ` · showing ${sh.rows.length} of ${sh.total_rows} rows`}</p>
      <div className="table-wrap">
        <table>
          <thead><tr>{(sh.rows[0] ?? []).map((c, i) => <th key={i}>{c}</th>)}</tr></thead>
          <tbody>{sh.rows.slice(1).map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </div>
  ));
}

const FILE_ICON = { pptx: "📽️", docx: "📄", xlsx: "📊", pdf: "📕" };

function FileCard({ name }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => { getPreview(name).then(setData).catch((e) => setError(e.message)); }, [name]);
  return (
    <article className="file-card">
      <header>
        <span className="file-name">{FILE_ICON[name.split(".").pop()] ?? "🗎"} {name}</span>
        <button type="button" className="download" onClick={() => downloadFile(name).catch((e) => setError(e.message))}>
          ⬇ Download
        </button>
      </header>
      {error && <p className="empty">Preview unavailable: {error}</p>}
      {!data && !error && <p className="empty">Loading preview…</p>}
      {data?.type === "pptx" && <SlidePreview slides={data.slides} />}
      {data?.type === "docx" && <DocPreview blocks={data.blocks} />}
      {data?.type === "xlsx" && <SheetPreview sheets={data.sheets} />}
      {data?.type === "text" && <pre className="text-preview">{data.text}</pre>}
    </article>
  );
}

// --- Phase 9: human approval. The run is PAUSED on the server until these decisions are sent. ---
const KIND_ICON = { email: "✉️", calendar: "📅", file: "🗑️" };

function ActionDetails({ action, edit, onEdit }) {
  const d = action.details;
  if (action.kind === "email") {
    if (edit) { // editing: the recipient stays fixed (it's what the agent proposed); only subject/body change
      return (
        <div className="paper email-preview email-edit">
          <p><b>To:</b> {d.to} <span className="locked">(can't be changed here)</span></p>
          <label><b>Subject</b>
            <input value={edit.subject} onChange={(e) => onEdit({ ...edit, subject: e.target.value })} />
          </label>
          <label><b>Body</b>
            <textarea rows={Math.min(14, edit.body.split("\n").length + 2)} value={edit.body}
                      onChange={(e) => onEdit({ ...edit, body: e.target.value })} />
          </label>
        </div>
      );
    }
    return (
      <div className="paper email-preview">
        <p><b>To:</b> {d.to}</p><p><b>Subject:</b> {d.subject}</p>
        <hr /><p className="email-body">{d.body}</p>
      </div>
    );
  }
  if (action.kind === "calendar") {
    return (
      <p className="action-meta">
        {d.action}{d.title ? ` · ${d.title}` : ""}{d.start ? ` · ${d.start} → ${d.end.slice(-5)}` : ""}
        {d.attendees?.length ? ` · with ${d.attendees.join(", ")}` : ""}{d.reason ? ` · reason: ${d.reason}` : ""}
      </p>
    );
  }
  return <p className="action-meta">This can't be undone (the slide is removed from the file).</p>;
}

function ApprovalPanel({ actions, onSubmit, busy }) {
  const [decisions, setDecisions] = useState({}); // id -> "approve" | "reject"; missing = reject (safe default)
  const [edits, setEdits] = useState({}); // id -> {subject, body} while the user edits an email
  const all = (value) => setDecisions(Object.fromEntries(actions.map((a) => [a.id, value])));
  const approved = actions.filter((a) => decisions[a.id] === "approve").length;

  function toggleEdit(a) {
    setEdits(({ [a.id]: open, ...rest }) => (open ? rest : { ...rest, [a.id]: { subject: a.details.subject, body: a.details.body } }));
  }

  function submit() {
    // An approved email with changed text is sent as {"decision": "approve", "edits": {...}}; the rest stay plain strings.
    onSubmit(Object.fromEntries(Object.entries(decisions).map(([id, decision]) => {
      const e = edits[id];
      const a = actions.find((x) => x.id === id);
      const changed = e && decision === "approve" &&
        Object.fromEntries(["subject", "body"].filter((k) => e[k] !== a.details[k]).map((k) => [k, e[k]]));
      return [id, changed && Object.keys(changed).length ? { decision, edits: changed } : decision];
    })));
  }
  return (
    <section className="panel approval" aria-live="assertive">
      <p className="eyebrow">⚠ YOUR APPROVAL IS NEEDED · nothing below has happened yet</p>
      {actions.map((a) => (
        <article key={a.id} className={`action ${decisions[a.id] ?? ""}`}>
          <header>
            <span className="action-title">{KIND_ICON[a.kind]} {a.summary}</span>
            <span className="choice" role="group" aria-label={`Decision for ${a.summary}`}>
              {a.kind === "email" && (
                <button type="button" className={edits[a.id] ? "on edit" : ""} aria-pressed={!!edits[a.id]}
                        onClick={() => toggleEdit(a)}>✎ {edits[a.id] ? "Editing" : "Edit"}</button>
              )}
              <button type="button" className={decisions[a.id] === "approve" ? "on approve" : ""} aria-pressed={decisions[a.id] === "approve"}
                      onClick={() => setDecisions({ ...decisions, [a.id]: "approve" })}>✓ Approve</button>
              <button type="button" className={decisions[a.id] === "reject" ? "on reject" : ""} aria-pressed={decisions[a.id] === "reject"}
                      onClick={() => setDecisions({ ...decisions, [a.id]: "reject" })}>✗ Reject</button>
            </span>
          </header>
          <ActionDetails action={a} edit={edits[a.id]} onEdit={(e) => setEdits({ ...edits, [a.id]: e })} />
        </article>
      ))}
      <div className="approval-row">
        <button type="button" className="ghost" onClick={() => all("approve")}>Approve all</button>
        <button type="button" className="ghost" onClick={() => all("reject")}>Reject all</button>
        <button type="button" className="launch" disabled={busy} onClick={submit}>
          Submit · {approved} approved, {actions.length - approved} rejected
        </button>
      </div>
    </section>
  );
}

// --- Phase 8: task history (threads saved by the checkpointer) and long-term profile ---
function HistoryPanel({ onOpen, currentId }) {
  const [threads, setThreads] = useState(null);
  useEffect(() => { getThreads().then(setThreads).catch(() => setThreads([])); }, []);
  if (!threads) return <p className="empty">Loading…</p>;
  if (!threads.length) return <p className="empty">No missions yet.</p>;
  return (
    <ul className="threads">
      {threads.map((t) => (
        <li key={t.id}>
          <button type="button" className={t.id === currentId ? "current" : ""} onClick={() => onOpen(t.id)}>
            <span>{t.title}</span><time>{t.updated.replace("T", " ").slice(0, 16)}</time>
          </button>
        </li>
      ))}
    </ul>
  );
}

const PROFILE_FIELDS = [
  ["name", "Name", "e.g. Alex Kumar"], ["email", "Email", "you@company.com"], ["role", "Role", "e.g. Product engineer"],
  ["sign_off", "Email sign-off", "Best regards,\nAlex"], ["preferences", "Preferences", "e.g. Keep answers short."],
];

function ProfilePanel() {
  const [profile, setProfile] = useState(null);
  const [status, setStatus] = useState("");
  useEffect(() => { getProfile().then(setProfile).catch(() => setProfile({})); }, []);
  if (!profile) return <p className="empty">Loading…</p>;
  async function save(e) {
    e.preventDefault();
    setStatus("Saving…");
    try { setProfile(await saveProfile(profile)); setStatus("Saved ✓ Agents will use this from the next request."); }
    catch (err) { setStatus(`Couldn't save: ${err.message}`); }
  }
  return (
    <form className="profile" onSubmit={save}>
      <p className="empty">Long-term memory: what every agent knows about you, in every mission.</p>
      {PROFILE_FIELDS.map(([key, label, hint]) => (
        <label key={key}>
          <span>{label}</span>
          {key === "sign_off" || key === "preferences"
            ? <textarea rows={2} value={profile[key] ?? ""} placeholder={hint} onChange={(e) => setProfile({ ...profile, [key]: e.target.value })} />
            : <input value={profile[key] ?? ""} placeholder={hint} onChange={(e) => setProfile({ ...profile, [key]: e.target.value })} />}
        </label>
      ))}
      <div className="profile-row"><button type="submit" className="launch">Save profile</button><span>{status}</span></div>
    </form>
  );
}

// --- User accounts: log in / create an account ---
function LoginScreen({ onLogin }) {
  const [mode, setMode] = useState("login"); // "login" | "register"
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true); setError("");
    try {
      if (mode === "register") await register(username.trim(), password);
      await login(username.trim(), password);
      onLogin(username.trim());
    } catch (err) {
      setError(err.message.includes("fetch") ? "Can't reach the API. Is it running on port 8000?" : err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page login-page">
      <form className="panel login" onSubmit={submit}>
        <p className="eyebrow">MULTI-AGENT ORCHESTRATOR</p>
        <h1>Mission Control</h1>
        <p className="empty">{mode === "login" ? "Log in to see your missions."
          : "Create your account. The first account on this server takes over the data that existed before accounts."}</p>
        <label>Username
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required
                 minLength={3} maxLength={32} autoFocus />
        </label>
        <label>Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8}
                 autoComplete={mode === "login" ? "current-password" : "new-password"} />
        </label>
        {error && <p className="login-error" role="alert">{error}</p>}
        <button type="submit" className="launch" disabled={busy}>{busy ? "…" : mode === "login" ? "Log in" : "Create account"}</button>
        <button type="button" className="ghost" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>
          {mode === "login" ? "New here? Create an account" : "Have an account? Log in"}
        </button>
      </form>
    </div>
  );
}

// The auth gate: checking -> login screen or Mission Control. key={user}: another user starts with a clean slate.
export default function App() {
  const [user, setUser] = useState(undefined); // undefined = still checking, null = logged out
  useEffect(() => { me().then((u) => setUser(u.username)).catch(() => setUser(null)); }, []);
  if (user === undefined) return <div className="page"><p className="empty">Loading…</p></div>;
  if (!user) return <LoginScreen onLogin={setUser} />;
  return <MissionControl key={user} username={user} onLoggedOut={() => setUser(null)} />;
}

function MissionControl({ username, onLoggedOut }) {
  const [descriptions, setDescriptions] = useState({});
  const [apiDown, setApiDown] = useState(false);
  const [message, setMessage] = useState(EXAMPLES[0]);
  const [phase, setPhase] = useState("idle"); // idle | running | done | error
  const [turn, setTurn] = useState(0);
  const [nodes, setNodes] = useState({}); // agent -> "working" | "done"
  const [log, setLog] = useState([]);
  const [files, setFiles] = useState([]); // workspace files this run created/changed (from the "files" event)
  const [threadId, setThreadId] = useState(null); // the conversation; follow-ups reuse it (short-term memory)
  const [convo, setConvo] = useState([]); // earlier turns of this conversation: [{q, a}]
  const [drawer, setDrawer] = useState(null); // "history" | "profile" | null
  const [pending, setPending] = useState([]); // actions the paused run is waiting on (Phase 9)
  const lastQuestion = useRef("");
  const abort = useRef(null);

  function resetRun() { setPhase("idle"); setTurn(0); setNodes({}); setLog([]); setFiles([]); setPending([]); }

  function newMission() { resetRun(); setThreadId(null); setConvo([]); setMessage(""); setDrawer(null); }

  async function openThread(id) {
    const { messages, pending_approval } = await getThread(id);
    const pairs = [];
    messages.forEach((m) => (m.role === "user" ? pairs.push({ q: m.content, a: "" }) : pairs.length && (pairs.at(-1).a = m.content)));
    resetRun(); setThreadId(id); setConvo(pairs); setMessage(""); setDrawer(null);
    if (pending_approval?.length) { setPending(pending_approval); setPhase("awaiting"); } // paused earlier: still waiting
  }

  // After an error or stop, agents that never reported back are no longer "working".
  const settle = () => setNodes((n) => Object.fromEntries(Object.entries(n).map(([k, v]) => [k, v === "working" ? "idle" : v])));

  const loadAgents = () =>
    getAgents()
      .then((list) => { setDescriptions(Object.fromEntries(list.map((a) => [a.name, a.description]))); setApiDown(false); })
      .catch((err) => (err instanceof LoggedOut ? onLoggedOut() : setApiDown(true)));

  useEffect(() => { loadAgents(); }, []);

  function onEvent(event, data) {
    if (event === "thread") {
      setThreadId(data.thread_id);
    } else if (event === "plan") {
      setTurn(data.turn);
      setNodes((n) => ({ ...n, ...Object.fromEntries(data.agents.map((a) => [a, "working"])) }));
      setLog((l) => [...l, { kind: "plan", turn: data.turn, agents: data.agents }]);
    } else if (event === "agent") {
      setNodes((n) => ({ ...n, [data.agent]: "done" }));
      setLog((l) => [...l, { kind: "agent", agent: data.agent, text: data.result }]);
    } else if (event === "final") {
      setLog((l) => [...l, { kind: "final", text: data.final_answer }]);
      setPhase("done");
    } else if (event === "approval") {
      setPending(data.actions);
      setPhase("awaiting");
      setLog((l) => [...l, { kind: "plan", turn: "⏸", agents: [], paused: true }]);
    } else if (event === "approved") {
      setLog((l) => [...l, ...data.outcomes.map((o) => ({ kind: "outcome", text: o }))]);
    } else if (event === "files") {
      setFiles((f) => [...new Set([...f, ...data.files])]);
    } else if (event === "error") {
      setLog((l) => [...l, { kind: "error", text: data.message }]);
      setPhase("error");
      settle();
    }
  }

  async function launch(e) {
    e.preventDefault();
    if (!message.trim() || phase === "running") return;
    abort.current = new AbortController();
    if (apiDown) loadAgents(); // the API may have started since the page loaded: re-check instead of staying "offline"
    const previous = log.find((x) => x.kind === "final");
    // Read the ref NOW: React runs state-updater functions later, after the ref below already holds the new question.
    const previousQuestion = lastQuestion.current;
    if (threadId && previous) setConvo((c) => [...c, { q: previousQuestion, a: previous.text }]); // move last turn up
    lastQuestion.current = message.trim();
    setPhase("running"); setTurn(0); setNodes({}); setLog([]); setFiles([]);
    try {
      await streamTask(message.trim(), threadId, onEvent, abort.current.signal);
      setPhase((p) => (p === "running" ? "done" : p));
      setMessage(""); // ready for a follow-up
    } catch (err) {
      if (err instanceof LoggedOut) return onLoggedOut(); // session expired: back to the login screen
      if (err.name === "AbortError") {
        setLog((l) => [...l, { kind: "error", text: "Stopped. (Agents already running on the server finish their current step.)" }]);
      } else {
        setLog((l) => [...l, { kind: "error", text: err.message.includes("fetch") ? "Can't reach the API. Is it running on port 8000?" : err.message }]);
      }
      setPhase("error");
      settle();
    }
  }

  async function decide(decisions) {
    abort.current = new AbortController();
    setPending([]);
    setPhase("running");
    try {
      await resumeTask(threadId, decisions, onEvent, abort.current.signal);
      setPhase((p) => (p === "running" ? "done" : p));
      setMessage("");
    } catch (err) {
      if (err instanceof LoggedOut) return onLoggedOut();
      setLog((l) => [...l, { kind: "error", text: err.message }]);
      setPhase("error");
    }
  }

  const final = log.find((x) => x.kind === "final");
  const used = Object.keys(nodes).length;

  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="eyebrow">MULTI-AGENT ORCHESTRATOR</p>
          <h1>Mission Control</h1>
        </div>
        <div className="top-actions">
          <button type="button" className={`ghost ${drawer === "history" ? "on" : ""}`} aria-expanded={drawer === "history"}
                  onClick={() => setDrawer(drawer === "history" ? null : "history")}>🕘 History</button>
          <button type="button" className={`ghost ${drawer === "profile" ? "on" : ""}`} aria-expanded={drawer === "profile"}
                  onClick={() => setDrawer(drawer === "profile" ? null : "profile")}>👤 {username}</button>
          <button type="button" className="ghost" onClick={async () => { await logout(); onLoggedOut(); }}>Log out</button>
          <span className={`pill ${apiDown ? "error" : phase}`} aria-live="polite">
            {apiDown ? "API offline" : { idle: "Standing by", running: "Mission in progress", awaiting: "Awaiting your approval",
              done: "Mission complete", error: "Mission aborted" }[phase]}
          </span>
        </div>
      </header>

      {drawer && (
        <section className="panel drawer">
          <p className="eyebrow">{drawer === "history" ? "PAST MISSIONS · click one to continue it" : "YOUR PROFILE"}</p>
          {drawer === "history" ? <HistoryPanel onOpen={openThread} currentId={threadId} /> : <ProfilePanel />}
        </section>
      )}

      {convo.length > 0 && (
        <section className="panel convo">
          <p className="eyebrow">EARLIER IN THIS MISSION</p>
          {convo.map((t, i) => (
            <details key={i} className="turn">
              <summary>🧑 {t.q}</summary>
              <div className="result"><Rich text={t.a || "(no answer saved)"} /></div>
            </details>
          ))}
        </section>
      )}

      {pending.length > 0 && <ApprovalPanel key={pending.map((a) => a.id).join()} actions={pending} onSubmit={decide} busy={phase === "running"} />}

      <form className="command" onSubmit={launch}>
        <div className="command-head">
          <label htmlFor="msg" className="eyebrow">{threadId ? "FOLLOW-UP (same mission, it remembers the conversation)" : "YOUR REQUEST"}</label>
          {threadId && phase !== "running" && <button type="button" className="ghost" onClick={newMission}>✦ New mission</button>}
        </div>
        <textarea id="msg" rows={3} value={message} maxLength={4000}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) launch(e); }}
                  placeholder={threadId ? "Ask a follow-up… e.g. “now add a slide about sick leave to that deck”" : "e.g. Create a presentation from survey_report.pdf"} />
        <div className="command-row">
          <div className="chips">
            {EXAMPLES.map((ex) => (
              <button type="button" key={ex} className="chip" onClick={() => setMessage(ex)} disabled={phase === "running"}>
                {ex.length > 46 ? ex.slice(0, 44) + "…" : ex}
              </button>
            ))}
          </div>
          {phase === "running"
            ? <button type="button" className="launch stop" onClick={() => abort.current?.abort()}>Stop</button>
            : <button type="submit" className="launch" disabled={!message.trim() || phase === "awaiting"}
                      title={phase === "awaiting" ? "Approve or reject the pending actions first" : undefined}>Launch ↗</button>}
        </div>
      </form>

      <main className="deck">
        <section className="panel">
          <Constellation nodes={nodes} phase={phase} turn={turn} descriptions={descriptions} />
          <p className="legend">
            <span className="dot working" /> working <span className="dot done" /> done <span className="dot idle" /> not needed
            {phase !== "idle" && <span className="count"> · {used} of {RING.length} agents used</span>}
          </p>
        </section>

        <section className="panel log" aria-live="polite">
          <p className="eyebrow">MISSION LOG</p>
          {log.length === 0 && <p className="empty">Launch a request to watch the orchestrator pick agents, pass results between them, and assemble the answer.</p>}
          <ol>
            {log.filter((x) => x.kind !== "final").map((item, i) => (
              <li key={i} className={`entry ${item.kind}`}>
                {item.kind === "plan" && (item.agents.length
                  ? <>🛰️ <b>Turn {item.turn}</b> · dispatch {item.agents.map((a) => `${META[a]?.icon} ${META[a]?.label ?? a}`).join("  +  ")}
                      {item.agents.length > 1 && <em> (in parallel)</em>}</>
                  : item.paused ? <>⏸ <b>Paused</b> · waiting for your approval (see the panel at the top)</>
                  : <>🧩 <b>Turn {item.turn}</b> · all results in, checking approvals and composing the answer</>)}
                {item.kind === "outcome" && <>🧑‍⚖️ {item.text}</>}
                {item.kind === "agent" && (
                  <details>
                    <summary>✓ {META[item.agent]?.icon} {META[item.agent]?.label ?? item.agent} reported back</summary>
                    <div className="result"><Rich text={item.text} /></div>
                  </details>
                )}
                {item.kind === "error" && <>⚠️ {item.text}</>}
              </li>
            ))}
            {phase === "running" && <li className="entry pending">⏳ working…</li>}
          </ol>
          {final && (
            <div className="final">
              <p className="eyebrow">FINAL ANSWER</p>
              <div className="result"><Rich text={final.text} /></div>
            </div>
          )}
        </section>
      </main>

      {files.length > 0 && (
        <section className="panel output">
          <p className="eyebrow">MISSION OUTPUT · {files.length} FILE{files.length > 1 ? "S" : ""} CREATED OR CHANGED</p>
          {files.map((f) => <FileCard key={f} name={f} />)}
        </section>
      )}
    </div>
  );
}
