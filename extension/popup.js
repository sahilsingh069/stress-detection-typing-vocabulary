"use strict";

const SERVER = "http://127.0.0.1:8765";
const LEVELS = ["Low", "Medium", "High"];
const SOURCES = { own: "your data", dreaddit: "Dreaddit", swell: "SWELL-KW, unvalidated", kaggle: "Kaggle, unvalidated", synthetic: "synthetic" };
const $ = (id) => document.getElementById(id);
let currentHost = null;

function ago(t) {
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  return `${Math.round(s / 3600)} h ago`;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function render(s) {
  $("enabled").checked = s.enabled;
  $("nudges").checked = s.nudges;
  $("pauseSite").checked = !!currentHost && s.pausedSites.includes(currentHost);
  $("pauseSite").disabled = !currentHost;

  const r = s.lastResult;
  const sm = r && r.smoothed;
  const level = $("level");
  level.className = "level" + (sm ? " " + sm.label : "");
  level.textContent = !s.enabled ? "Off" : sm ? sm.label : "No estimate yet";
  $("window").textContent = sm ? `(average of last ${sm.n})` : "";

  const bars = $("bars");
  bars.replaceChildren();
  if (sm) {
    LEVELS.forEach((l) => {
      const row = el("div", "bar");
      const tr = el("div", "track");
      const f = el("div", "fill " + l);
      f.style.width = `${Math.round(sm.probs[l] * 100)}%`;
      tr.appendChild(f);
      row.append(el("span", "", l), tr, el("span", "pct", `${Math.round(sm.probs[l] * 100)}%`));
      bars.appendChild(row);
    });
  } else if (s.enabled) {
    bars.appendChild(el("div", "muted small",
      "Type a few sentences in any text box (at least 15 keys), then pause for 2 seconds."));
  }
  $("method").textContent = r
    ? `${r.method}${r.validated ? "" : " · models not validated yet"}` : "";

  const mini = (id, part) => {
    const e = $(id);
    e.className = "mini" + (part ? " " + part.label : "");
    e.textContent = part ? part.label : "—";
    e.title = part ? `model: ${SOURCES[part.source] || part.source}` : "needs more input";
  };
  mini("typing", r && r.typingModel);
  mini("text", r && r.textModel);
  $("when").textContent = r ? ago(r.t) : "—";

  const strip = $("strip");
  strip.replaceChildren();
  const hist = s.history.slice(-30);
  if (!hist.length) strip.appendChild(el("span", "empty", "nothing yet"));
  hist.forEach((h) => {
    const sq = el("span", h.label);
    sq.title = `${h.label} · ${h.site} · ${ago(h.t)}`;
    strip.appendChild(sq);
  });
}

async function load() {
  render(await chrome.storage.local.get({
    enabled: true, pausedSites: [], nudges: true, history: [], lastResult: null,
  }));
}

async function checkServer() {
  const out = $("server");
  try {
    const st = await (await fetch(`${SERVER}/api/status`)).json();
    if (!st.ready) {
      out.textContent = "server running, models missing: run train_demo.py";
      out.className = "sub bad";
      return;
    }
    const m = st.meta.models;
    out.textContent = `connected · typing: ${SOURCES[m.typing.source] || m.typing.source}`;
    out.className = "sub";
  } catch (_) {
    out.textContent = "server not running: python app/server.py";
    out.className = "sub bad";
  }
}

async function init() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab && tab.url ? new URL(tab.url) : null;
    if (url && /^https?:$/.test(url.protocol)) currentHost = url.hostname;
  } catch (_) { currentHost = null; }
  $("site").textContent = currentHost || "this page (not a website)";

  $("enabled").addEventListener("change", (e) => chrome.storage.local.set({ enabled: e.target.checked }));
  $("nudges").addEventListener("change", (e) => chrome.storage.local.set({ nudges: e.target.checked }));
  $("pauseSite").addEventListener("change", async (e) => {
    const { pausedSites = [] } = await chrome.storage.local.get("pausedSites");
    const next = e.target.checked
      ? [...new Set([...pausedSites, currentHost])]
      : pausedSites.filter((h) => h !== currentHost);
    await chrome.storage.local.set({ pausedSites: next });
  });
  $("clear").addEventListener("click", async () => {
    await chrome.storage.local.set({ history: [], lastResult: null });
    chrome.runtime.sendMessage({ type: "refreshBadge" });
  });
  $("demo").addEventListener("click", () => chrome.tabs.create({ url: SERVER }));
  $("dashboard").addEventListener("click", () =>
    chrome.tabs.create({ url: chrome.runtime.getURL("dashboard.html") }));

  chrome.storage.onChanged.addListener((_, area) => { if (area === "local") load(); });
  await load();
  checkServer();
}

init();
