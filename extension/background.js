/*
  Background service worker.

  Receives typing bursts from content scripts, estimates stress here in the
  browser (engine.js), keeps a short history of ESTIMATES ONLY (never the text),
  shows a smoothed level on the toolbar badge and, if enabled, suggests a break
  when stress stays high. This extension makes no network requests.
*/
"use strict";

importScripts("model.js", "engine.js");

const HISTORY_DAYS = 30;                    // the dashboard can look back this far
const HISTORY_MAX = 5000;                   // safety cap on stored estimates
const SMOOTH_WINDOW_MS = 10 * 60 * 1000;   // estimates older than this are ignored
const SMOOTH_COUNT = 5;                     // average of the last N bursts
const NUDGE_STREAK = 3;                     // consecutive High bursts before a suggestion
const NUDGE_COOLDOWN_MS = 30 * 60 * 1000;

const LEVELS = ["Low", "Medium", "High"];
const BADGE = {
  Low: { text: "LOW", color: "#0F766E" },
  Medium: { text: "MED", color: "#B7791F" },
  High: { text: "HIGH", color: "#B3123F" },
};

const DEFAULTS = {
  enabled: true,
  pausedSites: [],
  nudges: true,
  history: [],
  lastResult: null,
  lastNudge: 0,
};

chrome.runtime.onInstalled.addListener(async () => {
  const current = await chrome.storage.local.get(null);
  const missing = Object.fromEntries(
    Object.entries(DEFAULTS).filter(([k]) => !(k in current)));
  await chrome.storage.local.set(missing);
  await refreshBadge();
});
chrome.runtime.onStartup.addListener(refreshBadge);

chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === "burst") handleBurst(msg);
  if (msg && msg.type === "refreshBadge") refreshBadge();
  return false;
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.enabled) refreshBadge();
});

/* ------------------------------------------------------------------ */

function pruneHistory(history, now = Date.now()) {
  const oldest = now - HISTORY_DAYS * 24 * 60 * 60 * 1000;
  return history.filter((h) => h.t >= oldest).slice(-HISTORY_MAX);
}

function smooth(history, now = Date.now()) {
  const recent = history.filter((h) => now - h.t <= SMOOTH_WINDOW_MS).slice(-SMOOTH_COUNT);
  if (!recent.length) return null;
  const probs = Object.fromEntries(LEVELS.map((l) => [
    l, recent.reduce((s, h) => s + (h.probs[l] || 0), 0) / recent.length,
  ]));
  const label = LEVELS.reduce((a, b) => (probs[b] > probs[a] ? b : a));
  return { label, probs, n: recent.length };
}

async function setBadge(text, color) {
  await chrome.action.setBadgeText({ text });
  if (color) await chrome.action.setBadgeBackgroundColor({ color });
}

async function refreshBadge() {
  const s = await chrome.storage.local.get(DEFAULTS);
  if (!s.enabled) return setBadge("OFF", "#8B9AA8");
  const sm = smooth(s.history);
  if (!sm) return setBadge("", null);
  return setBadge(BADGE[sm.label].text, BADGE[sm.label].color);
}

async function handleBurst(msg) {
  const s = await chrome.storage.local.get(DEFAULTS);
  if (!s.enabled || s.pausedSites.includes(msg.site)) return;

  const data = StressEngine.predict(msg.text, msg.events);
  const est = data.fused;
  // Typing alone, and unsure: too weak to store, show or count towards the break nudge.
  if (!est || (!data.text && data.typing && data.typing.unclear)) return refreshBadge();

  const entry = {
    t: Date.now(),
    site: msg.site,
    label: est.label,
    probs: est.probs,
    method: data.typing && data.text ? "late fusion" : "single modality",
    validated: !!est.validated,
    typingModel: data.typing ? { label: data.typing.label, probs: data.typing.probs, source: data.typing.source, unclear: data.typing.unclear } : null,
    textModel: data.text ? { label: data.text.label, probs: data.text.probs, source: data.text.source } : null,
    keys: msg.presses,
  };
  // A month of history is kept compact: per-model detail stays in lastResult only.
  const { typingModel, textModel, ...compact } = entry;
  const history = pruneHistory([...s.history, compact]);
  const smoothed = smooth(history);
  await chrome.storage.local.set({
    history,
    lastResult: { ...entry, smoothed },
  });
  await refreshBadge();
  await maybeNudge(history, s);
}

async function maybeNudge(history, s) {
  if (!s.nudges) return;
  const now = Date.now();
  if (now - s.lastNudge < NUDGE_COOLDOWN_MS) return;
  const recent = history.filter((h) => now - h.t <= SMOOTH_WINDOW_MS).slice(-NUDGE_STREAK);
  if (recent.length < NUDGE_STREAK || !recent.every((h) => h.label === "High")) return;
  await chrome.storage.local.set({ lastNudge: now });
  chrome.notifications.create(`nudge-${now}`, {
    type: "basic",
    iconUrl: "icons/icon128.png",
    title: "Time for a short break?",
    message: "Your recent typing suggests elevated stress. A few minutes away from the screen might help.",
    contextMessage: "Estimate from a student research prototype",
    priority: 0,
  });
}
