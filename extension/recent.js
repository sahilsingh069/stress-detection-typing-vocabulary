/*
  The "recent level": the average of the last few estimates from the last 10 minutes.
  Shared by the badge and the popup, and always worked out from the time now, so an old
  estimate can never be shown as recent.
*/
(() => {
  "use strict";

  const WINDOW_MS = 10 * 60 * 1000;   // estimates older than this are ignored
  const COUNT = 5;                    // average of the last N bursts
  const LEVELS = ["Low", "Medium", "High"];

  function recentOf(history, now = Date.now()) {
    return history.filter((h) => now - h.t <= WINDOW_MS).slice(-COUNT);
  }

  function smooth(history, now = Date.now()) {
    const recent = recentOf(history, now);
    if (!recent.length) return null;
    const probs = Object.fromEntries(LEVELS.map((l) => [
      l, recent.reduce((s, h) => s + (h.probs[l] || 0), 0) / recent.length,
    ]));
    const label = LEVELS.reduce((a, b) => (probs[b] > probs[a] ? b : a));
    return { label, probs, n: recent.length };
  }

  /** When the level can next change with no new typing, or null if nothing is recent. */
  function nextExpiry(history, now = Date.now()) {
    const recent = recentOf(history, now);
    return recent.length ? Math.min(...recent.map((h) => h.t)) + WINDOW_MS : null;
  }

  globalThis.StressRecent = { WINDOW_MS, smooth, nextExpiry };
})();
