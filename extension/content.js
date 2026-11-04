/*
  Content script: watches typing in ordinary text fields on the page and,
  after a pause, hands one "burst" of typing to the background worker.

  Privacy rules enforced here, before anything leaves the page:
    - password, credit-card and one-time-code fields are never observed
    - only the text typed during the burst is sent (rebuilt from the key
      presses), not what was already in the field
    - nothing is collected while the extension is off or paused for this site
*/
(() => {
  "use strict";

  const MIN_PRESSES = 15;   // fewer keys give unstable timing statistics
  const PAUSE_MS = 2000;    // a pause this long ends a burst
  const MAX_PRESSES = 400;  // very long bursts are split

  const host = location.hostname || "local file";
  let enabled = true;
  let paused = false;
  let burst = null;
  let timer = null;

  try {
    chrome.storage.local.get({ enabled: true, pausedSites: [] }, (s) => {
      enabled = s.enabled;
      paused = s.pausedSites.includes(host);
    });
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area !== "local") return;
      if (changes.enabled) enabled = changes.enabled.newValue;
      if (changes.pausedSites) paused = (changes.pausedSites.newValue || []).includes(host);
      if (!enabled || paused) reset();
    });
  } catch (_) {
    return; // extension context gone (e.g. reloaded): do nothing
  }

  const TEXT_INPUT_TYPES = new Set(["", "text", "search"]);
  const SENSITIVE_AUTOCOMPLETE = /(password|cc-|one-time-code)/i;

  function isObservable(el) {
    if (!el || el.nodeType !== 1) return false;
    const ac = el.getAttribute && (el.getAttribute("autocomplete") || "");
    if (SENSITIVE_AUTOCOMPLETE.test(ac)) return false;
    if (el.tagName === "TEXTAREA") return !el.readOnly && !el.disabled;
    if (el.tagName === "INPUT") {
      const type = (el.getAttribute("type") || "").toLowerCase();
      return TEXT_INPUT_TYPES.has(type) && !el.readOnly && !el.disabled;
    }
    return el.isContentEditable === true;
  }

  function targetOf(e) {
    const path = e.composedPath ? e.composedPath() : [];
    return path.length ? path[0] : e.target;
  }

  function reset() {
    clearTimeout(timer);
    burst = null;
  }

  function flush() {
    clearTimeout(timer);
    if (!burst) return;
    const b = burst;
    burst = null;
    if (b.presses < MIN_PRESSES) return;
    try {
      chrome.runtime.sendMessage({
        type: "burst",
        site: host,
        text: b.chars.join(""),
        events: b.events,
        presses: b.presses,
      });
    } catch (_) { /* extension reloaded; drop this burst */ }
  }

  document.addEventListener("keydown", (e) => {
    if (!enabled || paused || !e.isTrusted) return;
    const el = targetOf(e);
    if (!isObservable(el)) return;
    if (!burst || burst.el !== el) {
      flush();
      burst = { el, t0: performance.now(), events: [], chars: [], presses: 0 };
    }
    const t = performance.now() - burst.t0;
    burst.events.push({ ev: "down", key: e.key, code: e.code, t: +t.toFixed(2) });
    burst.presses++;

    // rebuild only what is typed now (approximate: ignores cursor moves)
    if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) burst.chars.push(e.key);
    else if (e.key === "Backspace") burst.chars.pop();
    else if (e.key === "Enter") burst.chars.push("\n");

    clearTimeout(timer);
    if (burst.presses >= MAX_PRESSES) flush();
    else timer = setTimeout(flush, PAUSE_MS);
  }, true);

  document.addEventListener("keyup", (e) => {
    if (!burst || !e.isTrusted || targetOf(e) !== burst.el) return;
    const t = performance.now() - burst.t0;
    burst.events.push({ ev: "up", key: e.key, code: e.code, t: +t.toFixed(2) });
  }, true);

  window.addEventListener("pagehide", flush);
  document.addEventListener("visibilitychange", () => { if (document.hidden) flush(); });
})();
