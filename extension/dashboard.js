"use strict";

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const ANY_UNVALIDATED = "Includes estimates from models not validated yet, so read this as a demonstration, not a measurement.";
const HEAT_STEPS = [[0, 20], [20, 40], [40, 60], [60, 80], [80, 101]];
const NICE_MINUTES = [1, 2, 5, 10, 15, 30, 60, 120, 180, 240, 360];
let history = [];
let period = null;
let activeRange = "today";
let activeSite = null;

function svg(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVG, tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  if (parent) parent.appendChild(e);
  return e;
}

function html(tag, cls, text, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

function pct(n, total) {
  return total ? `${Math.round((n / total) * 100)}%` : "0%";
}

const toScore = (v) => (v === null || v === undefined ? "—" : String(Math.round(v)));
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function toInput(t) {
  const d = new Date(t);
  const m = String(d.getMonth() + 1).padStart(2, "0");
  return `${d.getFullYear()}-${m}-${String(d.getDate()).padStart(2, "0")}`;
}

function fromInput(value) {
  const [y, m, d] = value.split("-").map(Number);
  return new Date(y, m - 1, d).getTime();
}

function showTip(text, x, y) {
  const tip = $("tooltip");
  tip.replaceChildren();
  const [lead, ...rest] = text.split("\n");
  html("strong", "", lead, tip);
  rest.forEach((line) => html("div", "", line, tip));
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.max(8, Math.min(x + 12, innerWidth - r.width - 8))}px`;
  tip.style.top = `${Math.max(8, y - r.height - 12)}px`;
}

function hideTip() { $("tooltip").hidden = true; }

function withTip(el, text) {
  el.addEventListener("mousemove", (e) => showTip(text, e.clientX, e.clientY));
  el.addEventListener("mouseleave", hideTip);
  el.addEventListener("focus", () => {
    const r = el.getBoundingClientRect();
    showTip(text, r.left + r.width / 2, r.top);
  });
  el.addEventListener("blur", hideTip);
}

function levelsLine(b) {
  return `High ${pct(b.High, b.n)}, Medium ${pct(b.Medium, b.n)}, Low ${pct(b.Low, b.n)}`;
}

function binTip(score, title, b) {
  return `${toScore(score)} average score\n${title}\n${plural(b.n, "estimate")}: ${levelsLine(b)}`
    + (b.n < MIN_RELIABLE ? "\nToo few to rely on" : "");
}

/* --------------------------------------------------------------- time chart */

/* Round clock times across the visible span. */
function timeTicks(from, to, width) {
  const want = Math.max(2, Math.floor(width / 90));
  const minutes = NICE_MINUTES.find((m) => (to - from) / (m * 60000) <= want) || 360;
  const step = minutes * 60000;
  const first = startOfDay(from) + Math.ceil((from - startOfDay(from)) / step) * step;
  const ticks = [];
  for (let t = first; t <= to; t += step) ticks.push(t);
  return ticks;
}

/* A day's estimates on a clock axis zoomed to when they happened. */
function dayPoints(entries) {
  const pts = withRolling(entries);
  const first = pts[0].t;
  const last = pts[pts.length - 1].t;
  const span = Math.max(last - first, 20 * 60000);
  const middle = (first + last) / 2;
  const from = Math.max(middle - span * 0.55, startOfDay(first));
  const to = Math.min(middle + span * 0.55, addDays(startOfDay(first), 1));
  return { pts, from, to };
}

function drawTimeChart(host, cfg) {
  host.replaceChildren();
  const width = Math.max(host.clientWidth, 280);
  const height = 260;
  const m = { top: 12, right: cfg.endLabel ? 48 : 12, bottom: 26, left: 34 };
  const plotW = width - m.left - m.right;
  const plotH = height - m.top - m.bottom;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img", "aria-label": cfg.aria }, host);
  const X = (v) => m.left + (cfg.to > cfg.from ? ((v - cfg.from) / (cfg.to - cfg.from)) * plotW : plotW / 2);
  const Y = (v) => m.top + plotH - (v / 100) * plotH;

  [0, 25, 50, 75, 100].forEach((v) => {
    svg("line", { x1: m.left, x2: width - m.right, y1: Y(v), y2: Y(v), class: v ? "grid" : "axis" }, root);
    svg("text", { x: m.left - 6, y: Y(v) + 4, "text-anchor": "end" }, root).textContent = String(v);
  });
  cfg.ticks(plotW).forEach(([v, label]) => {
    svg("text", { x: X(v), y: height - 7, "text-anchor": "middle" }, root).textContent = label;
  });

  cfg.lines.forEach((run) => {
    if (run.length < 2) return;
    svg("polyline", { points: run.map(([x, y]) => `${X(x)},${Y(y)}`).join(" "), class: "trend" }, root);
  });
  const cross = svg("line", { y1: m.top, y2: m.top + plotH, class: "cross", visibility: "hidden" }, root);
  cfg.points.forEach((p) => {
    svg("circle", { cx: X(p.x), cy: Y(p.y), r: 5, class: `dot ${p.cls}${p.thin ? " thin" : ""}` }, root);
  });
  if (cfg.endLabel) {
    const y = Math.min(Math.max(Y(cfg.endLabel.y) + 4, m.top + 10), m.top + plotH - 2);
    svg("text", { x: X(cfg.endLabel.x) + 10, y, class: "endlabel" }, root).textContent = cfg.endLabel.text;
  }

  // The pointer only has to be nearest; arrow keys walk the same points.
  const overlay = svg("rect", { x: m.left, y: 0, width: plotW, height: m.top + plotH, class: "hit", tabindex: 0,
    "aria-label": `${cfg.aria} Use the left and right arrow keys to read each point.` }, root);
  let index = cfg.points.length - 1;
  const show = (i, cx, cy) => {
    index = i;
    const p = cfg.points[i];
    cross.setAttribute("x1", X(p.x));
    cross.setAttribute("x2", X(p.x));
    cross.setAttribute("visibility", "visible");
    showTip(p.tip, cx, cy);
  };
  const nearest = (px) => cfg.points.reduce((best, p, i) =>
    (Math.abs(X(p.x) - px) < Math.abs(X(cfg.points[best].x) - px) ? i : best), 0);
  const hide = () => { cross.setAttribute("visibility", "hidden"); hideTip(); };
  overlay.addEventListener("mousemove", (e) => {
    const box = root.getBoundingClientRect();
    show(nearest(((e.clientX - box.left) / box.width) * width), e.clientX, e.clientY);
  });
  overlay.addEventListener("mouseleave", hide);
  overlay.addEventListener("blur", hide);
  const fromKeys = (i) => {
    const box = root.getBoundingClientRect();
    const p = cfg.points[i];
    show(i, box.left + (X(p.x) / width) * box.width, box.top + (Y(p.y) / height) * box.height);
  };
  overlay.addEventListener("focus", () => fromKeys(index));
  overlay.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    fromKeys(Math.max(0, Math.min(cfg.points.length - 1, index + (e.key === "ArrowRight" ? 1 : -1))));
  });
}

function drawDay(host, entries) {
  const { pts, from, to } = dayPoints(entries);
  const last = pts[pts.length - 1];
  drawTimeChart(host, {
    from,
    to,
    aria: "Stress score of each estimate through the day, with a rolling average. The table view has the same numbers by hour.",
    ticks: (w) => timeTicks(from, to, w).map((t) => [t, clock(t)]),
    lines: pts.length >= 2 ? [pts.map((p) => [p.t, p.rolling])] : [],
    points: pts.map((p) => ({
      x: p.t,
      y: p.score,
      cls: `seg-${p.label}`,
      tip: `${toScore(p.score)} stress score\n${p.label}, ${clock(p.t)}\n${p.site}\nAverage of last ${ROLLING}: ${toScore(p.rolling)}`,
    })),
    endLabel: pts.length >= 2 ? { x: last.t, y: last.rolling, text: `avg ${toScore(last.rolling)}` } : null,
  });
}

function drawDays(host, bins) {
  const step = (w) => Math.max(1, Math.ceil(bins.length / Math.max(Math.floor(w / 62), 1)));
  const runs = [];
  bins.forEach((b, i) => {
    if (!b.n) return;
    const previous = i > 0 && bins[i - 1].n;
    if (!previous) runs.push([]);
    runs[runs.length - 1].push([i, b.avg]);
  });
  drawTimeChart(host, {
    from: -0.5,
    to: bins.length - 0.5,
    aria: "Average stress score per day. Gaps are days with no estimates. The table view has the same numbers.",
    ticks: (w) => bins.map((b, i) => [i, b.label]).filter((_, i) => i % step(w) === 0),
    lines: runs,
    points: bins.map((b, i) => [b, i]).filter(([b]) => b.n).map(([b, i]) => ({
      x: i,
      y: b.avg,
      cls: "seg-score",
      thin: b.n < MIN_RELIABLE,
      tip: binTip(b.avg, b.detail, b),
    })),
    endLabel: null,
  });
}

/* --------------------------------------------------------------- other marks */

function renderMix(s) {
  const bar = $("mix");
  bar.replaceChildren();
  const legend = $("mixLegend");
  legend.replaceChildren();
  ["Low", "Medium", "High"].forEach((l) => {
    if (s[l]) {
      const seg = html("div", `seg bg-${l}`, undefined, bar);
      seg.style.flexGrow = s[l];
      seg.tabIndex = 0;
      withTip(seg, `${pct(s[l], s.n)} ${l}\n${plural(s[l], "estimate")}`);
    }
    const item = html("span", "", undefined, legend);
    html("i", `bg-${l}`, undefined, item);
    item.append(`${l} ${pct(s[l], s.n)}`);
    html("em", "", `(${s[l]})`, item);
  });
}

function renderSites(rows) {
  const host = $("sites");
  host.replaceChildren();
  const head = html("div", "siterow sitehead", undefined, host);
  ["Site", "", "Score", "Est."].forEach((t, i) => html("span", i > 1 ? "value" : "name", t, head));
  rows.forEach((r) => {
    const row = html(r.isOther ? "div" : "button", `siterow${r.n < MIN_RELIABLE ? " thin" : ""}`, undefined, host);
    if (!r.isOther) {
      row.type = "button";
      row.addEventListener("click", () => { hideTip(); activeSite = r.site; render(); });
    }
    html("span", "name", r.site, row).title = r.site;
    const track = html("span", "track", undefined, row);
    html("span", "fill", undefined, track).style.width = `${Math.max(r.avg, 1)}%`;
    html("span", "value", toScore(r.avg), row);
    html("span", "count", `${r.n}`, row);
    withTip(row, binTip(r.avg, r.site, r));
  });
}

function heatStep(avg) {
  return HEAT_STEPS.findIndex(([a, b]) => avg >= a && avg < b);
}

function renderHeat(grid) {
  const table = $("heat");
  table.replaceChildren();
  const head = table.insertRow();
  html("th", "", "", head);
  grid[0].forEach((_, c) => {
    html("th", "", String(c * BLOCK_HOURS).padStart(2, "0"), head).title = blockLabel(c);
  });
  grid.forEach((row, r) => {
    const tr = table.insertRow();
    html("th", "", WEEKDAYS[r], tr);
    row.forEach((cell, c) => {
      const td = tr.insertCell();
      if (!cell.n) return;
      td.className = cell.n < MIN_RELIABLE ? "few" : `step-${heatStep(cell.avg)}`;
      td.tabIndex = 0;
      withTip(td, binTip(cell.avg, `${WEEKDAYS[r]}, ${blockLabel(c)}`, cell));
    });
  });
  const legend = $("heatLegend");
  legend.replaceChildren();
  html("span", "", "Score", legend);
  HEAT_STEPS.forEach(([a, b], i) => {
    const item = html("span", "", undefined, legend);
    html("i", `step-${i}`, undefined, item);
    item.append(`${a}–${Math.min(b, 100)}`);
  });
  const few = html("span", "", undefined, legend);
  html("i", "fewkey", undefined, few);
  few.append(`Fewer than ${MIN_RELIABLE} estimates`);
}

function tile(key, value, detail) {
  const t = html("div", "tile");
  html("div", "k", key, t);
  html("div", "v", value, t);
  html("div", "d", detail, t);
  return t;
}

function fillTable(table, firstColumn, rows) {
  table.replaceChildren();
  const head = table.insertRow();
  [firstColumn, "Estimates", "Average score", "High", "Medium", "Low"].forEach((h) => html("th", "", h, head));
  rows.forEach(([name, b]) => {
    const r = table.insertRow();
    [name, b.n, toScore(b.avg), `${pct(b.High, b.n)} (${b.High})`, `${pct(b.Medium, b.n)} (${b.Medium})`,
      `${pct(b.Low, b.n)} (${b.Low})`].forEach((v) => { r.insertCell().textContent = v; });
  });
}

function previousName(days) {
  if (activeRange === "today") return "yesterday";
  if (activeRange === "yesterday") return "the day before";
  return days === 1 ? "the previous day" : `the previous ${days} days`;
}

function renderLegend(single, count) {
  const host = $("legend");
  host.replaceChildren();
  host.hidden = !single;
  if (!single) return;
  LEVELS_DESC.forEach((l) => {
    const s = html("span", "", undefined, host);
    html("b", `round bg-${l}`, undefined, s);
    s.append(`${l} estimate`);
  });
  if (count < 2) return;
  const s = html("span", "", undefined, host);
  html("u", "linekey", undefined, s);
  s.append(`Average of last ${ROLLING}`);
}

/* --------------------------------------------------------------- page */

function render() {
  const { start, end } = period;
  $("from").value = toInput(start);
  $("to").value = toInput(end - 1);
  document.querySelectorAll("[data-range]").forEach((b) => {
    const on = b.dataset.range === activeRange;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on);
  });
  const chip = $("siteChip");
  chip.hidden = !activeSite;
  chip.textContent = activeSite ? `${activeSite}  ✕` : "";
  chip.title = "Show all sites";

  const days = dayCount(start, end);
  const entries = inRange(history, start, end, activeSite);
  const earlier = inRange(history, addDays(start, -days), start, activeSite);
  const s = stats(entries);

  $("caveat").hidden = !entries.some((h) => !h.validated);
  $("caveat").textContent = ANY_UNVALIDATED;
  $("content").hidden = !s.n;
  $("empty").hidden = !!s.n;
  $("empty").textContent = activeSite
    ? `No estimates from ${activeSite} in this period.`
    : "No estimates in this period. Type in any text box with the server running.";
  if (!s.n) { hideTip(); return; }

  // overall
  $("score").textContent = toScore(s.avg);
  $("verdict").textContent = verdict(s);
  const before = stats(earlier);
  const diff = Math.round(s.avg) - Math.round(before.avg);
  $("delta").textContent = before.n >= MIN_RELIABLE && s.n >= MIN_RELIABLE
    ? `${diff === 0 ? "Same as" : `${diff > 0 ? "▲" : "▼"} ${Math.abs(diff)} ${diff > 0 ? "above" : "below"}`} ${previousName(days)}`
    : `No comparison: fewer than ${MIN_RELIABLE} estimates ${before.n < MIN_RELIABLE ? `from ${previousName(days)}` : "in this period"}`;
  renderMix(s);

  // sentences
  const notes = insights(entries, earlier, previousName(days), days);
  const list = $("insights");
  list.replaceChildren();
  notes.forEach((text) => html("li", "", text, list));
  if (!notes.length) {
    html("li", "", s.n < MIN_RELIABLE
      ? `Only ${plural(s.n, "estimate")} so far. Patterns appear here from ${MIN_RELIABLE} on.`
      : `No difference of ${MIN_DIFF}+ points stands out yet.`, list);
  }
  $("insightNote").textContent = `Called out only for ${MIN_DIFF}+ points and ${MIN_RELIABLE}+ estimates. A pattern here may be chance, and it does not show what caused it.`;

  // tiles
  const peak = entries.reduce((a, b) => (scoreOf(b) > scoreOf(a) ? b : a));
  const sites = siteStats(entries);
  const realSites = sites.filter((r) => !r.isOther);
  const reliable = realSites.filter((r) => r.n >= MIN_RELIABLE);
  const whenPeak = days <= 1 ? clock(peak.t)
    : new Date(peak.t).toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
  $("tiles").replaceChildren(
    tile("Estimates", String(s.n), days <= 1 ? (s.n > 1 ? `${clock(entries[0].t)} to ${clock(entries[s.n - 1].t)}` : clock(entries[0].t)) : `on ${new Set(entries.map((h) => startOfDay(h.t))).size} of ${days} days`),
    tile("Highest reading", toScore(scoreOf(peak)), `${whenPeak}, ${peak.site}`),
    realSites.length < 2
      ? tile("Site", realSites[0] ? realSites[0].site : "—", activeSite ? "filtered" : "the only site in this period")
      : tile("Highest site", reliable.length ? reliable[0].site : "—",
        reliable.length ? `average ${toScore(reliable[0].avg)}, ${plural(reliable[0].n, "estimate")}` : `needs ${MIN_RELIABLE}+ estimates on a site`),
  );

  // over time
  const { bins, unit } = timelineBins(entries, start, end);
  const single = days <= 1;
  $("timelineTitle").textContent = single ? "Through the day" : "Day by day";
  $("timelineHint").textContent = single
    ? "Each dot is one estimate. The axis covers only the time you were typing."
    : `Average score per day. Faded: fewer than ${MIN_RELIABLE} estimates. Gaps: no estimates.`;
  renderLegend(single, s.n);
  if (single) drawDay($("timeline"), entries);
  else drawDays($("timeline"), bins);

  // where and when
  const showSites = sites.length >= 2;
  const showHeat = !single;
  $("sitePanel").hidden = !showSites;
  $("heatPanel").hidden = !showHeat;
  $("pair").hidden = !showSites && !showHeat;
  $("pair").classList.toggle("solo", showSites !== showHeat);
  if (showSites) renderSites(sites);
  const grid = heatGrid(entries);
  if (showHeat) renderHeat(grid);

  // tables
  fillTable($("table"), unit === "hour" ? "Hour" : "Day", bins.filter((b) => b.n).map((b) => [b.detail, b]));
  $("siteTableWrap").hidden = !showSites;
  if (showSites) fillTable($("siteTable"), "Site", sites.map((r) => [r.site, r]));
  $("heatTableWrap").hidden = !showHeat;
  if (showHeat) {
    const cells = [];
    grid.forEach((row, r) => row.forEach((cell, c) => {
      if (cell.n) cells.push([`${WEEKDAYS[r]}, ${blockLabel(c)}`, cell]);
    }));
    fillTable($("heatTable"), "Day and time", cells);
  }
}

function setRange(range) {
  activeRange = range;
  period = periodFor(range);
  render();
}

function setCustom() {
  const { value: a } = $("from");
  const { value: b } = $("to");
  if (!a || !b) return;
  let start = fromInput(a);
  let last = fromInput(b);
  if (last < start) [start, last] = [last, start];
  activeRange = "custom";
  period = { start, end: addDays(last, 1) };
  render();
}

async function load() {
  const s = await chrome.storage.local.get({ history: [] });
  history = s.history;
  render();
}

document.querySelectorAll("[data-range]").forEach((b) =>
  b.addEventListener("click", () => setRange(b.dataset.range)));
$("siteChip").addEventListener("click", () => { activeSite = null; render(); });
$("from").addEventListener("change", setCustom);
$("to").addEventListener("change", setCustom);
$("clear").addEventListener("click", async () => {
  if (!confirm("Delete all stored estimates? This cannot be undone.")) return;
  await chrome.storage.local.set({ history: [], lastResult: null });
  chrome.runtime.sendMessage({ type: "refreshBadge" });
});
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.history) {
    history = changes.history.newValue || [];
    render();
  }
});
let resizeTimer;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(render, 120);
});

period = periodFor("today");
load();
