/*
  Turns stored estimates into what the dashboard draws and says. No DOM here.
  All day and hour boundaries use the viewer's local time.
*/
"use strict";

const LEVELS_DESC = ["High", "Medium", "Low"];
const MIN_RELIABLE = 3;      // fewer estimates than this: drawn faded, never called out
const MIN_DIFF = 10;         // score points; smaller differences are not called out
const ROLLING = 5;           // matches the toolbar badge, which averages the last 5
const MAX_SITES = 8;

const DAY_MS = 24 * 60 * 60 * 1000;
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const BLOCK_HOURS = 3;
const PARTS = [["Nights", 0], ["Mornings", 6], ["Afternoons", 12], ["Evenings", 18]];

function startOfDay(t) {
  const d = new Date(t);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

function addDays(t, n) {            // calendar days, so clock changes cannot shift a day
  const d = new Date(t);
  d.setDate(d.getDate() + n);
  return d.getTime();
}

/* A period as [start, end) in ms. */
function periodFor(range, now = Date.now()) {
  const today = startOfDay(now);
  if (range === "today") return { start: today, end: addDays(today, 1) };
  if (range === "yesterday") return { start: addDays(today, -1), end: today };
  return { start: addDays(today, 1 - Number(range)), end: addDays(today, 1) };
}

function dayCount(start, end) {
  return Math.round((end - start) / DAY_MS);
}

/* 0 to 100: the chance of High plus half the chance of Medium. */
function scoreOf(h) {
  if (h.probs && typeof h.probs.High === "number") {
    return 100 * (h.probs.High + 0.5 * (h.probs.Medium || 0));
  }
  return { Low: 0, Medium: 50, High: 100 }[h.label] ?? 0;
}

function stats(entries) {
  const out = { n: 0, High: 0, Medium: 0, Low: 0, score: 0 };
  entries.forEach((h) => {
    if (!(h.label in { High: 1, Medium: 1, Low: 1 })) return;
    out[h.label]++;
    out.n++;
    out.score += scoreOf(h);
  });
  out.avg = out.n ? out.score / out.n : null;
  return out;
}

function inRange(history, start, end, site) {
  return history
    .filter((h) => h.t >= start && h.t < end && (!site || h.site === site))
    .sort((a, b) => a.t - b.t);
}

/* Each point with the average of itself and the previous few. */
function withRolling(entries) {
  return entries.map((h, i) => {
    const recent = entries.slice(Math.max(0, i - ROLLING + 1), i + 1);
    return { ...h, score: scoreOf(h), rolling: stats(recent).avg };
  });
}

/* One bin per hour for a single day (for the table), otherwise one per day. */
function timelineBins(entries, start, end) {
  const groups = new Map();
  const single = dayCount(start, end) <= 1;
  entries.forEach((h) => {
    const key = single ? new Date(h.t).getHours() : startOfDay(h.t);
    groups.set(key, [...(groups.get(key) || []), h]);
  });
  const bins = [];
  if (single) {
    const day = new Date(start).toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
    for (let i = 0; i < 24; i++) {
      const hour = String(i).padStart(2, "0");
      bins.push({ key: i, label: hour, detail: `${day}, ${hour}:00`, ...stats(groups.get(i) || []) });
    }
    return { bins, unit: "hour" };
  }
  for (let t = start; t < end; t = addDays(t, 1)) {
    const d = new Date(t);
    bins.push({
      key: t,
      label: d.toLocaleDateString(undefined, { day: "numeric", month: "short" }),
      detail: d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" }),
      ...stats(groups.get(t) || []),
    });
  }
  return { bins, unit: "day" };
}

/* Sites, highest score first; those with too few estimates follow the rest. */
function siteStats(entries) {
  const groups = new Map();
  entries.forEach((h) => groups.set(h.site, [...(groups.get(h.site) || []), h]));
  const rows = [...groups].map(([site, list]) => ({ site, ...stats(list) }))
    .sort((a, b) => (b.n >= MIN_RELIABLE) - (a.n >= MIN_RELIABLE) || b.avg - a.avg);
  if (rows.length <= MAX_SITES) return rows;
  const rest = new Set(rows.slice(MAX_SITES - 1).map((r) => r.site));
  const other = stats(entries.filter((h) => rest.has(h.site)));
  return [...rows.slice(0, MAX_SITES - 1), { site: `Other sites (${rest.size})`, isOther: true, ...other }];
}

/* Rows Monday to Sunday, columns of BLOCK_HOURS hours. */
function heatGrid(entries) {
  const columns = 24 / BLOCK_HOURS;
  const lists = WEEKDAYS.map(() => Array.from({ length: columns }, () => []));
  entries.forEach((h) => {
    const d = new Date(h.t);
    lists[(d.getDay() + 6) % 7][Math.floor(d.getHours() / BLOCK_HOURS)].push(h);
  });
  return lists.map((row) => row.map(stats));
}

function blockLabel(i) {
  const from = i * BLOCK_HOURS;
  return `${String(from).padStart(2, "0")}:00 to ${String((from + BLOCK_HOURS) % 24).padStart(2, "0")}:00`;
}

function verdict(s) {
  if (!s.n) return null;
  if (s.n < MIN_RELIABLE) return "Too few estimates";
  if (s.Low / s.n >= 0.6) return "Mostly low";
  if (s.High / s.n >= 0.5) return "Mostly high";
  return "Mixed";
}

function clock(t) {
  return new Date(t).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
}

/* Longest run of consecutive High estimates, or null if shorter than 3. */
function highStreak(entries) {
  let best = null;
  let from = -1;
  entries.forEach((h, i) => {
    if (h.label !== "High") { from = -1; return; }
    if (from < 0) from = i;
    if (!best || i - from + 1 > best.n) best = { n: i - from + 1, start: entries[from].t, end: h.t };
  });
  return best && best.n >= MIN_RELIABLE ? best : null;
}

/* Plain sentences, only for differences large enough and on enough estimates. */
function insights(entries, previous, previousName, days) {
  const now = stats(entries);
  const out = [];
  if (now.n < MIN_RELIABLE) return out;
  const before = stats(previous);
  const round = Math.round;

  if (before.n >= MIN_RELIABLE) {
    const d = round(now.avg) - round(before.avg);
    out.push(Math.abs(d) >= MIN_DIFF
      ? `Your average score is ${Math.abs(d)} points ${d > 0 ? "higher" : "lower"} than ${previousName} (${round(now.avg)} vs. ${round(before.avg)}).`
      : `Your average score is about the same as ${previousName} (${round(now.avg)} vs. ${round(before.avg)}).`);
  }
  if (now.n >= 2 * MIN_RELIABLE) {
    const half = Math.floor(now.n / 2);
    const a = round(stats(entries.slice(0, half)).avg);
    const b = round(stats(entries.slice(-half)).avg);
    if (Math.abs(b - a) >= MIN_DIFF) {
      out.push(`Your score ${b > a ? "rose" : "fell"} over this period, from ${a} in the earlier estimates to ${b} in the later ones.`);
    }
  }
  const streak = highStreak(entries);
  if (streak) {
    const when = days <= 1
      ? `${clock(streak.start)} to ${clock(streak.end)}`
      : new Date(streak.end).toLocaleDateString(undefined, { day: "numeric", month: "short" });
    out.push(`Longest run of High estimates: ${streak.n} in a row (${when}).`);
  }
  const sites = siteStats(entries).filter((s) => s.n >= MIN_RELIABLE && !s.isOther);
  if (sites.length >= 2) {
    const top = sites[0];
    const low = sites[sites.length - 1];
    const d = round(top.avg) - round(low.avg);
    if (d >= MIN_DIFF) out.push(`${top.site} scored highest (${round(top.avg)}), ${d} points above ${low.site}.`);
  }
  if (days >= 2) {
    const parts = PARTS.map(([name, from], i) => ({
      name,
      ...stats(entries.filter((h) => {
        const hour = new Date(h.t).getHours();
        return hour >= from && hour < (PARTS[i + 1] ? PARTS[i + 1][1] : 24);
      })),
    })).filter((p) => p.n >= MIN_RELIABLE).sort((a, b) => b.avg - a.avg);
    if (parts.length >= 2) {
      const top = parts[0];
      const low = parts[parts.length - 1];
      if (round(top.avg) - round(low.avg) >= MIN_DIFF) {
        out.push(`${top.name} scored highest (${round(top.avg)}) and ${low.name.toLowerCase()} lowest (${round(low.avg)}).`);
      }
    }
  }
  return out;
}
