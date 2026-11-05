/*
  Runs the stress models inside the browser, so typing never leaves this computer.
  A port of features/text_features.py, features/burst_features.py and NLTK's VADER;
  tests/test_engine_parity.py checks that the answers match Python.
*/
"use strict";

(() => {
  const M = globalThis.STRESS_MODEL;
  const LEVELS = ["Low", "Medium", "High"];

  /* ------------------------------------------------------------ helpers */
  function round(x, digits) {        // like Python: an exact tie goes to the even digit
    const exact = x.toFixed(60);
    const cut = exact.indexOf(".") + 1 + digits;
    if (!/^50*$/.test(exact.slice(cut))) return Number(x.toFixed(digits));
    const kept = exact.slice(0, cut);
    const even = Number(kept[kept.length - 1]) % 2 === 0;
    return even ? Number(kept) : Number(kept) + Math.sign(x) * 10 ** -digits;
  }
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const mean = (a) => sum(a) / a.length;
  const std = (a) => {
    const m = mean(a);
    return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length);
  };
  function percentile(values, q) {
    const a = [...values].sort((x, y) => x - y);
    const at = (q / 100) * (a.length - 1);
    const lo = Math.floor(at);
    return lo + 1 < a.length ? a[lo] + (at - lo) * (a[lo + 1] - a[lo]) : a[lo];
  }
  const softmax = (scores) => {
    const top = Math.max(...scores);
    const e = scores.map((s) => Math.exp(s - top));
    const total = sum(e);
    return e.map((v) => v / total);
  };
  const isUpper = (w) => w.toUpperCase() === w && w.toLowerCase() !== w;

  /* ------------------------------------------------------------ VADER */
  const V = M.vader;
  const LEX = new Map(Object.entries(V.lexicon));
  const BOOST = new Map(Object.entries(V.booster));
  const IDIOMS = new Map(Object.entries(V.idioms));
  const NEGATE = new Set(V.negate);
  const PUNCT = /[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g;
  const B_DECR = -0.293;
  const C_INCR = 0.733;
  const N_SCALAR = -0.74;
  const splitWords = (t) => t.split(/\s+/).filter(Boolean);

  function negated(word) {
    const w = word.toLowerCase();
    return NEGATE.has(w) || w.includes("n't");
  }

  function wordsAndEmoticons(text) {
    const plain = new Set(splitWords(text.replace(PUNCT, "")).filter((w) => [...w].length > 1));
    return splitWords(text).filter((w) => [...w].length > 1).map((we) => {
      for (const p of V.punc) if (we.endsWith(p) && plain.has(we.slice(0, -p.length))) return we.slice(0, -p.length);
      for (const p of V.punc) if (we.startsWith(p) && plain.has(we.slice(p.length))) return we.slice(p.length);
      return we;
    });
  }

  function scalar(word, valence, capDiff) {
    const w = word.toLowerCase();
    if (!BOOST.has(w)) return 0;
    let s = BOOST.get(w);
    if (valence < 0) s *= -1;
    if (isUpper(word) && capDiff) s += valence > 0 ? C_INCR : -C_INCR;
    return s;
  }

  function neverCheck(valence, w, startI, i) {
    if (startI === 0) {
      if (negated(w[i - 1])) valence *= N_SCALAR;
    } else if (startI === 1) {
      if (w[i - 2] === "never" && (w[i - 1] === "so" || w[i - 1] === "this")) valence *= 1.5;
      else if (negated(w[i - 2])) valence *= N_SCALAR;
    } else if (startI === 2) {
      if ((w[i - 3] === "never" && (w[i - 2] === "so" || w[i - 2] === "this")) || w[i - 1] === "so" || w[i - 1] === "this") valence *= 1.25;
      else if (negated(w[i - 3])) valence *= N_SCALAR;
    }
    return valence;
  }

  function idiomsCheck(valence, w, i) {
    // Negative indexes wrap around in Python; the same words are used here.
    const at = (k) => w[k < 0 ? w.length + k : k];
    const twoOne = `${at(i - 2)} ${at(i - 1)}`;
    const threeTwo = `${at(i - 3)} ${at(i - 2)}`;
    for (const seq of [`${at(i - 1)} ${w[i]}`, `${at(i - 2)} ${at(i - 1)} ${w[i]}`, twoOne, `${at(i - 3)} ${at(i - 2)} ${at(i - 1)}`, threeTwo]) {
      if (IDIOMS.has(seq)) { valence = IDIOMS.get(seq); break; }
    }
    if (w.length - 1 > i && IDIOMS.has(`${w[i]} ${w[i + 1]}`)) valence = IDIOMS.get(`${w[i]} ${w[i + 1]}`);
    if (w.length - 1 > i + 1 && IDIOMS.has(`${w[i]} ${w[i + 1]} ${w[i + 2]}`)) valence = IDIOMS.get(`${w[i]} ${w[i + 1]} ${w[i + 2]}`);
    if (BOOST.has(threeTwo) || BOOST.has(twoOne)) valence += B_DECR;
    return valence;
  }

  function leastCheck(valence, w, i) {
    const before = i > 0 ? w[i - 1].toLowerCase() : "";
    if (i > 1 && !LEX.has(before) && before === "least") {
      const two = w[i - 2].toLowerCase();
      if (two !== "at" && two !== "very") valence *= N_SCALAR;
    } else if (i > 0 && !LEX.has(before) && before === "least") {
      valence *= N_SCALAR;
    }
    return valence;
  }

  function vader(text) {
    const w = wordsAndEmoticons(text);
    const caps = w.filter(isUpper).length;
    const capDiff = w.length - caps > 0 && w.length - caps < w.length;
    const first = new Map();
    w.forEach((t, i) => { if (!first.has(t)) first.set(t, i); });
    let sentiments = [];
    w.forEach((item) => {
      const i = first.get(item);
      const low = item.toLowerCase();
      if ((i < w.length - 1 && low === "kind" && w[i + 1].toLowerCase() === "of") || BOOST.has(low)) {
        sentiments.push(0);
        return;
      }
      let valence = 0;
      if (LEX.has(low)) {
        valence = LEX.get(low);
        if (isUpper(item) && capDiff) valence += valence > 0 ? C_INCR : -C_INCR;
        for (let s = 0; s < 3; s++) {
          if (i > s && !LEX.has(w[i - (s + 1)].toLowerCase())) {
            let b = scalar(w[i - (s + 1)], valence, capDiff);
            if (s === 1 && b !== 0) b *= 0.95;
            if (s === 2 && b !== 0) b *= 0.9;
            valence += b;
            valence = neverCheck(valence, w, s, i);
            if (s === 2) valence = idiomsCheck(valence, w, i);
          }
        }
        valence = leastCheck(valence, w, i);
      }
      sentiments.push(valence);
    });
    const lows = w.map((x) => x.toLowerCase());
    const bi = lows.indexOf("but");
    if (bi >= 0) sentiments = sentiments.map((v, k) => (k < bi ? v * 0.5 : k > bi ? v * 1.5 : v));
    if (!sentiments.length) return { neg: 0, neu: 0, pos: 0, compound: 0 };

    let total = sum(sentiments);
    const ep = Math.min((text.match(/!/g) || []).length, 4) * 0.292;
    const qs = (text.match(/\?/g) || []).length;
    const amp = ep + (qs > 1 ? (qs <= 3 ? qs * 0.18 : 0.96) : 0);
    if (total > 0) total += amp;
    else if (total < 0) total -= amp;
    const compound = total / Math.sqrt(total * total + 15);
    let pos = 0;
    let neg = 0;
    let neu = 0;
    sentiments.forEach((v) => {
      if (v > 0) pos += v + 1;
      if (v < 0) neg += v - 1;
      if (v === 0) neu += 1;
    });
    if (pos > Math.abs(neg)) pos += amp;
    else if (pos < Math.abs(neg)) neg -= amp;
    const all = pos + Math.abs(neg) + neu;
    return { neg: round(Math.abs(neg / all), 3), neu: round(Math.abs(neu / all), 3),
      pos: round(Math.abs(pos / all), 3), compound: round(compound, 4) };
  }

  /* ------------------------------------------------------------ text */
  const T = M.text;
  const TERMS = new Map(Object.entries(T.terms));
  const STOP = new Set(T.stop);
  const WORDS = Object.fromEntries(Object.entries(T.words).map(([k, v]) => [k, new Set(v)]));

  function linguistic(text) {
    const tokens = text.toLowerCase().match(/[a-z']+/g) || [];
    const n = tokens.length;
    if (!n) {
      return { sent_compound: 0, sent_pos: 0, sent_neg: 0, sent_neu: 1, negative_word_ratio: 0, stress_word_ratio: 0,
        positive_word_ratio: 0, negation_ratio: 0, first_person_ratio: 0, lexical_diversity: 0, avg_word_len: 0,
        word_count: 0, exclamation_ratio: 0, question_ratio: 0, caps_ratio: 0 };
    }
    const s = vader(text);
    const share = (set) => tokens.filter((t) => set.has(t)).length / n;
    const chars = [...text];
    const size = Math.max(chars.length, 1);
    return { sent_compound: s.compound, sent_pos: s.pos, sent_neg: s.neg, sent_neu: s.neu,
      negative_word_ratio: share(WORDS.negative), stress_word_ratio: share(WORDS.stress),
      positive_word_ratio: share(WORDS.positive), negation_ratio: share(WORDS.negation),
      first_person_ratio: share(WORDS.first), lexical_diversity: new Set(tokens).size / n,
      avg_word_len: mean(tokens.map((t) => t.length)), word_count: n,
      exclamation_ratio: chars.filter((c) => c === "!").length / size,
      question_ratio: chars.filter((c) => c === "?").length / size,
      caps_ratio: chars.filter(isUpper).length / size };
  }

  function tfidfScore(text) {
    const words = (text.toLowerCase().match(/[\p{L}\p{M}\p{N}_]{2,}/gu) || []).filter((w) => !STOP.has(w));
    const counts = new Map();
    const add = (g) => { if (TERMS.has(g)) counts.set(g, (counts.get(g) || 0) + 1); };
    words.forEach((w, i) => {
      add(w);
      if (i + 1 < words.length) add(`${w} ${words[i + 1]}`);
    });
    let norm = 0;
    let score = 0;
    counts.forEach((c, g) => {
      const [idf, weight] = TERMS.get(g);
      const v = (1 + Math.log(c)) * idf;
      norm += v * v;
      score += v * weight;
    });
    return norm > 0 ? score / Math.sqrt(norm) : 0;
  }

  function textProbs(text) {
    const f = linguistic(text);
    let d = T.intercept + tfidfScore(text);
    T.ling.names.forEach((name, i) => { d += ((f[name] - T.ling.mean[i]) / T.ling.scale[i]) * T.ling.coef[i]; });
    const second = 1 / (1 + Math.exp(-d));
    const out = { Low: 0, Medium: 0, High: 0 };
    out[T.classes[0]] = 1 - second;
    out[T.classes[1]] = second;
    return out;
  }

  /* ------------------------------------------------------------ typing */
  function rhythm(gapsIn) {
    const gaps = gapsIn.filter((g) => g > 0 && g < 2000);
    if (gaps.length < 10) return { rhythm_step_median: NaN, rhythm_step_iqr: NaN, rhythm_lag1: NaN, rhythm_drift: NaN, rhythm_entropy: NaN, rhythm_fast_share: NaN };
    const log = gaps.map(Math.log);
    const step = log.slice(1).map((v, i) => Math.abs(v - log[i]));
    const m = mean(log);
    const c = log.map((v) => v - m);
    const denom = sum(c.map((v) => v * v));
    let lag = 0;
    for (let i = 0; i + 1 < c.length; i++) lag += c[i] * c[i + 1];
    const xm = (log.length - 1) / 2;
    let sxy = 0;
    let sxx = 0;
    log.forEach((v, i) => { sxy += (i - xm) * (v - m); sxx += (i - xm) * (i - xm); });
    let lo = Math.min(...log);
    let hi = Math.max(...log);
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    const bins = new Array(10).fill(0);
    log.forEach((v) => { bins[Math.min(9, Math.floor(((v - lo) / (hi - lo)) * 10))]++; });
    const shares = bins.filter((b) => b > 0).map((b) => b / log.length);
    return { rhythm_step_median: percentile(step, 50), rhythm_step_iqr: percentile(step, 75) - percentile(step, 25),
      rhythm_lag1: denom > 1e-12 ? lag / denom : 0, rhythm_drift: (sxy / sxx) * log.length,
      rhythm_entropy: -sum(shares.map((p) => p * Math.log(p))), rhythm_fast_share: gaps.filter((g) => g < 120).length / gaps.length };
  }

  function burstFeatures(times, backspaces) {
    const gaps = times.slice(1).map((t, i) => t - times[i]).filter((g) => g > 0);
    const duration = Math.max(times[times.length - 1] - times[0], 1);
    const g = gaps.length > 0;
    return { log_speed: Math.log(times.length / (duration / 60000)),
      log_median_gap: g ? Math.log(percentile(gaps, 50)) : NaN, log_mean_gap: g ? Math.log(mean(gaps)) : NaN,
      gap_cv: g ? std(gaps) / mean(gaps) : NaN, backspace_rate: mean(backspaces.map(Number)), ...rhythm(gaps) };
  }

  function typingProbs(times, backspaces) {
    const P = M.typing;
    const f = burstFeatures(times, backspaces);
    const x = P.features.map((name, i) => {
      const v = Number.isFinite(f[name]) ? f[name] : P.median[i];
      return (v - P.mean[i]) / P.scale[i];
    });
    const scores = P.coef.map((row, c) => P.intercept[c] + sum(row.map((w, i) => w * x[i])));
    const p = softmax(scores);
    return Object.fromEntries(P.classes.map((c, i) => [c, p[i]]));
  }

  /* ------------------------------------------------------------ one burst */
  function labelOf(probs) {
    return LEVELS.reduce((a, b) => (probs[b] > probs[a] ? b : a));
  }

  function result(probs, source, validated) {
    const label = labelOf(probs);
    return { label, probs, confidence: probs[label], source, validated };
  }

  const MODIFIERS = new Set(["shift", "control", "alt", "altgraph", "meta", "capslock", "fn"]);

  function predict(text, events) {
    // SWELL-KW logs a capital as one key, a browser as Shift plus the letter.
    const presses = (events || []).filter((e) => e.ev === "down" && !MODIFIERS.has(String(e.key).toLowerCase()));
    const typing = presses.length >= M.minKeys
      ? result(typingProbs(presses.map((e) => e.t), presses.map((e) => String(e.key).toLowerCase() === "backspace")), "swell", false)
      : null;
    if (typing) typing.unclear = typing.confidence < M.unclearBelow;
    const words = splitWords(text || "").length;
    const textEst = words >= M.minWords ? result(textProbs(text), "dreaddit", true) : null;
    let fused = null;
    if (typing && textEst) {
      const w = M.typingWeight;
      fused = result(Object.fromEntries(LEVELS.map((l) => [l, w * typing.probs[l] + (1 - w) * textEst.probs[l]])), "late fusion", false);
    } else if (typing || textEst) {
      fused = result((typing || textEst).probs, "single modality", false);
    }
    return { typing, text: textEst, fused };
  }

  globalThis.StressEngine = { predict, vader, linguistic, textProbs, typingProbs, burstFeatures };
})();
