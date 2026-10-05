/* Ntawusigara tools: rank uncertainty, coverage allocation, survey-round comparison.
   Pure functions, no network, used by index.html and the on-device assistant.

   Sampling assumption (documented in docs/METHODOLOGY.md): district estimates come
   from samples of about 400-500 households (EICV7) or adults (FinScope 2024). With
   cluster sampling, the effective sample size is smaller; we use N_EFF = 250 by default.
   If a district row carries poverty_2024_se / finscope_n, those are used instead. */
(function (root) {
  'use strict';
  var N_EFF = 250, DRAWS = 1000, SEED = 2026;
  var KEYS = ['poverty_2024', 'poor_people', 'not_formally_served', 'unbanked', 'progress_lag'];

  /* ---- seeded random numbers ---- */
  function rng(seed) { var a = seed >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function normal(r) { var u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function gamma(r, k) { // Marsaglia-Tsang
    if (k < 1) return gamma(r, k + 1) * Math.pow(r(), 1 / k);
    var d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) { var x, v; do { x = normal(r); v = 1 + c * x; } while (v <= 0); v = v * v * v; var u = r();
      if (u < 1 - 0.0331 * x * x * x * x || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
  }
  function betaPct(r, pct, n) { var p = Math.min(Math.max(pct / 100, 0), 1), a = gamma(r, p * n + 0.5), b = gamma(r, (1 - p) * n + 0.5); return 100 * a / (a + b); }
  function nFromSe(pct, se) { var p = pct / 100, s = se / 100; return s > 0 ? Math.max(20, p * (1 - p) / (s * s)) : N_EFF; }

  /* ---- 1. sampling draws and rank ranges ---- */
  // Each draw re-samples every district's survey estimates within their sampling error.
  function makeDraws(rows, opt) {
    opt = opt || {}; var D = opt.draws || DRAWS, r = rng(opt.seed || SEED), N = rows.length;
    var out = new Float64Array(D * N * 4); // pov24, pov17, banked, notFormal per draw/district
    for (var d = 0; d < D; d++) for (var i = 0; i < N; i++) {
      var row = rows[i], nP = row.poverty_2024_se ? nFromSe(row.poverty_2024, row.poverty_2024_se) : (opt.nEff || N_EFF), nF = row.finscope_n || opt.nEff || N_EFF;
      var g = [row.banked, row.other_formal_only, row.informal_only, row.excluded].map(function (p) { return gamma(r, Math.max(p, 0) / 100 * nF + 0.5); });
      var tot = g[0] + g[1] + g[2] + g[3], o = (d * N + i) * 4;
      out[o] = betaPct(r, row.poverty_2024, nP); out[o + 1] = betaPct(r, row.poverty_2017_modelled, nP);
      out[o + 2] = 100 * g[0] / tot; out[o + 3] = 100 * (g[2] + g[3]) / tot;
    }
    return { D: D, N: N, v: out };
  }
  function indicators(rows, get) { // raw indicator matrix from per-district values
    var N = rows.length, m = {}, drop = [];
    KEYS.forEach(function (k) { m[k] = new Float64Array(N); });
    for (var i = 0; i < N; i++) { var x = get(i); m.poverty_2024[i] = x.pov24; m.poor_people[i] = x.pov24 / 100 * rows[i].population_2022;
      m.not_formally_served[i] = x.nfs; m.unbanked[i] = 100 - x.banked; drop[i] = x.pov17 - x.pov24; }
    var mx = Math.max.apply(null, drop); for (var j = 0; j < N; j++) m.progress_lag[j] = mx - drop[j];
    return m;
  }
  function scores(m, W) {
    var N = m.poverty_2024.length, tot = KEYS.reduce(function (a, k) { return a + (W[k] || 0); }, 0) || 1, s = new Float64Array(N);
    KEYS.forEach(function (k) { var a = m[k], lo = Infinity, hi = -Infinity, i; for (i = 0; i < N; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; }
      var w = (W[k] || 0) / tot; for (i = 0; i < N; i++) s[i] += (hi > lo ? (a[i] - lo) / (hi - lo) : 0) * w * 100; });
    return s;
  }
  function ranksOf(s) { var idx = Array.from(s.keys ? s.keys() : Array(s.length).keys()).sort(function (a, b) { return s[b] - s[a]; }), r = new Int32Array(s.length); idx.forEach(function (i, k) { r[i] = k + 1; }); return r; }
  function rankRanges(rows, draws, W) {
    var N = draws.N, D = draws.D, all = []; for (var i = 0; i < N; i++) all.push(new Int32Array(D));
    for (var d = 0; d < D; d++) {
      var m = indicators(rows, function (i) { var o = (d * N + i) * 4; return { pov24: draws.v[o], pov17: draws.v[o + 1], banked: draws.v[o + 2], nfs: draws.v[o + 3] }; });
      var rk = ranksOf(scores(m, W)); for (var j = 0; j < N; j++) all[j][d] = rk[j];
    }
    var res = {};
    rows.forEach(function (row, i) { var a = Array.from(all[i]).sort(function (x, y) { return x - y; }), q = function (p) { return a[Math.min(D - 1, Math.floor(p * D))]; };
      var top5 = 0, top10 = 0; a.forEach(function (x) { if (x <= 5) top5++; if (x <= 10) top10++; });
      res[row.district] = { lo: q(0.05), hi: q(0.95), median: q(0.5), pTop5: top5 / D * 100, pTop10: top10 / D * 100 }; });
    return res;
  }

  /* ---- 2. coverage allocator ---- */
  // Shares `people` across districts in proportion to weight, never above a district's number of poor people.
  function allocate(rows, opt) {
    var people = Math.max(0, Math.round(opt.people || 0)), pool;
    if (opt.method === 'all') pool = rows.map(function (r) { return { r: r, w: r.poor_people * Math.max(r.score, 0) / 100 }; });
    else pool = rows.slice().sort(function (a, b) { return a.rank - b.rank; }).slice(0, opt.topN || 10).map(function (r) { return { r: r, w: r.poor_people }; });
    pool.forEach(function (p) { p.alloc = 0; });
    var left = people, open = pool.filter(function (p) { return p.w > 0; });
    for (var guard = 0; left > 0.5 && open.length && guard < 50; guard++) {
      var W = open.reduce(function (a, p) { return a + p.w; }, 0), next = [], given = 0;
      open.forEach(function (p) { var room = p.r.poor_people - p.alloc, add = Math.min(room, left * p.w / W); p.alloc += add; given += add; if (p.r.poor_people - p.alloc > 0.5) next.push(p); });
      left -= given; open = next;
    }
    // whole people, largest remainder
    var fl = pool.map(function (p) { return Math.floor(p.alloc); }), rest = Math.round(pool.reduce(function (a, p) { return a + p.alloc; }, 0)) - fl.reduce(function (a, b) { return a + b; }, 0);
    pool.map(function (p, i) { return { i: i, f: p.alloc - fl[i] }; }).sort(function (a, b) { return b.f - a.f; }).slice(0, rest).forEach(function (x) { fl[x.i]++; });
    var lines = pool.map(function (p, i) { return { district: p.r.district, rank: p.r.rank, poor: p.r.poor_people, people: fl[i], share: fl[i] / p.r.poor_people * 100 }; })
      .filter(function (l) { return l.people > 0; }).sort(function (a, b) { return b.people - a.people; });
    var placed = lines.reduce(function (a, l) { return a + l.people; }, 0);
    return { lines: lines, placed: placed, unplaced: people - placed, cost: opt.cost ? placed * opt.cost : null };
  }

  /* ---- 3. comparing survey rounds ---- */
  function parseCSV(text) {
    var lines = String(text).replace(/^\uFEFF/, '').split(/\r?\n/).filter(function (l) { return l.trim(); });
    if (!lines.length) return [];
    var sep = (lines[0].match(/;/g) || []).length > (lines[0].match(/,/g) || []).length ? ';' : ',';
    function split(l) { var out = [], cur = '', q = false; for (var i = 0; i < l.length; i++) { var ch = l[i];
      if (ch === '"') { if (q && l[i + 1] === '"') { cur += '"'; i++; } else q = !q; } else if (ch === sep && !q) { out.push(cur); cur = ''; } else cur += ch; }
      out.push(cur); return out.map(function (s) { return s.trim(); }); }
    var head = split(lines[0]).map(function (h) { return h.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, ''); });
    return lines.slice(1).map(function (l) { var c = split(l), o = {}; head.forEach(function (h, i) { o[h] = c[i]; }); return o; });
  }
  var ALIASES = { district: ['district', 'akarere', 'name'], poverty_2024: ['poverty_rate', 'poverty', 'poverty_2024', 'poverty_pct', 'ubukene'],
    banked: ['banked', 'banked_pct'], other_formal_only: ['other_formal_only', 'other_formal'], informal_only: ['informal_only', 'informal'],
    excluded: ['excluded', 'exclusion'], population_2022: ['population', 'population_2022'] };
  function pick(o, k) { for (var i = 0; i < ALIASES[k].length; i++) { var v = o[ALIASES[k][i]]; if (v != null && v !== '') return v; } return null; }
  function num(v) { if (v == null) return null; var n = parseFloat(String(v).replace(/\s/g, '').replace(',', '.')); return isFinite(n) ? n : null; }
  function seDiff(p1, p2, n) { var a = p1 / 100, b = p2 / 100; return 100 * Math.sqrt(a * (1 - a) / n + b * (1 - b) / n); }
  function compareRounds(rows, newRaw, W, opt) {
    opt = opt || {}; var n = opt.nEff || N_EFF, by = {}, unknown = [];
    rows.forEach(function (r) { by[r.district.toLowerCase()] = r; });
    var next = {};
    newRaw.forEach(function (o) { var d = String(pick(o, 'district') || '').trim(), r = by[d.toLowerCase()];
      if (!r) { if (d) unknown.push(d); return; }
      var x = {}; ['poverty_2024', 'banked', 'other_formal_only', 'informal_only', 'excluded', 'population_2022'].forEach(function (k) { var v = num(pick(o, k)); if (v != null) x[k] = v; });
      next[r.district] = x; });
    var hasFin = Object.keys(next).some(function (d) { return next[d].banked != null; });
    var merged = rows.map(function (r) { var x = next[r.district] || {}, m = {};
      for (var k in r) m[k] = r[k];
      ['poverty_2024', 'banked', 'other_formal_only', 'informal_only', 'excluded', 'population_2022'].forEach(function (k) { if (x[k] != null) m[k] = x[k]; });
      m.not_formally_served = m.informal_only + m.excluded; return m; });
    var oldS = scores(indicators(rows, function (i) { var r = rows[i]; return { pov24: r.poverty_2024, pov17: r.poverty_2017_modelled, banked: r.banked, nfs: r.not_formally_served }; }), W);
    var newS = scores(indicators(merged, function (i) { var r = merged[i]; return { pov24: r.poverty_2024, pov17: r.poverty_2017_modelled, banked: r.banked, nfs: r.not_formally_served }; }), W);
    var oldR = ranksOf(oldS), newR = ranksOf(newS);
    var res = rows.map(function (r, i) { var m = merged[i], line = { district: r.district, inNew: !!next[r.district], oldRank: oldR[i], newRank: newR[i] };
      [['poverty_2024', 'pov'], ['banked', 'bank'], ['not_formally_served', 'nfs']].forEach(function (p) {
        var d = m[p[0]] - r[p[0]], se = seDiff(r[p[0]], m[p[0]], n); line[p[1] + 'Old'] = r[p[0]]; line[p[1] + 'New'] = m[p[0]];
        line[p[1] + 'Delta'] = Math.round(d * 10) / 10; line[p[1] + 'Sig'] = Math.abs(d) > 1.96 * se; });
      return line; });
    var top = function (key) { return res.filter(function (l) { return l[key] <= 5; }).map(function (l) { return l.district; }); };
    var o5 = top('oldRank'), n5 = top('newRank');
    return { lines: res, matched: Object.keys(next).length, unknown: unknown, hasFinance: hasFin,
      enteredTop5: n5.filter(function (d) { return o5.indexOf(d) < 0; }), leftTop5: o5.filter(function (d) { return n5.indexOf(d) < 0; }),
      povFellSig: res.filter(function (l) { return l.povSig && l.povDelta < 0; }).map(function (l) { return l.district; }),
      povRoseSig: res.filter(function (l) { return l.povSig && l.povDelta > 0; }).map(function (l) { return l.district; }),
      bankRoseSig: res.filter(function (l) { return l.bankSig && l.bankDelta > 0; }).map(function (l) { return l.district; }) };
  }
  function templateCSV(rows) {
    return 'district,poverty_rate,banked,other_formal_only,informal_only,excluded,population\n' +
      rows.slice().sort(function (a, b) { return a.district < b.district ? -1 : 1; })
        .map(function (r) { return [r.district, r.poverty_2024, r.banked, r.other_formal_only, r.informal_only, r.excluded, r.population_2022].join(','); }).join('\n') + '\n';
  }
  function demoRound(rows, seed) { // one simulated re-survey: same truth, new sampling noise only
    var dr = makeDraws(rows, { draws: 1, seed: seed || 99 }), N = rows.length;
    return rows.map(function (r, i) { var o = i * 4, b = dr.v[o + 2], nfs = dr.v[o + 3], share = r.informal_only / Math.max(r.not_formally_served, 1e-9);
      return { district: r.district, poverty_rate: dr.v[o].toFixed(1), banked: b.toFixed(0), other_formal_only: Math.max(0, 100 - b - nfs).toFixed(0),
        informal_only: (nfs * (isFinite(share) ? share : 0.5)).toFixed(0), excluded: (nfs * (1 - (isFinite(share) ? share : 0.5))).toFixed(0) }; });
  }

  var api = { N_EFF: N_EFF, makeDraws: makeDraws, rankRanges: rankRanges, allocate: allocate, parseCSV: parseCSV,
    compareRounds: compareRounds, templateCSV: templateCSV, demoRound: demoRound, scores: scores, indicators: indicators };
  if (typeof module === 'object' && module.exports) module.exports = api; else root.NtwTools = api;
})(typeof self !== 'undefined' ? self : this);
