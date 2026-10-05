/* Ntawusigara on-device assistant. No API, no server, works offline.
   1. Two Naive Bayes text classifiers (app/ai-model.js, trained by
      analysis/train_assistant.py) find the question type and its language.
   2. Rules find districts, provinces, indicators, thresholds and direction.
   3. The answer is computed from the tool's own district table, so every
      figure is real and nothing is invented. Briefing notes are written the
      same way (data-to-text), in Kinyarwanda, English or French. */
(function (root) {
  'use strict';
  var M = root.NTW_MODEL;
  function load(m) { var idx = {}, v = m.v.split(' '), w = m.w.split('|'); for (var i = 0; i < v.length; i++) idx[v[i]] = w[i]; return { c: m.c, p: m.p, b: m.b, idx: idx, cache: {} }; }
  var NBI = load(M.intent), NBL = load(M.lang);
  function normQ(s) { return String(s).toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, ' ').trim(); }
  function feats(s) { var o = {}; normQ(s).split(' ').forEach(function (w) { if (!w) return; o['#' + w] = 1; var p = '<' + w + '>'; for (var n = 3; n <= 4; n++) for (var i = 0; i + n <= p.length; i++) o[p.substr(i, n)] = 1; }); return Object.keys(o); }
  function predict(m, s) {
    var sc = m.p.slice();
    feats(s).forEach(function (f) { var e = m.idx[f]; if (e == null) return; var w = m.cache[f];
      if (!w) { w = m.b.slice(); if (e) e.split(';').forEach(function (p) { var k = p.split(':'); w[+k[0]] += +k[1] / 10; }); m.cache[f] = w; }
      for (var k = 0; k < sc.length; k++) sc[k] += w[k]; });
    var mx = Math.max.apply(null, sc), Z = 0, best = 0; sc.forEach(function (x, k) { Z += Math.exp(x - mx); if (x > sc[best]) best = k; });
    return { label: m.c[best], p: 1 / Z };
  }

  /* ---------- national figures (EICV7, FinScope 2024) ---------- */
  var NAT = { poverty_2024: 27.4, poverty_2017_modelled: 39.8, poverty_drop_pp: 12.4, banked: 22, not_formally_served: 8, excluded: 4, unbanked: 78 };
  var PCT = { poverty_2024: 1, poverty_2017_modelled: 1, banked: 1, unbanked: 1, other_formal_only: 1, informal_only: 1, excluded: 1, not_formally_served: 1, urban_share_pct: 1, top5_robustness_pct: 1 };
  var LAB = {
    poverty_2024: { en: 'poverty rate (2024)', fr: 'taux de pauvreté (2024)', rw: "igipimo cy'ubukene (2024)" },
    poor_people: { en: 'people below the poverty line', fr: 'personnes sous le seuil de pauvreté', rw: "abantu bari munsi y'umurongo w'ubukene" },
    banked: { en: 'adults with a bank account', fr: 'adultes bancarisés', rw: 'abakuru bafite konti ya banki' },
    unbanked: { en: 'adults without a bank account', fr: 'adultes sans compte bancaire', rw: 'abakuru badafite konti ya banki' },
    other_formal_only: { en: 'adults using other formal services only (mostly mobile money and SACCOs)', fr: "adultes n'utilisant que d'autres services formels (surtout mobile money et SACCO)", rw: 'abakuru bakoresha izindi serivisi zemewe gusa (cyane cyane mobile money na SACCO)' },
    informal_only: { en: 'adults using informal finance only', fr: "adultes n'utilisant que la finance informelle", rw: 'abakuru bakoresha imari itemewe gusa' },
    excluded: { en: 'financially excluded adults', fr: 'adultes financièrement exclus', rw: "abakuru batagerwaho na serivisi z'imari" },
    not_formally_served: { en: 'adults not formally served', fr: 'adultes hors finance formelle', rw: "abakuru badakoresha serivisi z'imari zemewe" },
    poverty_drop_pp: { en: 'fall in poverty since 2017 (points)', fr: 'recul de la pauvreté depuis 2017 (points)', rw: "igabanuka ry'ubukene kuva 2017 (amanota)" },
    population_2022: { en: 'population (2022)', fr: 'population (2022)', rw: 'abaturage (2022)' },
    urban_share_pct: { en: 'urban share of the population', fr: 'part de population urbaine', rw: "ijanisha ry'abatuye mu mijyi" },
    score: { en: 'priority score (current weights)', fr: 'score de priorité (poids actuels)', rw: "amanota y'ibyihutirwa (uburemere buriho)" },
    top5_robustness_pct: { en: 'top-5 robustness', fr: 'robustesse top 5', rw: "inshuro kari mu 5 ba mbere" }
  };
  var FIN = ['banked', 'other_formal_only', 'informal_only', 'excluded'];
  var PROVS = ['City of Kigali', 'Southern', 'Western', 'Northern', 'Eastern'];
  var TYPS = ['Poor and under-served', 'Stalled progress', 'Lower poverty, finance gaps', 'Banked urban core'];

  /* ---------- slot finding ---------- */
  function slots(raw, rows) {
    var s = ' ' + normQ(raw) + ' ', o = { raw: raw, n: s };
    o.districts = [];
    rows.map(function (r) { return { d: r.district, i: s.indexOf(' ' + r.district.toLowerCase() + ' ') }; })
      .filter(function (x) { return x.i > -1; }).sort(function (a, b) { return a.i - b.i; }).forEach(function (x) { o.districts.push(x.d); });
    var P = [['Southern', / (southern|south|sud|amajyepfo) /], ['Western', / (western|west|ouest|iburengerazuba) /], ['Northern', / (northern|north|nord|amajyaruguru) /],
      ['Eastern', / (eastern|east|iburasirazuba) | de l est | province est /], ['City of Kigali', / kigali /]];
    P.forEach(function (p) { if (!o.province && p[1].test(s)) o.province = p[0]; });
    var I = [
      ['poor_people', /number of poor|how many poor|poor people|people below|nombre de pauvres|de pauvres|des pauvres|personnes pauvres| abakene /],
      ['top5_robustness_pct', /robust/],
      ['poverty_drop_pp', / fall| fell| drop|declin|reduc|progress|decreas|recul|baiss|diminu|gabanu|terambere/],
      ['other_formal_only', /mobile|sacco|other formal|autres services formels|izindi serivisi/],
      ['not_formally_served', /not formally|outside formal|formally served|hors (de la )?finance formelle|hors formel|zemewe/],
      ['unbanked', /unbanked|without (a )?bank|no bank|non bancaris|sans compte|badafite konti|nta konti/],
      ['informal_only', /informal|informel|ibimina|itemewe|tontine|vsla|savings group/],
      ['excluded', /exclu|ntibakoresha|batagerwaho/],
      ['banked', /bank|banque|bancaris|konti|banki/],
      ['population_2022', /population|residents|abaturage|habitants|people live|vit le plus/],
      ['urban_share_pct', / urban|urbain|mijyi/],
      ['score', /priority score|score|amanota y ibyihutirwa/],
      ['poverty_2024', /pover|poor|pauvr|bukene|kennye|rich|riche|kize|wealth/]];
    I.forEach(function (x) { if (!o.ind && x[1].test(s)) o.ind = x[0]; });
    o.finance = /financ|bank|banque|banki|imari|mobile|sacco|access|acces|inclusion/.test(s);
    o.asc = / (least|lowest|fewest|smallest|bottom|minimum|richest|wealthiest|moins|faible|faibles|bas|gake|bike|buke|hasi|nke|make|gakize|akize|ikize|kize|riche|riches) /.test(s);
    var tn = s.match(/ top (\d+) | (\d+) (districts|uturere|premiers) | uturere (\d+) /); if (tn) { var v = +(tn[1] || tn[2] || tn[4]); if (v > 0 && v <= 30) o.top = v; }
    var TY = [['Stalled progress', /stall|point mort|dindiye|hagaze/], ['Poor and under-served', /under served|mal desservi|serivisi nke|poor and/],
      ['Lower poverty, finance gaps', /finance gap|lacunes|icyuho mu mari|lower poverty/], ['Banked urban core', /urban core|coeur urbain|ukoresha banki|banked urban/]];
    TY.forEach(function (x) { if (!o.typ && x[1].test(s)) o.typ = x[0]; });
    var num = String(raw).match(/(\d{1,3}(?:[ ,.\u00a0\u202f]\d{3})+|\d+(?:[.,]\d+)?)\s*(%|percent|pour ?cent)?/);
    if (num) { var t = num[1]; o.thr = /[ ,.\u00a0\u202f]\d{3}$/.test(t) && t.length > 4 ? +t.replace(/[^\d]/g, '') : +t.replace(',', '.'); if (!num[2] && o.thr >= 2000 && o.thr <= 2030) delete o.thr; }
    o.cmp = /below|under|less than|lower than|fewer than|inferieur|moins de|en dessous|munsi ya|kitageze|bitageze/.test(s) ? 'lt' : 'gt';
    o.thanks = /thank|merci|murakoze/.test(s);
    o.domain = o.districts.length > 0 || !!o.province || !!o.ind || !!o.typ || /district|turere|karere|province|intara|financ|imari|eicv|finscope|census|ibarura|recensement|index|indice|weight|poids|uburemere|typolog|cluster|itsinda|rank|classement|rang|mwanya|priorit|ibyihutirwa|rwanda|data|donnee|amakuru|source|nkomoko|method|limit|imbogamizi|model|brief|note|incamake|national|gihugu|correl|link|lien|isano/.test(s);
    return o;
  }

  /* ---------- formatting in the answer's language L ---------- */
  function pk(L, o) { return o[L] || o.en; }
  function loc(L) { return L === 'en' ? 'en-US' : 'fr-FR'; }
  function dec(n, L, d) { return Number(n).toLocaleString(loc(L), { minimumFractionDigits: d || 0, maximumFractionDigits: d == null ? 1 : d }); }
  function int(n, L) { return Math.round(n).toLocaleString(loc(L)); }
  function pc(n, L) { return dec(n, L) + (L === 'fr' ? '\u00a0%' : '%'); }
  function r1(x) { return Math.round(x * 10) / 10; }
  function val(k, v, L) { if (PCT[k]) return pc(v, L); if (k === 'poverty_drop_pp') return pk(L, { en: dec(v, L) + ' points', fr: dec(v, L) + ' points', rw: 'amanota ' + dec(v, L) }); if (k === 'score') return dec(v, L, 1); return int(v, L); }
  function get(r, k) { return k === 'score' ? r.score : r[k]; }
  function join(a, L) { return a.length < 2 ? a.join('') : a.slice(0, -1).join(', ') + ' ' + pk(L, { en: 'and', fr: 'et', rw: 'na' }) + ' ' + a[a.length - 1]; }
  function ord(n, L) { if (L === 'fr') return n === 1 ? '1er' : n + 'e'; if (L === 'rw') return String(n); var s = ['th', 'st', 'nd', 'rd'], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); }
  function cap(x) { return x.charAt(0).toUpperCase() + x.slice(1); }
  function rankOf(rows, r, k, asc) { var v = get(r, k); return 1 + rows.filter(function (x) { return asc ? get(x, k) < v : get(x, k) > v; }).length; }
  function natLine(k, L) { return NAT[k] == null ? '' : ' ' + pk(L, { en: 'National: ', fr: 'National\u00a0: ', rw: 'Mu gihugu: ' }) + val(k, NAT[k], L) + '.'; }
  var CAVEAT = { en: 'Differences of a few points may not be meaningful: district figures come from samples of about 400 to 500.', fr: "Des écarts de quelques points peuvent ne pas être significatifs : les chiffres de district reposent sur des échantillons d'environ 400 à 500.", rw: "Itandukaniro ry'amanota make rishobora kutagira icyo rivuze: imibare y'uturere ishingiye ku ngero z'abantu 400 kugeza 500." };

  /* ---------- answers ---------- */
  var A = {};
  A.rank = function (L, o, c) {
    var k = o.ind || (o.finance ? 'not_formally_served' : 'poverty_2024'), n = o.top || 5;
    var s = c.rows.slice().sort(function (a, b) { return o.asc ? get(a, k) - get(b, k) : get(b, k) - get(a, k); }).slice(0, n);
    var head = o.asc ? pk(L, { en: 'From lowest, ', fr: 'Du plus bas au plus élevé, ', rw: 'Uhereye ku kiri hasi, ' }) : pk(L, { en: 'From highest, ', fr: 'Du plus élevé au plus bas, ', rw: 'Uhereye ku kiri hejuru, ' });
    var out = head + LAB[k][L] + (L === 'fr' ? '\u00a0:' : ':') + '\n' + s.map(function (r, i) { return (i + 1) + '. ' + r.district + ': ' + val(k, get(r, k), L); }).join('\n');
    if (o.finance && !o.ind) out += '\n' + pk(L, { en: '"Not formally served" means adults using only informal finance, or none.', fr: '« Hors finance formelle » : adultes qui n\'utilisent que l\'informel, ou rien.', rw: '"Badakoresha serivisi zemewe" ni abakoresha imari itemewe gusa cyangwa ntacyo.' });
    return out + (NAT[k] == null ? '' : '\n' + natLine(k, L).trim());
  };
  function fullProfile(L, r, c) {
    var T = c.T[L] || c.T.en, x = function (k) { return pc(r[k], L); };
    return pk(L, {
      en: r.district + ' (' + T.prov[r.province] + '): poverty ' + x('poverty_2024') + ' in 2024, against 27.4% nationally; it was ' + x('poverty_2017_modelled') + ' in 2017 (modelled), a fall of ' + dec(r.poverty_drop_pp, L) + ' points. About ' + int(r.poor_people, L) + ' people live below the poverty line, out of ' + int(r.population_2022, L) + ' residents. ' + cap(x('banked')) + ' of adults are banked, ' + x('other_formal_only') + ' use other formal services only, ' + x('informal_only') + ' informal finance only and ' + x('excluded') + ' are excluded. Typology: ' + T.typ[r.typology] + '; priority rank ' + r.rank + ' of 30 with the current weights.',
      fr: r.district + ' (' + T.prov[r.province] + ')\u00a0: pauvreté de ' + x('poverty_2024') + ' en 2024, contre 27,4\u00a0% au niveau national\u00a0; elle était de ' + x('poverty_2017_modelled') + ' en 2017 (estimation modélisée), soit un recul de ' + dec(r.poverty_drop_pp, L) + ' points. Environ ' + int(r.poor_people, L) + ' personnes vivent sous le seuil de pauvreté, sur ' + int(r.population_2022, L) + ' habitants. ' + x('banked') + ' des adultes sont bancarisés, ' + x('other_formal_only') + " n'utilisent que d'autres services formels, " + x('informal_only') + " uniquement l'informel et " + x('excluded') + ' sont exclus. Typologie\u00a0: ' + T.typ[r.typology] + '\u00a0; rang de priorité ' + r.rank + ' sur 30 avec les poids actuels.',
      rw: 'Akarere ka ' + r.district + ' (' + T.prov[r.province] + '): ubukene bwari ' + x('poverty_2024') + " mu 2024, mu gihe mu gihugu ari 27,4%; mu 2017 bwari " + x('poverty_2017_modelled') + ' (igereranya), bwagabanutseho amanota ' + dec(r.poverty_drop_pp, L) + '. Abantu bagera kuri ' + int(r.poor_people, L) + " bari munsi y'umurongo w'ubukene, mu baturage " + int(r.population_2022, L) + '. ' + x('banked') + " by'abakuru bafite konti ya banki, " + x('other_formal_only') + ' bakoresha izindi serivisi zemewe gusa, ' + x('informal_only') + ' bakoresha imari itemewe gusa naho ' + x('excluded') + " ntibakoresha serivisi z'imari na zimwe. Ubwoko: " + T.typ[r.typology] + "; ku mwanya wa " + r.rank + " muri 30 ku rutonde rw'ibyihutirwa hakurikijwe uburemere buriho."
    });
  }
  function oneInd(L, r, k, c) {
    var v = val(k, get(r, k), L), n = rankOf(c.rows, r, k, false), nat = NAT[k] == null ? '' : pk(L, { en: '; national figure ', fr: '\u00a0; niveau national\u00a0: ', rw: '; mu gihugu ni ' }) + val(k, NAT[k], L);
    if (k === 'poverty_drop_pp') return pk(L, {
      en: 'In ' + r.district + ', poverty fell ' + dec(r.poverty_drop_pp, L) + ' points, from ' + pc(r.poverty_2017_modelled, L) + ' in 2017 (modelled) to ' + pc(r.poverty_2024, L) + ' in 2024: the ' + ord(n, L) + ' largest fall of 30 districts. Nationally it fell 12.4 points.',
      fr: 'À ' + r.district + ', la pauvreté a reculé de ' + dec(r.poverty_drop_pp, L) + ' points, de ' + pc(r.poverty_2017_modelled, L) + ' en 2017 (estimation modélisée) à ' + pc(r.poverty_2024, L) + ' en 2024\u00a0: le ' + ord(n, L) + ' plus fort recul des 30 districts. Au niveau national, elle a reculé de 12,4 points.',
      rw: 'Muri ' + r.district + ', ubukene bwagabanutseho amanota ' + dec(r.poverty_drop_pp, L) + ', buva kuri ' + pc(r.poverty_2017_modelled, L) + ' mu 2017 (igereranya) bugera kuri ' + pc(r.poverty_2024, L) + ' mu 2024: ni ku mwanya wa ' + n + ' mu turere 30 mu kugabanuka cyane. Mu gihugu bwagabanutseho amanota 12,4.' });
    return pk(L, {
      en: r.district + ': ' + LAB[k][L] + ' ' + v + ', the ' + ord(n, L) + ' highest of 30 districts' + nat + '.',
      fr: r.district + '\u00a0: ' + LAB[k][L] + ' ' + v + ', au ' + ord(n, L) + ' rang des 30 districts (du plus élevé au plus bas)' + nat + '.',
      rw: r.district + ': ' + LAB[k][L] + ' ni ' + v + ', ku mwanya wa ' + n + ' mu turere 30 (uhereye ku kiri hejuru)' + nat + '.' });
  }
  A.profile = function (L, o, c) {
    if (!o.districts.length) return o.province ? A.province(L, o, c) : A.rank(L, o, c);
    if (o.districts.length > 1) return A.compare(L, o, c);
    var r = c.by[o.districts[0]];
    if (o.ind && o.ind !== 'score') return oneInd(L, r, o.ind, c);
    if (o.finance) return pk(L, { en: 'Financial access in ' + r.district + ' (FinScope 2024): ', fr: 'Accès financier à ' + r.district + ' (FinScope 2024)\u00a0: ', rw: "Serivisi z'imari muri " + r.district + ' (FinScope 2024): ' }) +
      FIN.map(function (k) { return LAB[k][L] + ' ' + pc(r[k], L); }).join('; ') + '.' + natLine('banked', L);
    return fullProfile(L, r, c);
  };
  A.compare = function (L, o, c) {
    var ds = o.districts.slice(0, 4);
    if (ds.length < 2) return ds.length ? A.profile(L, o, c) : pk(L, { en: 'Name two districts to compare, for example: Compare Nyamagabe and Nyaruguru.', fr: 'Nommez deux districts à comparer, par exemple\u00a0: Comparez Nyamagabe et Nyaruguru.', rw: 'Vuga uturere tubiri two kugereranya, urugero: Gereranya Nyamagabe na Nyaruguru.' });
    var rs = ds.map(function (d) { return c.by[d]; });
    var ks = o.ind && !(o.finance && FIN.indexOf(o.ind) > -1 && o.ind === 'banked' && /access|acces|imari|financ/.test(o.n)) ? [o.ind] : o.finance ? FIN : ['poverty_2024', 'poor_people', 'poverty_drop_pp', 'banked', 'not_formally_served'];
    var lines = ks.map(function (k) { return cap(LAB[k][L]) + ': ' + rs.map(function (r) { return r.district + ' ' + val(k, get(r, k), L); }).join(', ') + '.'; });
    if (ks.indexOf('poverty_2024') > -1 && rs.length === 2) { var a = rs[0], b = rs[1], hi = a.poverty_2024 >= b.poverty_2024 ? a : b, d = r1(Math.abs(a.poverty_2024 - b.poverty_2024));
      if (d > 0) lines.push(pk(L, { en: hi.district + ' has the higher poverty rate, by ' + dec(d, L) + ' points.', fr: hi.district + ' a le taux de pauvreté le plus élevé, de ' + dec(d, L) + ' points.', rw: hi.district + ' ifite ubukene buri hejuru ho amanota ' + dec(d, L) + '.' })); }
    lines.push(pk(L, CAVEAT));
    return lines.join('\n');
  };
  A.province = function (L, o, c) {
    var T = c.T[L] || c.T.en, k = o.ind && FIN.concat(['unbanked', 'not_formally_served']).indexOf(o.ind) > -1 ? o.ind : null;
    var st = PROVS.map(function (p) { var rs = c.rows.filter(function (r) { return r.province === p; }), pop = 0, poor = 0, sum = 0;
      rs.forEach(function (r) { pop += r.population_2022; poor += r.poor_people; if (k) sum += r[k]; }); return { p: p, rs: rs, pop: pop, poor: poor, rate: poor / pop * 100, avg: k ? sum / rs.length : 0 }; });
    if (o.province) { var x = st.filter(function (q) { return q.p === o.province; })[0];
      var ds = x.rs.slice().sort(function (a, b) { return b.poverty_2024 - a.poverty_2024; }).map(function (r) { return r.district + ' ' + pc(r.poverty_2024, L); }).join(', ');
      return pk(L, { en: T.prov[x.p] + ': about ' + int(x.poor, L) + ' poor people out of ' + int(x.pop, L) + ' residents, a poverty rate of ' + pc(r1(x.rate), L) + ' (district rates weighted by 2022 population). Districts: ' + ds + '.',
        fr: T.prov[x.p] + '\u00a0: environ ' + int(x.poor, L) + ' pauvres sur ' + int(x.pop, L) + ' habitants, soit un taux de ' + pc(r1(x.rate), L) + ' (taux des districts pondérés par la population 2022). Districts\u00a0: ' + ds + '.',
        rw: T.prov[x.p] + ': abakene bagera kuri ' + int(x.poor, L) + ' mu baturage ' + int(x.pop, L) + ", ubukene bwa " + pc(r1(x.rate), L) + " (ibipimo by'uturere bipimwe n'umubare w'abaturage 2022). Uturere: " + ds + '.' }); }
    if (k) { st.sort(function (a, b) { return b.avg - a.avg; });
      return cap(LAB[k][L]) + ':\n' + st.map(function (q) { return T.prov[q.p] + ': ' + pc(r1(q.avg), L); }).join('\n') + '\n' + pk(L, { en: 'Simple average of district values.', fr: 'Moyenne simple des valeurs des districts.', rw: "Impuzandengo isanzwe y'imibare y'uturere." }); }
    st.sort(function (a, b) { return b.rate - a.rate; });
    return st.map(function (q) { return T.prov[q.p] + ': ' + pc(r1(q.rate), L) + ', ' + pk(L, { en: int(q.poor, L) + ' poor people', fr: int(q.poor, L) + ' pauvres', rw: 'abakene ' + int(q.poor, L) }); }).join('\n') + '\n' +
      pk(L, { en: 'Poverty rates are district rates weighted by 2022 population.', fr: 'Taux des districts pondérés par la population 2022.', rw: "Ibipimo by'ubukene ni iby'uturere bipimwe n'umubare w'abaturage 2022." });
  };
  A.typology = function (L, o, c) {
    var T = c.T[L] || c.T.en, mem = function (ty) { return c.rows.filter(function (r) { return r.typology === ty; }).map(function (r) { return r.district; }); };
    if (o.typ) { var m = mem(o.typ); return '"' + T.typ[o.typ] + '": ' + T.typNote[o.typ] + '. ' + pk(L, { en: m.length + ' districts: ', fr: m.length + ' districts\u00a0: ', rw: 'Uturere ' + m.length + ': ' }) + join(m, L) + '.'; }
    if (o.districts.length) return o.districts.map(function (d) { var r = c.by[d]; return pk(L, { en: d + ' is in the "' + T.typ[r.typology] + '" group: ', fr: d + ' appartient au groupe «\u00a0' + T.typ[r.typology] + '\u00a0»\u00a0: ', rw: d + ' iri mu itsinda "' + T.typ[r.typology] + '": ' }) + T.typNote[r.typology] + '.'; }).join('\n');
    return pk(L, { en: 'K-means clustering groups the districts into four types:', fr: 'Un k-means regroupe les districts en quatre types\u00a0:', rw: 'K-means ihuza uturere mu matsinda ane:' }) + '\n' +
      TYPS.map(function (ty) { var m = mem(ty); return T.typ[ty] + ' (' + m.length + '): ' + T.typNote[ty] + '. ' + join(m, L) + '.'; }).join('\n');
  };
  A.priority = function (L, o, c) {
    var T = c.T[L] || c.T.en;
    if (o.districts.length) return o.districts.slice(0, 3).map(function (d) {
      var r = c.by[d], tot = c.IND.reduce(function (a, i) { return a + c.W[i.key]; }, 0) || 1;
      var con = c.IND.map(function (i) { return { k: i.key, v: r['n_' + i.key] * c.W[i.key] / tot * 100 }; }).sort(function (a, b) { return b.v - a.v; });
      var t1 = T.ind[con[0].k].toLowerCase(), t2 = T.ind[con[1].k].toLowerCase(), rob = r.top5_robustness_pct, strong = rob >= 50;
      return pk(L, {
        en: d + ' ranks ' + r.rank + ' of 30 with a priority score of ' + dec(r.score, L, 1) + ' under the current weights. Most of its score comes from ' + t1 + ' (' + dec(con[0].v, L, 1) + ' points) and ' + t2 + ' (' + dec(con[1].v, L, 1) + ' points). It is in the top five in ' + pc(rob, L) + ' of 10,000 random weightings' + (strong ? ', so it is a priority almost whatever the weights.' : '.'),
        fr: d + ' est au ' + ord(r.rank, L) + ' rang sur 30, avec un score de priorité de ' + dec(r.score, L, 1) + ' selon les poids actuels. Son score vient surtout de\u00a0: ' + t1 + ' (' + dec(con[0].v, L, 1) + ' points) et ' + t2 + ' (' + dec(con[1].v, L, 1) + ' points). Il figure dans le top 5 pour ' + pc(rob, L) + ' des 10\u00a0000 pondérations aléatoires' + (strong ? '\u00a0: il est donc prioritaire presque quels que soient les poids.' : '.'),
        rw: 'Akarere ka ' + d + ' kari ku mwanya wa ' + r.rank + " muri 30, gafite amanota " + dec(r.score, L, 1) + " y'ibyihutirwa hakurikijwe uburemere buriho. Amanota yako aturuka cyane kuri " + t1 + ' (amanota ' + dec(con[0].v, L, 1) + ') na ' + t2 + ' (amanota ' + dec(con[1].v, L, 1) + "). Kaza mu 5 ba mbere muri " + pc(rob, L) + " by'ibipimo 10 000 byatoranyijwe ku buryo bw'impanuka" + (strong ? ", bityo karihutirwa hafi uko uburemere bwaba bumeze kose." : '.') }) + (c.ranges && c.ranges[d] ? ' ' + pk(L, {
        en: 'Allowing for survey sampling error, its likely rank is ' + c.ranges[d].lo + '–' + c.ranges[d].hi + '.',
        fr: 'Compte tenu de l\'erreur d\'échantillonnage, son rang probable est ' + c.ranges[d].lo + '–' + c.ranges[d].hi + '.',
        rw: "Hitawe ku makosa y'icyitegererezo, birashoboka ko kari ku mwanya wa " + c.ranges[d].lo + '–' + c.ranges[d].hi + '.' }) : '');
    }).join('\n');
    var s = c.rows.slice().sort(function (a, b) { return a.rank - b.rank; }).slice(0, o.top || 5);
    return pk(L, { en: 'Top ' + s.length + ' with the current weights:', fr: 'Top ' + s.length + ' avec les poids actuels\u00a0:', rw: 'Uturere ' + s.length + " twa mbere hakurikijwe uburemere buriho:" }) + '\n' +
      s.map(function (r) { return r.rank + '. ' + r.district + ' (' + dec(r.score, L, 1) + ')'; }).join('\n') + '\n' + T.robust + c.S.robust_top5.join(', ') + '.';
  };
  A.filter = function (L, o, c) {
    if (o.thr == null) return A.rank(L, o, c);
    var k = o.ind || 'poverty_2024', thr = o.thr, gt = o.cmp === 'gt';
    var m = c.rows.filter(function (r) { var v = get(r, k); return gt ? v > thr : v < thr; }).sort(function (a, b) { return gt ? get(b, k) - get(a, k) : get(a, k) - get(b, k); });
    var lim = cap(LAB[k][L]) + ' ' + pk(L, gt ? { en: 'above ', fr: 'supérieur à ', rw: 'hejuru ya ' } : { en: 'below ', fr: 'inférieur à ', rw: 'munsi ya ' }) + val(k, thr, L);
    if (!m.length) return lim + ': ' + pk(L, { en: 'no district.', fr: 'aucun district.', rw: 'nta karere na kamwe.' });
    return lim + ': ' + pk(L, { en: m.length + (m.length > 1 ? ' districts' : ' district'), fr: m.length + ' district' + (m.length > 1 ? 's' : ''), rw: 'uturere ' + m.length }) + '\n' +
      m.map(function (r) { return r.district + ' (' + val(k, get(r, k), L) + ')'; }).join(', ') + '.';
  };
  A.allocate = function (L, o, c) {
    var T = root.NtwTools; if (!T) return A.other(L);
    var people = o.thr && o.thr >= 100 ? o.thr : 50000, all = /all (30 )?districts|every district|twose|tous les districts/.test(o.n);
    var a = T.allocate(c.rows, { people: people, method: all ? 'all' : 'top', topN: o.top || 10 });
    var top = a.lines.slice(0, 6).map(function (l) { return l.district + ' ' + int(l.people, L); }).join(', ');
    var head = pk(L, { en: 'Sharing ' + int(people, L) + ' people ' + (all ? 'across all 30 districts (weighted by priority score and poor people)' : 'across the top ' + (o.top || 10) + ' priority districts, in proportion to their poor people') + ': ',
      fr: 'Répartition de ' + int(people, L) + ' personnes ' + (all ? 'entre les 30 districts (pondérée par score de priorité et nombre de pauvres)' : 'entre les ' + (o.top || 10) + ' districts prioritaires, au prorata de leurs pauvres') + '\u00a0: ',
      rw: 'Kugabanya abantu ' + int(people, L) + ' ' + (all ? "mu turere twose 30 (hakurikijwe amanota y'ibyihutirwa n'abakene)" : "mu turere " + (o.top || 10) + " tw'ibanze, hakurikijwe abakene baho") + ': ' });
    var miss = a.unplaced > 0 ? ' ' + pk(L, { en: int(a.unplaced, L) + ' could not be placed; include more districts.', fr: int(a.unplaced, L) + ' non répartis\u00a0; incluez plus de districts.', rw: 'Abantu ' + int(a.unplaced, L) + ' ntibabonewe umwanya; ongeramo utundi turere.' }) : '';
    return head + top + (a.lines.length > 6 ? ', …' : '') + '.' + miss + '\n' + pk(L, {
      en: 'Change the number, cost per person or method in "Plan coverage" above; it is a starting point, not a decision.',
      fr: 'Modifiez le nombre, le coût ou la méthode dans « Planifier la couverture »\u00a0; c\'est un point de départ, pas une décision.',
      rw: 'Hindura umubare, ikiguzi cyangwa uburyo muri "Gutegura abazagerwaho" haruguru; ni intangiriro, si icyemezo.' });
  };
  A.uncertainty = function (L, o, c) {
    var R = c.ranges || {};
    if (!Object.keys(R).length) return (c.T[L] || c.T.en).m4;
    if (o.districts.length) return o.districts.slice(0, 3).map(function (d) { var r = c.by[d], x = R[d]; return pk(L, {
      en: d + ' ranks ' + r.rank + ' with the current weights, but allowing for survey sampling error it lands between ' + x.lo + ' and ' + x.hi + ' in 90% of 1,000 simulations, and in the top 5 in ' + pc(Math.round(x.pTop5), L) + ' of them.',
      fr: d + ' est au rang ' + r.rank + ' avec les poids actuels, mais compte tenu de l\'erreur d\'échantillonnage il se situe entre ' + x.lo + ' et ' + x.hi + ' dans 90\u00a0% de 1\u00a0000 simulations, et dans le top 5 dans ' + pc(Math.round(x.pTop5), L) + ' d\'entre elles.',
      rw: d + ' iri ku mwanya wa ' + r.rank + " hakurikijwe uburemere buriho, ariko hitawe ku makosa y'icyitegererezo iza hagati y'umwanya wa " + x.lo + " n'uwa " + x.hi + " muri 90% by'igeragezwa 1 000, no mu 5 ba mbere muri " + pc(Math.round(x.pTop5), L) + ' byabyo.' }); }).join('\n');
    var top = c.rows.slice().sort(function (a, b) { return a.rank - b.rank; }).slice(0, 6);
    return pk(L, { en: 'Rank ranges once survey sampling error is allowed for (90% of 1,000 simulations):', fr: 'Rangs probables compte tenu de l\'erreur d\'échantillonnage (90\u00a0% de 1\u00a0000 simulations)\u00a0:', rw: "Imyanya ishoboka hitawe ku makosa y'icyitegererezo (90% by'igeragezwa 1 000):" }) + '\n' +
      top.map(function (r) { var x = R[r.district]; return r.rank + '. ' + r.district + ': ' + x.lo + '–' + x.hi; }).join('\n') + '\n' + pk(L, {
      en: 'The leading districts overlap heavily, so treat them as a priority group rather than a strict order.',
      fr: 'Les premiers districts se chevauchent fortement\u00a0: traitez-les comme un groupe prioritaire plutôt que comme un ordre strict.',
      rw: "Uturere twa mbere turasa cyane, bityo dufatwe nk'itsinda ry'ibyihutirwa aho kuba urutonde ndakuka." });
  };
  A.national = function (L, o, c) { var S = c.S; return pk(L, {
    en: "Rwanda's poverty rate was 27.4% in 2024, down from 39.8% in 2017 (modelled). About " + int(S.estimated_poor_people, L) + ' people live below the poverty line, out of ' + int(S.total_population_2022, L) + ' residents (2022 Census). FinScope 2024: 96% of adults are financially included, 92% formally served and 22% banked.',
    fr: 'Le taux de pauvreté du Rwanda était de 27,4\u00a0% en 2024, contre 39,8\u00a0% en 2017 (estimation modélisée). Environ ' + int(S.estimated_poor_people, L) + ' personnes vivent sous le seuil de pauvreté, sur ' + int(S.total_population_2022, L) + ' habitants (recensement 2022). FinScope 2024\u00a0: 96\u00a0% des adultes sont inclus financièrement, 92\u00a0% servis par le secteur formel et 22\u00a0% bancarisés.',
    rw: "Ubukene mu Rwanda bwari 27,4% mu 2024, buvuye kuri 39,8% mu 2017 (igereranya). Abantu bagera kuri " + int(S.estimated_poor_people, L) + " bari munsi y'umurongo w'ubukene, mu baturage " + int(S.total_population_2022, L) + " (Ibarura 2022). FinScope 2024: 96% by'abakuru bagerwaho na serivisi z'imari, 92% bakoresha serivisi zemewe naho 22% bafite konti ya banki." }); };
  A.correlation = function (L, o, c) { var C = c.S.correlations, b = C.poverty_vs_banked, n = C.poverty_vs_not_formally_served; return pk(L, {
    en: 'Across the 30 districts, poorer districts have fewer banked adults (Spearman rho = ' + b.spearman_rho + ', p = ' + b.p + ') and somewhat more adults outside formal finance (rho = ' + n.spearman_rho + ', p = ' + n.p + ', not significant at the 5% level). This shows an overlap, not a cause.',
    fr: 'Sur les 30 districts, les plus pauvres comptent moins d\'adultes bancarisés (rho de Spearman = ' + dec(b.spearman_rho, L, 2) + ', p = ' + dec(b.p, L, 3) + ') et un peu plus d\'adultes hors finance formelle (rho = ' + dec(n.spearman_rho, L, 2) + ', p = ' + dec(n.p, L, 3) + ', non significatif au seuil de 5\u00a0%). Cela montre un recoupement, pas une cause.',
    rw: "Mu turere 30, uturere dukennye dufite abakuru bake bafite konti ya banki (Spearman rho = " + dec(b.spearman_rho, L, 2) + ', p = ' + dec(b.p, L, 3) + ") n'abakuru benshi gato badakoresha serivisi zemewe (rho = " + dec(n.spearman_rho, L, 2) + ', p = ' + dec(n.p, L, 3) + ", ntibihagije ku rugero rwa 5%). Ibi byerekana ko bijyana, si uko kimwe gitera ikindi." }); };
  A.method = function (L, o, c) {
    var T = c.T[L] || c.T.en, s = o.n;
    if (/weight|poids|uburemere/.test(s)) return pk(L, { en: 'Default weights: ', fr: 'Poids par défaut\u00a0: ', rw: 'Uburemere busanzwe: ' }) +
      c.IND.map(function (i) { return T.ind[i.key] + ' ' + pc(Math.round(c.S.default_weights[i.key] * 100), L); }).join(', ') + '. ' + T.m1;
    if (/robust/.test(s)) return T.m2;
    if (/limit|reliab|fiab|imbogamizi|accura|precis|sample|echantillon/.test(s)) return T.m4;
    if (/model|predict|prevision|regress|forest/.test(s)) return T.m5;
    if (/source|data|donnee|amakuru|finscope|eicv|census|recensement|ibarura|nkomoko/.test(s)) return T.m6;
    if (/typolog|cluster|k means|group|groupe|itsinda/.test(s)) return T.m3;
    return T.m1;
  };
  A.brief = function (L, o, c) {
    if (!o.districts.length) return pk(L, { en: 'Name a district, for example: Write a briefing note on Gisagara.', fr: 'Nommez un district, par exemple\u00a0: Rédigez une note sur Gisagara.', rw: 'Vuga akarere, urugero: Andika incamake ku karere ka Gisagara.' });
    return brief(o.districts[0], L, c);
  };
  A.greet = function (L, o, c) {
    if (o.thanks) return pk(L, { en: "You're welcome! Ask another question any time.", fr: 'Avec plaisir\u00a0! Posez une autre question quand vous voulez.', rw: 'Murakoze namwe! Mushobora kubaza ikindi kibazo igihe cyose.' });
    var ex = (c.T[L] || c.T.en).qs[0];
    return pk(L, { en: 'Hello! I answer questions about poverty and financial inclusion in Rwanda\'s 30 districts, using only the figures in this tool. Try: "' + ex + '"',
      fr: 'Bonjour\u00a0! Je réponds aux questions sur la pauvreté et l\'inclusion financière dans les 30 districts du Rwanda, uniquement à partir des chiffres de cet outil. Essayez\u00a0: «\u00a0' + ex + '\u00a0»',
      rw: "Muraho! Nsubiza ibibazo ku bukene na serivisi z'imari mu turere 30 tw'u Rwanda, nkoresheje gusa imibare iri muri iki gikoresho. Gerageza: \"" + ex + '"' });
  };
  A.other = function (L) { return pk(L, {
    en: 'I can only answer questions about poverty, financial inclusion and the district figures in this tool: rankings, district profiles, comparisons, provinces, typologies and the method.',
    fr: "Je ne peux répondre qu'aux questions sur la pauvreté, l'inclusion financière et les chiffres de districts de cet outil\u00a0: classements, profils, comparaisons, provinces, typologies et méthode.",
    rw: "Nshobora gusubiza gusa ibibazo ku bukene, serivisi z'imari n'imibare y'uturere iri muri iki gikoresho: urutonde, imiterere y'akarere, kugereranya, intara, amatsinda n'uburyo bwakoreshejwe." }); };

  /* ---------- briefing note (data-to-text) ---------- */
  function options(r, L, c) { var T = c.T[L] || c.T.en, t = function (k, n) { return String(T[k]).split('{n}').join(n); }, o = [];
    if (r.poverty_drop_pp < 7) o.push(t('o1', dec(r.poverty_drop_pp, L)));
    if (r.not_formally_served >= 12) o.push(t('o2', r.not_formally_served));
    if (r.excluded >= 6) o.push(t('o3', r.excluded));
    if (r.banked <= 12 && r.poverty_2024 >= 30) o.push(t('o4', r.banked));
    if (r.poor_people >= 150000) o.push(t('o5', int(r.poor_people, L)));
    if (r.banked >= 40) o.push(t('o6', int(r.poor_people, L)));
    if (!o.length) o.push(T.o7); return o.slice(0, 3); }
  function brief(d, L, c) {
    var r = c.by[d]; if (!r) return A.other(L);
    var T = c.T[L] || c.T.en, rows = c.rows, x = function (k) { return pc(r[k], L); };
    var kp = rankOf(rows, r, 'poverty_2024'), kn = rankOf(rows, r, 'poor_people'), kb = rankOf(rows, r, 'banked', true), kf = rankOf(rows, r, 'not_formally_served');
    var gap = r1(r.poverty_2024 - NAT.poverty_2024), ag = dec(Math.abs(gap), L), dd = r.poverty_drop_pp - NAT.poverty_drop_pp;
    var cmp = Math.abs(gap) < 1.5 ? pk(L, { en: 'close to', fr: 'proche de', rw: "hafi y'" }) : gap > 0 ? pk(L, { en: ag + ' points above', fr: ag + ' points au-dessus de', rw: 'amanota ' + ag + " hejuru y'" }) : pk(L, { en: ag + ' points below', fr: ag + ' points en dessous de', rw: 'amanota ' + ag + " munsi y'" });
    var pace = Math.abs(dd) < 1.5 ? pk(L, { en: 'in line with', fr: 'au même rythme que', rw: "ku muvuduko ungana n'" }) : dd > 0 ? pk(L, { en: 'faster than', fr: 'plus vite que', rw: 'ku muvuduko uruta ' }) : pk(L, { en: 'slower than', fr: 'plus lentement que', rw: "ku muvuduko uri munsi y'" });
    var p1 = pk(L, {
      en: d + ' District (' + T.prov[r.province] + ') had a poverty rate of ' + x('poverty_2024') + ' in 2024, ' + cmp + ' the national 27.4%, the ' + ord(kp, L) + ' highest of 30 districts. Poverty fell ' + dec(r.poverty_drop_pp, L) + ' points from ' + x('poverty_2017_modelled') + ' in 2017 (modelled), ' + pace + ' the national fall of 12.4 points. With ' + int(r.population_2022, L) + ' residents in 2022, about ' + int(r.poor_people, L) + ' people live below the poverty line, the ' + ord(kn, L) + ' largest number of any district.',
      fr: 'Le district de ' + d + ' (' + T.prov[r.province] + ') avait un taux de pauvreté de ' + x('poverty_2024') + ' en 2024, ' + cmp + ' la moyenne nationale de 27,4\u00a0%, au ' + ord(kp, L) + ' rang des 30 districts. La pauvreté a reculé de ' + dec(r.poverty_drop_pp, L) + ' points depuis 2017 (' + x('poverty_2017_modelled') + ', estimation modélisée), ' + pace + ' le recul national de 12,4 points. Avec ' + int(r.population_2022, L) + ' habitants en 2022, environ ' + int(r.poor_people, L) + ' personnes vivent sous le seuil de pauvreté, ' + (kn === 1 ? 'le nombre le plus élevé' : 'le ' + ord(kn, L) + ' nombre le plus élevé') + ' des districts.',
      rw: 'Mu 2024, akarere ka ' + d + ' (' + T.prov[r.province] + ') kari gafite ubukene bwa ' + x('poverty_2024') + ', ' + cmp + "impuzandengo y'igihugu ya 27,4%, kakaba ku mwanya wa " + kp + ' mu turere 30 dukennye cyane. Kuva 2017 ubukene bwagabanutseho amanota ' + dec(r.poverty_drop_pp, L) + ' (buvuye kuri ' + x('poverty_2017_modelled') + ', igereranya), ' + pace + "igabanuka ry'igihugu ry'amanota 12,4. Mu baturage " + int(r.population_2022, L) + ' bo mu 2022, abagera kuri ' + int(r.poor_people, L) + " bari munsi y'umurongo w'ubukene, akarere kakaba ku mwanya wa " + kn + ' mu kugira abakene benshi.' });
    var so = r.banked >= 40 ? pk(L, { en: 'Bank-led inclusion stands out: the share of banked adults is among the highest in the country.', fr: "L'inclusion portée par les banques se distingue\u00a0: la part d'adultes bancarisés est parmi les plus élevées du pays.", rw: "Icyihariye ni uko serivisi z'imari zishingiye kuri banki: abafite konti ya banki ni bamwe mu benshi mu gihugu." })
      : kb <= 6 ? pk(L, { en: 'Bank use is in the lowest fifth of districts.', fr: 'La bancarisation est dans le cinquième le plus faible des districts.', rw: 'Abafite konti ya banki ni bake: akarere kari mu gatanu ka nyuma mu turere twose.' })
      : kf <= 6 ? pk(L, { en: 'The share of adults outside formal finance (' + x('not_formally_served') + ') is in the highest fifth of districts.', fr: "La part d'adultes hors finance formelle (" + x('not_formally_served') + ') est dans le cinquième le plus élevé des districts.', rw: "Abakuru badakoresha serivisi zemewe (" + x('not_formally_served') + ') ni benshi: akarere kari mu gatanu ka mbere mu turere twose.' })
      : pk(L, { en: 'Formal access sits near the middle of the district range and rests mostly on mobile money and SACCOs.', fr: "L'accès formel se situe dans la moyenne des districts et repose surtout sur le mobile money et les SACCO.", rw: "Kugerwaho na serivisi zemewe biri hagati ugereranyije n'utundi turere, kandi bishingiye cyane kuri mobile money na SACCO." });
    var p2 = pk(L, {
      en: 'FinScope 2024 shows that ' + x('banked') + ' of adults in ' + d + ' have a bank account, against 22% nationally. ' + cap(x('other_formal_only')) + ' use other formal services only, mostly mobile money and Umurenge SACCOs, ' + x('informal_only') + ' rely only on informal finance and ' + x('excluded') + ' are fully excluded. ' + so,
      fr: 'Selon FinScope 2024, ' + x('banked') + ' des adultes de ' + d + ' ont un compte bancaire, contre 22\u00a0% au niveau national. ' + x('other_formal_only') + " n'utilisent que d'autres services formels, surtout le mobile money et les SACCO Umurenge, " + x('informal_only') + " uniquement l'informel et " + x('excluded') + ' sont totalement exclus. ' + so,
      rw: "Nk'uko FinScope 2024 ibigaragaza, " + x('banked') + " by'abakuru bo muri " + d + ' bafite konti ya banki, mu gihe mu gihugu ari 22%. ' + x('other_formal_only') + ' bakoresha izindi serivisi zemewe gusa, cyane cyane mobile money na Umurenge SACCO, ' + x('informal_only') + ' bakoresha imari itemewe gusa, naho ' + x('excluded') + " ntibakoresha serivisi z'imari na zimwe. " + so });
    var p3 = pk(L, { en: 'For planning: ', fr: 'Pour la planification\u00a0: ', rw: 'Mu igenamigambi: ' }) + options(r, L, c).join(' ') + ' ' + pk(L, {
      en: 'On the priority index it ranks ' + r.rank + ' of 30 with the current weights' + (c.ranges && c.ranges[d] ? ' (likely ' + c.ranges[d].lo + '–' + c.ranges[d].hi + ' once survey sampling error is allowed for)' : '') + ', and it is in the top five in ' + x('top5_robustness_pct') + ' of 10,000 random weightings. These are options to discuss, not tested solutions.',
      fr: "Sur l'indice de priorité, il est au " + ord(r.rank, L) + ' rang sur 30 avec les poids actuels' + (c.ranges && c.ranges[d] ? ' (rang probable ' + c.ranges[d].lo + '–' + c.ranges[d].hi + ', erreur d\'échantillonnage comprise)' : '') + ', et figure dans le top 5 pour ' + x('top5_robustness_pct') + ' des 10\u00a0000 pondérations aléatoires. Ce sont des pistes de discussion, pas des solutions éprouvées.',
      rw: "Ku rutonde rw'ibyihutirwa kari ku mwanya wa " + r.rank + ' muri 30 hakurikijwe uburemere buriho' + (c.ranges && c.ranges[d] ? " (birashoboka hagati ya " + c.ranges[d].lo + '–' + c.ranges[d].hi + " hitawe ku makosa y'icyitegererezo)" : '') + ', kandi kaza mu 5 ba mbere muri ' + x('top5_robustness_pct') + " by'ibipimo 10 000 byatoranyijwe ku buryo bw'impanuka. Ibi ni ibitekerezo byo kuganiraho, si ibisubizo byageragejwe." });
    var src = pk(L, { en: 'Sources: NISR EICV7 (2023/24), FinScope Rwanda 2024, Population and Housing Census 2022.', fr: 'Sources\u00a0: NISR EICV7 (2023/24), FinScope Rwanda 2024, Recensement de la population 2022.', rw: 'Inkomoko: NISR EICV7 (2023/24), FinScope Rwanda 2024, Ibarura rusange 2022.' });
    return [p1, p2, p3, src].join('\n\n');
  }

  /* ---------- router ---------- */
  var ctxMem = null;
  function prep(c) { c.by = {}; c.rows.forEach(function (r) { c.by[r.district] = r; }); return c; }
  function answer(q, c) {
    prep(c);
    var o = slots(q, c.rows), n = normQ(q), words = n.split(' ').length;
    var lp = predict(NBL, q), L = (lp.p >= 0.6 && words > 1) || lp.p >= 0.9 ? lp.label : c.lang;
    var pi = predict(NBI, q), it = pi.label;
    var marker = /^(and|what about|how about|et|et pour|et a|et en|na|naho|ese na|none se)\b/.test(n);
    var follow = ctxMem && (marker || (words <= 3 && pi.p < 0.8 && !(it === 'other' && pi.p >= 0.5))) && (o.districts.length || o.province || o.ind);
    if (follow) { it = ctxMem.it; if (!o.ind && ctxMem.o.ind && !(o.districts.length && it === 'profile')) o.ind = ctxMem.o.ind; if (o.asc === false && ctxMem.o.asc && !o.ind) o.asc = true; }
    else if (it !== 'greet' && it !== 'other' && !o.domain) it = 'other';
    // rules that sharpen the classifier
    // planning tools (rules; work in all three languages)
    if (/allocat|budget|distribut|split|share (them|people|beneficiar)|support \d|reach \d|beneficiar|gabany|ingengo|repart|benefici/.test(n) &&
        (o.thr != null || /people|beneficiar|abantu|personnes|households|ingo|menages/.test(n))) it = 'allocate';
    else if (it !== 'other' || o.domain) {
      if (/how (sure|certain|confident|reliable)|certain|uncertain|rank range|likely rank|sampling|margin|kwizerwa|gushidikanya|incertitude|marge|fiab|precis/.test(n) && it !== 'method') it = 'uncertainty';
    }
    if (['brief', 'other', 'greet', 'allocate', 'uncertainty'].indexOf(it) < 0) {
      if (/correl|linked|link |relationship|lien|isano|go with|bijyana/.test(' ' + n + ' ') || (!/which|quel|utuhe|akahe|where|hehe|ou /.test(n) && /(poor|pauvr|kenn).*(less|fewer|moins|bake|nke)/.test(n) && o.finance)) it = 'correlation';
      else if ((o.districts.length === 1 || (o.districts.length && it !== 'compare')) && /rank|ranked|number one|first|top|priorit|mwanya|rang|classement|ibyihutirwa|mbere/.test(n)) it = 'priority';
      else if (o.districts.length && /typolog|type|group|groupe|cluster|itsinda|bwoko/.test(n)) it = 'typology';
      else if (o.districts.length >= 2 && it !== 'priority' && it !== 'typology') it = 'compare';
      else if (o.districts.length === 1 && (it === 'rank' || it === 'filter' || it === 'national' || it === 'province')) it = 'profile';
      else if (!o.districts.length && o.province && it !== 'method') it = 'province';
      else if (it === 'filter' && o.thr == null) it = 'rank';
      else if (it === 'rank' && o.thr != null && o.thr !== o.top && /above|over|more than|greater|below|under|less than|plus de|moins de|superieur|inferieur|renze|renga|munsi/.test(n)) it = 'filter';
    }
    var text = (A[it] || A.other)(L, o, c);
    if (it !== 'other' && it !== 'greet') ctxMem = { it: it, o: o };
    return { text: text, lang: L, intent: it };
  }
  root.NtwLocalAI = { answer: answer, brief: function (d, L, c) { return brief(d, L, prep(c)); } };
})(typeof self !== 'undefined' ? self : this);
