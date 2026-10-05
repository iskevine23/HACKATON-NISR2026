/* Shared prompt builder for the Ntawusigara AI assistant.
   Used in the browser (window.NtwPrompt) and by api/ask.js (require). Keeping one
   copy means the assistant follows the same rules wherever it runs. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.NtwPrompt = factory();
})(typeof self !== "undefined" ? self : this, function () {
  var LANG = { rw: "Kinyarwanda", en: "English", fr: "French" };
  var MAX_Q = 500;

  function table(ctx) {
    return [ctx.columns.join(",")].concat(ctx.rows.map(function (r) { return r.join(","); })).join("\n");
  }

  function rules(lang) {
    return [
      "You are the assistant inside Ntawusigara, a district targeting tool for poverty reduction and financial inclusion in Rwanda, built for the NISR 2026 Big Data Hackathon.",
      "RULES:",
      "- Use ONLY the DATA below. Never invent figures, districts, programmes or causes.",
      "- If the data cannot answer, say so plainly and point to the relevant NISR report (EICV7, FinScope 2024, Census 2022).",
      "- You may do simple arithmetic on the data (sums, differences, ratios) and show the numbers used.",
      "- Describe differences and overlaps, not causes. District figures come from samples of about 400 to 500, so small gaps may not be meaningful.",
      "- Columns: poverty in % of people (2017 is NISR's modelled estimate); banked, other_formal_only (mostly mobile money and SACCOs), informal_only and excluded are % of adults 16+ from FinScope 2024 and sum to about 100; poor_people = poverty rate x 2022 population; priority_rank uses the tool's default weights; top5_robustness_pct = share of 10,000 random weightings placing the district in the top 5.",
      "- Write in " + LANG[lang] + " unless the user clearly writes in another of Kinyarwanda, English or French; then reply in that language.",
      "- Plain text only: no markdown, no tables, no bullet symbols.",
      "- Politely decline anything unrelated to poverty, financial inclusion or these districts."
    ].join("\n");
  }

  function build(opts) {
    var lang = LANG[opts.lang] ? opts.lang : "en";
    var ctx = opts.ctx;
    var head = rules(lang) + "\n\nNATIONAL: poverty 27.4% in 2024 (39.8% modelled for 2017); 96% of adults financially included, 92% formally served, 22% banked; estimated poor people " +
      ctx.summary.estimated_poor_people + ".\n\nDATA (CSV):\n" + table(ctx) + "\n\n";
    if (opts.mode === "brief") {
      var d = String(opts.district || "").slice(0, 40);
      return head + "TASK: Write a briefing note on " + d + " District for district officials and SACCO managers. Three short paragraphs, about 160 words in total: (1) where the district stands on poverty and its trend since 2017, compared with the national figures; (2) how adults access finance, and what stands out; (3) two or three practical points for planning, grounded only in the data and clearly worded as options, not proven solutions. End with one line naming the sources.";
    }
    var q = String(opts.question || "").slice(0, MAX_Q);
    return head + "QUESTION:\n" + q + "\n\nAnswer in 2 to 5 short sentences.";
  }

  return { build: build, LANG: LANG, MAX_Q: MAX_Q };
});
