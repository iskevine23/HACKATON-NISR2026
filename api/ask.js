// Serverless endpoint for the Ntawusigara AI assistant (Vercel-style handler).
// The API key stays on the server. The prompt and data are built here, so the
// endpoint only answers questions about this tool's data and cannot be used as
// a general chatbot.
//
// Environment variables (set them in your host's settings, never in code):
//   GEMINI_API_KEY     your Google AI Studio key  -> uses Gemini
//   ANTHROPIC_API_KEY  alternative                -> uses Claude
//   NTW_MODEL          optional model name (default gemini-2.5-flash or claude-haiku-4-5-20251001)
//   NTW_ALLOWED_ORIGIN optional, e.g. https://yourname.github.io
const ctx = require("../data/processed/ai_context.json");
const { build, LANG, MAX_Q } = require("../app/ai-prompt.js");

const DISTRICTS = new Set(ctx.rows.map((r) => r[0]));
const hits = new Map(); // simple per-instance limit: 20 requests per 10 minutes per IP

async function gemini(prompt) {
  const model = process.env.NTW_MODEL || "gemini-2.5-flash";
  const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-goog-api-key": process.env.GEMINI_API_KEY },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { maxOutputTokens: 800, temperature: 0.3 },
    }),
  });
  if (r.status === 429) return { code: "rate_limited" };
  if (!r.ok) return { code: "upstream_error" };
  const data = await r.json();
  const cand = (data.candidates || [])[0];
  if (!cand) return { code: "refused" };
  const text = ((cand.content && cand.content.parts) || []).map((p) => p.text || "").join("").trim();
  return text ? { text } : { code: cand.finishReason === "SAFETY" ? "refused" : "empty_completion" };
}

async function claude(prompt) {
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": process.env.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01" },
    body: JSON.stringify({ model: process.env.NTW_MODEL || "claude-haiku-4-5-20251001", max_tokens: 600, messages: [{ role: "user", content: prompt }] }),
  });
  if (r.status === 429) return { code: "rate_limited" };
  if (!r.ok) return { code: "upstream_error" };
  const data = await r.json();
  const text = (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join("\n").trim();
  return text ? { text } : { code: "empty_completion" };
}

module.exports = async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", process.env.NTW_ALLOWED_ORIGIN || "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).json({ code: "invalid_request" });
  if (!process.env.GEMINI_API_KEY && !process.env.ANTHROPIC_API_KEY) return res.status(503).json({ code: "not_configured" });

  const ip = (req.headers["x-forwarded-for"] || "").split(",")[0] || "unknown";
  const now = Date.now(), list = (hits.get(ip) || []).filter((t) => now - t < 600000);
  if (list.length >= 20) return res.status(429).json({ code: "rate_limited" });
  list.push(now); hits.set(ip, list);

  let body = req.body || {};
  if (typeof body === "string") { try { body = JSON.parse(body || "{}"); } catch (e) { return res.status(400).json({ code: "invalid_request" }); } }
  const mode = body.mode === "brief" ? "brief" : "ask";
  const lang = LANG[body.lang] ? body.lang : "en";
  if (mode === "brief" && !DISTRICTS.has(body.district)) return res.status(400).json({ code: "invalid_request" });
  if (mode === "ask" && (!body.question || String(body.question).length > MAX_Q)) return res.status(400).json({ code: "invalid_request" });

  const prompt = build({ mode, lang, question: body.question, district: body.district, ctx });
  try {
    const out = process.env.GEMINI_API_KEY ? await gemini(prompt) : await claude(prompt);
    if (out.text) return res.status(200).json({ text: out.text });
    return res.status(out.code === "rate_limited" ? 429 : 502).json({ code: out.code });
  } catch (e) {
    return res.status(502).json({ code: "upstream_error" });
  }
};
