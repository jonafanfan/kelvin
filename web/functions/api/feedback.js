// Saves one "How did we do?" answer. The page sends it here with
// POST /api/feedback. This runs on Cloudflare (a "Pages Function"), and
// env.DB is the feedback database set up in wrangler.jsonc.

const RATINGS = ["great", "okay", "bad"];
const MAX_COMMENT = 2000;

export async function onRequestPost({ request, env }) {
  let body;
  try {
    body = await request.json();
  } catch {
    return new Response("Expected JSON", { status: 400 });
  }
  const { token, rating, comment, aiSliders, finalSliders } = body;
  if (!RATINGS.includes(rating)) return new Response("Unknown rating", { status: 400 });
  const text = typeof comment === "string" ? comment.trim().slice(0, MAX_COMMENT) || null : null;
  const sliders = (values) => (values && typeof values === "object" ? JSON.stringify(values).slice(0, 1000) : null);

  // A first answer creates a row and gets a random key back. Adding a comment
  // (or changing the rating) afterwards sends that key to update the same row.
  // The key is random, so nobody can guess someone else's and overwrite it.
  if (typeof token === "string") {
    await env.DB.prepare("UPDATE feedback SET rating = ?, comment = COALESCE(?, comment) WHERE token = ?")
      .bind(rating, text, token)
      .run();
    return Response.json({ token });
  }
  const newToken = crypto.randomUUID();
  await env.DB.prepare("INSERT INTO feedback (token, rating, comment, ai_sliders, final_sliders) VALUES (?, ?, ?, ?, ?)")
    .bind(newToken, rating, text, sliders(aiSliders), sliders(finalSliders))
    .run();
  return Response.json({ token: newToken });
}
