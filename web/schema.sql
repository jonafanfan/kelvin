-- The feedback table. Set it up (once) with:  npm run setup-db
-- One row per "How did we do?" answer. No photos: only the rating, the
-- optional comment, and the slider values, which show how far the person
-- moved away from the AI's choices.
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  token TEXT NOT NULL UNIQUE, -- a random key, so only the same page can add a comment later
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  rating TEXT NOT NULL,      -- "great", "okay" or "bad"
  comment TEXT,              -- what's good or bad, if they wrote anything
  ai_sliders TEXT,           -- the AI's slider values, as JSON
  final_sliders TEXT         -- the slider values when they rated it, as JSON
);
