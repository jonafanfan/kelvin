// Kelvin's web app: the browser version of app.py. Everything happens on the
// visitor's device: the editor (editor.js), the AI (ai.js) and the histogram
// (histogram.js). No photo is ever uploaded anywhere.

import { SLIDERS, applyEdits, neutralParams } from "./editor.js";
import { aiSuggest, loadAI } from "./ai.js";
import { drawHistogram } from "./histogram.js";

// Big phone photos have 12+ million pixels. Editing a smaller copy on screen
// keeps the sliders smooth; Save applies the same edits to the full photo.
const PREVIEW_MAX_SIDE = 1024;
// While a slider is being dragged, redraw a half-size copy (a quarter of the
// pixels, so about 4 times faster), then the full preview once it stops.
const QUICK_SCALE = 0.5;
const FULL_REDRAW_AFTER = 150; // ms without slider movement

const $ = (id) => document.getElementById(id);
const state = {
  photo: null, // the opened photo, full size (an ImageBitmap)
  name: "",
  preview: null, // the shrunk original as r,g,b,r,g,b... from 0 to 1
  input: null, // the same with a whisper of noise added: what actually gets edited (see dither)
  quick: null, // { input, canvas }: the half-size copy for redrawing while dragging
  ai: null, // { params, reasons }: the AI's suggestion at full strength, once Auto has run
  edited: null,
  values: neutralParams(),
};

// ---------------------------------------------------------------------------
// Sliders
// ---------------------------------------------------------------------------

const rows = {};
for (const s of SLIDERS) {
  const row = document.createElement("div");
  row.className = "k-slider";
  row.innerHTML = `
    <div class="k-slider-top"><label for="slider-${s.name}">${s.label}</label><output>0.00</output></div>
    <p>${s.help}</p>
    <input type="range" id="slider-${s.name}" min="-1" max="1" step="0.01" value="0" disabled>`;
  const input = row.querySelector("input");
  input.addEventListener("input", () => {
    setSlider(s.name, Number(input.value), "you");
    scheduleRender();
  });
  // Double-click (or double-tap) the slider to put it back to 0
  input.addEventListener("dblclick", () => {
    setSlider(s.name, 0, "you");
    scheduleRender();
  });
  $("sliders").append(row);
  rows[s.name] = row;
}

// who: "you" or "ai". The fill is orange for your changes and blue for the AI's.
function setSlider(name, value, who) {
  state.values[name] = value;
  const row = rows[name];
  const input = row.querySelector("input");
  input.value = value;
  const at = `${((value + 1) / 2) * 100}%`;
  input.style.setProperty("--from", value < 0 ? at : "50%");
  input.style.setProperty("--to", value > 0 ? at : "50%");
  row.querySelector("output").textContent = (value > 0 ? "+" : "") + value.toFixed(2);
  row.classList.toggle("by-ai", who === "ai");
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

// Sliders fire many times a second; redraw at most once per screen refresh,
// using the quick half-size copy, and in full detail once they stop.
let pending = false;
let fullRedraw = null;
function scheduleRender() {
  if (!state.preview) return;
  clearTimeout(fullRedraw);
  fullRedraw = setTimeout(render, FULL_REDRAW_AFTER);
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    render({ quick: true });
  });
}

function render({ quick = false } = {}) {
  if (readyToShare) {
    readyToShare = null; // the photo has changed, so the one waiting to be saved is out of date
    $("save").textContent = "Save";
  }
  if (quick) {
    const edited = applyEdits(state.quick.input, state.values);
    putPixels(state.quick.canvas, edited);
    $("after").getContext("2d").drawImage(state.quick.canvas, 0, 0, $("after").width, $("after").height);
    drawHistogram($("histogram"), edited, state.preview);
    return;
  }
  clearTimeout(fullRedraw);
  state.edited = applyEdits(state.input, state.values);
  putPixels($("after"), state.edited);
  drawHistogram($("histogram"), state.edited, state.preview);
}

// ---------------------------------------------------------------------------
// Dithering: a whisper of random noise that hides "banding"
// ---------------------------------------------------------------------------
// Photos store 256 brightness levels per colour. When an edit stretches a
// smooth sky, neighbouring levels get pulled apart and the sky shows steps,
// like contour lines. Adding noise smaller than one level *before* editing
// scatters those steps into fine grain your eye doesn't see; rounding the
// result randomly up or down (instead of always to the nearest) does the same
// on the way out. Random numbers are slow to make, so we make a batch once.
const NOISE = Float32Array.from({ length: 65536 }, Math.random);
let noiseAt = 0;
const noise = () => NOISE[(noiseAt = (noiseAt + 1) & 65535)];

function dither(pixels) {
  const out = new Float32Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    const v = pixels[i] + (noise() - 0.5) / 255;
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

// Edited r,g,b floats -> RGBA bytes, rounding randomly up or down (see above)
function toBytes(pixels, rgba) {
  for (let i = 0, j = 0; i < pixels.length; i += 3, j += 4) {
    rgba[j] = pixels[i] * 255 + noise() - 0.5;
    rgba[j + 1] = pixels[i + 1] * 255 + noise() - 0.5;
    rgba[j + 2] = pixels[i + 2] * 255 + noise() - 0.5;
    rgba[j + 3] = 255;
  }
}

function putPixels(canvas, pixels) {
  const image = canvas.getContext("2d").createImageData(canvas.width, canvas.height);
  toBytes(pixels, image.data);
  canvas.getContext("2d").putImageData(image, 0, 0);
}

// RGBA bytes from the browser -> r,g,b floats from 0 to 1, as the editor wants
function toFloats(rgba) {
  const out = new Float32Array((rgba.length / 4) * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    out[j] = rgba[i] / 255;
    out[j + 1] = rgba[i + 1] / 255;
    out[j + 2] = rgba[i + 2] / 255;
  }
  return out;
}

function showReasons(rows) {
  let body;
  if (rows === undefined) body = "<p>Press <b>Auto</b> and Kelvin will set the sliders, and explain each change here.</p>";
  else if (!rows.length) body = "<p>Nothing needed fixing: this photo already looks balanced.</p>";
  else
    body = rows
      .map(([label, value, reason]) => `<div class="k-reason"><b>${label}</b><code>${value > 0 ? "+" : ""}${value.toFixed(2)}</code><span>${reason}</span></div>`)
      .join("");
  $("reasons").innerHTML = `<h3>What was changed</h3>${body}`;
}

// ---------------------------------------------------------------------------
// Opening a photo
// ---------------------------------------------------------------------------

async function openPhoto(file) {
  if (!file || !file.type.startsWith("image/")) return;
  let photo;
  try {
    photo = await createImageBitmap(file, { imageOrientation: "from-image" }); // phones store photos sideways plus a "rotate me" note
  } catch {
    alert("Kelvin couldn't open that file. Try a JPEG or PNG photo.");
    return;
  }
  state.photo = photo;
  state.name = file.name;
  $("filename").textContent = file.name;

  const scale = Math.min(1, PREVIEW_MAX_SIDE / Math.max(photo.width, photo.height));
  const width = Math.round(photo.width * scale), height = Math.round(photo.height * scale);
  for (const id of ["before", "after"]) {
    $(id).width = width;
    $(id).height = height;
  }
  const ctx = $("before").getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(photo, 0, 0, width, height);
  state.preview = toFloats(ctx.getImageData(0, 0, width, height).data);
  state.input = dither(state.preview);
  const quick = document.createElement("canvas");
  quick.width = Math.max(1, Math.round(width * QUICK_SCALE));
  quick.height = Math.max(1, Math.round(height * QUICK_SCALE));
  const quickCtx = quick.getContext("2d");
  quickCtx.imageSmoothingQuality = "high";
  quickCtx.drawImage(photo, 0, 0, quick.width, quick.height);
  state.quick = { canvas: quick, input: dither(toFloats(quickCtx.getImageData(0, 0, quick.width, quick.height).data)) };

  $("welcome").hidden = true;
  $("editor").hidden = false;
  $("compare").hidden = false;
  // Pulse the Auto button a few times, to invite a first tap
  $("auto").classList.add("k-pulse");
  closeFeedback(); // a new photo: feedback only makes sense once Auto has run on it
  for (const id of ["auto", "reset", "save"]) $(id).disabled = false;
  for (const input of document.querySelectorAll("input[type=range]")) input.disabled = false;
  reset();
}

function reset() {
  for (const s of SLIDERS) setSlider(s.name, 0, "you");
  state.ai = null;
  $("strength-row").hidden = true;
  showReasons();
  render();
}

// ---------------------------------------------------------------------------
// Auto strength: how much of the AI's edit to use
// ---------------------------------------------------------------------------
// The AI learned on flat, unprocessed camera files, and tends to push hard.
// Phone photos are already processed, so the full edit is often too much.
// 70% is the default; the person's own choice is remembered for next time.

const DEFAULT_STRENGTH = 0.7;
let strength = DEFAULT_STRENGTH;
try {
  strength = Number(localStorage.getItem("kelvin-strength") ?? DEFAULT_STRENGTH);
} catch {} // private browsing can block storage: just use the default
if (!(strength >= 0 && strength <= 1)) strength = DEFAULT_STRENGTH;

function showStrength() {
  $("strength").value = strength;
  $("strength").style.setProperty("--to", `${strength * 100}%`);
  $("strength-value").textContent = `${Math.round(strength * 100)}%`;
}

// Scale the AI's sliders by the strength. Sliders you've moved yourself
// (orange) are yours, so they stay as you left them.
function applyAI({ fresh = false } = {}) {
  const { params, reasons } = state.ai;
  for (const s of SLIDERS) {
    if (!fresh && !rows[s.name].classList.contains("by-ai")) continue;
    const value = Math.round(params[s.name] * strength * 100) / 100;
    setSlider(s.name, value, params[s.name] ? "ai" : "you");
  }
  showReasons(SLIDERS.filter((s) => s.name in reasons).map((s) => [s.label, state.values[s.name], reasons[s.name]]));
}

$("strength").addEventListener("input", () => {
  strength = Number($("strength").value);
  showStrength();
  try {
    localStorage.setItem("kelvin-strength", strength);
  } catch {}
  applyAI();
  scheduleRender();
});

$("open").addEventListener("click", () => $("file").click());
$("welcome-open").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => openPhoto(e.target.files[0]));
$("reset").addEventListener("click", reset);

// Drag and drop a photo anywhere on the page
addEventListener("dragover", (e) => {
  e.preventDefault();
  document.body.classList.add("dragging");
});
addEventListener("dragleave", () => document.body.classList.remove("dragging"));
addEventListener("drop", (e) => {
  e.preventDefault();
  document.body.classList.remove("dragging");
  openPhoto(e.dataTransfer.files[0]);
});

// ---------------------------------------------------------------------------
// Auto: the AI sets the sliders
// ---------------------------------------------------------------------------

$("auto").addEventListener("click", async () => {
  const button = $("auto");
  button.classList.remove("k-pulse");
  button.disabled = true;
  button.textContent = "Thinking…";
  try {
    state.ai = await aiSuggest(state.photo);
    showStrength();
    $("strength-row").hidden = false;
    applyAI({ fresh: true });
    askForFeedback(state.ai.params); // the AI's own choice, at full strength
    render();
  } catch (error) {
    console.error(error);
    alert("The AI couldn't load. Check your connection and try again.");
  } finally {
    button.disabled = false;
    button.textContent = "Auto";
  }
});

// ---------------------------------------------------------------------------
// "How did we do?": optional feedback after Auto. Picking a rating sends it
// straight away; "Tell us more" adds a comment to the same answer. Only the
// rating, the comment and the slider values are sent: never the photo.
// See functions/api/feedback.js for where it's saved.
// ---------------------------------------------------------------------------

const FEEDBACK_DELAY = 3000; // ms after Auto: time to look at the result first
const feedback = { token: null, rating: null, aiSliders: null, askedFor: null, timer: null };

function askForFeedback(aiSliders) {
  if (feedback.askedFor === state.photo) return; // once per photo, so pressing Auto again doesn't nag
  feedback.askedFor = state.photo;
  Object.assign(feedback, { token: null, rating: null, aiSliders: { ...aiSliders } });
  for (const b of document.querySelectorAll(".k-rating button")) b.setAttribute("aria-pressed", "false");
  $("feedback-note").hidden = true;
  $("feedback-more").hidden = true;
  $("feedback-comment").value = "";
  feedback.timer = setTimeout(() => {
    $("feedback").hidden = false;
    requestAnimationFrame(() => $("feedback").classList.add("k-show")); // slide up into view
  }, FEEDBACK_DELAY);
}

function closeFeedback() {
  clearTimeout(feedback.timer);
  $("feedback").classList.remove("k-show");
  $("feedback").hidden = true;
}

$("feedback-close").addEventListener("click", closeFeedback);
addEventListener("keydown", (e) => e.key === "Escape" && closeFeedback());

async function sendFeedback(extra) {
  const response = await fetch("api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: feedback.token, rating: feedback.rating, ...extra }),
  });
  if (!response.ok) throw new Error(`Feedback failed: ${response.status}`);
  feedback.token = (await response.json()).token;
}

function note(html) {
  $("feedback-note").innerHTML = html;
  $("feedback-note").hidden = false;
  $("feedback-more-link")?.addEventListener("click", () => {
    $("feedback-more").hidden = false;
    $("feedback-comment").focus();
    $("feedback-note").hidden = true;
  });
}

for (const button of document.querySelectorAll(".k-rating button")) {
  button.addEventListener("click", async () => {
    feedback.rating = button.dataset.rating;
    for (const b of document.querySelectorAll(".k-rating button")) b.setAttribute("aria-pressed", String(b === button));
    try {
      // The first answer also sends the sliders: the AI's choice, and where they are now
      await sendFeedback(feedback.token ? {} : { aiSliders: feedback.aiSliders, finalSliders: state.values });
      if ($("feedback-more").hidden) note('Thanks! <button type="button" class="k-link" id="feedback-more-link">Tell us more</button>');
    } catch {
      note("Couldn't send that. Check your connection and try again.");
    }
  });
}

$("feedback-more").addEventListener("submit", async (e) => {
  e.preventDefault();
  const comment = $("feedback-comment").value.trim();
  if (!comment) return;
  try {
    await sendFeedback({ comment });
    $("feedback-more").hidden = true;
    note("Thanks, that really helps.");
    feedback.timer = setTimeout(closeFeedback, 2500); // done: tidy it away
  } catch {
    note("Couldn't send that. Check your connection and try again.");
  }
});

// ---------------------------------------------------------------------------
// Before / after: drag anywhere on the photo to move the line
// ---------------------------------------------------------------------------

function moveSplit(e) {
  const box = $("compare").getBoundingClientRect();
  const at = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
  $("compare").style.setProperty("--split", `${at * 100}%`);
}
$("compare").addEventListener("pointerdown", (e) => {
  $("compare").setPointerCapture(e.pointerId);
  moveSplit(e);
});
$("compare").addEventListener("pointermove", (e) => {
  if ($("compare").hasPointerCapture(e.pointerId)) moveSplit(e);
});

// ---------------------------------------------------------------------------
// Save: apply the edits to the full-size photo. On a phone, open the share
// sheet so it's one tap to "Save Image" into the camera roll (websites aren't
// allowed to put photos there directly). On a computer, download it.
// ---------------------------------------------------------------------------

const onPhone = matchMedia("(hover: none)").matches;
let readyToShare = null; // a finished photo waiting for a second tap (see below)

$("save").addEventListener("click", async () => {
  const button = $("save");
  if (readyToShare) {
    const file = readyToShare;
    readyToShare = null;
    button.textContent = "Save";
    return share(file);
  }
  button.disabled = true;
  button.textContent = "Saving…";
  await new Promise((done) => setTimeout(done, 20)); // let the button update before the busy part
  try {
    const blob = await editFullSize();
    const file = new File([blob], state.name.replace(/\.[^.]+$/, "") + "-kelvin.jpg", { type: "image/jpeg" });
    if (onPhone && navigator.canShare?.({ files: [file] })) {
      // Phones only open the share sheet straight after a tap. Editing a big
      // photo can take long enough that the browser says no; then the button
      // asks for one more tap, which counts as a fresh one.
      if ((await share(file)) === "too late") {
        readyToShare = file;
        button.disabled = false;
        button.textContent = "Tap to save";
        return;
      }
    } else {
      download(file);
    }
  } finally {
    if (!readyToShare) {
      button.disabled = false;
      button.textContent = "Save";
    }
  }
});

async function share(file) {
  try {
    await navigator.share({ files: [file] });
  } catch (error) {
    if (error.name === "NotAllowedError") return "too late";
    if (error.name !== "AbortError") download(file); // AbortError: they closed the share sheet
  }
}

function download(file) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(file);
  link.download = file.name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 10000);
}

async function editFullSize() {
  const { width, height } = state.photo;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(state.photo, 0, 0);
  // A strip at a time, so a huge photo doesn't need huge amounts of memory at once
  const strip = 256;
  for (let top = 0; top < height; top += strip) {
    const rowsHere = Math.min(strip, height - top);
    const part = ctx.getImageData(0, top, width, rowsHere);
    toBytes(applyEdits(dither(toFloats(part.data)), state.values), part.data);
    ctx.putImageData(part, 0, top);
  }
  return new Promise((done) => canvas.toBlob(done, "image/jpeg", 0.92));
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

showReasons();
addEventListener("resize", () => state.edited && drawHistogram($("histogram"), state.edited, state.preview));
// Load the AI in the background, so the first press of Auto is quick
loadAI().catch(() => {});
// Works offline and can be installed as an app (see sw.js)
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js");
