// Kelvin's web app: the browser version of app.py. Everything happens on the
// visitor's device: the editor (editor.js), the AI (ai.js) and the histogram
// (histogram.js). No photo is ever uploaded anywhere.

import { SLIDERS, applyEdits, neutralParams } from "./editor.js";
import { aiSuggest, loadAI } from "./ai.js";
import { drawHistogram } from "./histogram.js";

// Big phone photos have 12+ million pixels. Editing a smaller copy on screen
// keeps the sliders smooth; Save applies the same edits to the full photo.
const PREVIEW_MAX_SIDE = 1024;

const $ = (id) => document.getElementById(id);
const state = {
  photo: null, // the opened photo, full size (an ImageBitmap)
  name: "",
  preview: null, // the shrunk original as r,g,b,r,g,b... from 0 to 1
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

// Sliders fire many times a second; redraw at most once per screen refresh.
let pending = false;
function scheduleRender() {
  if (pending || !state.preview) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    render();
  });
}

function render() {
  state.edited = applyEdits(state.preview, state.values);
  putPixels($("after"), state.edited);
  drawHistogram($("histogram"), state.edited, state.preview);
}

function putPixels(canvas, pixels) {
  const image = canvas.getContext("2d").createImageData(canvas.width, canvas.height);
  for (let i = 0, j = 0; i < pixels.length; i += 3, j += 4) {
    image.data[j] = pixels[i] * 255 + 0.5;
    image.data[j + 1] = pixels[i + 1] * 255 + 0.5;
    image.data[j + 2] = pixels[i + 2] * 255 + 0.5;
    image.data[j + 3] = 255;
  }
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
  $("reasons").innerHTML = `<h3>What the AI changed</h3>${body}`;
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

  $("empty").hidden = true;
  $("compare").hidden = false;
  for (const id of ["auto", "reset", "save"]) $(id).disabled = false;
  for (const input of document.querySelectorAll("input[type=range]")) input.disabled = false;
  reset();
}

function reset() {
  for (const s of SLIDERS) setSlider(s.name, 0, "you");
  showReasons();
  render();
}

$("open").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => openPhoto(e.target.files[0]));
$("reset").addEventListener("click", reset);

// Drag and drop a photo anywhere on the page
addEventListener("dragover", (e) => {
  e.preventDefault();
  $("photo").classList.add("dragging");
});
addEventListener("dragleave", () => $("photo").classList.remove("dragging"));
addEventListener("drop", (e) => {
  e.preventDefault();
  $("photo").classList.remove("dragging");
  openPhoto(e.dataTransfer.files[0]);
});

// ---------------------------------------------------------------------------
// Auto: the AI sets the sliders
// ---------------------------------------------------------------------------

$("auto").addEventListener("click", async () => {
  const button = $("auto");
  button.disabled = true;
  button.textContent = "Thinking…";
  try {
    const { params, reasons } = await aiSuggest(state.photo);
    for (const s of SLIDERS) setSlider(s.name, params[s.name], params[s.name] ? "ai" : "you");
    showReasons(SLIDERS.filter((s) => s.name in reasons).map((s) => [s.label, params[s.name], reasons[s.name]]));
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
// Save: apply the edits to the full-size photo and download it
// ---------------------------------------------------------------------------

$("save").addEventListener("click", async () => {
  const button = $("save");
  button.disabled = true;
  button.textContent = "Saving…";
  await new Promise((done) => setTimeout(done, 20)); // let the button update before the busy part
  try {
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
      const edited = applyEdits(toFloats(part.data), state.values);
      for (let i = 0, j = 0; i < edited.length; i += 3, j += 4) {
        part.data[j] = edited[i] * 255 + 0.5;
        part.data[j + 1] = edited[i + 1] * 255 + 0.5;
        part.data[j + 2] = edited[i + 2] * 255 + 0.5;
      }
      ctx.putImageData(part, 0, top);
    }
    const blob = await new Promise((done) => canvas.toBlob(done, "image/jpeg", 0.92));
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = state.name.replace(/\.[^.]+$/, "") + "-kelvin.jpg";
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
  } finally {
    button.disabled = false;
    button.textContent = "Save";
  }
});

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

showReasons();
addEventListener("resize", () => state.edited && drawHistogram($("histogram"), state.edited, state.preview));
// Load the AI in the background, so the first press of Auto is quick
loadAI().catch(() => {});
// Works offline and can be installed as an app (see sw.js)
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js");
