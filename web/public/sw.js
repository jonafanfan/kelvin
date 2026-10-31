// The "service worker": a small script the browser keeps running in the
// background for this site. It saves a copy of every file Kelvin needs, so
// the app opens and works with no internet connection.
//
// Strategy: answer from the saved copy straight away (fast, works offline),
// and quietly fetch a fresh copy for next time. So a new version shows up on
// the second visit after a deploy.

const CACHE = "kelvin";
const FILES = [
  "./", "index.html", "style.css", "app.js", "editor.js", "ai.js", "histogram.js",
  "manifest.webmanifest", "icon.png", "slider_net.onnx",
  "vendor/ort.wasm.bundle.min.mjs", "vendor/ort-wasm-simd-threaded.wasm",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(FILES)));
  self.skipWaiting();
});
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    caches.open(CACHE).then(async (cache) => {
      const saved = await cache.match(event.request, { ignoreSearch: true });
      const fresh = fetch(event.request)
        .then((response) => {
          // Also saves the fonts from Google, so they work offline too
          if (response.ok || response.type === "opaque") cache.put(event.request, response.clone());
          return response;
        })
        .catch(() => saved);
      return saved || fresh;
    }),
  );
});
