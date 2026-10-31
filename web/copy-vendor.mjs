// Copies the in-browser AI runtime (ONNX Runtime Web) from node_modules into
// public/vendor, so the site serves it itself and works offline.
// Run with:  npm run setup
import { copyFileSync, mkdirSync } from "node:fs";

const from = "node_modules/onnxruntime-web/dist/";
const files = ["ort.wasm.bundle.min.mjs", "ort-wasm-simd-threaded.wasm"];

mkdirSync("public/vendor", { recursive: true });
for (const file of files) copyFileSync(from + file, "public/vendor/" + file);
console.log(`Copied ${files.length} files to public/vendor`);
