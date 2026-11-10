/*
 * PNG Layer Decomposer - Photoshop UXP panel (manifest v5).
 *
 * Flow:
 *   1. Flatten a duplicate of the active document and export it as PNG (temp folder).
 *   2. POST the raw PNG bytes to the local FastAPI backend (/api/v1/decompose-png).
 *   3. Walk the returned layer tree (bottom -> top) and rebuild it in Photoshop:
 *        - background / object layers: download PNG -> placeEvent -> fit to bounds -> rasterize
 *        - text layers: native editable text layer via batchPlay, calibrated to the ink bounds
 *        - groups: Document.groupLayers()
 */
"use strict";

const { app, core, action, constants } = require("photoshop");
const uxp = require("uxp");

const { batchPlay } = action;
const fs = uxp.storage.localFileSystem;
const { formats } = uxp.storage;

// ------------------------------------------------------------------ state / DOM
const $ = (id) => document.getElementById(id);
const state = { busy: false, lastPsdUrl: null, lastJobId: null };

const ui = {
  serverUrl: $("serverUrl"),
  pingBtn: $("pingBtn"),
  serverDot: $("serverDot"),
  engineInfo: $("engineInfo"),
  optText: $("optText"),
  optObjects: $("optObjects"),
  optInpaint: $("optInpaint"),
  optHideSource: $("optHideSource"),
  optRasterText: $("optRasterText"),
  decomposeBtn: $("decomposeBtn"),
  pickFileBtn: $("pickFileBtn"),
  openPsdBtn: $("openPsdBtn"),
  progressFill: $("progressFill"),
  progressLabel: $("progressLabel"),
  progressPct: $("progressPct"),
  log: $("log"),
  clearLogBtn: $("clearLogBtn"),
};

// ------------------------------------------------------------------- logging / UI
function log(message, level = "info") {
  const line = document.createElement("div");
  line.className = `log__line log__line--${level}`;
  const time = new Date().toTimeString().slice(0, 8);
  line.innerHTML = `<span class="log__time">${time}</span> ${escapeHtml(message)}`;
  ui.log.appendChild(line);
  ui.log.scrollTop = ui.log.scrollHeight;
  (level === "err" ? console.error : console.log)(`[decomposer] ${message}`);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function setProgress(pct, label, kind) {
  const v = Math.max(0, Math.min(100, Math.round(pct)));
  ui.progressFill.style.width = `${v}%`;
  ui.progressFill.className = "progress__fill" + (kind ? ` progress__fill--${kind}` : "");
  ui.progressPct.textContent = `${v}%`;
  if (label) ui.progressLabel.textContent = label.toUpperCase();
}

/** Creeps the bar towards `ceiling` while a long request is in flight. */
function startCreep(from, ceiling, label) {
  let current = from;
  const timer = setInterval(() => {
    current += (ceiling - current) * 0.06;
    setProgress(current, label);
  }, 400);
  return () => clearInterval(timer);
}

function setBusy(busy) {
  state.busy = busy;
  ui.decomposeBtn.disabled = busy;
  if (ui.pickFileBtn) ui.pickFileBtn.disabled = busy;
  ui.pingBtn.disabled = busy;
  ui.openPsdBtn.disabled = busy || !state.lastPsdUrl;
}

function isChecked(el) {
  return !!el.checked;
}

function baseUrl() {
  return (ui.serverUrl.value || "http://127.0.0.1:8000").trim().replace(/\/+$/, "");
}

// ------------------------------------------------------------------ backend I/O
async function pingServer(quiet = false) {
  try {
    const res = await fetch(`${baseUrl()}/api/v1/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = await res.json();
    const e = body.engines_available || {};
    ui.serverDot.className = "dot dot--ok";
    ui.engineInfo.textContent =
      `v${body.version} · OCR: ${(e.ocr || []).join("/")} · Seg: ${(e.segmentation || []).join("/")} · ` +
      `Inpaint: ${(e.inpainting || []).join("/")}`;
    if (!quiet) log(`Backend online (${baseUrl()})`, "ok");
    return true;
  } catch (err) {
    ui.serverDot.className = "dot dot--err";
    ui.engineInfo.textContent = "Backend unreachable - start uvicorn on 127.0.0.1:8000";
    if (!quiet) log(`Backend unreachable: ${err.message}`, "err");
    return false;
  }
}

function buildQuery(params) {
  return Object.keys(params)
    .map((k) => `${encodeURIComponent(k)}=${encodeURIComponent(String(params[k]))}`)
    .join("&");
}

async function postDecompose(pngBuffer, options) {
  const url = `${baseUrl()}/api/v1/decompose-png?${buildQuery(options)}`;
  let res;
  try {
    res = await fetch(url, { method: "POST", headers: { "Content-Type": "image/png" }, body: pngBuffer });
  } catch (err) {
    // Some UXP builds cannot send binary bodies; retry as JSON + base64.
    log(`Binary upload failed (${err.message}); retrying as base64 JSON…`, "warn");
    res = await fetch(`${baseUrl()}/api/v1/decompose-png`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: arrayBufferToBase64(pngBuffer), options }),
    });
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      detail += `: ${typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)}`;
    } catch (_) {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const parts = [];
  let chunk = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const n = (bytes[i] << 16) | ((bytes[i + 1] || 0) << 8) | (bytes[i + 2] || 0);
    chunk += B64[(n >> 18) & 63] + B64[(n >> 12) & 63];
    chunk += i + 1 < bytes.length ? B64[(n >> 6) & 63] : "=";
    chunk += i + 2 < bytes.length ? B64[n & 63] : "=";
    if (chunk.length > 32768) {
      parts.push(chunk);
      chunk = "";
    }
  }
  parts.push(chunk);
  return parts.join("");
}

async function downloadToTemp(url, filename) {
  const fullUrl = url.startsWith("http") ? url : `${baseUrl()}${url.startsWith("/") ? "" : "/"}${url}`;
  const res = await fetch(fullUrl);
  if (!res.ok) throw new Error(`Asset download failed (${res.status}): ${fullUrl}`);
  const buf = await res.arrayBuffer();
  const tmp = await fs.getTemporaryFolder();
  const file = await tmp.createFile(filename, { overwrite: true });
  await file.write(buf, { format: formats.binary });
  return file;
}

// ------------------------------------------------------------- Photoshop helpers
async function modal(commandName, fn) {
  return core.executeAsModal(fn, { commandName });
}

/** Export a flattened 8-bit RGB PNG of `doc` without touching the original. */
async function exportCanvasPng(doc) {
  const tmp = await fs.getTemporaryFolder();
  const file = await tmp.createFile(`decompose_src_${Date.now()}.png`, { overwrite: true });

  await modal("Export canvas for decomposition", async () => {
    // 1. Direct saveAs.png as copy (works in standard UXP without modifying the open document)
    try {
      if (doc.saveAs && typeof doc.saveAs.png === "function") {
        await doc.saveAs.png(file, { compression: 6, interlaced: false }, true);
        return;
      }
    } catch (saveErr) {
      log(`Direct saveAs.png notice (${saveErr.message}), trying duplicate…`, "warn");
    }

    // 2. BatchPlay duplicate document, flatten, and save
    try {
      await batchPlay(
        [
          {
            _obj: "duplicate",
            _target: [{ _ref: "document", _enum: "ordinal", _value: "targetEnum" }],
            name: "__decompose_tmp__",
          },
        ],
        {}
      );
      const dup = app.activeDocument;
      try {
        if (dup.bitsPerChannel !== constants.BitsPerChannelType.EIGHT) {
          dup.bitsPerChannel = constants.BitsPerChannelType.EIGHT;
        }
        await dup.flatten();
        await dup.saveAs.png(file, { compression: 6, interlaced: false }, true);
      } finally {
        await dup.closeWithoutSaving();
        app.activeDocument = doc;
      }
      return;
    } catch (bpErr) {
      log(`Duplicate export fallback notice: ${bpErr.message}`, "warn");
    }

    // 3. Fallback: Quick PNG export via batchPlay
    const token = fs.createSessionToken(file);
    await batchPlay(
      [
        {
          _obj: "save",
          as: {
            _obj: "PNGFormat",
            method: { _enum: "PNGMethod", _value: "quick" },
            PNGInterlaceType: { _enum: "PNGInterlaceType", _value: "PNGInterlaceNone" },
            PNGFilter: { _enum: "PNGFilter", _value: "PNGFilterAdaptive" },
            compression: 6,
          },
          in: { _path: token, _kind: "local" },
          saveStage: { _enum: "saveStageType", _value: "saveBegin" },
          copy: true,
          _options: { dialogOptions: "dontDisplay" },
        },
      ],
      {}
    );
  });

  return file.read({ format: formats.binary });
}

const px = (v) => ({ _unit: "pixelsUnit", _value: v });

function boundsOf(layer) {
  const b = layer.boundsNoEffects || layer.bounds;
  return { left: b.left, top: b.top, width: b.right - b.left, height: b.bottom - b.top };
}

/** Make sure a freshly created layer sits at the top of the stack, outside any group. */
async function ensureTopLevel(doc, layer) {
  if (layer.parent && doc.layers.length && doc.layers[0].id !== layer.id) {
    layer.move(doc.layers[0], constants.ElementPlacement.PLACEBEFORE);
  }
}

/** Scale + move `layer` so its pixel bounds match `target` (document pixels). */
async function fitLayerToBounds(layer, target, mode = "stretch", align = "left") {
  let b = boundsOf(layer);
  if (b.width <= 0 || b.height <= 0) return;
  let sx = (target.width / b.width) * 100;
  let sy = (target.height / b.height) * 100;
  if (mode === "uniform") sx = sy;
  if (Math.abs(sx - 100) > 0.5 || Math.abs(sy - 100) > 0.5) {
    await layer.resize(sx, sy, constants.AnchorPosition.TOPLEFT);
    b = boundsOf(layer);
  }
  let dx = target.left - b.left;
  if (mode === "uniform" && align === "center") dx = target.left + target.width / 2 - (b.left + b.width / 2);
  if (mode === "uniform" && align === "right") dx = target.left + target.width - (b.left + b.width);
  const dy = target.top - b.top;
  if (Math.abs(dx) >= 0.5 || Math.abs(dy) >= 0.5) await layer.translate(dx, dy);
}

/** Place a PNG file as a layer, fit it to `bounds` and rasterize it. */
async function placePngLayer(doc, file, info) {
  const token = fs.createSessionToken(file);
  await batchPlay(
    [
      {
        _obj: "placeEvent",
        null: { _path: token, _kind: "local" },
        freeTransformCenterState: { _enum: "quadCenterState", _value: "QCSAverage" },
        offset: { _obj: "offset", horizontal: px(0), vertical: px(0) },
        _options: { dialogOptions: "dontDisplay" },
      },
    ],
    {}
  );
  const layer = doc.activeLayers[0];
  layer.name = info.name;
  await ensureTopLevel(doc, layer);
  await fitLayerToBounds(layer, info.bounds, "stretch");
  await layer.rasterize(constants.RasterizeType.ENTIRELAYER);
  layer.opacity = info.opacity == null ? 100 : info.opacity;
  layer.visible = info.visible !== false;
  return layer;
}

function hexToRgb(hex) {
  const clean = String(hex || "#000000").replace("#", "");
  const n = parseInt(clean.length === 3 ? clean.split("").map((c) => c + c).join("") : clean, 16) || 0;
  return { red: (n >> 16) & 255, grain: (n >> 8) & 255, blue: n & 255 };
}

/**
 * Create a real native editable Photoshop Type Layer (T icon) matching OCR text and ink bounds.
 * Sets paragraph justification (center, left, right) and calibrates position to the visual ink bounds.
 * NEVER creates a pixel cutout or layer mask.
 */
async function createTextLayer(doc, info) {
  const t = info.text || {};
  const b = info.bounds || { left: 0, top: 0, width: 100, height: 30 };
  const content = String(t.content || info.name || "Text").trim();
  const align = (t.alignment || "left").toLowerCase();
  // Use OCR-estimated font size (fallback to cap-height scaling, min 8px)
  const isAllUpper = !/[a-z]/.test(content) && /[A-Z]/.test(content);
  const fallbackSize = isAllUpper ? b.height / 0.72 : b.height / 0.88;
  const fontSizePx = Math.max(8, Math.round(t.font_size_px || fallbackSize));
  // Horizontal anchor based on alignment
  let posX = Math.round(b.left);
  if (align === "center") {
    posX = Math.round(b.left + b.width / 2);
  } else if (align === "right") {
    posX = Math.round(b.left + b.width);
  }
  // Baseline: align to bottom of OCR box (subtract small margin)
  const posY = Math.round(b.top + b.height - Math.round(fontSizePx * 0.2));
  const layerName = info.name || content;

  let layer = null;

  // 2. Primary: Official Photoshop UXP DOM Document.createTextLayer()
  try {
    layer = await doc.createTextLayer({
      name: layerName,
      contents: content,
      fontSize: fontSizePx,
      position: { x: posX, y: posY },
    });
  } catch (domErr) {
    log(`DOM createTextLayer notice (${domErr.message}); trying minimal DOM creation...`, "warn");
  }

  // 3. Fallback: Minimal DOM createTextLayer()
  if (!layer) {
    try {
      layer = await doc.createTextLayer({
        name: layerName,
        contents: content,
      });
      if (layer) {
        const curB = layer.boundsNoEffects || layer.bounds;
        const dx = posX - curB.left;
        const dy = posY - curB.top;
        if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
          await layer.translate(dx, dy);
        }
      }
    } catch (e2) {
      log(`Minimal createTextLayer notice: ${e2.message}`, "warn");
    }
  }

  // 4. Fallback: batchPlay make textLayer
  if (!layer) {
    try {
      await batchPlay(
        [
          {
            _obj: "make",
            _target: [{ _ref: "textLayer" }],
            using: {
              _obj: "textLayer",
              textKey: content,
              textClickPoint: {
                _obj: "paint",
                horizontal: { _unit: "pixelsUnit", _value: posX },
                vertical: { _unit: "pixelsUnit", _value: posY },
              },
            },
            _options: { dialogOptions: "dontDisplay" },
          },
        ],
        {}
      );
      layer = doc.activeLayers[0];
      if (layer) layer.name = layerName;
    } catch (bpErr) {
      log(`batchPlay make textLayer error: ${bpErr.message}`, "err");
    }
  }

  if (!layer) {
    throw new Error(`Could not create native Type layer for "${content}"`);
  }

  layer.name = layerName;

  // 5. Apply Character Styling (color, size) safely
  try {
    if (layer.textItem && layer.textItem.characterStyle) {
      if (t.color_hex) {
        const rgb = hexToRgb(t.color_hex);
        const col = new app.SolidColor();
        col.rgb.red = rgb.red;
        col.rgb.green = rgb.grain;
        col.rgb.blue = rgb.blue;
        layer.textItem.characterStyle.color = col;
      }
      if (fontSizePx) {
        layer.textItem.characterStyle.size = fontSizePx;
      }
      if (t.leading_pt) {
        try {
          layer.textItem.characterStyle.autoLeading = false;
          layer.textItem.characterStyle.leading = t.leading_pt;
        } catch (_) {}
      }
      if (typeof t.tracking === "number" && t.tracking !== 0) {
        try {
          layer.textItem.characterStyle.tracking = t.tracking;
        } catch (_) {}
      }
    }
  } catch (styleErr) {
    log(`Styling notice for "${layerName}": ${styleErr.message}`, "info");
  }

  // 6a. Apply Font Family (PostScript name) if provided by OCR
  try {
    if (t.font_family && layer.textItem && layer.textItem.fontPostScriptName !== undefined) {
      layer.textItem.fontPostScriptName = t.font_family;
    }
  } catch (fontErr) {
    log(`Font family notice for "${layerName}": ${fontErr.message}`, "info");
  }

  // 6. Set Paragraph Justification (LEFT, CENTER, RIGHT)
  try {
    if (layer.textItem && layer.textItem.paragraphStyle) {
      const justMap = {
        center: constants?.Justification?.CENTER || "CENTER",
        right: constants?.Justification?.RIGHT || "RIGHT",
        left: constants?.Justification?.LEFT || "LEFT",
      };
      layer.textItem.paragraphStyle.justification = justMap[align] || "LEFT";
    }
  } catch (pErr) {
    try {
      layer.textItem.paragraphStyle.justification = align.toUpperCase();
    } catch (_) {}
  }

  // 7. Bounds Alignment Calibration:
  // Measure actual rendered layer bounding box in Photoshop and adjust translation
  // so the text visually aligns with the detected ink region.
  try {
    const act = layer.boundsNoEffects || layer.bounds;
    if (act && act.right > act.left && act.bottom > act.top) {
      let dx = 0;
      if (align === "center") {
        const targetCX = b.left + b.width / 2;
        const actualCX = (act.left + act.right) / 2;
        dx = targetCX - actualCX;
      } else if (align === "right") {
        const targetRX = b.left + b.width;
        const actualRX = act.right;
        dx = targetRX - actualRX;
      } else {
        const targetLX = b.left;
        const actualLX = act.left;
        dx = targetLX - actualLX;
      }

      // Vertical alignment: align baseline / top with the detected box
      const targetTY = b.top;
      const actualTY = act.top;
      const dy = targetTY - actualTY;

      if (Math.abs(dx) >= 0.5 || Math.abs(dy) >= 0.5) {
        await layer.translate(Math.round(dx), Math.round(dy));
      }
    }
  } catch (calibErr) {
    log(`Bounds calibration notice for "${layerName}": ${calibErr.message}`, "info");
  }

  // 8. Rotation: tilt text layer if original ad had slanted text
  if (t.angle_deg && Math.abs(t.angle_deg) > 1.5) {
    try {
      await layer.rotate(t.angle_deg);
    } catch (rotErr) {
      log(`Rotation notice for "${layerName}": ${rotErr.message}`, "info");
    }
  }

  return layer;
}

async function groupLayers(doc, layers, name) {
  if (!layers.length) return null;
  const group = await doc.groupLayers(layers);
  group.name = name;
  return group;
}

/**
 * Recursively rebuild a layer-tree node. Returns the created Photoshop layer(s)
 * for this node so the parent can group them.
 */
async function buildNode(doc, node, layerById, ctx) {
  if (node.type === "group" || node.type === "document") {
    const created = [];
    for (const child of node.children) {
      const out = await buildNode(doc, child, layerById, ctx);
      if (out) created.push(out);
    }
    if (node.type === "document") return created;
    return groupLayers(doc, created, node.name);
  }

  const info = layerById[node.layer_id];
  if (!info) return null;
  if (info.type === "original") {
    ctx.done += 1;
    return null; // the source pixels already live in the document
  }

  let layer = null;

  if (info.type === "text") {
    // ALWAYS CREATE A REAL EDITABLE TYPE LAYER (T ICON)
    // ABSOLUTELY NEVER PLACE A PNG CUTOUT FOR TEXT!
    try {
      layer = await createTextLayer(doc, info);
      log(`Editable Type layer created: "${info.text?.content || info.name}"`, "ok");
    } catch (err) {
      log(`Error creating text layer "${info.name}": ${err.message}`, "err");
      try {
        layer = await doc.createTextLayer({
          name: info.name,
          contents: info.text?.content || info.name || "Text",
        });
      } catch (err3) {
        log(`Fatal text creation error: ${err3.message}`, "err");
      }
    }
  } else {
    // Pixel layer (background, object, shape)
    try {
      const file = await downloadToTemp(info.asset_url, `${ctx.jobId}_${info.id}.png`);
      layer = await placePngLayer(doc, file, info);
      log(`Placed ${info.type} layer "${info.name}"`, "info");
    } catch (err) {
      log(`Failed to place ${info.type} layer "${info.name}": ${err.message}`, "err");
    }
  }

  ctx.done += 1;
  setProgress(60 + (ctx.done / ctx.total) * 38, `Building layers ${ctx.done}/${ctx.total}`);
  return layer;
}

// --------------------------------------------------------------------- pipeline runner
async function runDecomposePipeline(pngBytes, sourceName, t0) {
  const options = {
    extract_text: isChecked(ui.optText),
    isolate_objects: isChecked(ui.optObjects),
    inpaint_background: isChecked(ui.optInpaint),
    extract_design: true,
    build_psd: true,
    include_original: false,
  };
  setProgress(12, "Decomposing (AI)");
  log(`POST /api/v1/decompose-png (${(pngBytes.byteLength / 1024).toFixed(0)} KB)…`);
  const stopCreep = startCreep(12, 58, "Decomposing (AI)");
  let result;
  try {
    result = await postDecompose(pngBytes, options);
  } finally {
    stopCreep();
  }
  const counts = result.layers.reduce((acc, l) => ((acc[l.type] = (acc[l.type] || 0) + 1), acc), {});
  log(
    `Job ${result.job_id}: ${JSON.stringify(counts)} · engines ocr=${result.engines.ocr} ` +
      `seg=${result.engines.segmentation} inpaint=${result.engines.inpainting} · ${result.timings_ms.total} ms`,
    "ok"
  );
  (result.warnings || []).forEach((w) => log(w, "warn"));
  state.lastPsdUrl = result.psd_url;
  state.lastJobId = result.job_id;

  // Primary: open backend-compiled PSD with native layers, text, shapes & inpainting
  if (result.psd_url) {
    setProgress(75, "Opening PSD");
    log(`Downloading and opening PSD…`);
    try {
      const psdFile = await downloadToTemp(result.psd_url, `decomposed_${result.job_id}.psd`);
      await modal("Open decomposed PSD", async () => {
        await app.open(psdFile);
      });
      setProgress(100, "Done", "ok");
      log(`Decomposed document opened with all layers! (${((Date.now() - t0) / 1000).toFixed(1)} s)`, "ok");
      return;
    } catch (psdErr) {
      log(`Direct PSD open notice (${psdErr.message}), trying DOM layer builder…`, "warn");
    }
  }

  // Fallback: Rebuild in a new document via DOM
  setProgress(60, "Building layers");
  const layerById = {};
  result.layers.forEach((l) => (layerById[l.id] = l));

  await modal("Build decomposed layers", async (executionContext) => {
    const newDoc = await app.documents.add({
      width: result.width,
      height: result.height,
      resolution: 72,
      mode: "RGBColor",
      fill: "transparent",
      name: `Decomposed · ${result.job_id.slice(0, 6)}`,
    });
    app.activeDocument = newDoc;

    const suspensionID = await executionContext.hostControl.suspendHistory({
      documentID: newDoc.id,
      name: "Decompose PNG to layers",
    });
    try {
      const ctx = { done: 0, total: result.layers.length, jobId: result.job_id };
      await buildNode(newDoc, result.layer_tree, layerById, ctx);
      await executionContext.hostControl.resumeHistory(suspensionID, true);
    } catch (err) {
      await executionContext.hostControl.resumeHistory(suspensionID, false);
      throw err;
    }
  });

  setProgress(100, "Done", "ok");
  log(`Finished in ${((Date.now() - t0) / 1000).toFixed(1)} s - Built with editable Text layers!`, "ok");
}

async function decomposeActiveCanvas() {
  if (state.busy) return;
  const doc = app.activeDocument;
  if (!doc) {
    log("No active canvas found. Open an image in Photoshop or use 'Select Image & Decompose'.", "err");
    return;
  }
  setBusy(true);
  const t0 = Date.now();
  try {
    setProgress(2, "Checking backend");
    if (!(await pingServer(true))) throw new Error(`Backend not reachable at ${baseUrl()}`);

    setProgress(6, "Exporting canvas");
    log(`Exporting active canvas "${doc.name}" (${doc.width}×${doc.height}px)…`);
    const png = await exportCanvasPng(doc);
    await runDecomposePipeline(png, doc.name, t0);
  } catch (err) {
    setProgress(100, "Failed", "err");
    log(err && err.message ? err.message : String(err), "err");
  } finally {
    setBusy(false);
  }
}

async function decomposeSelectedFile() {
  if (state.busy) return;
  try {
    const file = await fs.getFileForOpening({
      types: ["png", "jpg", "jpeg", "webp"],
    });
    if (!file) return;

    setBusy(true);
    const t0 = Date.now();
    setProgress(2, "Checking backend");
    if (!(await pingServer(true))) throw new Error(`Backend not reachable at ${baseUrl()}`);

    setProgress(6, "Reading file");
    log(`Selected image: "${file.name}"`);
    const bytes = await file.read({ format: formats.binary });
    await runDecomposePipeline(bytes, file.name, t0);
  } catch (err) {
    setProgress(100, "Failed", "err");
    log(err && err.message ? err.message : String(err), "err");
  } finally {
    setBusy(false);
  }
}

async function openCompiledPsd() {
  if (!state.lastPsdUrl) return;
  setBusy(true);
  try {
    log(`Downloading ${state.lastPsdUrl}`);
    const file = await downloadToTemp(state.lastPsdUrl, `decomposed_${state.lastJobId}.psd`);
    await modal("Open compiled PSD", async () => {
      await app.open(file);
    });
    log("Compiled PSD opened in a new document", "ok");
  } catch (err) {
    log(`Could not open PSD: ${err.message}`, "err");
  } finally {
    setBusy(false);
  }
}

// ------------------------------------------------------------------------ wiring
ui.decomposeBtn.addEventListener("click", decomposeActiveCanvas);
if (ui.pickFileBtn) ui.pickFileBtn.addEventListener("click", decomposeSelectedFile);
ui.openPsdBtn.addEventListener("click", openCompiledPsd);
ui.pingBtn.addEventListener("click", () => pingServer(false));
ui.clearLogBtn.addEventListener("click", () => (ui.log.innerHTML = ""));

setProgress(0, "Idle");
log("Panel ready. Start backend, open canvas or click Select Image.");
pingServer(true);
