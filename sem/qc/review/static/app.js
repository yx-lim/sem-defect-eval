"use strict";
// Plain-JS review frontend. Every URL is relative ("api/...") so the app works behind a sub-path proxy.

const S = {
  cfg: null, items: [], filtered: [], cur: null, curIdx: -1,
  w: 0, h: 0, scale: 1,
  imgBse: null, imgInlens: null,
  labels: null, base: null,        // Uint8Array w*h (current edit / pre-fill or 255)
  polygons: [],                    // [{class_name, subtype, points: [[x,y]] tile coords}]
  propPolygon: null,               // candidate proposal polygon (tile coords)
  undo: [], dirty: false, vlmViewed: false,
  tool: "none", drawClass: 0, drawSubtype: null, brush: 6,
  redrawMode: false, polyPts: [], painting: false, lastPt: null, hover: null,
  overlayOn: true, opacity: 0.45,
};
const $ = (id) => document.getElementById(id);
const IGNORE = 255;

async function api(path, opts = {}) {
  const r = await fetch("api/" + path, opts);
  if (!r.ok) {
    let msg = r.status + " " + r.statusText;
    try { const j = await r.json(); msg += ": " + JSON.stringify(j.detail ?? j); } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}
function flash(msg, isErr) {
  const el = $("status-msg");
  el.textContent = msg; el.style.color = isErr ? "#ff8a80" : "#ffe082";
  clearTimeout(flash.t); flash.t = setTimeout(() => { el.textContent = ""; }, isErr ? 8000 : 3000);
}
function loadImage(url) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error("image " + url)); im.src = url; });
}
async function loadLabelPng(url) {
  const r = await fetch(url);
  if (!r.ok) return null;
  const bmp = await createImageBitmap(await r.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const c = new OffscreenCanvas(bmp.width, bmp.height); const g = c.getContext("2d");
  g.drawImage(bmp, 0, 0);
  const d = g.getImageData(0, 0, bmp.width, bmp.height).data;
  const out = new Uint8Array(bmp.width * bmp.height);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4];
  return out;
}
function color(id) { return S.cfg.palette[String(id)] || [255, 255, 255]; }
function instColor(cls) {
  if (cls === "agglomerate") return [0, 255, 0];
  return color(S.cfg.classIds[cls] ?? 255);
}

// ---------- init ----------
async function init() {
  const cfg = await api("config");
  cfg.classIds = {}; for (const [id, n] of Object.entries(cfg.classes)) cfg.classIds[n] = Number(id);
  S.cfg = cfg;
  $("reviewer").value = localStorage.getItem("sem_review_reviewer") || "";
  $("reviewer").addEventListener("change", () => localStorage.setItem("sem_review_reviewer", $("reviewer").value.trim()));
  const fc = $("f-class");
  for (const c of cfg.candidate_classes) fc.add(new Option(c, c));
  const allClasses = [...Object.values(cfg.classes), "agglomerate"];
  for (const c of allClasses) $("relabel-class").add(new Option(c, c));
  fillSubtypes($("relabel-subtype")); fillSubtypes($("draw-subtype"));
  const leg = $("legend");
  for (const [id, n] of Object.entries(cfg.classes)) {
    const [r, g, b] = color(id); leg.insertAdjacentHTML("beforeend", `<span><i style="background:rgb(${r},${g},${b})"></i>${id} ${n}</span>`);
  }
  leg.insertAdjacentHTML("beforeend", `<span><i style="background:transparent"></i>i ignore</span><span><i style="background:#0f0"></i>agglomerate (polygon)</span>`);
  bindUi();
  await refreshList();
  const hash = location.hash.slice(1);
  if (hash && S.items.find((it) => it.item_id === hash)) await openItem(hash);
  else await nextUnreviewed();
}
function fillSubtypes(sel) {
  sel.innerHTML = ""; sel.add(new Option("(subtype)", ""));
  for (const s of S.cfg.artifact_subtypes) sel.add(new Option(s, s));
}
function setDrawClassOptions(kind) {
  const sel = $("draw-class"); sel.innerHTML = "";
  for (const [id, n] of Object.entries(S.cfg.classes)) sel.add(new Option(`${id} ${n}`, id));
  sel.add(new Option("i ignore", String(IGNORE)));
  sel.add(new Option("agglomerate (polygon only)", "agglomerate"));
  sel.value = String(S.drawClass);
  if (!sel.value) sel.value = "0";
}

// ---------- list / progress ----------
async function refreshList() {
  const q = new URLSearchParams();
  for (const [k, id] of [["kind", "f-kind"], ["class_name", "f-class"], ["status", "f-status"]]) if ($(id).value) q.set(k, $(id).value);
  const [all, filt] = await Promise.all([api("items"), api("items?" + q)]);
  S.items = all.items; S.filtered = filt.items;
  const ul = $("item-list"); ul.innerHTML = "";
  for (const it of S.filtered) {
    const li = document.createElement("li"); li.dataset.id = it.item_id;
    const st = it.status || "unreviewed";
    li.innerHTML = `<span class="badge ${st}">${st}</span><span>${it.kind === "candidate" ? (it.class_name + (it.subtype ? "/" + it.subtype : "")) : "tile"}</span><span style="color:#777">${it.source} · ${it.sampling_method} · ${it.split}</span>`;
    li.onclick = () => openItem(it.item_id);
    if (S.cur && it.item_id === S.cur.item_id) li.classList.add("active");
    ul.appendChild(li);
  }
  $("list-count").textContent = `${S.filtered.length} items in filter`;
  await refreshProgress();
}
async function refreshProgress() {
  const p = await api("progress");
  $("progress-fill").style.width = (p.total ? (100 * p.reviewed / p.total) : 0) + "%";
  $("progress-text").textContent = `${p.reviewed} / ${p.total} reviewed`;
  const row = (k, v) => `<tr><td>${k}</td><td>${v.reviewed}/${v.total}</td><td><span class="mini-bar"><span style="width:${v.total ? 100 * v.reviewed / v.total : 0}%"></span></span></td></tr>`;
  let html = "<table><tr><th>kind</th><th></th><th></th></tr>";
  for (const [k, v] of Object.entries(p.by_kind)) html += row(k, v);
  html += "<tr><th>class</th><th></th><th></th></tr>";
  for (const [k, v] of Object.entries(p.by_class)) html += row(k, v);
  html += "</table><div>" + Object.entries(p.by_status).map(([k, v]) => `<span class="badge ${k}">${k}: ${v}</span>`).join(" ") + "</div>";
  $("progress-detail").innerHTML = html;
}

// ---------- item ----------
async function openItem(id) {
  if (S.dirty && !confirm("Discard unsaved edits on the current item?")) return;
  const it = await api("items/" + encodeURIComponent(id));
  S.cur = it; S.dirty = false; S.undo = []; S.polyPts = []; S.redrawMode = false;
  history.replaceState(null, "", "#" + id);
  S.w = it.tile.w; S.h = it.tile.h;
  const v = Date.now();
  const [bse, inl] = await Promise.all([
    loadImage(`api/items/${id}/crop/BSE.png`), loadImage(`api/items/${id}/crop/Inlens.png`)]);
  S.imgBse = bse; S.imgInlens = inl;
  $("img-context").src = `api/items/${id}/context.png`;
  // label state
  const x0 = it.tile.x0, y0 = it.tile.y0;
  const toTile = (pts) => pts.map(([x, y]) => [x - x0, y - y0]);
  S.base = null;
  if (it.kind === "exhaustive_tile" && it.proposal.semantic_png) S.base = await loadLabelPng(`api/items/${id}/prefill.png?v=${v}`);
  if (!S.base) S.base = new Uint8Array(S.w * S.h).fill(IGNORE);
  const saved = it.has_mask ? await loadLabelPng(`api/items/${id}/mask.png?v=${v}`) : null;
  S.labels = saved || S.base.slice();
  const h = it.human;
  if (h && h.status === "redrawn" && h.polygons) S.polygons = h.polygons.map((p) => ({ class_name: p.class_name, subtype: p.subtype || null, points: toTile(p.points) }));
  else if (it.kind === "exhaustive_tile") S.polygons = (it.proposal.instances || []).filter((p) => p.polygon && p.polygon.length >= 3)
    .map((p) => ({ class_name: p.class_name, subtype: p.subtype || null, points: toTile(p.polygon) }));
  else S.polygons = [];
  S.propPolygon = it.proposal.polygon && it.proposal.polygon.length >= 3 ? toTile(it.proposal.polygon) : null;
  // ui
  const isTile = it.kind === "exhaustive_tile";
  $("actions-candidate").hidden = isTile; $("actions-tile").hidden = !isTile;
  $("btn-accept-tile").hidden = isTile && it.prefill === "blank";
  setDrawClassOptions(it.kind);
  $("relabel-class").value = it.proposal.class_name || "matrix_other";
  $("relabel-subtype").value = it.proposal.subtype || "";
  $("notes").value = (h && h.notes) || "";
  $("vlm-panel").open = false; S.vlmViewed = false;
  renderVlm(it.vlm_suggestion);
  renderMeta();
  setTool(isTile ? "brush" : "none");
  updateEditorVisibility();
  layoutCanvases(); redraw();
  $("empty-msg").hidden = true; $("item-view").hidden = false;
  for (const li of document.querySelectorAll("#item-list li")) li.classList.toggle("active", li.dataset.id === id);
  S.curIdx = S.filtered.findIndex((x) => x.item_id === id);
}
function renderMeta() {
  const it = S.cur, p = it.proposal, s = it.sampling;
  const fmt = (x) => (x === null || x === undefined) ? "–" : (typeof x === "number" ? x.toFixed(3) : x);
  $("item-meta").innerHTML =
    `<b>${it.item_id}</b> · ${it.kind} · stem ${it.stem} · ${it.batch} · split ${it.split} · tile x0=${it.tile.x0} y0=${it.tile.y0} ${it.tile.w}×${it.tile.h}<br>` +
    `proposal: <b>${p.class_name ?? "pre-filled label map"}</b>${p.subtype ? " / " + p.subtype : ""} from ${p.source} · score ${fmt(p.score)} · uncertainty ${fmt(p.uncertainty)}` +
    (p.duplicates && p.duplicates.length ? ` · also proposed by: ${p.duplicates.map((d) => d.source + ":" + d.class_name).join(", ")}` : "") +
    `<br>sampling: ${s.method} · stratum ${s.stratum} · weight ${fmt(s.weight)}`;
  const h = it.human;
  $("human-current").innerHTML = h && h.status
    ? `Current decision: <span class="badge ${h.status}">${h.status}</span> ${h.class_name ? "class " + h.class_name : ""}${h.subtype ? "/" + h.subtype : ""} by ${h.reviewer_id} at ${h.timestamp}${h.notes ? " — “" + escapeHtml(h.notes) + "”" : ""}`
    : `<span class="badge">unreviewed</span>`;
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function renderVlm(v) {
  const b = $("vlm-body");
  if (!v) { b.innerHTML = "<i>No VLM suggestion for this item.</i>"; return; }
  b.innerHTML = `<p><b>${escapeHtml(v.label || "VLM suggestion — not ground truth")}</b>. It does not pre-select any decision.</p>` +
    `<table><tr><td>model</td><td>${escapeHtml(v.model_id)}</td></tr><tr><td>class</td><td>${escapeHtml(v.class_name)}</td></tr>` +
    `<tr><td>artifact subtype</td><td>${escapeHtml(v.artifact_subtype ?? "–")}</td></tr><tr><td>confidence</td><td>${v.confidence}</td></tr>` +
    `<tr><td>rationale</td><td>${escapeHtml(v.rationale)}</td></tr><tr><td>prompt</td><td>${escapeHtml(v.prompt_version ?? "")}</td></tr></table>`;
}
function editing() { return S.cur && (S.cur.kind === "exhaustive_tile" || S.redrawMode); }
function updateEditorVisibility() {
  const show = editing();
  $("editor-tools").hidden = !show; $("instance-list").hidden = !show;
  $("btn-save-redraw").hidden = !(S.cur.kind === "candidate" && S.redrawMode);
  $("btn-redraw").classList.toggle("active-mode", S.redrawMode);
  $("draw-subtype-wrap").hidden = !(String(S.drawClass) === "7");
  renderInstances();
}
function renderInstances() {
  const el = $("instance-list");
  if (!editing()) { el.innerHTML = ""; return; }
  el.innerHTML = `<span class="tool-label">Polygons (${S.polygons.length}):</span>` + S.polygons.map((p, i) =>
    `<span class="inst" style="border-color:rgb(${instColor(p.class_name).join(",")})">${i + 1}. ${p.class_name}${p.subtype ? "/" + p.subtype : ""} (${p.points.length} pts) <button data-del="${i}">×</button></span>`).join("");
  for (const btn of el.querySelectorAll("button[data-del]")) btn.onclick = () => { pushUndo(); S.polygons.splice(Number(btn.dataset.del), 1); S.dirty = true; renderInstances(); redraw(); };
}

// ---------- canvas ----------
function layoutCanvases() {
  const maxSide = Math.max(260, Math.min(560, (window.innerWidth - 820) / 2));
  S.scale = Math.min(4, maxSide / S.w, maxSide / S.h);
  for (const id of ["cv-bse", "cv-inlens"]) {
    const c = $(id); c.width = Math.round(S.w * S.scale); c.height = Math.round(S.h * S.scale);
  }
}
let overlayCanvas = null;
function buildOverlay() {
  if (!overlayCanvas || overlayCanvas.width !== S.w || overlayCanvas.height !== S.h) overlayCanvas = new OffscreenCanvas(S.w, S.h);
  const g = overlayCanvas.getContext("2d"); const im = g.createImageData(S.w, S.h); const d = im.data;
  const lut = new Array(256).fill(null);
  for (const id of Object.keys(S.cfg.classes)) lut[Number(id)] = color(id);
  for (let i = 0; i < S.labels.length; i++) {
    const c = lut[S.labels[i]]; if (!c) continue;
    d[i * 4] = c[0]; d[i * 4 + 1] = c[1]; d[i * 4 + 2] = c[2]; d[i * 4 + 3] = 255;
  }
  g.putImageData(im, 0, 0);
}
function drawPoly(g, pts, rgb, fill, close = true, dash = null) {
  if (!pts || pts.length === 0) return;
  g.beginPath(); g.moveTo(pts[0][0] * S.scale, pts[0][1] * S.scale);
  for (const [x, y] of pts.slice(1)) g.lineTo(x * S.scale, y * S.scale);
  if (close) g.closePath();
  if (fill) { g.fillStyle = `rgba(${rgb.join(",")},${0.35 * S.opacity / 0.45})`; g.fill(); }
  g.setLineDash(dash || []); g.strokeStyle = `rgb(${rgb.join(",")})`; g.lineWidth = 2; g.stroke(); g.setLineDash([]);
}
function redraw() {
  if (!S.cur) return;
  const hasLabels = S.labels.some((v) => v !== IGNORE);
  if (S.overlayOn && hasLabels) buildOverlay();
  for (const [id, img] of [["cv-bse", S.imgBse], ["cv-inlens", S.imgInlens]]) {
    const c = $(id); const g = c.getContext("2d");
    g.imageSmoothingEnabled = S.scale < 1;
    g.clearRect(0, 0, c.width, c.height);
    g.globalAlpha = 1; g.drawImage(img, 0, 0, c.width, c.height);
    if (!S.overlayOn) continue;
    g.imageSmoothingEnabled = false;
    if (hasLabels) { g.globalAlpha = S.opacity; g.drawImage(overlayCanvas, 0, 0, c.width, c.height); g.globalAlpha = 1; }
    if (S.propPolygon) drawPoly(g, S.propPolygon, instColor(S.cur.proposal.class_name), !editing(), true, editing() ? [6, 4] : null);
    for (const p of S.polygons) drawPoly(g, p.points, instColor(p.class_name), false);
    if (S.polyPts.length) {
      drawPoly(g, S.hover ? [...S.polyPts, S.hover] : S.polyPts, [255, 255, 255], false, false, [4, 3]);
    }
    if (editing() && S.hover && (S.tool === "brush" || S.tool === "eraser")) {
      g.beginPath(); g.arc(S.hover[0] * S.scale, S.hover[1] * S.scale, S.brush * S.scale, 0, 2 * Math.PI);
      g.strokeStyle = S.tool === "eraser" ? "#fff" : `rgb(${color(S.drawClass).join(",")})`; g.lineWidth = 1; g.stroke();
    }
  }
}
function evtPt(e) {
  const r = e.target.getBoundingClientRect();
  return [Math.max(0, Math.min(S.w - 1, (e.clientX - r.left) / S.scale)), Math.max(0, Math.min(S.h - 1, (e.clientY - r.top) / S.scale))];
}
function pushUndo() {
  S.undo.push({ labels: S.labels.slice(), polygons: JSON.parse(JSON.stringify(S.polygons)) });
  if (S.undo.length > 40) S.undo.shift();
}
function doUndo() {
  if (S.polyPts.length) { S.polyPts.pop(); redraw(); return; }
  const u = S.undo.pop(); if (!u) return;
  S.labels = u.labels; S.polygons = u.polygons; S.dirty = true; renderInstances(); redraw();
}
function stamp(cx, cy) {
  const r = S.brush, r2 = r * r;
  const erase = S.tool === "eraser";
  const val = S.drawClass === "agglomerate" ? null : Number(S.drawClass);
  if (!erase && val === null) return;
  const isTile = S.cur.kind === "exhaustive_tile";
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(S.h - 1, Math.ceil(cy + r)); y++)
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(S.w - 1, Math.ceil(cx + r)); x++) {
      const dx = x - cx, dy = y - cy; if (dx * dx + dy * dy > r2) continue;
      const i = y * S.w + x;
      S.labels[i] = erase ? (isTile ? S.base[i] : IGNORE) : val;
    }
}
function strokeTo(p) {
  const [x0, y0] = S.lastPt || p; const [x1, y1] = p;
  const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / Math.max(1, S.brush / 2)));
  for (let k = 0; k <= n; k++) stamp(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n);
  S.lastPt = p;
}
function fillPolygon(pts, val) {
  const c = new OffscreenCanvas(S.w, S.h); const g = c.getContext("2d");
  g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (const [x, y] of pts.slice(1)) g.lineTo(x, y); g.closePath();
  g.fillStyle = "#fff"; g.fill();
  const d = g.getImageData(0, 0, S.w, S.h).data;
  for (let i = 0; i < S.labels.length; i++) if (d[i * 4 + 3] >= 128) S.labels[i] = val;
}
function closePolygon() {
  if (S.polyPts.length < 3) { flash("Polygon needs ≥3 points", true); return; }
  pushUndo();
  const pts = S.polyPts.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
  const cls = S.drawClass;
  const isTile = S.cur.kind === "exhaustive_tile";
  if (cls === "agglomerate") S.polygons.push({ class_name: "agglomerate", subtype: null, points: pts });
  else {
    const val = Number(cls);
    fillPolygon(pts, val);
    const name = val === IGNORE ? null : S.cfg.classes[String(val)];
    // Exhaustive tiles keep only agglomerate/artifact instances; candidate redraws keep every drawn polygon.
    if (name && (!isTile || name === "artifact")) S.polygons.push({ class_name: name, subtype: name === "artifact" ? (S.drawSubtype || null) : null, points: pts });
  }
  S.polyPts = []; S.dirty = true; renderInstances(); redraw();
}
function bindCanvas(c) {
  c.addEventListener("pointerdown", (e) => {
    if (!editing()) return;
    const p = evtPt(e);
    if (S.tool === "brush" || S.tool === "eraser") {
      if (S.tool === "brush" && S.drawClass === "agglomerate") { flash("agglomerate is polygon-only", true); return; }
      pushUndo(); S.painting = true; S.lastPt = null; strokeTo(p); S.dirty = true; c.setPointerCapture(e.pointerId); redraw();
    } else if (S.tool === "polygon") { S.polyPts.push(p); redraw(); }
  });
  c.addEventListener("pointermove", (e) => {
    S.hover = evtPt(e);
    if (S.painting) strokeTo(S.hover);
    if (editing()) redraw();
  });
  c.addEventListener("pointerup", () => { S.painting = false; S.lastPt = null; });
  c.addEventListener("pointerleave", () => { S.hover = null; if (editing()) redraw(); });
  c.addEventListener("dblclick", (e) => { if (S.tool === "polygon" && editing()) { e.preventDefault(); S.polyPts.pop(); closePolygon(); } });
}
function setTool(t) {
  S.tool = t; if (t !== "polygon") S.polyPts = [];
  for (const b of document.querySelectorAll("#editor-tools button[data-tool]")) b.classList.toggle("active-tool", b.dataset.tool === t);
  $("tool-hint").textContent = t === "polygon" ? "click to add points, Enter/double-click to close, Esc to cancel" : "";
  redraw();
}
function setDrawClass(v) {
  S.drawClass = (v === "agglomerate") ? "agglomerate" : Number(v);
  $("draw-class").value = String(v);
  $("draw-subtype-wrap").hidden = String(S.drawClass) !== "7";
}

// ---------- decisions ----------
function reviewer() {
  const r = $("reviewer").value.trim();
  if (!r) { flash("Enter a reviewer id first", true); $("reviewer").focus(); return null; }
  localStorage.setItem("sem_review_reviewer", r); return r;
}
function polysFullRes() {
  const { x0, y0 } = S.cur.tile;
  return S.polygons.map((p) => ({ class_name: p.class_name, subtype: p.subtype, points: p.points.map(([x, y]) => [x + x0, y + y0]) }));
}
async function uploadMask() {
  const r = await api(`items/${S.cur.item_id}/mask`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: S.labels });
  return r.semantic_png;
}
async function decide(status, extra = {}) {
  const rid = reviewer(); if (!rid || !S.cur) return;
  const body = { status, class_name: null, subtype: null, polygons: [], semantic_png: null, notes: $("notes").value, reviewer_id: rid, vlm_viewed: !!S.vlmViewed, ...extra };
  try {
    const res = await api(`items/${S.cur.item_id}/decision`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    S.dirty = false;
    flash(`Saved: ${status ?? "reset"} for ${S.cur.item_id}`);
    S.cur.human = res.human; renderMeta();
    await refreshList();
    if (status && $("auto-advance").checked) await nextUnreviewed(); else await openItem(S.cur.item_id);
  } catch (e) { flash("Save failed: " + e.message, true); }
}
async function saveDrawing(isTile) {
  const hasPixels = S.labels.some((v) => v !== IGNORE);
  if (!hasPixels && S.polygons.length === 0) { flash("Nothing drawn", true); return; }
  if (!reviewer()) return;
  try {
    const semantic_png = hasPixels ? await uploadMask() : null;
    const extra = { semantic_png, polygons: polysFullRes() };
    if (!isTile) {
      const sel = String(S.drawClass);
      const first = S.polygons[0];
      extra.class_name = sel === "agglomerate" ? "agglomerate" : (sel === "255" ? (first ? first.class_name : S.cur.proposal.class_name) : S.cfg.classes[sel]);
      extra.subtype = extra.class_name === "artifact" ? (S.drawSubtype || null) : null;
    }
    await decide("redrawn", extra);
  } catch (e) { flash("Save failed: " + e.message, true); }
}
async function nextUnreviewed() {
  const q = new URLSearchParams();
  if (S.cur) q.set("after", S.cur.item_id);
  if ($("f-kind").value) q.set("kind", $("f-kind").value);
  if ($("f-class").value) q.set("class_name", $("f-class").value);
  const r = await api("next_unreviewed?" + q);
  if (r.item_id) { await openItem(r.item_id); } else flash("No unreviewed items in this filter");
}
function step(d) {
  if (!S.filtered.length) return;
  const i = S.curIdx < 0 ? 0 : (S.curIdx + d + S.filtered.length) % S.filtered.length;
  openItem(S.filtered[i].item_id);
}

// ---------- bindings ----------
function bindUi() {
  bindCanvas($("cv-bse")); bindCanvas($("cv-inlens"));
  for (const id of ["f-kind", "f-class", "f-status"]) $(id).onchange = refreshList;
  $("btn-next-unrev").onclick = nextUnreviewed;
  $("btn-export").onclick = async () => {
    try { const r = await api("export", { method: "POST" }); flash(`Exported ${r.count} ground-truth items to ${r.path}`); } catch (e) { flash(e.message, true); }
  };
  $("btn-help").onclick = () => { $("help").hidden = !$("help").hidden; };
  $("btn-help-close").onclick = () => { $("help").hidden = true; };
  $("overlay-toggle").onchange = (e) => { S.overlayOn = e.target.checked; redraw(); };
  $("overlay-opacity").oninput = (e) => { S.opacity = Number(e.target.value) / 100; redraw(); };
  for (const b of document.querySelectorAll("#editor-tools button[data-tool]")) b.onclick = () => setTool(b.dataset.tool);
  $("draw-class").onchange = (e) => setDrawClass(e.target.value);
  $("draw-subtype").onchange = (e) => { S.drawSubtype = e.target.value || null; };
  $("brush-size").oninput = (e) => { S.brush = Number(e.target.value); $("brush-size-val").textContent = S.brush; };
  $("btn-undo").onclick = doUndo;
  $("btn-clear").onclick = () => {
    pushUndo(); S.labels = (S.cur.kind === "exhaustive_tile" ? new Uint8Array(S.w * S.h).fill(IGNORE) : new Uint8Array(S.w * S.h).fill(IGNORE));
    S.polygons = []; S.polyPts = []; S.dirty = true; renderInstances(); redraw();
  };
  $("btn-accept").onclick = () => decide("accepted");
  $("btn-reject").onclick = () => decide("rejected");
  $("btn-relabel").onclick = relabel;
  $("btn-redraw").onclick = toggleRedraw;
  $("btn-save-redraw").onclick = () => saveDrawing(false);
  $("btn-uncertain").onclick = () => decide("uncertain");
  $("btn-reset").onclick = () => decide(null);
  $("vlm-panel").addEventListener("toggle", () => { if ($("vlm-panel").open) S.vlmViewed = true; });
  $("btn-accept-tile").onclick = () => {
    if (S.cur && S.cur.prefill === "blank") { flash("Blank-start tile: label it and use Save edits", true); return; }
    if (S.dirty && !confirm("You have unsaved edits. 'Accept as is' records the pre-fill unchanged. Continue?")) return;
    decide("accepted");
  };
  $("btn-save-edits").onclick = () => saveDrawing(true);
  $("btn-uncertain-tile").onclick = () => decide("uncertain");
  $("btn-reset-tile").onclick = () => decide(null);
  window.addEventListener("resize", () => { if (S.cur) { layoutCanvases(); redraw(); } });
  window.addEventListener("beforeunload", (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ""; } });
  document.addEventListener("keydown", onKey);
}
function relabel() {
  const cls = $("relabel-class").value;
  const sub = cls === "artifact" ? ($("relabel-subtype").value || null) : null;
  decide("relabeled", { class_name: cls, subtype: sub });
}
function toggleRedraw() {
  if (S.cur.kind !== "candidate") return;
  S.redrawMode = !S.redrawMode;
  if (S.redrawMode) {
    const cls = S.cur.proposal.class_name;
    setDrawClass(cls === "agglomerate" ? "agglomerate" : String(S.cfg.classIds[cls] ?? 0));
    S.drawSubtype = S.cur.proposal.subtype || null; $("draw-subtype").value = S.drawSubtype || "";
    setTool(cls === "agglomerate" ? "polygon" : "brush");
  } else setTool("none");
  updateEditorVisibility(); redraw();
}
function onKey(e) {
  const tag = (e.target.tagName || "").toLowerCase();
  if (["input", "textarea", "select"].includes(tag)) { if (e.key === "Escape") e.target.blur(); return; }
  if (!S.cur) return;
  const isTile = S.cur.kind === "exhaustive_tile";
  const k = e.key;
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") { e.preventDefault(); doUndo(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const map = {
    a: () => (isTile ? $("btn-accept-tile").click() : decide("accepted")),
    r: () => { if (!isTile) decide("rejected"); },
    l: () => { if (!isTile) relabel(); },
    d: () => toggleRedraw(),
    s: () => (isTile ? saveDrawing(true) : (S.redrawMode && saveDrawing(false))),
    u: () => decide("uncertain"),
    n: () => nextUnreviewed(),
    j: () => step(1), k: () => step(-1), ArrowRight: () => step(1), ArrowLeft: () => step(-1),
    b: () => editing() && setTool("brush"), e: () => editing() && setTool("eraser"), p: () => editing() && setTool("polygon"),
    Escape: () => { S.polyPts = []; redraw(); },
    Enter: () => { if (S.tool === "polygon" && S.polyPts.length) closePolygon(); },
    z: () => doUndo(),
    o: () => { $("overlay-toggle").checked = !$("overlay-toggle").checked; S.overlayOn = $("overlay-toggle").checked; redraw(); },
    "[": () => { S.brush = Math.max(1, S.brush - 1); $("brush-size").value = S.brush; $("brush-size-val").textContent = S.brush; redraw(); },
    "]": () => { S.brush = Math.min(40, S.brush + 1); $("brush-size").value = S.brush; $("brush-size-val").textContent = S.brush; redraw(); },
    i: () => editing() && setDrawClass("255"),
    "?": () => { $("help").hidden = !$("help").hidden; },
  };
  if (/^[0-7]$/.test(k) && editing()) { setDrawClass(k); e.preventDefault(); return; }
  if (map[k]) { e.preventDefault(); map[k](); }
}

init().catch((e) => flash("Init failed: " + e.message, true));
