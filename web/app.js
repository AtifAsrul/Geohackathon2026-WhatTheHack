/* Exploration Copilot - viewer + chat. Plain JavaScript, no build step. */
"use strict";
const $ = (id) => document.getElementById(id);
const S = {
  vols: [], vol: null, ref: null, il: null, xl: null, twt: null, axis: "i", mapMode: "slice",
  horizon: null, attr: "time", cmaps: {}, gain: 1,
  layers: { horizons: true, faults: false, facies: false, wells: true, leads: true, similar: false },
  interp: { ready: false }, wells: [], evidence: {}, chat: [], busy: false, hlines: [], outlines: [],
  hzTime: null, project: {}, status: {}, secData: null, mapData: null,
};
const seq = { sec: 0, map: 0 };

// ------------------------------------------------------------------ small helpers
function nearestIdx(arr, v) {
  let lo = 0, hi = arr.length - 1;
  if (arr.length < 2 || arr[0] > arr[hi]) { let b = 0, d = Infinity; for (let i = 0; i < arr.length; i++) { const e = Math.abs(arr[i] - v); if (e < d) { d = e; b = i; } } return b; }
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (arr[m] <= v) lo = m; else hi = m; }
  return Math.abs(arr[lo] - v) <= Math.abs(arr[hi] - v) ? lo : hi;
}
const vol = () => S.vols.find((v) => v.name === S.vol) || null;
const refVol = () => S.vols.find((v) => v.name === S.ref) || S.vols[0] || null;
const edges = (a) => { const d = a.length > 1 ? (a[a.length - 1] - a[0]) / (a.length - 1) : 1; return [a[0] - d / 2, a[a.length - 1] + d / 2]; };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, n = 3) => (v == null || isNaN(v) ? "n/a" : Math.abs(v) >= 1000 ? v.toFixed(0) : Number(v.toPrecision(n)).toString());
async function getJSON(url, opt) { const r = await fetch(url, opt); if (!r.ok) { let d = r.statusText; try { d = (await r.json()).detail || d; } catch (e) {} throw new Error(d); } return r.json(); }

function makeLut(stops) {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 0; while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k], [t1, c1] = stops[k + 1]; const f = Math.min(1, Math.max(0, (t - t0) / (t1 - t0 || 1)));
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = c0[c] + (c1[c] - c0[c]) * f;
  }
  return lut;
}
const LUT = {
  gray: makeLut([[0, [0, 0, 0]], [1, [255, 255, 255]]]),
  seismic: makeLut([[0, [20, 40, 160]], [0.5, [250, 250, 250]], [1, [170, 20, 20]]]),
  viridis: makeLut([[0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]], [0.75, [94, 201, 98]], [1, [253, 231, 37]]]),
  rainbow: makeLut([[0, [40, 40, 160]], [0.25, [40, 170, 200]], [0.5, [80, 190, 90]], [0.75, [240, 220, 60]], [1, [200, 40, 40]]]),
  terrain: makeLut([[0, [180, 30, 30]], [0.25, [240, 160, 60]], [0.5, [245, 240, 150]], [0.75, [90, 180, 120]], [1, [40, 70, 160]]]),
};
const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

function toCanvas(w, h, fill) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d"); const im = ctx.createImageData(w, h); fill(im.data); ctx.putImageData(im, 0, 0); return c;
}
function colorize(buf, shape, lut) {
  const [h, w] = shape;
  return toCanvas(w, h, (d) => { for (let i = 0, n = w * h; i < n; i++) { const q = buf[i] * 3; d[i * 4] = lut[q]; d[i * 4 + 1] = lut[q + 1]; d[i * 4 + 2] = lut[q + 2]; d[i * 4 + 3] = 255; } });
}
function overlayCanvas(layer, buf, shape) {
  const [h, w] = shape; const fc = (S.interp.facies_colors || []).map(hexRgb);
  return toCanvas(w, h, (d) => {
    for (let i = 0, n = w * h; i < n; i++) {
      const q = buf[i]; let r = 0, g = 0, b = 0, a = 0;
      if (layer === "faults") { if (q > 70) { r = 255; g = 60; b = 60; a = Math.min(230, (q - 70) * 2.2); } }
      else if (layer === "facies") { const c = fc[q % Math.max(1, fc.length)] || [128, 128, 128]; r = c[0]; g = c[1]; b = c[2]; a = 105; }
      else if (layer === "similar") { if (q > 150) { r = 255; g = 215; b = 0; a = Math.min(210, (q - 150) * 2.4); } }
      d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = a;
    }
  });
}
async function fetchBin(url, Type = Uint8Array) {
  const r = await fetch(url); if (!r.ok) throw new Error("not available");
  const shape = r.headers.get("X-Shape").split(",").map(Number);
  const ext = r.headers.get("X-Extent"); const range = r.headers.get("X-Range");
  return { shape, buf: new Type(await r.arrayBuffer()), ext: ext ? JSON.parse(ext) : null, range: range ? range.split(",").map(Number) : null };
}

// ------------------------------------------------------------------ a zoomable picture with axes
class Panel {
  constructor(canvas) {
    this.cv = canvas; this.ctx = canvas.getContext("2d"); this.img = null; this.ext = null; this.view = null; this.overlays = [];
    this.m = { l: 50, t: 20, r: 8, b: 8 }; this.onDraw = null; this.onPick = null; this.onStep = null; this.onHover = null;
    let down = null, moved = false;
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault(); if (!this.img) return;
      if (e.ctrlKey || e.metaKey) this.zoom(e.deltaY < 0 ? 1 / 1.25 : 1.25, ...this.mouse(e));
      else if (this.onStep) this.onStep(e.deltaY > 0 ? 1 : -1);
    }, { passive: false });
    canvas.addEventListener("mousedown", (e) => { down = { p: this.mouse(e), v: { ...this.view } }; moved = false; });
    window.addEventListener("mousemove", (e) => {
      if (!this.img) return;
      const [x, y] = this.mouse(e);
      if (down) {
        if (Math.hypot(x - down.p[0], y - down.p[1]) > 4) moved = true;
        if (moved && this.zoomed()) {
          const dh = (x - down.p[0]) / this.pw() * (down.v.h1 - down.v.h0), dv = (y - down.p[1]) / this.ph() * (down.v.v1 - down.v.v0);
          this.view = { h0: down.v.h0 - dh, h1: down.v.h1 - dh, v0: down.v.v0 - dv, v1: down.v.v1 - dv }; this.clamp(); this.draw();
        }
      } else if (e.target === canvas && this.onHover) this.onHover(this.H(x), this.V(y));
    });
    window.addEventListener("mouseup", (e) => {
      if (down && !(moved && this.zoomed()) && e.target === canvas && this.img && this.onPick) { const [x, y] = this.mouse(e); if (this.inside(x, y)) this.onPick(this.H(x), this.V(y)); }
      down = null;
    });
    canvas.addEventListener("dblclick", () => { if (this.ext) { this.view = this.full(); this.draw(); } });
  }
  mouse(e) { const r = this.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  pw() { return Math.max(10, this.w - this.m.l - this.m.r); } ph() { return Math.max(10, this.h - this.m.t - this.m.b); }
  full() { return { h0: this.ext.h[0], h1: this.ext.h[1], v0: this.ext.v[0], v1: this.ext.v[1] }; }
  zoomed() { const f = this.full(); return Math.abs(this.view.h1 - this.view.h0) < Math.abs(f.h1 - f.h0) * 0.999 || Math.abs(this.view.v1 - this.view.v0) < Math.abs(f.v1 - f.v0) * 0.999; }
  inside(x, y) { return x >= this.m.l && x <= this.w - this.m.r && y >= this.m.t && y <= this.h - this.m.b; }
  X(h) { return this.m.l + (h - this.view.h0) / (this.view.h1 - this.view.h0) * this.pw(); }
  Y(v) { return this.m.t + (v - this.view.v0) / (this.view.v1 - this.view.v0) * this.ph(); }
  H(x) { return this.view.h0 + (x - this.m.l) / this.pw() * (this.view.h1 - this.view.h0); }
  V(y) { return this.view.v0 + (y - this.m.t) / this.ph() * (this.view.v1 - this.view.v0); }
  clamp() {
    const f = this.full(), v = this.view;
    for (const [a, b] of [["h0", "h1"], ["v0", "v1"]]) {
      const span = Math.min(v[b] - v[a], f[b] - f[a]);
      if (v[a] < f[a]) { v[a] = f[a]; v[b] = f[a] + span; } if (v[b] > f[b]) { v[b] = f[b]; v[a] = f[b] - span; }
    }
  }
  zoom(k, x, y) {
    const h = this.H(x), v = this.V(y), vw = this.view;
    this.view = { h0: h - (h - vw.h0) * k, h1: h + (vw.h1 - h) * k, v0: v - (v - vw.v0) * k, v1: v + (vw.v1 - v) * k }; this.clamp(); this.draw();
  }
  resize() {
    const r = this.cv.parentElement.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    this.w = r.width; this.h = r.height; this.cv.width = Math.max(1, r.width * d); this.cv.height = Math.max(1, r.height * d);
    this.ctx.setTransform(d, 0, 0, d, 0, 0); this.draw();
  }
  setImage(img, ext, keepView) {
    const same = this.ext && keepView && this.ext.h[0] === ext.h[0] && this.ext.h[1] === ext.h[1] && this.ext.v[0] === ext.v[0] && this.ext.v[1] === ext.v[1];
    this.img = img; this.ext = ext; if (!same || !this.view) this.view = this.full(); this.draw();
  }
  clear() { this.img = null; this.overlays = []; this.draw(); }
  draw() {
    const c = this.ctx; if (!this.w) return;
    c.clearRect(0, 0, this.w, this.h); if (!this.img) return;
    c.save(); c.beginPath(); c.rect(this.m.l, this.m.t, this.pw(), this.ph()); c.clip();
    const put = (im, e, smooth) => { c.imageSmoothingEnabled = smooth; const x0 = this.X(e.h[0]), y0 = this.Y(e.v[0]); c.drawImage(im, x0, y0, this.X(e.h[1]) - x0, this.Y(e.v[1]) - y0); };
    put(this.img, this.ext, true);
    for (const o of this.overlays) put(o.img, o.ext, o.smooth);
    if (this.onDraw) this.onDraw(c, this);
    c.restore();
    // axes
    c.fillStyle = "#8fa1af"; c.strokeStyle = "#2a3640"; c.font = "11px Segoe UI, sans-serif"; c.lineWidth = 1;
    const ticks = (a, b, n) => { const span = Math.abs(b - a), raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw))), s = [1, 2, 5, 10].map((q) => q * p).find((q) => q >= raw) || raw; const out = []; for (let t = Math.ceil(Math.min(a, b) / s) * s; t <= Math.max(a, b); t += s) out.push(t); return out; };
    c.textAlign = "center"; c.textBaseline = "bottom";
    for (const t of ticks(this.view.h0, this.view.h1, Math.max(3, this.pw() / 90))) { const x = this.X(t); c.fillText(Math.round(t), x, this.m.t - 3); c.beginPath(); c.moveTo(x, this.m.t - 2); c.lineTo(x, this.m.t); c.stroke(); }
    c.textAlign = "right"; c.textBaseline = "middle";
    for (const t of ticks(this.view.v0, this.view.v1, Math.max(3, this.ph() / 60))) { const y = this.Y(t); c.fillText(Math.round(t), this.m.l - 5, y); c.beginPath(); c.moveTo(this.m.l - 3, y); c.lineTo(this.m.l, y); c.stroke(); }
    c.save(); c.translate(11, this.m.t + this.ph() / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.fillText(this.vlabel || "", 0, 0); c.restore();
    c.textAlign = "right"; c.textBaseline = "bottom"; c.fillStyle = "#5f7180"; c.fillText(this.hlabel || "", this.w - 6, this.h - 4);
  }
}
const sec = new Panel($("cv-section")), map = new Panel($("cv-map"));

// ------------------------------------------------------------------ display range
function range(v) {
  const s = v.stats;
  if (s.bipolar) { const c = Math.max(Math.abs(s.lo), Math.abs(s.hi)) / S.gain; return [-c, c]; }
  const mid = (s.lo + s.hi) / 2, half = (s.hi - s.lo) / 2 / S.gain; return [mid - half, mid + half];
}
const cmapFor = (v) => S.cmaps[v.name] || (v.stats.bipolar ? "gray" : "viridis");

// ------------------------------------------------------------------ section
async function updateSection() {
  const v = vol(); const tok = ++seq.sec;
  if (!v || S.il == null) { sec.clear(); return; }
  const num = S.axis === "i" ? S.il : S.xl, arr = S.axis === "i" ? v.ilines : v.xlines, lat = S.axis === "i" ? v.xlines : v.ilines;
  const [lo, hi] = range(v);
  try {
    const r = await fetchBin(`/api/slice?vol=${encodeURIComponent(v.name)}&axis=${S.axis}&idx=${nearestIdx(arr, num)}&lo=${lo}&hi=${hi}`);
    if (tok !== seq.sec) return;
    S.secData = { buf: r.buf, shape: r.shape, lo, hi, lat, samples: v.samples, label: v.label };
    sec.hlabel = S.axis === "i" ? "crossline →" : "inline →"; sec.vlabel = "two-way time (ms)";
    const ext = { h: edges(lat), v: edges(v.samples) };
    const ov = [];
    if (S.interp.ready) {
      for (const layer of ["facies", "faults", "similar"]) if (S.layers[layer]) {
        try { const o = await fetchBin(`/api/overlay?layer=${layer}&axis=${S.axis}&number=${num}`); ov.push({ img: overlayCanvas(layer, o.buf, o.shape), ext: o.ext, smooth: layer !== "facies" }); } catch (e) {}
      }
      S.hlines = S.layers.horizons ? await getJSON(`/api/hlines?axis=${S.axis}&number=${num}`).catch(() => []) : [];
    } else S.hlines = [];
    if (tok !== seq.sec) return;
    sec.overlays = ov; sec.setImage(colorize(r.buf, r.shape, LUT[cmapFor(v)]), ext, true);
    $("empty-section").textContent = "";
  } catch (e) { if (tok === seq.sec) { sec.clear(); $("empty-section").textContent = "This picture could not be loaded: " + e.message; } }
}
sec.onDraw = (c, p) => {
  // horizons
  c.lineWidth = 1.6; c.font = "bold 11px Segoe UI, sans-serif"; c.textBaseline = "bottom"; c.textAlign = "left";
  for (const h of S.hlines) {
    c.strokeStyle = h.color; c.beginPath();
    h.pos.forEach((q, k) => { const x = p.X(q), y = p.Y(h.twt[k]); k ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke();
    c.fillStyle = h.color; c.fillText(h.name, p.m.l + 4, Math.max(p.m.t + 12, p.Y(h.twt[Math.max(0, nearestIdx(h.pos, p.view.h0))]) - 2));
  }
  // wells close to this section
  if (S.layers.wells) {
    const v = refVol(); const lines = S.axis === "i" ? v.ilines : v.xlines; const step = Math.abs(lines[1] - lines[0]) || 1;
    const here = S.axis === "i" ? S.il : S.xl, tol = 8 * step;
    for (const w of S.wells) {
      if (!w.path || !w.path.length) continue;
      const pts = w.path.filter((q) => q[2] != null && !isNaN(q[2]) && Math.abs((S.axis === "i" ? q[0] : q[1]) - here) <= tol);
      if (pts.length < 2) continue;
      c.strokeStyle = "#ffe14d"; c.lineWidth = 2; c.beginPath();
      pts.forEach((q, k) => { const x = p.X(S.axis === "i" ? q[1] : q[0]), y = p.Y(q[2]); k ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke();
      c.fillStyle = "#ffe14d"; c.fillText(w.name, p.X(S.axis === "i" ? pts[0][1] : pts[0][0]) + 4, Math.max(p.m.t + 12, p.Y(pts[0][2])));
    }
  }
  // crosshair
  c.strokeStyle = "rgba(0,255,255,.85)"; c.lineWidth = 1; c.setLineDash([5, 4]);
  const x = p.X(S.axis === "i" ? S.xl : S.il), y = p.Y(S.twt);
  c.beginPath(); c.moveTo(x, p.m.t); c.lineTo(x, p.h); c.moveTo(p.m.l, y); c.lineTo(p.w, y); c.stroke(); c.setLineDash([]);
};
sec.onPick = (h, t) => setPos(S.axis === "i" ? { xl: h, twt: t } : { il: h, twt: t });
sec.onStep = (d) => stepLine(d);
sec.onHover = (h, t) => {
  const d = S.secData; if (!d) return;
  const a = nearestIdx(d.lat, h), k = nearestIdx(d.samples, t), q = d.buf[k * d.shape[1] + a];
  const val = d.lo + q / 255 * (d.hi - d.lo);
  readout(S.axis === "i" ? S.il : d.lat[a], S.axis === "i" ? d.lat[a] : S.xl, d.samples[k], `${d.label} ≈ ${fmt(val)}`);
};

// ------------------------------------------------------------------ map
async function updateMap() {
  const v = vol(); const tok = ++seq.map;
  if (!v || S.il == null) { map.clear(); return; }
  map.hlabel = "crossline →"; map.vlabel = "inline";
  try {
    const ov = [];
    if (S.mapMode === "horizon" && S.interp.ready && S.horizon) {
      const r = await fetchBin(`/api/map?horizon=${S.horizon}&attr=${S.attr}`, Float32Array);
      if (S.attr === "time") S.hzTime = r; else S.hzTime = await fetchBin(`/api/map?horizon=${S.horizon}&attr=time`, Float32Array).catch(() => null);
      S.outlines = S.layers.leads ? await getJSON(`/api/outlines?horizon=${S.horizon}`).catch(() => []) : [];
      if (tok !== seq.map) return;
      const [h, w] = r.shape, [lo, hi] = r.range, fc = (S.interp.facies_colors || []).map(hexRgb);
      const lut = S.attr === "time" ? LUT.terrain : LUT.viridis; const step = (hi - lo) / 14 || 1;
      const img = toCanvas(w, h, (d) => {
        for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
          const k = i * w + j, val = r.buf[k]; let c;
          if (S.attr === "facies") c = fc[Math.round(val) % Math.max(1, fc.length)] || [90, 90, 90];
          else { const q = Math.max(0, Math.min(255, Math.round((val - lo) / (hi - lo || 1) * 255))) * 3; c = [lut[q], lut[q + 1], lut[q + 2]]; }
          let dim = 1;
          if (S.attr === "time" && j + 1 < w && i + 1 < h) { const b = Math.floor(val / step); if (b !== Math.floor(r.buf[k + 1] / step) || b !== Math.floor(r.buf[k + w] / step)) dim = 0.55; }
          d[k * 4] = c[0] * dim; d[k * 4 + 1] = c[1] * dim; d[k * 4 + 2] = c[2] * dim; d[k * 4 + 3] = 255;
        }
      });
      const dh = (r.ext.h[1] - r.ext.h[0]) / Math.max(1, w - 1) / 2, dv = (r.ext.v[1] - r.ext.v[0]) / Math.max(1, h - 1) / 2;
      S.mapData = { kind: "horizon", buf: r.buf, shape: r.shape, ext: r.ext, label: (S.interp.map_labels || {})[S.attr] || S.attr };
      map.overlays = []; map.setImage(img, { h: [r.ext.h[0] - dh, r.ext.h[1] + dh], v: [r.ext.v[0] - dv, r.ext.v[1] + dv] }, true);
    } else {
      const [lo, hi] = range(v);
      const r = await fetchBin(`/api/slice?vol=${encodeURIComponent(v.name)}&axis=t&idx=${nearestIdx(v.samples, S.twt)}&lo=${lo}&hi=${hi}`);
      if (tok !== seq.map) return;
      if (S.interp.ready) for (const layer of ["facies", "faults", "similar"]) if (S.layers[layer]) {
        try { const o = await fetchBin(`/api/overlay?layer=${layer}&axis=t&number=${S.twt}`); ov.push({ img: overlayCanvas(layer, o.buf, o.shape), ext: o.ext, smooth: layer !== "facies" }); } catch (e) {}
      }
      if (tok !== seq.map) return;
      S.outlines = []; S.mapData = { kind: "slice", buf: r.buf, shape: r.shape, lo, hi, il: v.ilines, xl: v.xlines, label: v.label };
      map.overlays = ov; map.setImage(colorize(r.buf, r.shape, LUT[cmapFor(v)]), { h: edges(v.xlines), v: edges(v.ilines) }, true);
    }
    $("empty-map").textContent = "";
  } catch (e) { if (tok === seq.map) { map.clear(); $("empty-map").textContent = "This picture could not be loaded: " + e.message; } }
}
function hzTimeAt(il, xl) {
  const t = S.hzTime; if (!t) return null; const [h, w] = t.shape;
  const i = Math.round((il - t.ext.v[0]) / (t.ext.v[1] - t.ext.v[0] || 1) * (h - 1)), j = Math.round((xl - t.ext.h[0]) / (t.ext.h[1] - t.ext.h[0] || 1) * (w - 1));
  if (i < 0 || j < 0 || i >= h || j >= w) return null; return t.buf[i * w + j];
}
map.onDraw = (c, p) => {
  c.font = "bold 11px Segoe UI, sans-serif"; c.textBaseline = "middle"; c.textAlign = "left";
  if (S.mapMode === "horizon" && S.layers.leads) for (const o of S.outlines) {
    c.fillStyle = "#ffffff"; let sx = 0, sy = 0;
    for (let k = 0; k < o.il.length; k++) { const x = p.X(o.xl[k]), y = p.Y(o.il[k]); c.fillRect(x - 1, y - 1, 2, 2); sx += x; sy += y; }
    const cx = sx / o.il.length, cy = sy / o.il.length; c.fillStyle = "rgba(0,0,0,.65)"; c.fillRect(cx - 12, cy - 9, 26, 18); c.fillStyle = "#fff"; c.fillText(o.id, cx - 8, cy);
  }
  if (S.layers.wells) for (const w of S.wells) {
    if (w.il == null) continue; let q = [w.il, w.xl];
    if (w.path && w.path.length) { let best = Infinity; for (const pt of w.path) { if (pt[2] == null) continue; const d = Math.abs(pt[2] - S.twt); if (d < best) { best = d; q = pt; } } }
    const x = p.X(q[1]), y = p.Y(q[0]);
    c.fillStyle = "#111"; c.strokeStyle = "#ffe14d"; c.lineWidth = 2; c.beginPath(); c.arc(x, y, 4.5, 0, 6.3); c.fill(); c.stroke();
    c.fillStyle = "#ffe14d"; c.fillText(w.name, x + 8, y);
  }
  c.strokeStyle = "rgba(0,255,255,.85)"; c.lineWidth = 1; c.setLineDash([5, 4]);
  const x = p.X(S.xl), y = p.Y(S.il); c.beginPath(); c.moveTo(x, p.m.t); c.lineTo(x, p.h); c.moveTo(p.m.l, y); c.lineTo(p.w, y); c.stroke(); c.setLineDash([]);
  // north arrow
  const a = (refVol() || {}).affine;
  if (a) {
    const det = a.xx * a.yi - a.xi * a.yx; if (Math.abs(det) > 1e-9) {
      const dxl = -a.xi / det, dil = a.xx / det;      // (crossline, inline) step that moves 1 m north
      let vx = dxl * (p.X(1) - p.X(0)), vy = dil * (p.Y(1) - p.Y(0)); const n = Math.hypot(vx, vy) || 1; vx /= n; vy /= n;
      const ox = p.w - 30, oy = p.m.t + 30; c.strokeStyle = "#fff"; c.fillStyle = "#fff"; c.lineWidth = 2;
      c.beginPath(); c.moveTo(ox - vx * 14, oy - vy * 14); c.lineTo(ox + vx * 14, oy + vy * 14); c.stroke();
      c.beginPath(); c.arc(ox + vx * 14, oy + vy * 14, 3, 0, 6.3); c.fill(); c.textAlign = "center"; c.fillText("N", ox + vx * 25, oy + vy * 25);
    }
  }
};
map.onPick = (h, v) => { const p = { il: v, xl: h }; if (S.mapMode === "horizon") { const t = hzTimeAt(v, h); if (t != null) p.twt = t; } setPos(p); };
map.onStep = (d) => { if (S.mapMode === "slice") stepTime(d); };
map.onHover = (h, v) => {
  const d = S.mapData; if (!d) return;
  if (d.kind === "slice") { const i = nearestIdx(d.il, v), j = nearestIdx(d.xl, h), q = d.buf[i * d.shape[1] + j]; readout(d.il[i], d.xl[j], S.twt, `${d.label} ≈ ${fmt(d.lo + q / 255 * (d.hi - d.lo))}`); }
  else { const [hh, w] = d.shape; const i = Math.round((v - d.ext.v[0]) / (d.ext.v[1] - d.ext.v[0] || 1) * (hh - 1)), j = Math.round((h - d.ext.h[0]) / (d.ext.h[1] - d.ext.h[0] || 1) * (w - 1));
    if (i >= 0 && j >= 0 && i < hh && j < w) readout(Math.round(v), Math.round(h), hzTimeAt(v, h), `${S.horizon} ${d.label}: ${fmt(d.buf[i * w + j], 4)}`); }
};
function readout(il, xl, twt, extra) {
  const v = refVol(); let xy = "";
  if (v && v.affine) { const a = v.affine; xy = `   X ${(a.x0 + a.xi * il + a.xx * xl).toFixed(0)}  Y ${(a.y0 + a.yi * il + a.yx * xl).toFixed(0)}`; }
  $("readout").textContent = `IL ${Math.round(il)}   XL ${Math.round(xl)}   TWT ${twt == null ? "–" : Math.round(twt)} ms   ${extra || ""}${xy}`;
}

// ------------------------------------------------------------------ position + controls
function setPos(p, opt = {}) {
  const v = refVol(); if (!v) return;
  const old = { il: S.il, xl: S.xl, twt: S.twt };
  if (p.il != null && !isNaN(p.il)) S.il = v.ilines[nearestIdx(v.ilines, +p.il)];
  if (p.xl != null && !isNaN(p.xl)) S.xl = v.xlines[nearestIdx(v.xlines, +p.xl)];
  if (p.twt != null && !isNaN(p.twt)) S.twt = v.samples[nearestIdx(v.samples, +p.twt)];
  syncControls();
  const lineChanged = S.axis === "i" ? old.il !== S.il : old.xl !== S.xl;
  if (lineChanged || opt.force) updateSection(); else sec.draw();
  if ((S.mapMode === "slice" && old.twt !== S.twt) || opt.force) updateMap(); else map.draw();
}
function syncControls() {
  const v = refVol(); if (!v) return;
  const lines = S.axis === "i" ? v.ilines : v.xlines, cur = S.axis === "i" ? S.il : S.xl;
  $("line-num").value = cur; $("line-slider").max = lines.length - 1; $("line-slider").value = nearestIdx(lines, cur);
  $("time-num").value = Math.round(S.twt); $("time-slider").max = v.samples.length - 1; $("time-slider").value = nearestIdx(v.samples, S.twt);
}
function stepLine(d) { const v = refVol(); if (!v) return; const lines = S.axis === "i" ? v.ilines : v.xlines; const i = Math.max(0, Math.min(lines.length - 1, nearestIdx(lines, S.axis === "i" ? S.il : S.xl) + d)); setPos(S.axis === "i" ? { il: lines[i] } : { xl: lines[i] }); }
function stepTime(d) { const v = refVol(); if (!v) return; const i = Math.max(0, Math.min(v.samples.length - 1, nearestIdx(v.samples, S.twt) + d)); setPos({ twt: v.samples[i] }); }
$("line-prev").onclick = () => stepLine(-1); $("line-next").onclick = () => stepLine(1);
$("time-prev").onclick = () => stepTime(-1); $("time-next").onclick = () => stepTime(1);
$("line-num").onchange = (e) => setPos(S.axis === "i" ? { il: +e.target.value } : { xl: +e.target.value });
$("time-num").onchange = (e) => setPos({ twt: +e.target.value });
$("line-slider").oninput = (e) => { const v = refVol(); const lines = S.axis === "i" ? v.ilines : v.xlines; setPos(S.axis === "i" ? { il: lines[+e.target.value] } : { xl: lines[+e.target.value] }); };
$("time-slider").oninput = (e) => { const v = refVol(); setPos({ twt: v.samples[+e.target.value] }); };
function setAxis(a) { S.axis = a; document.querySelectorAll("#tabs-axis button").forEach((b) => b.classList.toggle("on", b.dataset.axis === a)); sec.view = null; syncControls(); updateSection(); }
document.querySelectorAll("#tabs-axis button").forEach((b) => (b.onclick = () => setAxis(b.dataset.axis)));
function setMapMode(m) {
  if (m === "horizon" && !S.interp.ready) { note("Horizon maps appear after you run the interpretation."); m = "slice"; }
  S.mapMode = m; document.querySelectorAll("#tabs-map button").forEach((b) => b.classList.toggle("on", b.dataset.map === m));
  $("time-stepper").classList.toggle("hidden", m !== "slice"); $("time-slider").classList.toggle("hidden", m !== "slice"); $("hz-controls").classList.toggle("hidden", m !== "horizon");
  updateMap();
}
document.querySelectorAll("#tabs-map button").forEach((b) => (b.onclick = () => setMapMode(b.dataset.map)));
$("sel-hz").onchange = (e) => { S.horizon = e.target.value; updateMap(); };
$("sel-attr").onchange = (e) => { S.attr = e.target.value; updateMap(); };
$("sel-vol").onchange = (e) => { S.vol = e.target.value; $("sel-cmap").value = cmapFor(vol()); updateSection(); updateMap(); };
$("sel-cmap").onchange = (e) => { S.cmaps[S.vol] = e.target.value; updateSection(); updateMap(); };
$("gain-up").onclick = () => { S.gain = Math.min(16, S.gain * 1.4); updateSection(); updateMap(); };
$("gain-down").onclick = () => { S.gain = Math.max(0.1, S.gain / 1.4); updateSection(); updateMap(); };
document.querySelectorAll("#layers .chip").forEach((b) => (b.onclick = () => {
  const l = b.dataset.layer; if (!S.interp.ready && l !== "wells") { note("This layer appears after you run the interpretation."); return; }
  S.layers[l] = !S.layers[l]; b.classList.toggle("on", S.layers[l]); updateSection(); updateMap();
}));
window.addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "ArrowRight") stepLine(1); else if (e.key === "ArrowLeft") stepLine(-1); else if (e.key === "ArrowDown") stepTime(1); else if (e.key === "ArrowUp") stepTime(-1); else return;
  e.preventDefault();
});
window.addEventListener("resize", () => { sec.resize(); map.resize(); });

// jump to a place: {il, xl, twt, horizon, vol(role), axis, volrole, layer, map:{horizon, attr}}
function go(w) {
  if (!w) return;
  const role = w.vol || w.volrole;
  if (role) { const v = S.vols.find((q) => q.role === role || q.name === role); if (v) { S.vol = v.name; $("sel-vol").value = v.name; $("sel-cmap").value = cmapFor(v); } }
  if (w.axis && w.axis !== S.axis) { S.axis = w.axis; document.querySelectorAll("#tabs-axis button").forEach((b) => b.classList.toggle("on", b.dataset.axis === w.axis)); sec.view = null; }
  if (w.layer === "similar") { S.layers.similar = true; $("chip-similar").classList.remove("hidden"); $("chip-similar").classList.add("on"); }
  const hz = (w.map && w.map.horizon) || w.horizon;
  if (hz && S.interp.ready && (S.interp.horizons || []).some((h) => h.name === hz)) {
    S.horizon = hz; $("sel-hz").value = hz;
    const want = (w.map && w.map.attr) || { sand: "sand_below", ai: "ai_below", vpvs: "vpvs_below", far: "far_over_near", near: "far_over_near" }[role];
    if (want && (S.interp.map_attrs || []).includes(want)) { S.attr = want; $("sel-attr").value = want; }
    S.mapMode = "horizon"; document.querySelectorAll("#tabs-map button").forEach((b) => b.classList.toggle("on", b.dataset.map === "horizon"));
    $("time-stepper").classList.add("hidden"); $("time-slider").classList.add("hidden"); $("hz-controls").classList.remove("hidden");
  }
  setPos({ il: w.il, xl: w.xl, twt: w.twt }, { force: true });
}

// ------------------------------------------------------------------ state, import, run
function note(text) { const t = $("tip"); t.textContent = text; t.style.left = "50%"; t.style.top = "64px"; t.style.transform = "translateX(-50%)"; t.classList.remove("hidden"); clearTimeout(note.t); note.t = setTimeout(() => { t.classList.add("hidden"); t.style.transform = ""; }, 3200); }
function pill(id, cls, text) { const p = $(id); p.className = "pill " + cls; p.querySelector("span").textContent = text; }

async function loadState(first) {
  const st = await getJSON("/api/state");
  const hadVols = S.vols.length;
  S.vols = st.volumes; S.ref = st.ref; S.wells = st.wells || []; S.interp = st.interp || { ready: false }; S.project = st.project || {}; S.status = st.status || {};
  S.defaultDir = st.default_data_dir;
  if (!S.vols.find((v) => v.name === S.vol)) S.vol = S.ref;
  const sv = $("sel-vol"); sv.innerHTML = S.vols.map((v) => `<option value="${esc(v.name)}">${esc(v.label)}</option>`).join(""); if (S.vol) { sv.value = S.vol; $("sel-cmap").value = cmapFor(vol()); }
  pill("pill-data", S.vols.length ? "ok" : "", S.vols.length ? `${S.vols.length} volumes · ${S.wells.length} wells` : "No data yet");
  pill("pill-llm", st.llm.ok ? "ok" : "bad", st.llm.ok ? "Assistant ready" : "Assistant offline"); $("pill-llm").title = `${st.llm.model} — ${st.llm.msg}`;
  if (S.interp.ready) {
    pill("pill-interp", "ok", "Interpreted " + (S.interp.built || ""));
    $("sel-hz").innerHTML = S.interp.horizons.map((h) => `<option>${esc(h.name)}</option>`).join("");
    if (!S.interp.horizons.find((h) => h.name === S.horizon)) S.horizon = (S.interp.horizons[0] || {}).name;
    $("sel-hz").value = S.horizon;
    $("sel-attr").innerHTML = S.interp.map_attrs.map((a) => `<option value="${a}">${esc((S.interp.map_labels || {})[a] || a)}</option>`).join("");
    if (!S.interp.map_attrs.includes(S.attr)) S.attr = "time"; $("sel-attr").value = S.attr;
    const k = await getJSON("/api/knowledge"); S.evidence = {}; for (const e of k.evidence) S.evidence[e.id] = e; S.know = k;
  } else pill("pill-interp", "", "Not interpreted");
  $("btn-run").textContent = S.interp.ready ? "Re-run interpretation" : "Run interpretation";
  const v = refVol();
  if (v && (S.il == null || !hadVols)) { S.il = v.ilines[v.ilines.length >> 1]; S.xl = v.xlines[v.xlines.length >> 1]; S.twt = v.samples[v.samples.length >> 1]; }
  const msg = !S.vols.length ? (S.status.state === "running" ? "Converting the volumes… pictures appear here as soon as the first one is ready." : "No data loaded yet.\nClick “Import data”, choose the folder with your SEG-Y files, then click “Run interpretation”.") : "";
  $("empty-section").textContent = msg; $("empty-map").textContent = msg;
  syncControls(); renderLeads(); renderAbout(); renderSuggest();
  sec.resize(); map.resize(); updateSection(); updateMap();
  showProgress(S.status);
  if (S.status.state === "running") pollStatus();
  if (first && !S.chat.length) greet();
}
function showProgress(s) {
  const box = $("progress"); if (!s || !s.state || (s.state === "done" && !showProgress.keep)) { box.classList.add("hidden"); return; }
  if (s.state === "running") showProgress.keep = true;
  box.classList.remove("hidden");
  const ids = (s.stages || []).map((q) => q.id), cur = ids.indexOf(s.stage);
  $("prog-steps").innerHTML = (s.stages || []).map((q, i) => `<span class="${s.state === "done" || i < cur ? "done" : i === cur ? (s.state === "error" ? "fail" : "now") : ""}">${s.state === "done" || i < cur ? "✓ " : ""}${esc(q.label)}</span>`).join("");
  $("prog-msg").textContent = s.state === "error" ? "Stopped: " + (s.error || s.message || "") : s.state === "done" ? "Finished. You can close this bar by reloading the page." : (s.message || "");
  if (s.state === "running") pill("pill-interp", "busy", "Interpreting…"); else if (s.state === "error") pill("pill-interp", "bad", "Run stopped");
  $("btn-run").disabled = s.state === "running";
}
let polling = false;
async function pollStatus() {
  if (polling) return; polling = true; let nvol = S.vols.length, lastStage = S.status.stage;
  while (true) {
    await new Promise((r) => setTimeout(r, 2000));
    let s; try { s = await getJSON("/api/status"); } catch (e) { continue; }
    showProgress(s);
    if (s.state !== "running") { polling = false; await loadState(); if (s.state === "done") addBot("The interpretation is finished. Ask me anything about it, or open the **Leads** tab.", true); return; }
    if (s.stage !== lastStage) { lastStage = s.stage; const st = await getJSON("/api/state"); if (st.volumes.length !== nvol) { polling = false; nvol = st.volumes.length; await loadState(); return; } }
  }
}
// import dialog
async function browse(path) {
  const b = await getJSON("/api/browse?path=" + encodeURIComponent(path || ""));
  $("imp-path").value = b.path;
  $("imp-list").innerHTML = `<div class="dir" data-p="${esc(b.parent)}">⬑ up one level</div>` + b.dirs.map((d) => `<div class="dir" data-p="${esc(b.path + "/" + d)}">📁 ${esc(d)}</div>`).join("") + b.files.map((f) => `<div class="file">${esc(f.name)} <span style="float:right">${f.mb} MB</span></div>`).join("");
  $("imp-found").textContent = b.segy_found ? `Found in this folder: ${b.segy_found} SEG-Y volume(s) and ${b.las_found} well log file(s).` : "No SEG-Y files in this folder yet - open the folder that contains them.";
  $("imp-use").disabled = !b.segy_found;
  document.querySelectorAll("#imp-list .dir").forEach((d) => (d.onclick = () => browse(d.dataset.p)));
}
$("btn-import").onclick = () => { $("modal").classList.remove("hidden"); browse(S.project.data_dir || S.defaultDir || ""); };
$("imp-go").onclick = () => browse($("imp-path").value); $("imp-path").onkeydown = (e) => { if (e.key === "Enter") browse($("imp-path").value); };
$("imp-cancel").onclick = () => $("modal").classList.add("hidden");
$("imp-use").onclick = async () => {
  try { const r = await getJSON("/api/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: $("imp-path").value }) });
    S.project.data_dir = r.data_dir; S.project.sources = r.segy; $("modal").classList.add("hidden"); note(`${r.segy.length} volumes selected. Now click “Run interpretation”.`); }
  catch (e) { $("imp-found").textContent = e.message; }
};
$("btn-run").onclick = async () => {
  if (!S.project.sources || !S.project.sources.length) { $("btn-import").onclick(); return; }
  const force = S.interp.ready; if (force && !confirm("Run the interpretation again from scratch? The converted volumes are kept; this can take a while.")) return;
  try { const r = await getJSON("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ force }) }); if (!r.ok) { note(r.msg); return; } S.status = { state: "running" }; showProgress.keep = true; pollStatus(); showProgress(await getJSON("/api/status")); }
  catch (e) { note(e.message); }
};

// ------------------------------------------------------------------ side panel: leads, about
document.querySelectorAll("#tabs-side button").forEach((b) => (b.onclick = () => {
  document.querySelectorAll("#tabs-side button").forEach((q) => q.classList.toggle("on", q === b));
  for (const s of ["chat", "leads", "about"]) $("side-" + s).classList.toggle("hidden", s !== b.dataset.side);
}));
function showSide(name) { document.querySelector(`#tabs-side button[data-side="${name}"]`).click(); }
function renderLeads() {
  const box = $("side-leads");
  if (!S.interp.ready) { box.innerHTML = `<div class="card"><h3>No leads yet</h3><div class="sub">Run the interpretation to find and rank traps.</div></div>`; return; }
  const bar = (v) => `<div class="bar"><i class="${v >= 0.6 ? "" : v >= 0.4 ? "mid" : "low"}" style="width:${Math.round(v * 100)}%"></i></div><span>${v.toFixed(2)}</span>`;
  box.innerHTML = `<div class="sub" style="color:var(--muted);font-size:12.5px">Ranked by a simple, transparent screening score (not a probability of success). Click a lead to see it; ask the assistant to argue for and against it.</div>` +
    S.interp.leads.map((l) => `<div class="card"><h3>${l.id} · ${esc(l.kind)} <span class="tag ${l.tested ? "drilled" : "new"}">${l.tested ? "drilled: " + esc(l.wells_inside.join(", ")) : "undrilled"}</span></h3>
      <div class="sub">${esc(l.horizon)} · crest IL ${l.where.il} / XL ${l.where.xl} at ${Math.round(l.where.twt)} ms · ${l.area_km2 != null ? l.area_km2 + " km²" : ""} · relief ${Math.round(l.relief_ms)} ms</div>
      <div class="bars"><span>Trap</span>${bar(l.scores.trap)}<span>Reservoir</span>${bar(l.scores.reservoir)}<span>Seal</span>${bar(l.scores.seal)}<span>Hydrocarbon signs</span>${bar(l.scores.hydrocarbon_indicators)}<span><b>Overall</b></span>${bar(l.scores.overall)}</div>
      <ul>${l.for.slice(0, 3).map((e) => `<li>${evChip(e)} ${esc(((S.evidence[e] || {}).text || "").replace(/^L\d+: /, ""))}</li>`).join("")}</ul>
      <div class="row"><button class="small ghost" data-show="${l.id}">Show on map</button><button class="small ghost" data-explain="${l.id}">Explain</button><button class="small ghost" data-debate="${l.id}">Argue both sides</button></div></div>`).join("");
  box.querySelectorAll("[data-show]").forEach((b) => (b.onclick = () => { const l = S.interp.leads.find((q) => q.id === b.dataset.show); go({ ...l.where, map: { horizon: l.horizon, attr: S.interp.map_attrs.includes("sand_below") ? "sand_below" : "time" } }); }));
  box.querySelectorAll("[data-explain]").forEach((b) => (b.onclick = () => { showSide("chat"); ask(`Explain lead ${b.dataset.explain}: what is it, what supports it, what are the risks, and what would you do next?`); }));
  box.querySelectorAll("[data-debate]").forEach((b) => (b.onclick = () => { showSide("chat"); ask(`Argue both sides for lead ${b.dataset.debate}: first the strongest case FOR drilling it, then the strongest case AGAINST, each with evidence, then give your verdict.`); }));
  bindChips(box);
}
function renderAbout() {
  const box = $("side-about"); const k = S.know || {}, m = S.interp.methods || {};
  const vols = S.vols.map((v) => `<li><b>${esc(v.label)}</b> — ${v.shape[0]} × ${v.shape[1]} × ${v.shape[2]}, converted with ${esc(v.converter || "?")}</li>`).join("");
  box.innerHTML = `<div class="about">
    <h4>What this app does</h4><p>It converts your SEG-Y volumes to an AI-ready format, interprets them automatically, ties them to the wells, and lets you question the result. The assistant only states what the data supports, cites its evidence, offers an alternative explanation, and tells you how sure it is.</p>
    <h4>Volumes</h4><ul>${vols || "<li>none yet</li>"}</ul>
    ${S.interp.ready ? `<h4>How the interpretation was made</h4><ul>
      <li><b>Horizons:</b> ${esc(m.rgt || "")} → ${esc(m.horizons || "")}</li><li><b>Faults:</b> ${esc(m.faults || "")}</li>
      <li><b>Seismic facies:</b> ${esc(m.facies || "")}</li><li><b>Traps:</b> ${esc(m.traps || "")}</li>
      <li><b>Assistant:</b> Gemma on this HPC, reading the evidence list and pictures of the data</li></ul>
    <h4>Seismic facies</h4><ul>${(S.interp.facies || []).map((f) => `<li><span class="swatch" style="background:${S.interp.facies_colors[(f.facies - 1) % S.interp.facies_colors.length]}"></span>Facies ${f.facies}: ${esc(f.description)} (${Math.round(f.share * 100)}%)</li>`).join("")}</ul>
    <h4>Well checks</h4><ul>${(k.calibration || []).map((c) => `<li>${esc(c.check)}: ${esc(c.verdict)}${c.correlation != null ? ` (r = ${c.correlation})` : ""}</li>`).join("") || "<li>none</li>"}</ul>
    <h4>Limits to keep in mind</h4><ul>${(S.interp.data_quality || []).map((q) => `<li>${esc(q)}</li>`).join("") || "<li>none recorded</li>"}</ul>` : `<h4>Next step</h4><p>Import a data folder, then click “Run interpretation”.</p>`}
    <h4>Privacy</h4><p>The app runs only on this machine. Data never leaves the HPC: the only model it talks to is the Gemma instance hosted here.</p></div>`;
}

// ------------------------------------------------------------------ chat
function evChip(id) { return `<span class="ev" data-ev="${id}">${id}</span>`; }
function renderMd(text) {
  let t = esc(text);
  t = t.replace(/\[\[\s*IL\s*([\d.]+)\s*[,/]\s*XL\s*([\d.]+)(?:\s*[,/]\s*TWT\s*([\d.]+)(?:\s*ms)?)?(?:\s*,\s*vol\s*=\s*([\w-]+))?\s*\]\]/gi,
    (m, il, xl, tw, v) => `<span class="go" data-il="${il}" data-xl="${xl}" data-twt="${tw || ""}" data-vol="${v || ""}">⌖ IL ${Math.round(il)} · XL ${Math.round(xl)}${tw ? " · " + Math.round(tw) + " ms" : ""}</span>`);
  t = t.replace(/[\[(]((?:E\d+)(?:\s*[,;]\s*E\d+)*)[\])]/g, (m, ids) => ids.split(/\s*[,;]\s*/).map(evChip).join(""));
  t = t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|\s)\*(\S[^*\n]*?)\*(?=\s|[.,;:]|$)/g, "$1<em>$2</em>");
  const out = []; let list = null;
  for (const line of t.split(/\n/)) {
    const li = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (li) { if (!list) { list = []; out.push(list); } list.push(li[1]); continue; }
    list = null; const h = line.match(/^\s*#{1,4}\s+(.*)$/);
    if (h) out.push(`<p><strong>${h[1]}</strong></p>`); else if (line.trim()) out.push(`<p>${line}</p>`);
  }
  return out.map((o) => (Array.isArray(o) ? `<ul>${o.map((x) => `<li>${x}</li>`).join("")}</ul>` : o)).join("");
}
function bindChips(root) {
  root.querySelectorAll(".ev").forEach((c) => {
    const e = S.evidence[c.dataset.ev];
    c.onmouseenter = (ev) => { if (!e) return; const t = $("tip"); t.style.transform = ""; t.textContent = e.text; t.classList.remove("hidden"); const r = c.getBoundingClientRect(); t.style.left = Math.max(8, Math.min(window.innerWidth - 400, r.left - 150)) + "px"; t.style.top = (r.bottom + 6) + "px"; };
    c.onmouseleave = () => $("tip").classList.add("hidden");
    c.onclick = () => { if (e && e.where) go(e.where); else if (e) note("This fact is about the whole survey, not one place."); };
  });
  root.querySelectorAll(".go").forEach((c) => (c.onclick = () => go({ il: +c.dataset.il, xl: +c.dataset.xl, twt: c.dataset.twt ? +c.dataset.twt : undefined, vol: c.dataset.vol || undefined })));
}
function addUser(text) { const d = document.createElement("div"); d.className = "msg user"; d.textContent = text; $("messages").appendChild(d); d.scrollIntoView({ block: "end" }); }
function addBot(text, final) {
  const d = document.createElement("div"); d.className = "msg bot"; d.innerHTML = `<div class="steps"></div><div class="body"></div>`; $("messages").appendChild(d);
  if (text != null) { d.querySelector(".body").innerHTML = renderMd(text); bindChips(d); } d.scrollIntoView({ block: "end" }); return d;
}
function finishBot(d, text) {
  const cited = [...new Set((text.match(/E\d+/g) || []).filter((id) => S.evidence[id]))];
  const row = document.createElement("div"); row.className = "msg-actions";
  if (cited.length) { const b = document.createElement("button"); b.textContent = cited.length === 1 ? "Show the fact used" : `Show the ${cited.length} facts used`; b.onclick = () => { let l = d.querySelector(".evlist"); if (l) { l.remove(); return; } l = document.createElement("div"); l.className = "evlist"; l.innerHTML = cited.map((id) => `<div>${evChip(id)} ${esc(S.evidence[id].text)}</div>`).join(""); d.appendChild(l); bindChips(l); l.scrollIntoView({ block: "nearest" }); }; row.appendChild(b); }
  const c = document.createElement("button"); c.textContent = "Challenge this"; c.onclick = () => ask("Challenge your last answer the way a sceptical senior reviewer would: give the strongest alternative interpretation, the evidence that supports it, and then your final verdict and the one piece of new data that would settle it."); row.appendChild(c);
  d.appendChild(row);
}
function viewCtx() { const v = vol(); return v && S.il != null ? { il: S.il, xl: S.xl, twt: S.twt, axis: S.axis, vol: v.name, vol_label: v.label } : null; }
async function ask(text, opt = {}) {
  text = (text || "").trim(); if (!text || S.busy) return;
  S.busy = true; $("chat-send").disabled = true; $("suggest").classList.add("hidden");
  addUser(text); S.chat.push({ role: "user", content: text });
  const d = addBot(), steps = d.querySelector(".steps"), body = d.querySelector(".body"); body.classList.add("typing");
  let acc = "", err = false;
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages: S.chat, view: viewCtx(), attach_view: opt.attach != null ? opt.attach : $("attach-view").checked }) });
    const rd = r.body.getReader(), dec = new TextDecoder(); let buf = "";
    while (true) {
      const { done, value } = await rd.read(); if (done) break; buf += dec.decode(value, { stream: true });
      let nl; while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl); buf = buf.slice(nl + 1); if (!line.trim()) continue; const ev = JSON.parse(line);
        if (ev.type === "step") { const s = document.createElement("span"); s.textContent = ev.text; steps.appendChild(s); }
        else if (ev.type === "view") go(ev.action);
        else if (ev.type === "token") { acc += ev.text; body.innerHTML = renderMd(acc); }
        else if (ev.type === "reset") { acc = ""; body.innerHTML = ""; }
        else if (ev.type === "error") { err = true; d.classList.add("err"); body.innerHTML = renderMd(ev.text); }
        const box = $("messages"); if (box.scrollHeight - box.scrollTop - box.clientHeight < 160) box.scrollTop = box.scrollHeight;
      }
    }
  } catch (e) { err = true; d.classList.add("err"); body.textContent = "The app did not answer: " + e.message; }
  body.classList.remove("typing"); bindChips(d);
  if (!err && acc) { S.chat.push({ role: "assistant", content: acc }); finishBot(d, acc); } else if (err) S.chat.pop();
  S.busy = false; $("chat-send").disabled = false;
}
$("chat-form").onsubmit = (e) => { e.preventDefault(); const t = $("chat-input").value; $("chat-input").value = ""; ask(t); };
$("chat-input").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chat-form").requestSubmit(); } };
function renderSuggest() {
  const q = S.interp.ready ? ["Give me the big picture of this survey", "Which lead is best, and why?", "What did the wells find?", "What are the biggest risks?", "Where would you drill next?", "Write a short interpretation report"] : [];
  const box = $("suggest"); box.innerHTML = q.map((s) => `<button>${esc(s)}</button>`).join(""); box.classList.toggle("hidden", !q.length || S.chat.length > 0);
  box.querySelectorAll("button").forEach((b) => (b.onclick = () => ask(b.textContent)));
}
function greet() {
  addBot(S.interp.ready ? "Hello. I have studied this survey: horizons, faults, seismic facies, wells and a ranked list of leads. Ask me anything — I will show the evidence behind every statement, and you can click any blue tag to see it in the viewer."
    : S.vols.length ? "Hello. The volumes are loaded and you can browse them on the left. Click **Run interpretation** so I can study the survey; after that I can answer questions with evidence."
    : "Hello. To begin: click **Import data**, choose the folder with your SEG-Y volumes and wells, then click **Run interpretation**.");
}
$("btn-explain").onclick = () => { showSide("chat"); ask("Explain what is at my crosshair: what am I looking at, what does it mean geologically, and is it interesting for exploration?", { attach: true }); };
$("btn-similar").onclick = async () => {
  if (!S.interp.ready) { note("Look-alike search is available after you run the interpretation."); return; }
  try {
    const r = await getJSON(`/api/lookalike?il=${S.il}&xl=${S.xl}&twt=${S.twt}`); if (r.error) { note(r.error); return; }
    go({ layer: "similar", il: S.il, xl: S.xl, twt: S.twt }); showSide("chat");
    const txt = `Places whose seismic character resembles IL ${S.il} / XL ${S.xl} / ${Math.round(S.twt)} ms are now highlighted in yellow (${Math.round(r.share_of_volume_similar * 100)}% of the volume). Largest look-alike areas:\n` +
      (r.regions.length ? r.regions.slice(0, 6).map((g) => `- [[IL ${g.il}, XL ${g.xl}, TWT ${g.twt_ms}]] similarity ${g.similarity}`).join("\n") : "- none above the threshold") + `\n\nMethod: ${r.method}. Ask me what these places have in common.`;
    addBot(txt, true); S.chat.push({ role: "user", content: `(I ran a look-alike search at IL ${S.il}, XL ${S.xl}, TWT ${S.twt} ms.)` }, { role: "assistant", content: txt });
  } catch (e) { note(e.message); }
};

loadState(true).catch((e) => { $("empty-section").textContent = "The app could not start: " + e.message; });
