"use strict";
// Posture Coach dashboard. Talks only to the local server that served this page.

const TOKEN = document.querySelector('meta[name="posture-token"]').content;
const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";

let META = { rules: [] };
let settingsLoaded = false;
let lastEventKey = null;
let previewOn = true;
let lastCalibPhase = null;

async function api(path, body) {
  const opts = { headers: { "X-Token": TOKEN } };
  if (body !== undefined) {
    opts.method = "POST";
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function fmtDur(s) {
  if (s == null) return "–";
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

function ruleMeta(key) { return META.rules.find((r) => r.key === key) || { label: key, message: "" }; }

// ---------------------------------------------------------------- live status
function setPill(kind, icon, text) {
  const p = $("status-pill");
  p.className = `pill pill-${kind}`;
  p.querySelector(".pill-icon").textContent = icon;
  p.querySelector(".pill-text").textContent = text;
}

let lastNoCam = false;
function renderStatus(s) {
  $("pause-btn").textContent = s.paused ? "Resume" : "Pause";
  const noCam = s.state === "no_camera";
  if (noCam !== lastNoCam) {
    lastNoCam = noCam;
    setPreview($("preview-toggle").checked);
  }

  if (s.paused) setPill("neutral", "❚❚", "Paused");
  else if (s.state === "no_camera") setPill("bad", "!", "No camera frames");
  else if (!s.calibrated) setPill("warn", "!", "Calibrate to start");
  else if (!s.present) setPill("neutral", "–", "Away");
  else if (s.bad) setPill("bad", "!", "Adjust posture");
  else setPill("good", "✓", "Good posture");

  $("score").textContent = s.score == null ? "–" : s.score;
  $("score-sub").textContent = s.calibrated ? "posture score vs. your baseline" : "calibrate a baseline to get a score";
  $("perf").textContent = s.perf ? `${s.perf.fps || 0} fps · ${s.perf.proc_ms} ms/frame` : "";

  const alertIn = s.bad ? `${fmtDur(s.bad_for)} / ${fmtDur(s.alert_delay)}` : "–";
  $("t-bad").textContent = alertIn;
  $("t-still").textContent = s.present ? fmtDur(s.still_for) : "–";
  $("t-present").textContent = s.present ? fmtDur(s.present_for) : "–";
  $("t-profile").textContent = s.calibrated ? s.profile : "none";

  // Advice for the most severe current issue
  const adv = $("advice");
  if (s.bad && s.issues.length) {
    const m = ruleMeta(s.issues[0]);
    adv.className = "advice";
    adv.replaceChildren();
    const strong = document.createElement("strong");
    strong.textContent = m.label;
    adv.append(strong, document.createTextNode(m.message));
    adv.hidden = false;
  } else if (s.calibrated && s.present && s.still_for > 20 * 60) {
    adv.className = "advice info";
    adv.textContent = "You've been in a similar position for a while. A quick change of position or stretch helps.";
    adv.hidden = false;
  } else adv.hidden = true;

  renderRules(s);
  renderCalibration(s);
  renderProfiles(s);
  if (!settingsLoaded && s.settings) { loadSettings(s.settings); settingsLoaded = true; }
  handleEvents(s);
}

function renderRules(s) {
  const ul = $("rules");
  const enabled = new Set((s.settings || {}).enabled_rules || []);
  const items = (s.rules || []).filter((r) => r.key !== "ear_depth" || r.deviation != null);
  ul.replaceChildren(...items.map((r) => {
    const li = document.createElement("li");
    const on = enabled.has(r.key);
    li.className = "rule" + (r.severity != null && r.severity >= 1 && on ? " over" : "") + (on ? "" : " off");
    const name = document.createElement("span");
    name.className = "rule-name";
    name.textContent = r.label + (on ? "" : " (alerts off)");
    const val = document.createElement("span");
    val.className = "rule-val";
    val.textContent = r.deviation == null ? "not visible" : `${r.deviation > 0 ? "+" : ""}${r.deviation.toFixed(1)}${r.unit} / ${r.threshold.toFixed(r.unit === "%" ? 0 : 0)}${r.unit}`;
    const bar = document.createElement("div");
    bar.className = "rule-bar";
    const fill = document.createElement("div");
    fill.className = "fill";
    const sev = r.severity == null ? 0 : Math.max(0, Math.min(1.5, r.severity));
    fill.style.width = `${(sev / 1.5) * 100}%`;
    const tick = document.createElement("div");
    tick.className = "tick";
    tick.title = "Alert threshold";
    bar.append(fill, tick);
    li.append(name, val, bar);
    return li;
  }));
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "muted small";
    li.textContent = s.calibrated ? "Waiting for you to be in view…" : "No baseline yet: follow the steps under “Calibrate your baseline”.";
    ul.append(li);
  }
}

function renderCalibration(s) {
  const c = s.calibration;
  const ov = $("calib-overlay");
  const btn = $("calib-btn");
  if (c && (c.phase === "countdown" || c.phase === "capturing")) {
    ov.hidden = false;
    btn.textContent = "Cancel";
    btn.dataset.mode = "cancel";
    $("calib-count").textContent = Math.max(1, Math.ceil(c.phase_remaining));
    $("calib-label").textContent = c.phase === "countdown" ? "Settle into your relaxed, supported posture…" : "Hold still and breathe normally";
    $("calib-progress").style.width = `${Math.round(c.progress * 100)}%`;
  } else {
    ov.hidden = true;
    btn.textContent = "Capture baseline";
    btn.dataset.mode = "start";
  }
  if (c && (c.phase === "ok" || c.phase === "failed") && lastCalibPhase !== JSON.stringify(c)) {
    lastCalibPhase = JSON.stringify(c);
    const box = $("calib-result");
    box.hidden = false;
    box.className = "calib-result " + (c.phase === "ok" ? "ok" : "fail");
    box.replaceChildren();
    const head = document.createElement("strong");
    head.textContent = c.phase === "ok" ? `Baseline saved for “${c.profile}”.` : `Calibration didn't work: ${c.error}`;
    box.append(head);
    if (c.warnings && c.warnings.length) {
      const ul = document.createElement("ul");
      for (const w of c.warnings) { const li = document.createElement("li"); li.textContent = w; ul.append(li); }
      box.append(ul);
    }
  }
}

function renderProfiles(s) {
  const wrap = $("profiles");
  const names = Object.keys(s.profiles || {});
  const key = names.join("|") + "#" + s.profile;
  if (wrap.dataset.key === key) return;
  wrap.dataset.key = key;
  wrap.replaceChildren(...names.map((n) => {
    const chip = document.createElement("span");
    chip.className = "profile-chip" + (n === s.profile ? " active" : "");
    const use = document.createElement("button");
    use.textContent = n;
    use.title = "Use this profile";
    use.onclick = () => api("/api/profile", { profile: n }).then(poll);
    const del = document.createElement("button");
    del.textContent = "✕";
    del.title = `Delete profile ${n}`;
    del.setAttribute("aria-label", `Delete profile ${n}`);
    del.onclick = () => { if (confirm(`Delete the “${n}” baseline?`)) api("/api/profile/delete", { profile: n }).then(poll); };
    chip.append(use, del);
    return chip;
  }));
}

function handleEvents(s) {
  const evs = s.events || [];
  if (!evs.length) return;
  const last = evs[evs.length - 1];
  const key = `${last.kind}@${last.t}`;
  if (lastEventKey === null) { lastEventKey = key; return; }
  if (key === lastEventKey) return;
  lastEventKey = key;
  if (last.kind === "posture_alert" && s.settings && s.settings.sound) chime();
  loadHistory();
}

let audioCtx = null;
function chime() {
  try {
    audioCtx = audioCtx || new AudioContext();
    const t = audioCtx.currentTime;
    for (const [f, d] of [[660, 0], [880, 0.18]]) {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type = "sine"; o.frequency.value = f;
      g.gain.setValueAtTime(0, t + d);
      g.gain.linearRampToValueAtTime(0.08, t + d + 0.03);
      g.gain.exponentialRampToValueAtTime(0.0001, t + d + 0.6);
      o.connect(g).connect(audioCtx.destination);
      o.start(t + d); o.stop(t + d + 0.65);
    }
  } catch (e) { /* audio unavailable */ }
}

// ---------------------------------------------------------------- settings
function loadSettings(st) {
  const f = $("settings-form");
  for (const el of f.elements) {
    if (!el.name || !(el.name in st)) continue;
    if (el.type === "checkbox") el.checked = !!st[el.name];
    else el.value = st[el.name];
  }
  const box = $("rule-toggles");
  box.replaceChildren(...META.rules.map((r) => {
    const l = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = r.key; cb.dataset.rule = "1";
    cb.checked = st.enabled_rules.includes(r.key);
    l.append(cb, document.createTextNode(r.label));
    return l;
  }));
}

let saveTimer = null;
function saveSettings() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const f = $("settings-form");
    const body = {};
    for (const el of f.elements) {
      if (el.dataset.rule) continue;
      if (!el.name) continue;
      if (el.type === "checkbox") body[el.name] = el.checked;
      else if (el.type === "number") { if (el.value !== "") body[el.name] = Number(el.value); }
      else body[el.name] = el.value;
    }
    body.enabled_rules = [...f.querySelectorAll("input[data-rule]:checked")].map((x) => x.value);
    await api("/api/settings", body);
    const sv = $("saved"); sv.hidden = false; setTimeout(() => (sv.hidden = true), 1500);
  }, 300);
}

// ---------------------------------------------------------------- history & charts
const tip = $("tooltip");
function showTip(evt, value, label) {
  tip.replaceChildren();
  const s = document.createElement("strong"); s.textContent = value;
  const l = document.createElement("span"); l.textContent = label;
  tip.append(s, l);
  tip.hidden = false;
  const r = evt.target.getBoundingClientRect ? evt.target.getBoundingClientRect() : null;
  const x = evt.clientX ?? (r ? r.left + r.width / 2 : 0);
  const y = evt.clientY ?? (r ? r.top : 0);
  const w = tip.offsetWidth;
  tip.style.left = `${Math.min(window.innerWidth - w - 8, Math.max(8, x - w / 2))}px`;
  tip.style.top = `${Math.max(8, y - tip.offsetHeight - 12)}px`;
}
function hideTip() { tip.hidden = true; }

function el(name, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}

function yAxis(svg, x0, x1, y, ticks) {
  const g = el("g", { class: "axis" }, svg);
  for (const v of ticks) {
    el("line", { class: "gridline", x1: x0, x2: x1, y1: y(v), y2: y(v) }, g);
    const t = el("text", { x: x0 - 6, y: y(v) + 4, "text-anchor": "end" }, g);
    t.textContent = `${v}%`;
  }
}

function dailyChart(days) {
  const box = $("daily-chart");
  const W = 520, H = 170, L = 36, R = 6, T = 8, B = 22;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const data = days.slice(-14);
  const y = (v) => T + (H - T - B) * (1 - v / 100);
  yAxis(svg, L, W - R, y, [0, 50, 100]);
  if (!data.length) {
    const t = el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "empty" }, svg);
    t.textContent = "No history yet. Your daily progress will appear here.";
  }
  const slot = (W - L - R) / Math.max(data.length, 7);
  const bw = Math.min(28, slot - 6);
  data.forEach((d, i) => {
    const x = L + slot * i + (slot - bw) / 2;
    const v = d.good_pct ?? 0;
    const h = Math.max(2, y(0) - y(v));
    const r = Math.min(4, bw / 2, h);
    // Rounded data-end, square baseline
    const path = `M${x},${y(0)} V${y(0) - h + r} Q${x},${y(0) - h} ${x + r},${y(0) - h} H${x + bw - r} Q${x + bw},${y(0) - h} ${x + bw},${y(0) - h + r} V${y(0)} Z`;
    const bar = el("path", { d: path, class: "bar", tabindex: "0", "aria-label": `${d.date}: ${v}% good posture` }, svg);
    const label = `${d.date} · ${d.monitored_min} min tracked · ${d.alerts} alerts`;
    const show = (e) => showTip(e, `${v}% good`, label);
    bar.addEventListener("pointermove", show);
    bar.addEventListener("focus", show);
    bar.addEventListener("pointerleave", hideTip);
    bar.addEventListener("blur", hideTip);
    if (i === 0 || i === data.length - 1 || data.length <= 7) {
      const t = el("text", { x: x + bw / 2, y: H - 6, "text-anchor": "middle", class: "axis" }, svg);
      t.textContent = d.date.slice(5);
      t.setAttribute("fill", "var(--muted)");
      t.setAttribute("font-size", "11");
    }
  });
  box.replaceChildren(svg);
}

function todayChart(mins) {
  const box = $("today-chart");
  const W = 520, H = 150, L = 36, R = 6, T = 8, B = 22;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const pts = mins.filter((m) => m.score != null);
  const y = (v) => T + (H - T - B) * (1 - v / 100);
  yAxis(svg, L, W - R, y, [0, 50, 100]);
  if (pts.length < 2) {
    const t = el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "empty" }, svg);
    t.textContent = "Today's minute-by-minute score will appear here.";
    box.replaceChildren(svg);
    return;
  }
  const t0 = pts[0].ts, t1 = pts[pts.length - 1].ts;
  const x = (ts) => L + (W - L - R) * ((ts - t0) / Math.max(60, t1 - t0));
  // Break the line across gaps > 5 minutes (away / paused)
  let d = "", area = "", prev = null, segStart = null;
  const flush = () => { if (segStart != null) area += `L${x(prev.ts)},${y(0)} L${x(segStart)},${y(0)} Z `; };
  for (const p of pts) {
    if (!prev || p.ts - prev.ts > 300) { flush(); d += `M${x(p.ts)},${y(p.score)} `; area += `M${x(p.ts)},${y(p.score)} `; segStart = p.ts; }
    else { d += `L${x(p.ts)},${y(p.score)} `; area += `L${x(p.ts)},${y(p.score)} `; }
    prev = p;
  }
  flush();
  el("path", { d: area, class: "area" }, svg);
  el("path", { d, class: "line" }, svg);
  const fmt = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  for (const [ts, anchor] of [[t0, "start"], [t1, "end"]]) {
    const t = el("text", { x: x(ts), y: H - 6, "text-anchor": anchor, fill: "var(--muted)", "font-size": "11" }, svg);
    t.textContent = fmt(ts);
  }
  const cross = el("line", { class: "crosshair", y1: T, y2: y(0), visibility: "hidden" }, svg);
  const dot = el("circle", { class: "dot", r: 4, visibility: "hidden" }, svg);
  const hit = el("rect", { x: L, y: T, width: W - L - R, height: y(0) - T, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const sx = ((e.clientX - r.left) / r.width) * W;
    let best = pts[0];
    for (const p of pts) if (Math.abs(x(p.ts) - sx) < Math.abs(x(best.ts) - sx)) best = p;
    cross.setAttribute("x1", x(best.ts)); cross.setAttribute("x2", x(best.ts));
    dot.setAttribute("cx", x(best.ts)); dot.setAttribute("cy", y(best.score));
    cross.setAttribute("visibility", "visible"); dot.setAttribute("visibility", "visible");
    showTip(e, `score ${best.score}`, `${fmt(best.ts)} · ${best.good_pct}% of minute in good posture`);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); hideTip(); });
  box.replaceChildren(svg);
}

function historyTable(days) {
  const wrap = $("history-table");
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const h of ["Date", "Tracked (min)", "Good %", "Avg score", "Alerts", "Corrected", "Reminders", "Top issue"]) {
    const th = document.createElement("th"); th.textContent = h; head.append(th);
  }
  const body = table.createTBody();
  for (const d of [...days].reverse()) {
    const row = body.insertRow();
    const top = Object.keys(d.issues_min || {})[0];
    for (const v of [d.date, d.monitored_min, d.good_pct ?? "–", d.avg_score ?? "–", d.alerts, d.corrected, d.movement_reminders,
      top ? `${ruleMeta(top).label} (${d.issues_min[top]} min)` : "–"]) {
      row.insertCell().textContent = v;
    }
  }
  wrap.replaceChildren(table);
}

async function loadHistory() {
  let h;
  try { h = await api("/api/history"); } catch (e) { return; }
  const today = h.daily.length ? h.daily[h.daily.length - 1] : null;
  const isToday = today && today.date === new Date().toLocaleDateString("en-CA");
  $("h-today").textContent = isToday && today.good_pct != null ? `${Math.round(today.good_pct)}%` : "–";
  $("h-week").textContent = h.good_share_7d != null ? `${Math.round(h.good_share_7d)}%` : "–";
  $("h-alerts").textContent = isToday ? today.alerts : "0";
  $("h-corrected").textContent = isToday ? today.corrected : "0";
  dailyChart(h.daily);
  todayChart(h.today);
  historyTable(h.daily);
  const ul = $("events");
  const names = { posture_alert: "Alert", movement: "Movement reminder", break: "Break reminder", corrected: "Corrected" };
  ul.replaceChildren(...h.events.map((e) => {
    const li = document.createElement("li");
    const t = document.createElement("time");
    t.textContent = new Date(e.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const s = document.createElement("span");
    const what = names[e.kind] || e.kind;
    const issues = (e.issues || []).map((k) => ruleMeta(k).label.toLowerCase()).join(", ");
    s.textContent = issues ? `${what}: ${issues}` + (e.data && e.data.latency_s ? ` (in ${fmtDur(e.data.latency_s)})` : "") : what;
    li.append(t, s);
    return li;
  }));
  if (!h.events.length) {
    const li = document.createElement("li"); li.textContent = "Nothing yet."; ul.append(li);
  }
}

// ---------------------------------------------------------------- wiring
async function poll() {
  try { renderStatus(await api("/api/status")); }
  catch (e) { setPill("bad", "!", "Not connected to the app"); }
}

function setPreview(on) {
  previewOn = on;
  const img = $("preview");
  const off = $("preview-off");
  if (on && !lastNoCam) {
    img.src = `/stream.mjpg?token=${encodeURIComponent(TOKEN)}`;
    img.hidden = false;
    off.hidden = true;
    return;
  }
  img.removeAttribute("src");
  img.hidden = true;
  off.hidden = false;
  off.textContent = lastNoCam
    ? "No frames from the camera. Check that it's connected, allowed, and not in use by another app."
    : "Camera preview hidden. Tracking continues.";
}

async function init() {
  try { META = await api("/api/meta"); } catch (e) { /* retried by poll */ }
  $("pause-btn").onclick = async () => { const s = await api("/api/status"); await api("/api/pause", { paused: !s.paused }); poll(); };
  $("calib-btn").onclick = async () => {
    if ($("calib-btn").dataset.mode === "cancel") await api("/api/calibrate/cancel", {});
    else { $("calib-result").hidden = true; await api("/api/calibrate", { profile: $("profile-name").value.trim() || "sitting" }); }
    poll();
  };
  $("preview-toggle").onchange = (e) => setPreview(e.target.checked);
  $("settings-form").addEventListener("change", saveSettings);
  $("test-notify").onclick = () => api("/api/test-notification", {});
  $("clear-history").onclick = async () => { if (confirm("Delete all saved posture history? This can't be undone.")) { await api("/api/history/clear", {}); loadHistory(); } };
  $("table-toggle").onclick = (e) => {
    const t = $("history-table"); t.hidden = !t.hidden;
    e.target.textContent = t.hidden ? "Show table" : "Hide table";
    e.target.setAttribute("aria-pressed", String(!t.hidden));
  };
  document.addEventListener("visibilitychange", () => setPreview(!document.hidden && $("preview-toggle").checked));
  setPreview(true);
  poll();
  loadHistory();
  setInterval(poll, 500);
  setInterval(loadHistory, 30000);
}
init();
