/* METAGROSS operator console — vanilla JS, no dependencies.
 *
 * Input  (packet = metagross.autonomy.link.codec.to_jsonable output, one per telemetry.jsonl line):
 *   window.consoleReset(mission)        mission: {mission_id, goal_xy_a:[x,y], goal_sigma_m, success_radius_m}
 *   window.pushTelemetry(packet[, meta]) packet object or JSON string; meta.latency_ms optional
 *   window.consoleReplay(lines, opts)   array of packets / JSON lines; opts.realtime (default true)
 *   window.consoleSeek(t)               deterministic: apply every loaded replay packet with t_pkt <= t
 *   ?replay=<url>&mission=<url>         fetch + play (needs http(s); use drag-drop for local files)
 *   ?deterministic=1                    no wall clock, no timers (stable screenshots)
 * Output:
 *   window.onOperatorCommand(cmd) hook and a window 'operator-command' CustomEvent,
 *   cmd = {t, action: GO|HOLD|RESUME|ESTOP, goal_xy_a?} (metagross.contracts.messages.OperatorCmd).
 *
 * Frames: A-frame (mission) x forward / y left. The 64x64 ego costmap is body-aligned: row 0 is
 * the far-forward edge (+8 m), column 0 the left edge (+8 m), 0.25 m cells, vehicle at the centre.
 * The map view is track-up and centred on the latest packet pose.
 */
(function () {
  "use strict";

  // ----------------------------------------------------------------- colour key
  // Must equal metagross.contracts.messages.CELL_COLORS (tests/test_video_console.py checks this block).
  /*CELL_RGB_BEGIN*/
  const CELL_RGB = {
    UNSEEN: [203, 208, 214],
    GROUND: [34, 160, 90],
    POSITIVE: [220, 50, 47],
    DEPRESSION: [220, 50, 47],
    DITCH_CANDIDATE: [200, 40, 160],
    CREST_SHADOW: [240, 160, 30],
    OCCLUDED: [150, 156, 164],
    WATER: [20, 140, 190],
    DYNAMIC: [250, 90, 20]
  };
  /*CELL_RGB_END*/
  // Drive-mode colours = docs/DESIGN_TOKENS.md (metagross/eval/plot_style.MODE_COLORS; video/style.MODE_HEX).
  const MODE_RGB = {
    NOMINAL: "#15803D", CAUTION: "#CA8A04", DEGRADED: "#EA580C", STOP_AND_LOOK: "#DC2626",
    SAFE_STOP: "#7F1D1D", ARRIVED: "#1D4ED8", HOLD: "#64748B"
  };
  // Canvas colours for the white theme (DESIGN_TOKENS neutrals; ink = text colour used on light map cells).
  const THEME = { well: "#F8FAFC", grid: "#E2E8F0", ink: "#0F172A", text2: "#475569", text3: "#64748B",
    accent: "#1D4ED8", vo: "#2563EB", white: "#FFFFFF", accentFill: "rgba(29,78,216,0.14)" };
  const OLIVE = [110, 120, 40]; // high-cost seen ground (same as video.style.u4_palette)
  const COSTMAP_N = 64, COSTMAP_RES = 0.25;

  /** 4-bit code -> RGB, identical to video.style.u4_palette(). */
  function buildU4Palette() {
    const p = new Array(16);
    p[0] = CELL_RGB.UNSEEN;
    for (let i = 0; i < 8; i++) {
      const a = i / 7;
      p[1 + i] = CELL_RGB.GROUND.map((g, k) => Math.round(g + (OLIVE[k] - g) * a));
    }
    p[9] = CELL_RGB.OCCLUDED; p[10] = CELL_RGB.CREST_SHADOW; p[11] = CELL_RGB.WATER;
    p[12] = CELL_RGB.DITCH_CANDIDATE; p[13] = CELL_RGB.DYNAMIC; p[14] = CELL_RGB.DEPRESSION; p[15] = CELL_RGB.POSITIVE;
    return p;
  }
  const U4 = buildU4Palette();
  const LEGEND = [
    ["Seen ground", CELL_RGB.GROUND], ["Unseen", CELL_RGB.UNSEEN], ["Ditch (missing ground)", CELL_RGB.DITCH_CANDIDATE],
    ["Lethal", CELL_RGB.POSITIVE], ["Crest shadow", CELL_RGB.CREST_SHADOW], ["Water / mud", CELL_RGB.WATER],
    ["Occluded", CELL_RGB.OCCLUDED], ["Dynamic", CELL_RGB.DYNAMIC]
  ];

  // ----------------------------------------------------------------- state
  const params = new URLSearchParams(location.search);
  const DETERMINISTIC = params.get("deterministic") === "1";
  const V_MAX = 2.0; // m/s, platform cap (defaults.VEHICLE.max_speed_mps)
  const R_MAX = 12.0; // m, camera max range (defaults.CAM_MAX_RANGE_M)
  const KBPS_WINDOW_S = 10.0;
  const SPARK_WINDOW_S = 60.0;
  const STALE_S = 2.0;

  const S = {
    mission: null, last: null, trail: [], packets: [], log: [], mode: "HOLD", modeSince: 0,
    extent: 24, replay: [], replayTimer: null, lastWall: 0, lastCmd: null, costCanvas: null, meta: {}
  };
  const $ = (id) => document.getElementById(id);
  const fmtClock = (t) => { t = Math.max(0, t || 0); const m = Math.floor(t / 60); const s = t - 60 * m;
    return `T+${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`; };
  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

  // ----------------------------------------------------------------- decoding
  /** costmap_u4_hex -> Uint8Array(4096) of codes (high nibble first, as codec.pack_u4). */
  function decodeCostmap(hex) {
    if (!hex) return null;
    const out = new Uint8Array(COSTMAP_N * COSTMAP_N);
    for (let i = 0, j = 0; i < hex.length; i += 2, j += 2) {
      const b = parseInt(hex.substr(i, 2), 16);
      out[j] = b >> 4; out[j + 1] = b & 15;
    }
    return out;
  }
  /** A-frame point -> body frame of pose (x fwd, y left). */
  function toBody(pose, x, y) {
    const c = Math.cos(pose[2]), s = Math.sin(pose[2]); const dx = x - pose[0], dy = y - pose[1];
    return [c * dx + s * dy, -s * dx + c * dy];
  }

  // ----------------------------------------------------------------- public API
  function consoleReset(mission) {
    if (typeof mission === "string") mission = JSON.parse(mission);
    stopReplay();
    Object.assign(S, { mission: mission || null, last: null, trail: [], packets: [], log: [], mode: "HOLD", modeSince: 0, lastCmd: null });
    $("mission-id").textContent = (mission && mission.mission_id) || "—";
    $("cmd-status").textContent = "no command sent";
    document.querySelectorAll(".btn").forEach((b) => b.classList.remove("sent"));
    addLog(0, "HOLD", mission ? `Mission loaded · B at ${goalText(mission)}` : "Awaiting mission");
    render();
  }

  function goalText(m) {
    if (!m || !m.goal_xy_a) return "—";
    const [x, y] = m.goal_xy_a; const r = Math.hypot(x, y); const b = Math.atan2(y, x) * 180 / Math.PI;
    return `${r.toFixed(1)} m / ${b >= 0 ? "+" : ""}${b.toFixed(1)}° (σ ${(+m.goal_sigma_m || 0).toFixed(1)} m)`;
  }

  function pushTelemetry(pkt, meta) {
    if (typeof pkt === "string") pkt = JSON.parse(pkt);
    S.meta = meta || {};
    const prevMode = S.last ? S.last.mode : null;
    pkt._costmap = decodeCostmap(pkt.costmap_u4_hex);
    S.last = pkt;
    S.packets.push({ t: pkt.t, seq: pkt.seq, bytes: pkt.packet_bytes || 0, speed: pkt.speed_mps, vcap: pkt.v_cap_mps });
    if (S.packets.length > 2000) S.packets.shift();
    S.trail.push([pkt.pose[0], pkt.pose[1]]);
    if (S.trail.length > 4000) S.trail.shift();
    if (pkt.mode !== prevMode) {
      addLog(pkt.t, pkt.mode, pkt.reason || "", prevMode);
      S.mode = pkt.mode; S.modeSince = pkt.t;
    }
    S.lastWall = performance.now();
    render();
  }

  function addLog(t, mode, reason, prev) {
    S.log.unshift({ t, mode, reason, prev });
    if (S.log.length > 200) S.log.pop();
  }

  function sendCommand(action) {
    const t = S.last ? S.last.t : 0;
    const cmd = { t, action };
    if (action === "GO" && S.mission && S.mission.goal_xy_a) cmd.goal_xy_a = S.mission.goal_xy_a;
    S.lastCmd = cmd;
    S.log.unshift({ t, mode: "OPERATOR", reason: `${action} sent${cmd.goal_xy_a ? " · B " + goalText(S.mission) : ""}`, op: true });
    document.querySelectorAll(".btn").forEach((b) => b.classList.toggle("sent", b.dataset.action === action));
    $("cmd-status").textContent = `${action} sent at ${fmtClock(t)} · awaiting ack in telemetry`;
    if (typeof window.onOperatorCommand === "function") window.onOperatorCommand(cmd);
    window.dispatchEvent(new CustomEvent("operator-command", { detail: cmd }));
    renderLog();
  }

  // ----------------------------------------------------------------- replay
  function parseLines(lines) {
    if (typeof lines === "string") lines = lines.split(/\r?\n/);
    return lines.filter((l) => l && (typeof l !== "string" || l.trim())).map((l) => (typeof l === "string" ? JSON.parse(l) : l))
      .sort((a, b) => a.t - b.t);
  }
  function stopReplay() { if (S.replayTimer) clearTimeout(S.replayTimer); S.replayTimer = null; }
  function consoleReplay(lines, opts) {
    opts = Object.assign({ realtime: !DETERMINISTIC, speed: 1.0 }, opts || {});
    stopReplay();
    S.replay = parseLines(lines);
    $("replay-badge").hidden = false;
    if (!opts.realtime) { S.replay.forEach((p) => pushTelemetry(p)); return; }
    let i = 0; const t0 = S.replay.length ? S.replay[0].t : 0; const w0 = performance.now();
    const step = () => {
      const now = t0 + (performance.now() - w0) / 1000 * opts.speed;
      while (i < S.replay.length && S.replay[i].t <= now) pushTelemetry(S.replay[i++]);
      if (i < S.replay.length) S.replayTimer = setTimeout(step, Math.max(10, (S.replay[i].t - now) * 1000 / opts.speed));
    };
    step();
  }
  function consoleSeek(t) {
    stopReplay();
    const pk = S.replay;
    const mission = S.mission;
    Object.assign(S, { last: null, trail: [], packets: [], log: [], mode: "HOLD", modeSince: 0 });
    if (mission) addLog(0, "HOLD", `Mission loaded · B at ${goalText(mission)}`);
    pk.filter((p) => p.t <= t + 1e-9).forEach((p) => pushTelemetry(p));
    S.replay = pk;
  }

  // ----------------------------------------------------------------- rendering
  function render() { renderBanner(); renderTop(); renderSide(); renderMap(); renderLog(); }

  function renderTop() {
    const p = S.last;
    $("clock-mission").textContent = fmtClock(p ? p.t : 0);
    const pk = S.packets;
    if (!pk.length) return;
    const tNow = pk[pk.length - 1].t;
    const win = pk.filter((q) => q.t > tNow - KBPS_WINDOW_S);
    const span = Math.max(1e-3, Math.min(KBPS_WINDOW_S, tNow - (win[0] ? win[0].t : tNow)) + 0.5);
    const kbps = win.reduce((a, q) => a + q.bytes, 0) * 8 / 1000 / span;
    const seqs = pk.map((q) => q.seq); const expected = seqs[seqs.length - 1] - seqs[0] + 1;
    const lost = Math.max(0, expected - new Set(seqs).size);
    const lossPct = expected > 0 ? 100 * lost / expected : 0;
    const lat = S.meta.latency_ms != null ? S.meta.latency_ms : (p.health && p.health.latency_ms != null ? p.health.latency_ms : null);
    $("link-kbps").textContent = `${kbps.toFixed(2)} kbps`;
    $("link-loss").textContent = `loss ${lossPct.toFixed(0)}%`;
    $("link-lat").textContent = lat != null ? `lat ${Math.round(lat)} ms` : "lat —";
    $("pk-n").textContent = pk.length; $("pk-last").textContent = `${pk[pk.length - 1].bytes} B`;
    $("pk-mean").textContent = `${Math.round(pk.reduce((a, q) => a + q.bytes, 0) / pk.length)} B`;
    $("pk-lost").textContent = String(lost);
    const dot = $("link-dot");
    dot.className = "dot " + (lossPct > 30 ? "bad" : lossPct > 10 ? "warn" : "ok");
  }

  function renderBanner() {
    const p = S.last; const mode = p ? p.mode : S.mode;
    $("banner").dataset.mode = mode;
    $("mode-chip").textContent = mode.replace(/_/g, " ");
    $("mode-reason").textContent = p ? (p.reason || "—") : (S.mission ? "Mission loaded · awaiting GO" : "Awaiting mission");
    $("mode-since").textContent = `${(p ? Math.max(0, p.t - S.modeSince) : 0).toFixed(1)} s`;
    if (p && S.mission && S.mission.goal_xy_a) {
      $("dist-goal").textContent = `${Math.hypot(S.mission.goal_xy_a[0] - p.pose[0], S.mission.goal_xy_a[1] - p.pose[1]).toFixed(1)} m`;
    } else $("dist-goal").textContent = "—";
    document.documentElement.style.setProperty("--mode", MODE_RGB[mode] || MODE_RGB.HOLD);
  }

  function setBar(id, frac, marker) {
    const el = $(id); const f = Math.max(0, Math.min(1, frac || 0));
    el.querySelector(".fill").style.width = `calc(${(f * 100).toFixed(2)}% - 2px)`;
    const m = el.querySelector(".marker");
    if (m) m.style.left = `${(Math.max(0, Math.min(1, marker || 0)) * 100).toFixed(2)}%`;
  }

  function renderSide() {
    const p = S.last;
    const h = (p && p.health) || {};
    $("speed-val").textContent = p ? `${p.speed_mps.toFixed(2)} / ${p.v_cap_mps.toFixed(2)} m/s` : "— / — m/s";
    setBar("speed-bar", p ? p.speed_mps / V_MAX : 0, p ? p.v_cap_mps / V_MAX : 0);
    $("rcert-val").textContent = p ? `${p.r_cert_m.toFixed(1)} m` : "— m";
    setBar("rcert-bar", p ? p.r_cert_m / R_MAX : 0);
    const q = h.q != null ? h.q : (h.p_fail != null ? 1 - h.p_fail : null);
    const arc = $("q-arc"); const L = Math.PI * 50;
    arc.style.strokeDasharray = `${(q != null ? Math.max(0, Math.min(1, q)) : 0) * L} 999`;
    arc.style.stroke = MODE_RGB[p ? p.mode : "HOLD"];
    $("q-text").textContent = q != null ? q.toFixed(2) : "—";
    $("pfail-val").textContent = q != null ? (1 - q).toFixed(2) : "—";
    $("sigma-val").textContent = p ? `${p.pos_sigma_m.toFixed(2)} m` : "—";
    $("vo-val").textContent = h.vo_ok != null ? (h.vo_ok >= 0.5 ? "OK" : "LOST") : "—";
    $("slip-val").textContent = h.slip != null ? h.slip.toFixed(2) : "—";
    renderSpark();
  }

  function ticks(id, max, step, fmt) {
    const el = $(id); if (el.childElementCount) return;
    for (let v = 0; v <= max + 1e-9; v += step) { const s = document.createElement("span"); s.textContent = fmt(v); el.appendChild(s); }
  }

  function fitCanvas(cv) {
    const r = cv.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: r.height };
  }

  function renderSpark() {
    const { ctx, w, h } = fitCanvas($("spark"));
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = THEME.well; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = THEME.grid; ctx.lineWidth = 1;
    for (let v = 0.5; v < V_MAX; v += 0.5) { const y = h - (v / V_MAX) * h; ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); ctx.stroke(); }
    const pk = S.packets; if (pk.length < 2) return;
    const tN = pk[pk.length - 1].t;
    const pts = pk.filter((q) => q.t >= tN - SPARK_WINDOW_S);
    const X = (t) => w - (tN - t) / SPARK_WINDOW_S * w, Y = (v) => h - Math.min(v, V_MAX) / V_MAX * (h - 4) - 2;
    const line = (key, col, width) => {
      ctx.strokeStyle = col; ctx.lineWidth = width; ctx.beginPath();
      pts.forEach((q, i) => { const x = X(q.t), y = Y(q[key]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    };
    line("vcap", THEME.accent, 2); line("speed", THEME.ink, 1.5);
  }

  function renderMap() {
    const cv = $("map"); const { ctx, w, h } = fitCanvas(cv);
    ctx.fillStyle = THEME.well; ctx.fillRect(0, 0, w, h);
    const size = Math.min(w, h); const ppm = size / S.extent; const cx = w / 2, cy = h / 2;
    const B2P = (b) => [cx - b[1] * ppm, cy - b[0] * ppm]; // body (x fwd, y left) -> canvas
    // metric grid (every 2 m / 4 m / 8 m depending on zoom), aligned to the vehicle
    const g = S.extent <= 16 ? 2 : S.extent <= 24 ? 2 : 4;
    ctx.strokeStyle = THEME.grid; ctx.lineWidth = 1;
    for (let k = -Math.ceil(w / ppm / g); k <= Math.ceil(w / ppm / g); k++) {
      const x = Math.round(cx + k * g * ppm) + 0.5; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let k = -Math.ceil(h / ppm / g); k <= Math.ceil(h / ppm / g); k++) {
      const y = Math.round(cy + k * g * ppm) + 0.5; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    const p = S.last;
    if (!p) {
      ctx.fillStyle = THEME.text3; ctx.font = `14px ${getComputedStyle(document.body).fontFamily}`; ctx.textAlign = "center";
      ctx.fillText("No telemetry yet", cx, cy); return;
    }
    // costmap (body-aligned, centred on the vehicle)
    if (p._costmap) {
      if (!S.costCanvas) { S.costCanvas = document.createElement("canvas"); S.costCanvas.width = COSTMAP_N; S.costCanvas.height = COSTMAP_N; }
      const cctx = S.costCanvas.getContext("2d"); const img = cctx.createImageData(COSTMAP_N, COSTMAP_N);
      for (let i = 0; i < p._costmap.length; i++) { const c = U4[p._costmap[i]]; img.data.set([c[0], c[1], c[2], 255], i * 4); }
      cctx.putImageData(img, 0, 0);
      const half = COSTMAP_N * COSTMAP_RES / 2 * ppm;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(S.costCanvas, cx - half, cy - half, 2 * half, 2 * half);
      ctx.strokeStyle = THEME.text2; ctx.lineWidth = 1; ctx.strokeRect(cx - half + 0.5, cy - half + 0.5, 2 * half - 1, 2 * half - 1);
    }
    const pose = p.pose;
    // trail (A-frame history)
    if (S.trail.length > 1) {
      ctx.strokeStyle = THEME.vo; ctx.lineWidth = 2.5; ctx.lineJoin = "round"; ctx.beginPath();
      S.trail.forEach((q, i) => { const c = B2P(toBody(pose, q[0], q[1])); i ? ctx.lineTo(c[0], c[1]) : ctx.moveTo(c[0], c[1]); });
      ctx.stroke();
    }
    // next waypoints
    if (p.waypoints && p.waypoints.length) {
      const pts = [[0, 0]].concat(p.waypoints.map((q) => toBody(pose, q[0], q[1]))).map(B2P);
      ctx.strokeStyle = THEME.white; ctx.lineWidth = 6; ctx.beginPath(); pts.forEach((c, i) => (i ? ctx.lineTo(c[0], c[1]) : ctx.moveTo(c[0], c[1]))); ctx.stroke();
      ctx.strokeStyle = THEME.accent; ctx.lineWidth = 3; ctx.beginPath(); pts.forEach((c, i) => (i ? ctx.lineTo(c[0], c[1]) : ctx.moveTo(c[0], c[1]))); ctx.stroke();
      pts.slice(1).forEach((c) => { ctx.fillStyle = THEME.ink; ctx.beginPath(); ctx.arc(c[0], c[1], 5, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = THEME.white; ctx.beginPath(); ctx.arc(c[0], c[1], 3, 0, 2 * Math.PI); ctx.fill(); });
    }
    drawGoal(ctx, pose, B2P, ppm, w, h);
    drawVehicle(ctx, cx, cy, ppm);
    drawAxis(ctx, w - 40, 40, pose[2]);
    drawScale(ctx, 14, h - 16, ppm);
    // stale / sigma annotations
    ctx.font = `12px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
    ctx.fillStyle = THEME.text2; ctx.textAlign = "left";
    ctx.fillText(`pose ${pose[0].toFixed(1)}, ${pose[1].toFixed(1)} m · ψ ${(pose[2] * 180 / Math.PI).toFixed(0)}° · σ ${p.pos_sigma_m.toFixed(2)} m`, 14, 22);
  }

  function drawVehicle(ctx, cx, cy, ppm) {
    const L = 0.8 * ppm, W = 0.6 * ppm; // defaults.VEHICLE length / width
    ctx.fillStyle = THEME.ink; ctx.strokeStyle = THEME.white; ctx.lineWidth = 2;
    ctx.fillRect(cx - W / 2, cy - L / 2, W, L); ctx.strokeRect(cx - W / 2, cy - L / 2, W, L);
    ctx.fillStyle = THEME.white; ctx.beginPath();
    ctx.moveTo(cx, cy - L / 2 - Math.max(6, 0.18 * ppm)); ctx.lineTo(cx - Math.max(5, 0.16 * ppm), cy - L / 2 + 2); ctx.lineTo(cx + Math.max(5, 0.16 * ppm), cy - L / 2 + 2);
    ctx.closePath(); ctx.fill();
  }

  function drawGoal(ctx, pose, B2P, ppm, w, h) {
    const m = S.mission; if (!m || !m.goal_xy_a) return;
    const b = toBody(pose, m.goal_xy_a[0], m.goal_xy_a[1]); const c = B2P(b); const r = (+m.goal_sigma_m || 0) * ppm;
    const dist = Math.hypot(b[0], b[1]);
    const inside = c[0] > -r && c[0] < w + r && c[1] > -r && c[1] < h + r;
    ctx.font = `12px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
    if (inside) {
      ctx.fillStyle = THEME.accentFill; ctx.beginPath(); ctx.arc(c[0], c[1], r, 0, 2 * Math.PI); ctx.fill();
      ctx.setLineDash([7, 5]); ctx.strokeStyle = THEME.accent; ctx.lineWidth = 2; ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = THEME.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(c[0], c[1], 8, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(c[0] - 13, c[1]); ctx.lineTo(c[0] + 13, c[1]); ctx.moveTo(c[0], c[1] - 13); ctx.lineTo(c[0], c[1] + 13); ctx.stroke();
      label(ctx, c[0] + 14, c[1] - 16, `B · ${dist.toFixed(1)} m · σ ${(+m.goal_sigma_m).toFixed(1)} m`);
    } else {
      const dx = c[0] - w / 2, dy = c[1] - h / 2; const k = Math.min((w / 2 - 24) / Math.abs(dx || 1e-6), (h / 2 - 24) / Math.abs(dy || 1e-6));
      const ex = w / 2 + dx * k, ey = h / 2 + dy * k; const n = Math.hypot(dx, dy) || 1; const ux = dx / n, uy = dy / n;
      ctx.fillStyle = THEME.accent; ctx.beginPath(); ctx.moveTo(ex + ux * 12, ey + uy * 12);
      ctx.lineTo(ex - ux * 8 - uy * 9, ey - uy * 8 + ux * 9); ctx.lineTo(ex - ux * 8 + uy * 9, ey - uy * 8 - ux * 9); ctx.closePath(); ctx.fill();
      label(ctx, Math.min(Math.max(ex - ux * 34, 8), w - 120), Math.min(Math.max(ey - uy * 30, 14), h - 14), `B ${dist.toFixed(0)} m`);
    }
  }

  function label(ctx, x, y, s) {
    const tw = ctx.measureText(s).width;
    ctx.fillStyle = THEME.ink; ctx.fillRect(x, y - 9, tw + 10, 18);
    ctx.fillStyle = THEME.white; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(s, x + 5, y + 1); ctx.textBaseline = "alphabetic";
  }

  function drawAxis(ctx, x, y, yaw) {
    // +x_A drawn relative to the track-up view: rotate by the vehicle yaw
    ctx.fillStyle = THEME.ink; ctx.strokeStyle = THEME.ink; ctx.beginPath(); ctx.arc(x, y, 22, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    const ux = Math.sin(yaw), uy = -Math.cos(yaw); // screen dir of +x_A
    const nx = -uy, ny = ux;
    ctx.fillStyle = THEME.white; ctx.beginPath(); ctx.moveTo(x + ux * 15, y + uy * 15); ctx.lineTo(x - ux * 8 + nx * 7, y - uy * 8 + ny * 7);
    ctx.lineTo(x - ux * 3, y - uy * 3); ctx.lineTo(x - ux * 8 - nx * 7, y - uy * 8 - ny * 7); ctx.closePath(); ctx.fill();
    ctx.fillStyle = THEME.text3; ctx.font = "bold 11px " + getComputedStyle(document.body).fontFamily; ctx.textAlign = "center";
    ctx.fillText("x_A", x, y + 36); ctx.fillText("FWD ↑", x - 58, y + 4);
  }

  function drawScale(ctx, x, y, ppm) {
    const L = S.extent <= 16 ? 2 : S.extent <= 24 ? 5 : 10; const n = L * ppm;
    ctx.fillStyle = THEME.ink; ctx.fillRect(x, y - 2, n, 3); ctx.fillRect(x, y - 7, 2, 10); ctx.fillRect(x + n - 2, y - 7, 2, 10);
    ctx.font = `12px ${getComputedStyle(document.body).getPropertyValue("--mono")}`; ctx.textAlign = "center"; ctx.fillText(`${L} m`, x + n / 2, y - 9);
  }

  function renderLog() {
    const ol = $("log"); ol.textContent = "";
    S.log.slice(0, 60).forEach((e) => {
      const li = document.createElement("li");
      const t = document.createElement("span"); t.className = "t mono"; t.textContent = fmtClock(e.t);
      const m = document.createElement("span"); m.className = "m";
      const sw = document.createElement("i"); sw.style.background = e.op ? THEME.accent : (MODE_RGB[e.mode] || MODE_RGB.HOLD);
      m.appendChild(sw); m.appendChild(document.createTextNode(e.op ? "OPERATOR" : (e.prev ? `${e.prev.replace(/_/g, " ")} → ${e.mode.replace(/_/g, " ")}` : e.mode.replace(/_/g, " "))));
      const r = document.createElement("span"); r.className = "r mono"; r.textContent = e.reason || "";
      li.append(t, m, r); ol.appendChild(li);
    });
  }

  function buildLegend() {
    const el = $("legend");
    LEGEND.forEach(([name, c]) => { const d = document.createElement("div"); const s = document.createElement("span"); s.className = "sw"; s.style.background = rgb(c); d.append(s, name); el.appendChild(d); });
  }

  // ----------------------------------------------------------------- wiring
  function tickWall() {
    if (DETERMINISTIC) return;
    const d = new Date(); $("clock-wall").textContent = d.toTimeString().slice(0, 8);
    if (S.last && !S.replayTimer && performance.now() - S.lastWall > STALE_S * 1000 && $("replay-badge").hidden) $("link-dot").className = "dot warn";
  }

  function readFiles(files) {
    const arr = Array.from(files);
    const mission = arr.find((f) => /mission.*\.json$/i.test(f.name));
    const tel = arr.find((f) => /\.jsonl$/i.test(f.name));
    const go = () => tel && tel.text().then((txt) => consoleReplay(txt));
    if (mission) mission.text().then((txt) => { consoleReset(JSON.parse(txt)); go(); }); else { if (!S.mission) consoleReset(null); go(); }
  }

  function init() {
    buildLegend();
    ticks("speed-ticks", V_MAX, 0.5, (v) => v.toFixed(1));
    ticks("rcert-ticks", R_MAX, 2, (v) => v.toFixed(0));
    document.querySelectorAll("#zoom button").forEach((b) => b.addEventListener("click", () => {
      S.extent = +b.dataset.extent; document.querySelectorAll("#zoom button").forEach((x) => x.classList.toggle("on", x === b)); renderMap();
    }));
    document.querySelectorAll(".btn").forEach((b) => b.addEventListener("click", () => sendCommand(b.dataset.action)));
    window.addEventListener("resize", () => render());
    window.addEventListener("dragover", (e) => { e.preventDefault(); $("drop-hint").hidden = false; });
    window.addEventListener("dragleave", () => { $("drop-hint").hidden = true; });
    window.addEventListener("drop", (e) => { e.preventDefault(); $("drop-hint").hidden = true; readFiles(e.dataTransfer.files); });
    if (DETERMINISTIC) $("wall-wrap").hidden = true; else { tickWall(); setInterval(tickWall, 1000); }
    consoleReset(null);
    const rp = params.get("replay"), ms = params.get("mission");
    const load = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.text(); });
    if (ms) load(ms).then((t) => consoleReset(JSON.parse(t))).then(() => rp && load(rp).then((t) => consoleReplay(t)))
      .catch((e) => addLog(0, "HOLD", String(e)));
    else if (rp) load(rp).then((t) => consoleReplay(t)).catch((e) => { addLog(0, "HOLD", String(e)); renderLog(); });
    window.__consoleReady = true;
  }

  Object.assign(window, { consoleReset, pushTelemetry, consoleReplay, consoleSeek, consoleSendCommand: sendCommand });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
