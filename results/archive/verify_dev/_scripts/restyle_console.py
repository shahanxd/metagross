"""One-off: switch the operator console to the white DESIGN_TOKENS theme."""
from pathlib import Path

UI = Path(r"D:\Downloads\sih again\metagross\operator_ui")

css = (UI / "style.css").read_text(encoding="utf-8")
lines = css.split("\n")
assert lines[0].startswith("/* METAGROSS operator console") and "One accent" in lines[1]
lines[0] = "/* METAGROSS operator console - clean white theme from docs/DESIGN_TOKENS.md (same tokens as the video)."
lines[1] = "   One accent (#1D4ED8); every other colour is neutral or semantic (cell states, drive modes). */"
css = "\n".join(lines)
css_rep = [
    ("--bg: #0B1220;", "--bg: #F1F5F9;"),
    ("--panel: #111A2E;", "--panel: #FFFFFF;"),
    ("--well: #0E1628;", "--well: #F8FAFC;"),
    ("--border: #1E293B;", "--border: #E2E8F0;"),
    ("--text: #E2E8F0;", "--text: #0F172A;"),
    ("--text-2: #94A3B8;", "--text-2: #475569;"),
    ("--text-3: #64748B;", "--text-3: #64748B;"),
    ("--accent: #3B82F6;", "--accent: #1D4ED8;"),
    ("--m-nominal: #16A34A;", "--m-nominal: #15803D;"),
    ("--m-caution: #F59E0B;", "--m-caution: #CA8A04;"),
    ("--m-degraded: #EA580C;", "--m-degraded: #EA580C;"),
    ("--m-stop: #7C3AED;", "--m-stop: #DC2626;"),
    ("--m-safe: #DC2626;", "--m-safe: #7F1D1D;"),
    ("--m-arrived: #0EA5E9;", "--m-arrived: #1D4ED8;"),
    (".badge-deny { color: #FCA5A5; background: #3B1219; border-color: #7F1D1D; }",
     ".badge-deny { color: #B91C1C; background: #FEF2F2; border-color: #FCA5A5; }"),
    ('#banner[data-mode="CAUTION"] .mode-chip { color: #0F172A; }', '#banner[data-mode="CAUTION"] .mode-chip { color: #fff; }'),
    ("#log li:nth-child(odd) { background: rgba(14, 22, 40, 0.6); }", "#log li:nth-child(odd) { background: rgba(241, 245, 249, 0.8); }"),
    (".btn.go { background: #0F2A1A; border-color: #166534; color: #BBF7D0; }", ".btn.go { background: #F0FDF4; border-color: #15803D; color: #15803D; }"),
    (".btn.estop { background: var(--m-safe); border-color: #991B1B; color: #fff; }", ".btn.estop { background: #DC2626; border-color: #991B1B; color: #fff; }"),
    ("background: rgba(11, 18, 32, 0.92);", "background: rgba(255, 255, 255, 0.92);"),
]
for a, b in css_rep:
    assert a in css, a
    css = css.replace(a, b)
(UI / "style.css").write_text(css, encoding="utf-8")

js = (UI / "app.js").read_text(encoding="utf-8")
js_rep = [
    ('''  const MODE_RGB = {
    NOMINAL: "#16A34A", CAUTION: "#F59E0B", DEGRADED: "#EA580C", STOP_AND_LOOK: "#7C3AED",
    SAFE_STOP: "#DC2626", ARRIVED: "#0EA5E9", HOLD: "#64748B"
  };''',
     '''  // Drive-mode colours = docs/DESIGN_TOKENS.md (metagross/eval/plot_style.MODE_COLORS; video/style.MODE_HEX).
  const MODE_RGB = {
    NOMINAL: "#15803D", CAUTION: "#CA8A04", DEGRADED: "#EA580C", STOP_AND_LOOK: "#DC2626",
    SAFE_STOP: "#7F1D1D", ARRIVED: "#1D4ED8", HOLD: "#64748B"
  };
  // Canvas colours for the white theme (DESIGN_TOKENS neutrals; ink = text colour used on light map cells).
  const THEME = { well: "#F8FAFC", grid: "#E2E8F0", ink: "#0F172A", text2: "#475569", text3: "#64748B",
    accent: "#1D4ED8", vo: "#2563EB", white: "#FFFFFF", accentFill: "rgba(29,78,216,0.14)" };'''),
    ('ctx.fillStyle = "#0E1628"; ctx.fillRect(0, 0, w, h);\n    ctx.strokeStyle = "#1E293B"; ctx.lineWidth = 1;\n    for (let v',
     'ctx.fillStyle = THEME.well; ctx.fillRect(0, 0, w, h);\n    ctx.strokeStyle = THEME.grid; ctx.lineWidth = 1;\n    for (let v'),
    ('line("vcap", "#3B82F6", 2); line("speed", "#E2E8F0", 1.5);', 'line("vcap", THEME.accent, 2); line("speed", THEME.ink, 1.5);'),
    ('ctx.fillStyle = "#0E1628"; ctx.fillRect(0, 0, w, h);\n    const size', 'ctx.fillStyle = THEME.well; ctx.fillRect(0, 0, w, h);\n    const size'),
    ('ctx.strokeStyle = "#1E293B"; ctx.lineWidth = 1;\n    for (let k', 'ctx.strokeStyle = THEME.grid; ctx.lineWidth = 1;\n    for (let k'),
    ('ctx.fillStyle = "#64748B"; ctx.font = `14px', 'ctx.fillStyle = THEME.text3; ctx.font = `14px'),
    ('ctx.strokeStyle = "#475569"; ctx.lineWidth = 1; ctx.strokeRect(', 'ctx.strokeStyle = THEME.text2; ctx.lineWidth = 1; ctx.strokeRect('),
    ('ctx.strokeStyle = "#93C5FD"; ctx.lineWidth = 2.5;', 'ctx.strokeStyle = THEME.vo; ctx.lineWidth = 2.5;'),
    ('ctx.strokeStyle = "#0F172A"; ctx.lineWidth = 6;', 'ctx.strokeStyle = THEME.white; ctx.lineWidth = 6;'),
    ('ctx.strokeStyle = "#3B82F6"; ctx.lineWidth = 3;', 'ctx.strokeStyle = THEME.accent; ctx.lineWidth = 3;'),
    ('pts.slice(1).forEach((c) => { ctx.fillStyle = "#0F172A";', 'pts.slice(1).forEach((c) => { ctx.fillStyle = THEME.ink;'),
    ('ctx.fillStyle = "#FFFFFF"; ctx.beginPath(); ctx.arc(c[0], c[1], 3,', 'ctx.fillStyle = THEME.white; ctx.beginPath(); ctx.arc(c[0], c[1], 3,'),
    ('ctx.fillStyle = "#94A3B8"; ctx.textAlign = "left";', 'ctx.fillStyle = THEME.text2; ctx.textAlign = "left";'),
    ('ctx.fillStyle = "#0F172A"; ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 2;', 'ctx.fillStyle = THEME.ink; ctx.strokeStyle = THEME.white; ctx.lineWidth = 2;'),
    ('ctx.fillStyle = "#FFFFFF"; ctx.beginPath();\n    ctx.moveTo(cx,', 'ctx.fillStyle = THEME.white; ctx.beginPath();\n    ctx.moveTo(cx,'),
    ('ctx.fillStyle = "rgba(59,130,246,0.14)";', 'ctx.fillStyle = THEME.accentFill;'),
    ('ctx.setLineDash([7, 5]); ctx.strokeStyle = "#3B82F6";', 'ctx.setLineDash([7, 5]); ctx.strokeStyle = THEME.accent;'),
    ('ctx.strokeStyle = "#E2E8F0"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(c[0], c[1], 8,', 'ctx.strokeStyle = THEME.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(c[0], c[1], 8,'),
    ('ctx.fillStyle = "#3B82F6"; ctx.beginPath(); ctx.moveTo(ex', 'ctx.fillStyle = THEME.accent; ctx.beginPath(); ctx.moveTo(ex'),
    ('ctx.fillStyle = "#0F172A"; ctx.fillRect(x, y - 9, tw + 10, 18);', 'ctx.fillStyle = THEME.ink; ctx.fillRect(x, y - 9, tw + 10, 18);'),
    ('ctx.fillStyle = "#FFFFFF"; ctx.textAlign = "left"; ctx.textBaseline = "middle";', 'ctx.fillStyle = THEME.white; ctx.textAlign = "left"; ctx.textBaseline = "middle";'),
    ('ctx.fillStyle = "#0F172A"; ctx.strokeStyle = "#1E293B"; ctx.beginPath(); ctx.arc(x, y, 22,', 'ctx.fillStyle = THEME.ink; ctx.strokeStyle = THEME.ink; ctx.beginPath(); ctx.arc(x, y, 22,'),
    ('ctx.fillStyle = "#FFFFFF"; ctx.beginPath(); ctx.moveTo(x + ux * 15', 'ctx.fillStyle = THEME.white; ctx.beginPath(); ctx.moveTo(x + ux * 15'),
    ('ctx.fillStyle = "#94A3B8"; ctx.font = "bold 11px "', 'ctx.fillStyle = THEME.text3; ctx.font = "bold 11px "'),
    ('ctx.fillStyle = "#E2E8F0"; ctx.fillRect(x, y - 2, n, 3);', 'ctx.fillStyle = THEME.ink; ctx.fillRect(x, y - 2, n, 3);'),
    ('sw.style.background = e.op ? "#3B82F6" : (MODE_RGB[e.mode] || "#64748B");', 'sw.style.background = e.op ? THEME.accent : (MODE_RGB[e.mode] || MODE_RGB.HOLD);'),
]
for a, b in js_rep:
    assert js.count(a) == 1, (js.count(a), a)
    js = js.replace(a, b)
(UI / "app.js").write_text(js, encoding="utf-8")
print("ok")
