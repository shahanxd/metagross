"""Check slot figures: min text size, text-text overlaps, text outside canvas, PNG pixel sizes."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(r"D:\Downloads\sih again\metagross")
sys.path.insert(0, str(REPO))
SLOTS = REPO / "deck_assets" / "slots"
EXPECT = {"s2_how_it_works": (3584, 712), "s3_missing_ground": (2360, 768), "s3_models": (2360, 572),
          "s4_envelope": (1464, 696), "s2_ditch_theory": (1176, 440)}
MIN_PX = 14.0


def overlaps(boxes, tol=0.5):
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ix = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            iy = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
            if ix > tol and iy > tol:
                out.append((a["t"], b["t"], round(ix, 1), round(iy, 1)))
    return out


def check_svgs():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        page = br.new_page()
        for name in ("s2_how_it_works", "s3_missing_ground", "s3_models"):
            svg = (SLOTS / f"{name}.svg").read_text(encoding="utf-8")
            page.set_content(f'<html><body style="margin:0">{svg}</body></html>')
            page.evaluate("document.fonts.ready")
            boxes = page.evaluate("""() => {
              const svg = document.querySelector('svg'); const o = svg.getBoundingClientRect();
              return [...svg.querySelectorAll('text')].map(t => { const r = t.getBoundingClientRect();
                // tight vertical box: use font metrics approx (cap height region) via getBBox height scaled
                return {t: t.textContent, fs: parseFloat(t.getAttribute('font-size')), x0: r.left - o.left,
                        y0: r.top - o.top, x1: r.right - o.left, y1: r.bottom - o.top}; });
            }""")
            W, H = int(svg.split('width="', 1)[1].split('"', 1)[0]), int(svg.split('height="', 1)[1].split('"', 1)[0])
            print(f"== {name} ({W}x{H}) texts={len(boxes)} min_fs={min(b['fs'] for b in boxes)}")
            for b in boxes:
                if b["fs"] < MIN_PX:
                    print("  SMALL", b["t"], b["fs"])
                if b["x0"] < 0 or b["y0"] < 0 or b["x1"] > W or b["y1"] > H:
                    print("  OUT", b["t"], [round(b[k], 1) for k in ("x0", "y0", "x1", "y1")])
            # shrink boxes vertically to ~ascender-descender of glyphs (getBoundingClientRect uses line box)
            tight = []
            for b in boxes:
                h = b["y1"] - b["y0"]
                pad = h * 0.12
                tight.append({**b, "y0": b["y0"] + pad, "y1": b["y1"] - pad})
            for o in overlaps(tight):
                print("  OVERLAP", o)
        br.close()


def check_charts():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.text import Text
    from metagross.eval import plot_style as ps
    from metagross.eval import theory as th
    p = th.TheoryParams()
    with ps.deck_style():
        for name, maker in th.SLOT_FIGURES.items():
            fig = maker(p)
            fig.set_dpi(150)  # 1 px == 1 slide px
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            W, H = fig.bbox.width, fig.bbox.height
            boxes = []
            for t in fig.findobj(Text):
                if not t.get_visible() or not t.get_text().strip():
                    continue
                bb = Text.get_window_extent(t, rend)  # text only (an Annotation's own extent includes its leader)
                if bb.x0 > W or bb.x1 < 0:  # tick labels outside the view limits are never drawn
                    continue
                px = t.get_fontsize() * 150 / 72
                boxes.append({"t": t.get_text(), "fs": px, "x0": bb.x0, "x1": bb.x1, "y0": H - bb.y1, "y1": H - bb.y0})
            print(f"== {name} ({W:.0f}x{H:.0f}) texts={len(boxes)} min_fs={min(b['fs'] for b in boxes):.1f}")
            for b in boxes:
                if b["fs"] < MIN_PX - 1e-6:
                    print("  SMALL", repr(b["t"]), round(b["fs"], 1))
                if b["x0"] < -0.5 or b["y0"] < -0.5 or b["x1"] > W + 0.5 or b["y1"] > H + 0.5:
                    print("  OUT", repr(b["t"]), [round(b[k], 1) for k in ("x0", "y0", "x1", "y1")])
            for o in overlaps(boxes):
                print("  OVERLAP", o)
            import matplotlib.pyplot as plt
            plt.close(fig)


def check_pngs():
    from PIL import Image
    for name, (w, h) in EXPECT.items():
        f = SLOTS / f"{name}.png"
        if not f.exists():
            print("MISSING", f)
            continue
        sz = Image.open(f).size
        print(f"{name}.png {sz} {'OK' if sz == (w, h) else 'EXPECTED ' + str((w, h))}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["svg", "chart", "png"]
    if "svg" in what:
        check_svgs()
    if "chart" in what:
        check_charts()
    if "png" in what:
        check_pngs()
