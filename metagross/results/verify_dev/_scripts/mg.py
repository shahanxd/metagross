def diagram_missing_ground() -> Svg:
    W, H = 1840, 720
    s = Svg(W, H)
    green, grey, mag = C[CellState.GROUND], C[CellState.UNSEEN], C[CellState.DITCH_CANDIDATE]
    g_y = 520.0  # ground line [px]
    cam = (190.0, 270.0)  # camera optical centre [px]
    lip, far = 820.0, 990.0  # near lip / far wall x [px] (w = far - lip)
    depth_y = 650.0

    s.card(20, 20, 1180, 680, fill=T["bg"], stroke=T["border"])
    s.text(44, 60, "SIDE VIEW (SCHEMATIC)", size=12, weight=700, fill=T["text_muted"], ls=1.2)
    s.text(44, 98, "A ditch is missing ground, not an object", size=24, weight=700, fill=T["navy"])
    s.text(44, 128, "Its opening shrinks as 1/R² and its interior is never observed.", size=16, fill=T["text_secondary"])

    # terrain body, then the surface outline
    s.polygon([(40, g_y), (lip, g_y), (lip, depth_y), (far, depth_y), (far, g_y), (1180, g_y), (1180, 690), (40, 690)],
              fill=T["surface"])
    # the grazing ray through the near lip meets the far wall at y_hit
    slope = (g_y - cam[1]) / (lip - cam[0])
    y_hit = g_y + slope * (far - lip)
    # visible-angle wedge between the near-lip and far-lip rays
    s.polygon([(cam[0] + 18, cam[1]), (lip, g_y), (far, g_y)], fill=T["accent"], opacity=0.10)
    # hidden interior (never observed), with a light hatch
    s.polygon([(lip, g_y), (far, y_hit), (far, depth_y), (lip, depth_y)], fill=grey)
    for k in range(10):
        x1 = lip + 6 + k * 17
        s.line(x1, depth_y - 3, min(x1 + 50, far - 3), depth_y - 3 - (min(x1 + 50, far - 3) - x1), stroke=T["bg"], sw=1.1)
    s.path(f"M40,{g_y} L{lip},{g_y} L{lip},{depth_y} L{far},{depth_y} L{far},{g_y} L1180,{g_y}", stroke=T["text"], sw=2.4)
    s.line(far, g_y, far, y_hit, stroke=mag, sw=7)  # visible part of the far wall
    s.line(330, g_y, lip, g_y, stroke=green, sw=7)
    s.line(far, g_y, 1178, g_y, stroke=green, sw=7)

    # vehicle, mast, camera
    s.rect(110, g_y - 60, 170, 42, fill=T["surface"], stroke=T["text_secondary"], sw=1.6, rx=10)
    for wx_ in (140, 250):
        s.circle(wx_, g_y - 16, 16, fill=T["text"], stroke=T["bg"], sw=2)
    s.line(cam[0], g_y - 60, cam[0], cam[1] + 12, stroke=T["text_secondary"], sw=4)
    s.rect(cam[0] - 26, cam[1] - 14, 52, 28, fill=T["bg"], stroke=T["text"], sw=2, rx=6)
    s.circle(cam[0] + 14, cam[1], 7, fill=T["accent"])

    # rays
    for gx in (380, 520, 670):
        s.line(cam[0] + 18, cam[1], gx, g_y, stroke=green, sw=1.4, dash="5 5")
        s.circle(gx, g_y, 4.5, fill=green)
    s.line(cam[0] + 18, cam[1], far, y_hit, stroke=mag, sw=2)
    s.circle(far, y_hit, 5, fill=mag)
    s.line(cam[0] + 18, cam[1], far, g_y, stroke=T["accent"], sw=1.6)
    for gx in (1070, 1150):
        s.line(cam[0] + 18, cam[1], gx, g_y, stroke=green, sw=1.4, dash="5 5")
        s.circle(gx, g_y, 4.5, fill=green)

    # theta callout, anchored on the wedge
    a1 = math.atan2(g_y - cam[1], lip - cam[0])
    a2 = math.atan2(g_y - cam[1], far - cam[0])
    r_arc = 600.0
    p1 = (cam[0] + 18 + r_arc * math.cos(a1), cam[1] + r_arc * math.sin(a1))
    p2 = (cam[0] + 18 + r_arc * math.cos(a2), cam[1] + r_arc * math.sin(a2))
    s.path(f"M{p2[0]:.1f},{p2[1]:.1f} A{r_arc},{r_arc} 0 0 1 {p1[0]:.1f},{p1[1]:.1f}", stroke=T["accent"], sw=3)
    bx, by = 560.0, 190.0
    s.rect(bx, by, 300, 88, fill=T["bg"], stroke=T["accent"], sw=1.5, rx=12)
    s.text(bx + 20, by + 38, "θ ≈ H·w / R²", size=24, weight=700, mono=True, fill=T["accent"])
    s.text(bx + 20, by + 68, "visible angle of the opening", size=14, fill=T["text_secondary"])
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    s.line(bx + 150, by + 90, mx, my - 6, stroke=T["accent"], sw=1.4)

    # dimensions: H (left of the vehicle), R (below ground), w (above the ditch)
    hx = 70.0
    s.line(hx, cam[1] + 2, hx, g_y - 2, stroke=T["text"], sw=1.4, arrow=True)
    s.line(hx, g_y - 2, hx, cam[1] + 2, stroke=T["text"], sw=1.4, arrow=True)
    s.line(hx - 8, cam[1], cam[0] - 30, cam[1], stroke=T["text_muted"], sw=1, dash="3 4")
    s.text(hx - 12, (cam[1] + g_y) / 2 + 7, "H", size=20, weight=700, anchor="end", fill=T["text"])
    ry = 590.0
    s.line(cam[0], ry, lip - 2, ry, stroke=T["text"], sw=1.4, arrow=True)
    s.line(lip - 2, ry, cam[0] + 2, ry, stroke=T["text"], sw=1.4, arrow=True)
    s.rect((cam[0] + lip) / 2 - 20, ry - 16, 40, 30, fill=T["surface"])
    s.text((cam[0] + lip) / 2, ry + 7, "R", size=20, weight=700, anchor="middle", fill=T["text"])
    wy_ = g_y - 36
    s.line(lip + 2, wy_, far - 2, wy_, stroke=T["text"], sw=1.4, arrow=True)
    s.line(far - 2, wy_, lip + 2, wy_, stroke=T["text"], sw=1.4, arrow=True)
    s.rect((lip + far) / 2 - 14, wy_ - 34, 28, 26, fill=T["bg"])
    s.text((lip + far) / 2, wy_ - 14, "w", size=20, weight=700, anchor="middle", fill=T["text"])

    # callouts
    s.text(lip - 10, g_y + 30, "near lip", size=14, anchor="end", fill=T["text"], weight=700)
    s.lines((lip + far) / 2, depth_y - 40, ["hidden interior:", "never observed"], size=14, lh=19, anchor="middle",
            fill=T["text"], weight=700)
    s.lines(far + 16, y_hit - 22, ["far wall reappears", "below lip height"], size=14, lh=19, fill=mag, weight=700)
    s.text(44, 640, "Seen ground is certified. The gap between lip and far wall is", size=15, fill=T["text"])
    s.text(44, 662, "MISSING GROUND: kept unseen / ditch, never planned as free.", size=15, weight=700, fill=T["text"])

    # ---------------------------------------------------------------- inset: column profile
    ix, iy, iw, ih = 1230, 20, 590, 680
    s.card(ix, iy, iw, ih, fill=T["bg"], stroke=T["border"])
    s.text(ix + 24, iy + 40, "ONE IMAGE COLUMN, BOTTOM → UP", size=12, weight=700, fill=T["text_muted"], ls=1.2)
    s.text(ix + 24, iy + 68, "Measured range per image row", size=18, weight=700, fill=T["text"])
    px0, py0, pw_, ph_ = ix + 70, iy + 100, iw - 110, 340  # plot area
    s.line(px0, py0 + ph_, px0 + pw_, py0 + ph_, stroke=T["border"], sw=1.4)
    s.line(px0, py0 + ph_, px0, py0, stroke=T["border"], sw=1.4)
    s.text(px0 + pw_ / 2, py0 + ph_ + 30, "image row  (near → far)", size=14, anchor="middle", fill=T["text_secondary"])
    s.raw(f'<text x="{px0 - 22}" y="{py0 + ph_ / 2}" font-family="{FONT}" font-size="14" fill="{T["text_secondary"]}" '
          f'text-anchor="middle" transform="rotate(-90 {px0 - 22} {py0 + ph_ / 2})">range</text>')
    # Schematic profile: ground rows rise smoothly; the vertical far wall is seen over several rows at a
    # constant range ~R + w, then far ground resumes. Illustrative numbers (not to scale).
    r_lip, w_d, n = 3.0, 1.4, 44

    def ground_range(u: float) -> float:  # flat-ground range at normalised row u (0 = bottom row)
        ang = math.radians(38.0) * (1 - u) + math.radians(9.0) * u
        return D.CAM_HEIGHT_M / math.tan(ang)

    r_top = ground_range(1.0) + 0.3

    def to_px(u: float, r: float) -> tuple[float, float]:
        return px0 + 10 + u * (pw_ - 20), py0 + ph_ - r / r_top * (ph_ - 10)

    us = [i / (n - 1) for i in range(n)]
    for u in us:
        r = ground_range(u)
        if r < r_lip:
            s.circle(*to_px(u, r), 4, fill=green)
        elif r < r_lip + w_d:
            s.circle(*to_px(u, r_lip + w_d), 4.5, fill=mag)  # far wall: constant range
        else:
            s.circle(*to_px(u, r), 4, fill=green)
    u_lip = next(u for u in us if ground_range(u) >= r_lip)
    ja, jb = to_px(u_lip, r_lip), to_px(u_lip, r_lip + w_d)
    s.line(ja[0] - 16, ja[1] + 2, ja[0] - 16, jb[1] + 4, stroke=T["text"], sw=1.6, arrow=True)
    s.lines(ja[0] - 26, (ja[1] + jb[1]) / 2 + 2, ["range", "jump ≈ w"], size=14, lh=18, anchor="end", fill=T["text"],
            weight=700)
    s.text(to_px(0.62, 0)[0], jb[1] - 16, "far wall: flat range", size=13, fill=mag, weight=700, anchor="middle")
    ty = py0 + ph_ + 76
    s.lines(ix + 24, ty, ["Ground returns rise smoothly with the row, then jump.",
                          "The rows in between would have seen the ditch floor."], size=14, lh=20, fill=T["text"])
    s.swatch(ix + 24, ty + 44, mag, size=14)
    s.text(ix + 46, ty + 56, "reappears near lip height → DITCH CANDIDATE", size=13.5, fill=T["text"])
    s.swatch(ix + 24, ty + 70, C[CellState.CREST_SHADOW], size=14)
    s.text(ix + 46, ty + 82, "reappears well below / slopes away → CREST", size=13.5, fill=T["text"])
    s.swatch(ix + 24, ty + 96, grey, size=14)
    s.text(ix + 46, ty + 108, "never seen → UNSEEN (never free)", size=13.5, fill=T["text"])
    return s


