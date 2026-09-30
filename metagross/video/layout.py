"""1920 x 1080 dashboard compositor for the demo video (white theme, ``docs/DESIGN_TOKENS.md``).

Single-run layout (pixels)::

    +--------------------------------------------------------------------------------+
    | METAGROSS | MISSION id | GNSS DENIED | PASSIVE STEREO | CONFIG  [SIMULATED] T+ |
    +-------------------------------------------+------------------------------------+
    | SIMULATOR VIEW (1152 x 648)               | HEALTH  mode chip . reason . q     |
    |  third-person chase camera, re-rendered   +------------------------------------+
    |  from the GT log (not a robot sensor)     | SEEN-GROUND MAP (track-up BEV)     |
    |                                           |  cells in the contract colour key, |
    |   [lower-third caption]                   |  MPPI rollouts, plan, VO/GT trails,|
    +------------------+------------------------+  goal + uncertainty circle, legend |
    | ONBOARD LEFT CAM | OPERATOR LINK          +------------------------------------+
    |  320x200 debug   |  2 Hz 4-bit costmap    | SPEED GOVERNOR  R_cert dial,       |
    |  image + overlays|  packet, rate, load    |  speed vs v_cap, q / sigma / ms    |
    +------------------+------------------------+------------------------------------+
    | [SIMULATED . seed N . config X]     narration subtitle     METAGROSS . SIH26126 |
    +--------------------------------------------------------------------------------+

The BEV panel is drawn relative to the autonomy's *own* pose estimate (what the robot
believes), dead-reckoned between debug ticks with the GT motion so the view is smooth at
30 fps; the true (GT) position is a ring, so drift is visible as an offset. The operator-link
panel shows only what crossed the simulated radio: the held telemetry packet (A-frame pose,
mode, v_cap, R_cert, waypoints and the ego-centred 64 x 64 4-bit costmap, 0.25 m cells).

:func:`compose_split` places two runs side by side (same seed, different configs).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from metagross.autonomy.link.codec import COSTMAP_N, COSTMAP_RES_M
from metagross.config import defaults
from metagross.contracts.messages import CellState, DriveMode
from metagross.contracts.scenario import DEV_SEEDS, EVAL_SEEDS
from video import style
from video.draw import Painter, arc_points, blend, text_width
from video.replay import LINK_WINDOW_S, PanelData

log = logging.getLogger(__name__)

W, H = 1920, 1080
MARGIN = 24
GAP = 12
TOPBAR_H = 64
BODY_Y0 = TOPBAR_H + GAP
FOOTER_Y0 = 1028

# Panel rectangles (x0, y0, x1, y1)
CHASE = (MARGIN, BODY_Y0, MARGIN + 1152, BODY_Y0 + 648)
ROW2_Y0 = CHASE[3] + GAP
ROW2_Y1 = FOOTER_Y0 - GAP
LEFTCAM = (MARGIN, ROW2_Y0, MARGIN + 432, ROW2_Y1)
LINK = (LEFTCAM[2] + GAP, ROW2_Y0, CHASE[2], ROW2_Y1)
RIGHT_X0 = CHASE[2] + GAP
BANNER = (RIGHT_X0, BODY_Y0, W - MARGIN, BODY_Y0 + 88)
BEV = (RIGHT_X0, BANNER[3] + GAP, W - MARGIN, 760)
GAUGES = (RIGHT_X0, BEV[3] + GAP, W - MARGIN, ROW2_Y1)
CHASE_SIZE = (CHASE[2] - CHASE[0], CHASE[3] - CHASE[1])

HEADER_H = 34  # panel title strip
PAD = 8
RADIUS = 10  # panel corner radius (DESIGN_TOKENS: 10 px inner boxes)

BEV_PX_PER_M = 26.0  # BEV panel scale at the reference width BEV_REF_W_PX
BEV_REF_W_PX = 692.0
BEV_VEHICLE_FRAC = 0.70  # vehicle position down the map (track-up view)
LEGEND_ROW_H = 22
RCERT_LABEL_MIN_M = 1.0  # BEV: label the R_cert arc only when it is at least this long, m
DEFAULT_T_R_S = 1.0 / defaults.CAMERA_HZ_DEMO + defaults.ACTUATOR_LAG_S  # reaction time if not logged, s
DARK_INK = style.TEXT  # map-overlay ink on light cell colours
PLAN_RGB = style.ACCENT
VO_RGB = style.ACCENT_LIGHT
GT_RGB = style.WHITE
LINK_CELL_PX = 3.5  # operator-link costmap: display pixels per 0.25 m cell (64 cells -> 224 px)
DIAL_SWEEP_RAD = 1.5 * math.pi  # R_cert dial: 270 degree arc
DIAL_START_RAD = 0.75 * math.pi  # arc start (image convention: 0 = +x, +pi/2 = down)


@dataclass(frozen=True)
class Caption:
    """Lower-third caption (title + optional subtitle)."""

    title: str
    subtitle: str = ""


# ============================================================================ helpers
def _ang_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def view_pose(pd: PanelData, replay_gt_at_tick: Optional[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Display pose: the held autonomy estimate advanced by the GT motion since that tick."""
    ex, ey, eyaw = pd.est_pose_a
    if replay_gt_at_tick is None:
        return ex, ey, eyaw
    gx0, gy0, gyaw0 = replay_gt_at_tick
    gx, gy, gyaw = pd.gt_pose_a
    c0, s0 = math.cos(gyaw0), math.sin(gyaw0)
    dxb = c0 * (gx - gx0) + s0 * (gy - gy0)  # motion since tick, in the tick body frame
    dyb = -s0 * (gx - gx0) + c0 * (gy - gy0)
    ce, se = math.cos(eyaw), math.sin(eyaw)
    return ex + ce * dxb - se * dyb, ey + se * dxb + ce * dyb, eyaw + _ang_diff(gyaw, gyaw0)


class BevView:
    """Track-up map projection: A-frame metres <-> panel pixels (vehicle at (cx, cy), heading up)."""

    def __init__(self, pose: tuple[float, float, float], cx: float, cy: float, px_per_m: float) -> None:
        self.x, self.y, self.yaw = pose
        self.cx, self.cy, self.s = cx, cy, px_per_m
        self.c, self.sn = math.cos(self.yaw), math.sin(self.yaw)

    def a_to_px(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) A-frame -> (N, 2) pixels."""
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        dx, dy = p[:, 0] - self.x, p[:, 1] - self.y
        xb = self.c * dx + self.sn * dy
        yb = -self.sn * dx + self.c * dy
        return self.body_to_px(np.column_stack([xb, yb]))

    def body_to_px(self, pts: np.ndarray) -> np.ndarray:
        """(..., 2) body (x fwd, y left) -> pixels (x right, y down)."""
        p = np.asarray(pts, np.float64)
        return np.stack([self.cx - p[..., 1] * self.s, self.cy - p[..., 0] * self.s], axis=-1)

    def inverse_affine_to_grid(self, origin_xy: tuple[float, float], res: float) -> np.ndarray:
        """2x3 matrix mapping panel pixel (u, v) -> fractional (col, row) of an axis-aligned
        A-frame grid (rows = +y, cols = +x, ``origin_xy`` = corner of cell [0, 0])."""
        k = 1.0 / (self.s * res)
        # body: xb = (cy - v)/s ; yb = (cx - u)/s ; A: X = x + c xb - sn yb ; Y = y + sn xb + c yb
        a_u, a_v = self.sn * k, -self.c * k
        a0 = (self.x - origin_xy[0]) / res + (self.c * self.cy - self.sn * self.cx) * k - 0.5
        b_u, b_v = -self.c * k, -self.sn * k
        b0 = (self.y - origin_xy[1]) / res + (self.sn * self.cy + self.c * self.cx) * k - 0.5
        return np.array([[a_u, a_v, a0], [b_u, b_v, b0]], np.float64)


def colorize_map_crop(state: np.ndarray, confirmed: Optional[np.ndarray]) -> np.ndarray:
    """CellState grid -> RGB (confirmed ditch candidates drawn as lethal red)."""
    rgb = style.cell_palette()[state]
    if confirmed is not None:
        rgb[confirmed.astype(bool) & (state == CellState.DITCH_CANDIDATE)] = style.CONFIRMED_DITCH_RGB
    return rgb


def project_ground_points(pts_body: np.ndarray, T_body_cam: np.ndarray, fx: float, fy: float, cx: float,
                          cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Ground points (N, 2) body frame (z = 0) -> (N, 2) pixel coords + valid mask (in front of camera)."""
    R, t = T_body_cam[:3, :3], T_body_cam[:3, 3]
    p = np.column_stack([pts_body, np.zeros(len(pts_body))])
    pc = (p - t) @ R  # R^T (p - t)
    z = pc[:, 2]
    ok = z > 0.4
    zs = np.where(ok, z, 1.0)
    return np.column_stack([fx * pc[:, 0] / zs + cx, fy * pc[:, 1] / zs + cy]), ok


def ribbon(path_xy: np.ndarray, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Left / right edges of a ribbon of ``half_width`` around a polyline (same frame)."""
    p = np.asarray(path_xy, np.float64)
    d = np.gradient(p, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(n, 1e-9)
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    return p + half_width * nrm, p - half_width * nrm


def fmt_clock(t: float) -> str:
    t = max(t, 0.0)
    return f"T+{int(t // 60):02d}:{t % 60:04.1f}"


def a_to_ego_px(pts_a: np.ndarray, pose: tuple[float, float, float], x0: float, y0: float, px_per_cell: float,
                n: int = COSTMAP_N, res: float = COSTMAP_RES_M) -> np.ndarray:
    """A-frame points -> pixels of the ego costmap image drawn at (x0, y0) (row 0 = far forward, col 0 = left)."""
    p = np.asarray(pts_a, np.float64).reshape(-1, 2)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = p[:, 0] - pose[0], p[:, 1] - pose[1]
    xb, yb = c * dx + s * dy, -s * dx + c * dy
    half = n * res / 2.0
    return np.column_stack([x0 + (half - yb) / res * px_per_cell, y0 + (half - xb) / res * px_per_cell])


# ============================================================================ compositor
class Dashboard:
    """Composites :class:`PanelData` into 1920 x 1080 RGB frames (see module docstring).

    ``gt_at`` is an optional callable ``t -> (x_a, y_a, yaw_a)`` giving the GT pose at a debug
    tick time; with it the BEV view is dead-reckoned smoothly between ticks.
    """

    def __init__(self, config_label: Optional[str] = None, gt_at=None, playback: float = 1.0) -> None:
        self.config_label = config_label
        self.gt_at = gt_at
        self.playback = playback  # run seconds per video second; != 1 is shown in the top bar
        self._base: Optional[np.ndarray] = None
        self._calib = defaults.stereo_calibration()

    # ------------------------------------------------------------------ static chrome
    def _chrome(self) -> np.ndarray:
        if self._base is not None:
            return self._base
        img = np.empty((H, W, 3), np.uint8)
        img[...] = style.BG
        p = Painter(img)
        p.rect(0, 0, W, TOPBAR_H, style.PANEL)
        p.rect(0, TOPBAR_H, W, TOPBAR_H + 1, style.BORDER)
        for r, title, note in ((LINK, "OPERATOR LINK", f"{defaults.TELEMETRY_HZ:.0f} Hz packets · no video"),
                               (BEV, "SEEN-GROUND MAP", "what the robot believes · 0.2 m cells"),
                               (GAUGES, "SPEED GOVERNOR", "stop distance ≤ R_cert")):
            p.panel(*r, fill=style.PANEL, border=style.BORDER, radius=RADIUS)
            p.rect(r[0] + 1, r[1] + HEADER_H, r[2] - 1, r[1] + HEADER_H + 1, style.BORDER)
            p.text(r[0] + 16, r[1] + HEADER_H / 2 + 1, title, 14, style.TEXT_2, "bold", "lm")
            p.text(r[2] - 16, r[1] + HEADER_H / 2 + 1, note, 13, style.TEXT_3, "regular", "rm")
        p.panel(*LEFTCAM, fill=style.PANEL, border=style.BORDER, radius=RADIUS)
        p.panel(*BANNER, fill=style.PANEL, border=style.BORDER, radius=RADIUS)
        p.panel(*CHASE, fill=style.PANEL_2, border=style.BORDER, radius=0)  # the chase image is rectangular
        p.rect(0, FOOTER_Y0, W, H, style.PANEL)
        p.rect(0, FOOTER_Y0, W, FOOTER_Y0 + 1, style.BORDER)
        self._base = p.flush().copy()
        return self._base

    # ------------------------------------------------------------------ public
    def compose(self, pd: PanelData, caption: Optional[Caption] = None, subtitle: str = "") -> np.ndarray:
        """Render one frame; returns (1080, 1920, 3) uint8 RGB."""
        img = self._chrome().copy()
        p = Painter(img)
        self._topbar(p, pd)
        self._chase(p, pd, CHASE)
        self._banner(p, pd, BANNER)
        self._bev(p, pd, (BEV[0] + 1, BEV[1] + HEADER_H + 1, BEV[2] - 1, BEV[3] - 2 * LEGEND_ROW_H - 14),
                  legend_y=BEV[3] - 2 * LEGEND_ROW_H - 4)
        self._gauges(p, pd, GAUGES)
        self._leftcam(p, pd, LEFTCAM)
        self._link(p, pd, LINK)
        self._footer(p, pd, subtitle)
        if caption is not None:
            lower_third(p, caption, CHASE[0] + 24, CHASE[3] - 24, CHASE[2] - CHASE[0] - 48)
        return p.flush()

    # ------------------------------------------------------------------ top bar / footer
    def _topbar(self, p: Painter, pd: PanelData, config_text: Optional[str] = None) -> None:
        y = TOPBAR_H / 2
        wordmark(p, MARGIN, y)
        x = MARGIN + 36 + text_width("METAGROSS", 22, "bold") + 20
        p.rect(int(x), 20, int(x) + 1, 44, style.BORDER)
        x += 20
        p.text(x, y, "MISSION", 12, style.TEXT_3, "bold", "lm")
        x += text_width("MISSION", 12, "bold") + 8
        p.text(x, y, pd.mission.mission_id, 18, style.TEXT, "mono", "lm")
        x += text_width(pd.mission.mission_id, 18, "mono") + 24
        red = style.MODE_COLORS[DriveMode.STOP_AND_LOOK]
        x = chip(p, x, y, "GNSS DENIED", fg=red, bg=style.tint(red, 0.92), border=style.tint(red, 0.55), size=14) + 10
        sensor = "PASSIVE STEREO" if pd.mission.sensor_mode == "stereo" else f"SENSOR {pd.mission.sensor_mode.upper()}"
        x = chip(p, x, y, sensor, fg=style.TEXT_2, bg=style.PANEL_2, border=style.BORDER, size=14) + 10
        cfg = config_text or self.config_label or pd.mission.config_name
        x = chip(p, x, y, f"CONFIG {cfg}", fg=style.TEXT_2, bg=style.PANEL_2, border=style.BORDER, size=14) + 10
        if abs(self.playback - 1.0) > 1e-6:
            chip(p, x, y, f"PLAYBACK {self.playback:g}×", fg=style.WHITE, bg=style.NAVY, border=style.NAVY, size=14)
        clock = fmt_clock(pd.t)
        cw = text_width(clock, 24, "mono")
        p.text(W - MARGIN, y, clock, 24, style.TEXT, "mono", "rm")
        p.text(W - MARGIN - cw - 12, y, "SIM TIME", 12, style.TEXT_3, "bold", "rm")
        sx = W - MARGIN - cw - 12 - text_width("SIM TIME", 12, "bold") - 20
        simulated_chip(p, sx, y, pd.mission.demo_fake, right=True)

    def _footer(self, p: Painter, pd: PanelData, subtitle: str, config_text: Optional[str] = None) -> None:
        y = (FOOTER_Y0 + H) / 2
        m = pd.mission
        tag = "SIMULATED · DEMO_FAKE DATA" if m.demo_fake else "SIMULATED"
        honesty = f"{tag} · {seed_label(m.seed)} · config {config_text or self.config_label or m.config_name}"
        x1 = outlined_chip(p, MARGIN, y, honesty, style.SIMULATED_RGB, size=14)
        right = "METAGROSS · SIH26126 · BEL"
        p.text(W - MARGIN, y, right, 14, style.TEXT_3, "bold", "rm")
        if subtitle:
            avail = W - 2 * max(x1 + 24, text_width(right, 14, "bold") + MARGIN + 24)
            size = 22 if text_width(subtitle, 22) <= avail else 18
            p.text(W / 2, y, subtitle, size, style.TEXT, "regular", "mm")

    # ------------------------------------------------------------------ chase
    def _chase(self, p: Painter, pd: PanelData, r: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = r
        w, h = x1 - x0, y1 - y0
        if pd.chase_rgb is not None:
            p.blit(pd.chase_rgb, x0, y0, w, h, cv2.INTER_AREA)
            label = "SIMULATOR VIEW · chase camera, ground-truth replay"
        else:
            self._chase_placeholder(p, pd, r)
            label = "SIMULATOR VIEW · top-down (chase renderer not attached)"
        chip(p, x0 + 16, y0 + 26, label, fg=style.TEXT, bg=style.PANEL, border=style.BORDER, size=13, weight="bold")
        p.outline(x0, y0, x1, y1, style.BORDER)

    def _chase_placeholder(self, p: Painter, pd: PanelData, r: tuple[int, int, int, int]) -> None:
        """Clean top-down placeholder: 2 m grid, trails, vehicle, goal (A-frame, x up)."""
        x0, y0, x1, y1 = r
        sub = p.img[y0:y1, x0:x1]
        sub[...] = style.PANEL_2
        gx, gy, gyaw = pd.gt_pose_a
        s = 22.0  # px/m
        cxp, cyp = (x1 - x0) / 2, (y1 - y0) * 0.62
        view = BevView((gx, gy, 0.0), cxp, cyp, s)  # A-frame x up (not track-up)
        sp = Painter(sub)
        step = 2.0
        xs = np.arange(math.floor((gx - 20) / step) * step, gx + 20, step)
        ys = np.arange(math.floor((gy - 30) / step) * step, gy + 30, step)
        for xv in xs:
            sp.line(view.a_to_px(np.array([[xv, gy - 30], [xv, gy + 30]])), style.BORDER, 1)
        for yv in ys:
            sp.line(view.a_to_px(np.array([[gx - 20, yv], [gx + 20, yv]])), style.BORDER, 1)
        _draw_goal(sp, view, pd, dark=False)
        if len(pd.est_trail_a) > 1:
            sp.line(view.a_to_px(pd.est_trail_a), VO_RGB, 2)
        if len(pd.gt_trail_a) > 1:
            sp.dashed(view.a_to_px(pd.gt_trail_a), style.TEXT_2, 2, 8, 6)
        _draw_vehicle(sp, view, (gx, gy, gyaw), fill=style.ACCENT, edge=style.WHITE)
        sp.flush()

    # ------------------------------------------------------------------ banner (health mode)
    def _banner(self, p: Painter, pd: PanelData, r: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = r
        cy = (y0 + y1) / 2
        col = style.mode_color(pd.mode)
        p.rect(x0 + 1, y0 + 12, x0 + 6, y1 - 12, col, radius=2)
        p.text(x0 + 22, y0 + 20, "HEALTH MODE", 12, style.TEXT_3, "bold", "lm")
        mode = pd.mode.replace("_", " ")
        mw = int(text_width(mode, 24, "bold")) + 36
        p.rect(x0 + 22, int(cy - 12), x0 + 22 + mw, int(cy + 30), col, radius=21)
        p.text(x0 + 22 + mw / 2, cy + 9, mode, 24, style.mode_text_color(pd.mode), "bold", "mm")
        tx = x0 + 22 + mw + 22
        p.text(tx, y0 + 20, "REASON", 12, style.TEXT_3, "bold", "lm")
        reason = pd.reason or "—"
        size = 18 if text_width(reason, 18, "mono") < x1 - tx - 150 else 14
        p.text(tx, cy + 9, reason, size, style.TEXT, "mono", "lm")
        p.text(x1 - 18, y0 + 20, "INTEGRITY q", 12, style.TEXT_3, "bold", "rm")
        p.text(x1 - 18, cy + 9, f"{pd.q:4.2f}" if math.isfinite(pd.q) else "—", 24, style.TEXT, "mono", "rm")

    # ------------------------------------------------------------------ BEV
    def _bev(self, p: Painter, pd: PanelData, r: tuple[int, int, int, int], legend_y: Optional[int]) -> None:
        x0, y0, x1, y1 = r
        w, h = x1 - x0, y1 - y0
        sub = np.ascontiguousarray(p.img[y0:y1, x0:x1])
        tick = pd.tick
        gt_tick = self.gt_at(tick.t) if (self.gt_at is not None and tick is not None) else None
        pose = view_pose(pd, gt_tick)
        view = BevView(pose, w / 2, h * BEV_VEHICLE_FRAC, BEV_PX_PER_M * (w / BEV_REF_W_PX))
        sub[...] = style.UNSEEN_RGB
        border = tuple(int(c) for c in style.UNSEEN_RGB)
        drawn = False
        if tick is not None and "extra_map_crop_state" in tick.arrays:
            st = tick.arrays["extra_map_crop_state"].astype(np.uint8)
            conf = tick.arrays.get("extra_map_crop_confirmed")
            origin = tuple(float(v) for v in tick.extras.get("map_crop_origin", tick.arrays.get("extra_map_crop_origin", (0, 0))))
            res = float(tick.extras.get("map_res_m", defaults.MAP_RES_M))
            rgb = colorize_map_crop(st, conf)
            M = view.inverse_affine_to_grid(origin, res)  # type: ignore[arg-type]
            sub[...] = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=border)
            drawn = True
        elif pd.costmap_u4 is not None:  # 4-bit downlink map, body-aligned around the telemetry pose
            tel_pose = tuple(pd.telemetry["pose"]) if pd.telemetry else pose
            rgb = style.u4_palette()[pd.costmap_u4]
            M = _ego_grid_affine(view, tel_pose, rgb.shape[0], COSTMAP_RES_M)  # type: ignore[arg-type]
            sub[...] = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=border)
            drawn = True
        sp = Painter(sub)
        if not drawn:
            sp.text(w / 2, h / 2, "no map data", 16, DARK_INK, "regular", "mm")
        # MPPI rollouts (faint), body frame of the tick pose
        if tick is not None and "rollouts_xy" in tick.arrays:
            ro = tick.arrays["rollouts_xy"].astype(np.float64)
            tp = tick.pose_xy_yaw
            c, s = math.cos(tp[2]), math.sin(tp[2])
            ro_a = np.stack([tp[0] + c * ro[..., 0] - s * ro[..., 1], tp[1] + s * ro[..., 0] + c * ro[..., 1]], -1)
            layer = sub.copy()
            lp = Painter(layer)
            for k in range(ro_a.shape[0]):
                lp.line(view.a_to_px(ro_a[k]), DARK_INK, 1)
            blend(sub, layer, 0.30)
        # R_cert arc
        if pd.r_cert_m > 0.05:
            rr = pd.r_cert_m * view.s
            arc = arc_points((view.cx, view.cy), rr, -math.pi / 2 - 0.75, -math.pi / 2 + 0.75)
            sp.dashed(arc, DARK_INK, 2, 7, 5)
            if pd.r_cert_m >= RCERT_LABEL_MIN_M:  # a tiny arc would put the label on top of the vehicle
                _map_label(sp, arc[-1][0] + 6, arc[-1][1], f"R_cert {pd.r_cert_m:.1f} m")
        # trails
        if len(pd.gt_trail_a) > 1:
            gtp = view.a_to_px(pd.gt_trail_a[-1500:])
            sp.line(gtp, DARK_INK, 4)
            sp.dashed(gtp, GT_RGB, 2, 8, 6)
        if len(pd.est_trail_a) > 1:
            sp.line(view.a_to_px(np.vstack([pd.est_trail_a[-600:], [pose[0], pose[1]]])), VO_RGB, 3)
        # chosen plan
        if tick is not None and "plan_xy" in tick.arrays:
            pl = view.a_to_px(tick.arrays["plan_xy"])
            sp.line(pl, style.WHITE, 7)
            sp.line(pl, PLAN_RGB, 4)
        _draw_goal(sp, view, pd, dark=True)
        # true position ring + vehicle (estimate)
        gpx = view.a_to_px(np.array([pd.gt_pose_a[:2]]))[0]
        sp.circle(tuple(gpx), 7, DARK_INK, 4)
        sp.circle(tuple(gpx), 7, GT_RGB, 2)
        _draw_vehicle(sp, view, pose, fill=DARK_INK, edge=style.WHITE)
        # A-frame axis indicator + scale bar
        _axis_indicator(sp, w - 44, 44, -view.yaw)
        _scale_bar(sp, 16, h - 18, view.s)
        sp.flush()
        p.img[y0:y1, x0:x1] = sub
        if legend_y is not None:
            _legend(p, x0 + 16, legend_y, x1 - x0 - 32)

    # ------------------------------------------------------------------ gauges
    def _gauges(self, p: Painter, pd: PanelData, r: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = r
        tick = pd.tick
        vmax = defaults.VEHICLE.max_speed_mps
        rmax = defaults.CAM_MAX_RANGE_M
        t_r = float(tick.extras.get("t_r_s", DEFAULT_T_R_S)) if tick is not None else DEFAULT_T_R_S
        a = float(tick.extras.get("a_mps2", defaults.BRAKE_DECEL_MPS2)) if tick is not None else defaults.BRAKE_DECEL_MPS2
        v = max(pd.speed_mps, 0.0)
        d_stop = stop_distance_m(v, t_r, a)
        # R_cert dial (left)
        body_h = y1 - y0 - HEADER_H
        dcx, dcy = x0 + 128, y0 + HEADER_H + body_h / 2 + 8
        rcert_dial(p, dcx, dcy, 82, pd.r_cert_m, rmax, d_stop)
        # speed vs v_cap (right)
        gx0, gx1 = x0 + 262, x1 - 18
        y = y0 + HEADER_H + 26
        p.text(gx0, y, "SPEED", 12, style.TEXT_3, "bold", "lm")
        val = f"{pd.speed_mps:4.2f} / {pd.v_cap_mps:4.2f} m/s"
        p.text(gx1, y, val, 20, style.TEXT, "mono", "rm")
        p.text(gx1 - text_width(val, 20, "mono") - 10, y, "v / v_cap", 13, style.TEXT_2, "regular", "rm")
        by = int(y + 16)
        bar(p, gx0, by, gx1, by + 14, pd.speed_mps / vmax, style.TEXT, marker=pd.v_cap_mps / vmax, marker_color=style.ACCENT)
        for vv in np.arange(0.0, vmax + 1e-6, 0.5):
            p.text(gx0 + (gx1 - gx0) * vv / vmax, by + 26, f"{vv:.1f}", 11, style.TEXT_3, "mono", "mm")
        p.rect(gx0, by + 40, gx0 + 12, by + 43, style.ACCENT)
        p.text(gx0 + 18, by + 42, "v_cap: fastest speed that still stops inside R_cert", 12, style.TEXT_2, "regular", "lm")
        # tiles: q, sigma, compute
        ty = by + 60
        tw = (gx1 - gx0 - 2 * 10) / 3
        tiles = (("INTEGRITY q", f"{pd.q:4.2f}" if math.isfinite(pd.q) else "—", f"p_fail {1 - pd.q:4.2f}" if math.isfinite(pd.q) else ""),
                 ("POSITION σ", f"{pd.pos_sigma_m:4.2f} m" if math.isfinite(pd.pos_sigma_m) else "—", "1-sigma"),
                 ("COMPUTE", f"{pd.compute_ms:4.0f} ms" if math.isfinite(pd.compute_ms) else "—", "per tick"))
        for i, (lab, val, sub) in enumerate(tiles):
            tx0 = gx0 + i * (tw + 10)
            p.panel(int(tx0), int(ty), int(tx0 + tw), int(y1 - 14), fill=style.PANEL_2, border=style.BORDER, radius=8)
            p.text(tx0 + 12, ty + 18, lab, 11, style.TEXT_3, "bold", "lm")
            p.text(tx0 + 12, ty + 46, val, 22, style.TEXT, "mono", "lm")
            p.text(tx0 + 12, ty + 74, sub, 11, style.TEXT_2, "regular", "lm")

    # ------------------------------------------------------------------ left camera
    def _leftcam(self, p: Painter, pd: PanelData, r: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = r
        bw = x1 - x0 - 2 * PAD
        bh = int(round(bw * self._calib.height / self._calib.width))
        bh = min(bh, y1 - y0 - 2 * PAD)
        bx, by = x0 + PAD, y0 + (y1 - y0 - bh) // 2
        tick = pd.left_tick if pd.left_tick is not None else pd.tick  # overlays from the bundle the image came from
        if pd.left_rgb is None and (tick is None or "semantic_mask" not in tick.arrays):
            p.rect(bx, by, bx + bw, by + bh, style.PANEL_2, radius=6)
            p.text(bx + bw / 2, by + bh / 2 + 12, "no onboard image in this run (depth-only tier-0 sensor)", 13, style.TEXT_3,
                   "regular", "mm")
            chip(p, bx + 10, by + 22, "ONBOARD LEFT CAMERA", fg=style.TEXT, bg=style.PANEL, border=style.BORDER, size=12)
            return
        src_h, src_w = self._calib.height, self._calib.width
        if pd.left_rgb is not None:
            img = cv2.resize(pd.left_rgb, (bw, bh), interpolation=cv2.INTER_AREA if pd.left_rgb.shape[1] > bw else cv2.INTER_LINEAR)
        else:
            img = np.full((bh, bw, 3), 40, np.uint8)
        if tick is not None and "semantic_mask" in tick.arrays:
            sem = cv2.resize(tick.arrays["semantic_mask"].astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST)
            tint_img = np.zeros_like(img)
            m = np.zeros(sem.shape, bool)
            for cid, col in style.SEM_TINT.items():
                if col is not None:
                    sel = sem == cid
                    tint_img[sel] = col
                    m |= sel
            blend(img, tint_img, 0.25, m)
        if tick is not None and "missing_ground_mask" in tick.arrays:
            mg = cv2.resize(tick.arrays["missing_ground_mask"].astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST) > 0
            mag = np.zeros_like(img)
            mag[...] = style.DITCH_RGB
            blend(img, mag, 0.65, mg)
        sp = Painter(img)
        if tick is not None and "plan_xy" in tick.arrays and len(tick.arrays["plan_xy"]) > 1:
            tp = tick.pose_xy_yaw
            pl = np.asarray(tick.arrays["plan_xy"], np.float64)
            c, s = math.cos(tp[2]), math.sin(tp[2])
            d = pl - np.array(tp[:2])
            pb = np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])
            pb = pb[pb[:, 0] > 0.6]
            if len(pb) > 1:
                le, ri = ribbon(pb, defaults.VEHICLE.width_m / 2)
                k = self._calib
                sx, sy = bw / src_w, bh / src_h
                pl_px, okl = project_ground_points(le, k.T_body_cam, k.fx * sx, k.fy * sy, k.cx * sx, k.cy * sy)
                pr_px, okr = project_ground_points(ri, k.T_body_cam, k.fx * sx, k.fy * sy, k.cx * sx, k.cy * sy)
                ok = okl & okr
                if ok.sum() > 1:
                    poly = np.vstack([pl_px[ok], pr_px[ok][::-1]])
                    layer = img.copy()
                    Painter(layer).poly(poly, style.ACCENT)
                    blend(img, layer, 0.28)
                    sp.line(pl_px[ok], style.WHITE, 2)
                    sp.line(pr_px[ok], style.WHITE, 2)
        sp.flush()
        # legend strip along the bottom of the image, on a translucent dark plate for legibility
        plate = img[bh - 26:bh, :]
        blend(plate, np.zeros_like(plate), 0.45)
        p.img[by:by + bh, bx:bx + bw] = img
        lx = bx + 10
        for lab, col in (("ground", style.SEM_TINT[4]), ("obstacle", style.SEM_TINT[1]), ("missing ground", style.DITCH_RGB),
                         ("plan", style.ACCENT)):
            p.rect(lx, by + bh - 18, lx + 10, by + bh - 8, col)  # type: ignore[arg-type]
            p.text(lx + 15, by + bh - 13, lab, 12, style.WHITE, "medium", "lm")
            lx += int(text_width(lab, 12, "medium")) + 32
        chip(p, bx + 10, by + 22, "ONBOARD LEFT CAMERA · what the robot sees", fg=style.TEXT, bg=style.PANEL, border=style.BORDER,
             size=12)

    # ------------------------------------------------------------------ operator link
    def _link(self, p: Painter, pd: PanelData, r: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = r
        side = int(round(COSTMAP_N * LINK_CELL_PX))
        cx0, cy0 = x0 + 16, y0 + HEADER_H + (y1 - y0 - HEADER_H - side) // 2
        tel = pd.telemetry
        if pd.costmap_u4 is not None:
            rgb = cv2.resize(style.u4_palette()[pd.costmap_u4], (side, side), interpolation=cv2.INTER_NEAREST)
            p.img[cy0:cy0 + side, cx0:cx0 + side] = rgb
        else:
            p.rect(cx0, cy0, cx0 + side, cy0 + side, style.PANEL_2)
            p.text(cx0 + side / 2, cy0 + side / 2, "no packet yet", 13, style.TEXT_3, "regular", "mm")
        p.outline(cx0 - 1, cy0 - 1, cx0 + side + 1, cy0 + side + 1, style.BORDER)
        lp = Painter(p.img)
        if tel is not None and tel.get("waypoints"):
            pose = tuple(float(v) for v in tel["pose"])
            wp = a_to_ego_px(np.vstack([[pose[0], pose[1]], np.asarray(tel["waypoints"], np.float64)]), pose, cx0, cy0, LINK_CELL_PX)  # type: ignore[arg-type]
            inside = (wp[:, 0] >= cx0) & (wp[:, 0] < cx0 + side) & (wp[:, 1] >= cy0) & (wp[:, 1] < cy0 + side)
            wp = wp[np.cumprod(inside).astype(bool)]  # stop at the first waypoint outside the map
            if len(wp) > 1:
                lp.line(wp, style.WHITE, 5)
                lp.line(wp, PLAN_RGB, 3)
                for q in wp[1:]:
                    lp.circle((float(q[0]), float(q[1])), 3.5, PLAN_RGB, fill=True)
        # ego vehicle marker at the map centre, heading up
        c = (cx0 + side / 2, cy0 + side / 2)
        lp.poly(np.array([[c[0], c[1] - 9], [c[0] + 6, c[1] + 6], [c[0], c[1] + 3], [c[0] - 6, c[1] + 6]]), DARK_INK)
        # tiles
        tx0 = cx0 + side + 20
        tw = (x1 - 16 - tx0 - 10) / 2
        th = 64
        ty = y0 + HEADER_H + 14
        age = f"{pd.packet_age_s:3.1f} s" if math.isfinite(pd.packet_age_s) else "—"
        tiles = (("PACKET", f"{pd.packet_bytes} B" if pd.packet_bytes else "—", "pose · mode · map · path"),
                 ("RATE", f"{pd.link_rate_hz:3.1f} Hz", f"last {LINK_WINDOW_S:.0f} s"),
                 ("LINK LOAD", f"{pd.link_kbps:4.1f} kbit/s", f"budget {defaults.LINK_KBPS:.1f}"),
                 ("PACKET AGE", age, "since last update"))
        for i, (lab, val, sub) in enumerate(tiles):
            gx = tx0 + (i % 2) * (tw + 10)
            gy = ty + (i // 2) * (th + 10)
            p.panel(int(gx), int(gy), int(gx + tw), int(gy + th), fill=style.PANEL_2, border=style.BORDER, radius=8)
            p.text(gx + 12, gy + 16, lab, 11, style.TEXT_3, "bold", "lm")
            p.text(gx + tw - 12, gy + 16, sub, 11, style.TEXT_2, "regular", "rm")
            p.text(gx + 12, gy + 43, val, 20, style.TEXT, "mono", "lm")
        # what the operator sees about health, from the packet itself
        ly = ty + 2 * (th + 10) + 12
        p.text(tx0, ly, "OPERATOR SEES", 11, style.TEXT_3, "bold", "lm")
        if tel is not None:
            mode = str(tel.get("mode", "HOLD"))
            xm = chip(p, tx0 + text_width("OPERATOR SEES", 11, "bold") + 10, ly, mode.replace("_", " "), fg=style.WHITE,
                      bg=style.mode_color(mode), border=style.mode_color(mode), size=12)
            p.text(xm + 10, ly, f"v_cap {float(tel.get('v_cap_mps', 0.0)):.2f} · R_cert {float(tel.get('r_cert_m', 0.0)):.1f} m", 12,
                   style.TEXT_2, "mono", "lm")
        p.text(tx0, ly + 26, f"ego costmap {COSTMAP_N}×{COSTMAP_N} · {COSTMAP_RES_M:.2f} m cells · 4 bit, same colour key", 12,
               style.TEXT_3, "regular", "lm")


# ============================================================================ shared widgets
def stop_distance_m(v: float, t_r: float, a: float) -> float:
    """Stopping distance v^2 / (2 a) + v t_r + governor margin, metres (v m/s, t_r s, a m/s^2)."""
    return v * v / (2 * a) + v * t_r + defaults.GOVERNOR_MARGIN_M


def _ego_grid_affine(view: BevView, tel_pose: tuple[float, float, float], n: int, res: float) -> np.ndarray:
    """Panel pixel -> (col, row) of the body-aligned ego costmap (row 0 = far forward, col 0 = left)."""
    half = n * res / 2.0
    k = 1.0 / view.s
    # panel -> view body: xb = (cy - v) k, yb = (cx - u) k ; view body -> A ; A -> tel body
    dyaw = view.yaw - tel_pose[2]
    c, s = math.cos(dyaw), math.sin(dyaw)
    ct, st = math.cos(tel_pose[2]), math.sin(tel_pose[2])
    dx, dy = view.x - tel_pose[0], view.y - tel_pose[1]
    ox, oy = ct * dx + st * dy, -st * dx + ct * dy  # view origin in tel body frame
    # tel body: xt = ox + c xb - s yb ; yt = oy + s xb + c yb
    # grid: row = (half - xt)/res - 0.5 ; col = (half - yt)/res - 0.5
    xb_u, xb_v, xb_0 = 0.0, -k, view.cy * k
    yb_u, yb_v, yb_0 = -k, 0.0, view.cx * k
    xt = (c * xb_u - s * yb_u, c * xb_v - s * yb_v, ox + c * xb_0 - s * yb_0)
    yt = (s * xb_u + c * yb_u, s * xb_v + c * yb_v, oy + s * xb_0 + c * yb_0)
    col = (-yt[0] / res, -yt[1] / res, (half - yt[2]) / res - 0.5)
    row = (-xt[0] / res, -xt[1] / res, (half - xt[2]) / res - 0.5)
    return np.array([col, row], np.float64)


def turbo(d: np.ndarray, vmax: float) -> np.ndarray:
    """Disparity (px) -> RGB turbo colour map; invalid (<= 0 / non-finite) -> near-black."""
    valid = np.isfinite(d) & (d > 0)
    u8 = np.clip(np.where(valid, d, 0) / vmax * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = (12, 16, 24)
    return rgb


def wordmark(p: Painter, x: float, cy: float) -> None:
    """METAGROSS logo mark + wordmark, left edge at x, centred on cy."""
    p.rect(int(x), int(cy - 12), int(x) + 24, int(cy + 12), style.ACCENT, radius=5)
    p.rect(int(x) + 7, int(cy - 5), int(x) + 17, int(cy + 5), style.PANEL, radius=2)
    p.text(x + 36, cy, "METAGROSS", 22, style.TEXT, "bold", "lm")


def chip(p: Painter, x: float, cy: float, label: str, fg=style.TEXT, bg=style.PANEL_2, border=style.BORDER, size: int = 14,
         weight: str = "bold", pad: int = 10) -> float:
    """Pill-shaped label, left edge at x, vertically centred on cy. Returns the right edge."""
    w = text_width(label, size, weight) + 2 * pad
    h = size + 12
    x0, y0 = int(x), int(cy - h / 2)
    p.rect(x0, y0, int(x0 + w), y0 + h, border, radius=h // 2)
    p.rect(x0 + 1, y0 + 1, int(x0 + w) - 1, y0 + h - 1, bg, radius=h // 2 - 1)
    p.text(x0 + w / 2, cy + 1, label, size, fg, weight, "mm")
    return x0 + w


def outlined_chip(p: Painter, x: float, cy: float, label: str, color, size: int = 14, right: bool = False) -> float:
    """Honesty-style chip (DESIGN_TOKENS): white fill, 2 px border and bold text in ``color``.
    ``x`` is the left edge (or the right edge if ``right``). Returns the far edge."""
    w = text_width(label, size, "bold") + 24
    h = size + 14
    x0 = int(x - w) if right else int(x)
    y0 = int(cy - h / 2)
    p.rect(x0, y0, int(x0 + w), y0 + h, color, radius=h // 2)
    p.rect(x0 + 2, y0 + 2, int(x0 + w) - 2, y0 + h - 2, style.WHITE, radius=h // 2 - 2)
    p.text(x0 + w / 2, cy + 1, label, size, color, "bold", "mm")
    return x0 if right else x0 + w


def seed_label(seed: Optional[int]) -> str:
    """'DEV seed 128' / 'EVAL seed 7' / 'seed 999' - which scenario split a run's seed belongs to."""
    if seed is None:
        return "seed –"
    split = "DEV " if seed in DEV_SEEDS else "EVAL " if seed in EVAL_SEEDS else ""
    return f"{split}seed {seed}"


def simulated_chip(p: Painter, x: float, cy: float, demo_fake: bool = False, right: bool = False) -> float:
    """The always-visible 'SIMULATED' honesty chip."""
    return outlined_chip(p, x, cy, "SIMULATED · DEMO DATA" if demo_fake else "SIMULATED", style.SIMULATED_RGB, 15, right)


def bar(p: Painter, x0: int, y0: int, x1: int, y1: int, frac: float, color, marker: Optional[float] = None,
        marker_color=style.ACCENT) -> None:
    """Horizontal bar gauge in a recessed well; optional vertical marker at ``marker`` fraction."""
    p.rect(x0, y0, x1, y1, style.GRID, radius=(y1 - y0) // 2)
    f = float(np.clip(frac if math.isfinite(frac) else 0.0, 0.0, 1.0))
    if f > 0:
        p.rect(x0, y0, int(x0 + (x1 - x0) * f), y1, color, radius=(y1 - y0) // 2)
    if marker is not None and math.isfinite(marker):
        mx = int(x0 + (x1 - x0) * float(np.clip(marker, 0.0, 1.0)))
        p.rect(mx - 2, y0 - 6, mx + 2, y1 + 6, marker_color, radius=1)


def rcert_dial(p: Painter, cx: float, cy: float, r: float, r_cert: float, r_max: float, d_stop: float) -> None:
    """270-degree dial: certified range R_cert (accent arc, 0..r_max m) with the current stopping
    distance as a tick; the governor keeps the tick inside the arc."""
    track = arc_points((cx, cy), r, DIAL_START_RAD, DIAL_START_RAD + DIAL_SWEEP_RAD, 96)
    p.line(track, style.GRID, 14)
    f = float(np.clip(r_cert / r_max, 0.0, 1.0)) if math.isfinite(r_cert) else 0.0
    if f > 0.005:
        p.line(arc_points((cx, cy), r, DIAL_START_RAD, DIAL_START_RAD + DIAL_SWEEP_RAD * f, 96), style.ACCENT, 14)
    for m in np.arange(0.0, r_max + 1e-6, 2.0):
        a = DIAL_START_RAD + DIAL_SWEEP_RAD * m / r_max
        p.text(cx + (r + 22) * math.cos(a), cy + (r + 22) * math.sin(a), f"{m:.0f}", 11, style.TEXT_3, "mono", "mm")
    fs = float(np.clip(d_stop / r_max, 0.0, 1.0))
    a = DIAL_START_RAD + DIAL_SWEEP_RAD * fs
    p.line(np.array([[cx + (r - 13) * math.cos(a), cy + (r - 13) * math.sin(a)], [cx + (r + 11) * math.cos(a), cy + (r + 11) * math.sin(a)]]),
           style.TEXT, 3)
    p.text(cx, cy - 16, "R_cert", 13, style.TEXT_2, "bold", "mm")
    p.text(cx, cy + 12, f"{r_cert:4.1f} m" if math.isfinite(r_cert) else "—", 28, style.TEXT, "mono", "mm")
    p.text(cx, cy + 42, f"stop {d_stop:3.1f} m", 12, style.TEXT_2, "mono", "mm")


def lower_third(p: Painter, cap: Caption, x: int, y_bottom: int, max_w: int) -> None:
    """Lower third: white plate with an accent bar, title + subtitle."""
    tw = max(text_width(cap.title, 30, "bold"), text_width(cap.subtitle, 20, "regular") if cap.subtitle else 0.0)
    w = int(min(max_w, tw + 52))
    h = 90 if cap.subtitle else 60
    y0 = y_bottom - h
    plate = p.img[y0:y_bottom, x:x + w]
    blend(plate, np.full_like(plate, style.PANEL), 0.94)
    p.outline(x, y0, x + w, y_bottom, style.BORDER)
    p.rect(x, y0, x + 6, y_bottom, style.ACCENT)
    p.text(x + 26, y0 + 16, cap.title, 30, style.TEXT, "bold", "la")
    if cap.subtitle:
        p.text(x + 26, y0 + 57, cap.subtitle, 20, style.TEXT_2, "regular", "la")


def _map_label(p: Painter, x: float, y: float, s: str) -> None:
    w = text_width(s, 12, "mono") + 10
    p.rect(int(x), int(y - 9), int(x + w), int(y + 9), DARK_INK, radius=4)
    p.text(x + 5, y + 1, s, 12, style.WHITE, "mono", "lm")


def _draw_vehicle(p: Painter, view: BevView, pose: tuple[float, float, float], fill, edge) -> None:
    """Vehicle footprint (VEHICLE.length_m x width_m) with a heading notch."""
    L, Wd = defaults.VEHICLE.length_m, defaults.VEHICLE.width_m
    c, s = math.cos(pose[2]), math.sin(pose[2])
    corners_b = np.array([[L / 2, Wd / 2], [L / 2, -Wd / 2], [-L / 2, -Wd / 2], [-L / 2, Wd / 2]])
    nose_b = np.array([[L / 2 + 0.18, 0.0], [L / 2 - 0.05, 0.16], [L / 2 - 0.05, -0.16]])
    to_a = lambda b: np.column_stack([pose[0] + c * b[:, 0] - s * b[:, 1], pose[1] + s * b[:, 0] + c * b[:, 1]])  # noqa: E731
    poly = view.a_to_px(to_a(corners_b))
    p.poly(poly, fill)
    p.line(poly, edge, 2, closed=True)
    p.poly(view.a_to_px(to_a(nose_b)), edge)


def _draw_goal(p: Painter, view: BevView, pd: PanelData, dark: bool) -> None:
    """Goal B marker with the operator's goal-uncertainty circle; off-panel -> edge arrow."""
    g = pd.mission.goal_xy_a
    if g is None:
        return
    h, w = p.img.shape[:2]
    gp = view.a_to_px(np.array([g]))[0]
    ink = DARK_INK if dark else style.TEXT
    rr = pd.mission.goal_sigma_m * view.s
    inside = -rr < gp[0] < w + rr and -rr < gp[1] < h + rr
    dist = math.hypot(g[0] - pd.gt_pose_a[0], g[1] - pd.gt_pose_a[1])
    if inside:
        circ = arc_points(tuple(gp), rr, 0, 2 * math.pi, 96)
        layer = p.img.copy()
        Painter(layer).poly(circ, style.ACCENT)
        blend(p.img, layer, 0.14)
        p.dashed(circ, style.ACCENT, 2, 8, 6)
        p.circle(tuple(gp), 9, ink, 2)
        p.line(np.array([[gp[0] - 14, gp[1]], [gp[0] + 14, gp[1]]]), ink, 2)
        p.line(np.array([[gp[0], gp[1] - 14], [gp[0], gp[1] + 14]]), ink, 2)
        if dark:
            _map_label(p, gp[0] + 14, gp[1] - 16, f"B · {dist:.1f} m")
        else:
            p.text(gp[0] + 14, gp[1] - 16, f"B · {dist:.1f} m", 14, ink, "mono", "lm")
    else:
        c = np.array([w / 2, h / 2])
        d = gp - c
        t = min((w / 2 - 26) / max(abs(d[0]), 1e-6), (h / 2 - 26) / max(abs(d[1]), 1e-6))
        e = c + d * t
        u = d / max(np.linalg.norm(d), 1e-6)
        nrm = np.array([-u[1], u[0]])
        tri = np.array([e + u * 12, e - u * 8 + nrm * 9, e - u * 8 - nrm * 9])
        p.poly(tri, style.ACCENT)
        lx = float(np.clip(e[0] - u[0] * 30, 10, w - 110))
        ly = float(np.clip(e[1] - u[1] * 30, 14, h - 14))
        if dark:
            _map_label(p, lx, ly, f"B {dist:.0f} m")
        else:
            p.text(lx, ly, f"B {dist:.0f} m", 14, ink, "mono", "lm")


def _axis_indicator(p: Painter, cx: float, cy: float, rot: float) -> None:
    """Arrow for the A-frame +x axis ('x_A'); ``rot`` = angle of +x_A relative to screen-up (rad, CCW)."""
    p.circle((cx, cy), 24, DARK_INK, fill=True)
    u = np.array([-math.sin(rot), -math.cos(rot)])  # screen direction of +x_A
    nrm = np.array([-u[1], u[0]])
    c = np.array([cx, cy])
    p.poly(np.array([c + u * 17, c - u * 9 + nrm * 8, c - u * 4, c - u * 9 - nrm * 8]), style.WHITE)
    p.text(cx, cy + 36, "x_A", 12, DARK_INK, "bold", "mm")


def _scale_bar(p: Painter, x: float, y: float, px_per_m: float) -> None:
    L = 5.0 if px_per_m * 5 < 220 else 2.0
    n = px_per_m * L
    p.rect(int(x) - 4, int(y) - 20, int(x + n) + 8, int(y) + 8, DARK_INK, radius=4)
    p.rect(int(x), int(y) - 2, int(x + n), int(y) + 2, style.WHITE)
    for xx in (x, x + n):
        p.rect(int(xx), int(y) - 6, int(xx) + 2, int(y) + 4, style.WHITE)
    p.text(x + n / 2, y - 11, f"{L:.0f} m", 12, style.WHITE, "mono", "mm")


def _legend(p: Painter, x: int, y: int, width: int) -> None:
    cols = 4
    cw = width / cols
    for i, (lab, col) in enumerate(style.LEGEND):
        cx = x + (i % cols) * cw
        cy = y + (i // cols) * LEGEND_ROW_H
        p.rect(int(cx), int(cy), int(cx) + 12, int(cy) + 12, col, radius=3)
        p.text(cx + 18, cy + 6, lab, 13, style.TEXT_2, "regular", "lm")


# ============================================================================ split screen
SPLIT_W = 936
SPLIT_CHASE_H = 526


def compose_split(dash_a: Dashboard, pa: PanelData, dash_b: Dashboard, pb: PanelData, label_a: str, label_b: str,
                  caption: Optional[Caption] = None, subtitle: str = "") -> np.ndarray:
    """Two runs side by side (same seed): chase view, mode chip, BEV and governor gauges each."""
    img = np.empty((H, W, 3), np.uint8)
    img[...] = style.BG
    p = Painter(img)
    p.rect(0, 0, W, TOPBAR_H, style.PANEL)
    p.rect(0, TOPBAR_H, W, TOPBAR_H + 1, style.BORDER)
    configs = f"{dash_a.config_label or pa.mission.config_name} | {dash_b.config_label or pb.mission.config_name}"
    dash_a._topbar(p, pa, configs)
    for i, (dash, pd, lab) in enumerate(((dash_a, pa, label_a), (dash_b, pb, label_b))):
        x0 = 16 + i * (SPLIT_W + 16)
        x1 = x0 + SPLIT_W
        # header: run label + mode chip
        hy = BODY_Y0
        p.panel(x0, hy, x1, hy + 52, fill=style.PANEL, border=style.BORDER, radius=RADIUS)
        accent = style.ACCENT if i == 1 else style.TEXT_3
        p.rect(x0 + 1, hy + 10, x0 + 6, hy + 42, accent, radius=2)
        p.text(x0 + 20, hy + 27, lab, 20, style.TEXT, "bold", "lm")
        mode = pd.mode.replace("_", " ")
        mw = text_width(mode, 18, "bold") + 28
        p.rect(int(x1 - 14 - mw), hy + 11, x1 - 14, hy + 41, style.mode_color(pd.mode), radius=15)
        p.text(x1 - 14 - mw / 2, hy + 27, mode, 18, style.mode_text_color(pd.mode), "bold", "mm")
        # chase
        cr = (x0, hy + 62, x1, hy + 62 + SPLIT_CHASE_H)
        p.panel(*cr, fill=style.PANEL_2, border=style.BORDER, radius=0)
        dash._chase(p, pd, cr)
        # BEV + gauges
        by0 = cr[3] + 12
        bev_r = (x0, by0, x0 + 520, FOOTER_Y0 - 12)
        p.panel(*bev_r, fill=style.PANEL, border=style.BORDER, radius=0)
        dash._bev(p, pd, (bev_r[0] + 1, bev_r[1] + 1, bev_r[2] - 1, bev_r[3] - 1), legend_y=None)
        g = (bev_r[2] + 12, by0, x1, FOOTER_Y0 - 12)
        p.panel(*g, fill=style.PANEL, border=style.BORDER, radius=RADIUS)
        _split_gauges(p, pd, g)
        for e in pd.events:
            et = str(e.get("type", "")).upper().replace("_", " ")
            if et:
                red = style.MODE_COLORS[DriveMode.STOP_AND_LOOK]
                chip(p, cr[0] + 16, cr[3] - 34, f"REFEREE: {et}", fg=style.WHITE, bg=red, border=red, size=16)
                break
    p.rect(0, FOOTER_Y0, W, H, style.PANEL)
    p.rect(0, FOOTER_Y0, W, FOOTER_Y0 + 1, style.BORDER)
    dash_a._footer(p, pa, subtitle, configs)
    if caption is not None:
        lower_third(p, caption, 40, BODY_Y0 + 62 + SPLIT_CHASE_H - 24, 860)
    return p.flush()


def _split_gauges(p: Painter, pd: PanelData, r: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = r
    gx0, gx1 = x0 + 16, x1 - 16
    vmax = defaults.VEHICLE.max_speed_mps
    rows = (("SPEED", f"{pd.speed_mps:4.2f} m/s", pd.speed_mps / vmax, style.TEXT, pd.v_cap_mps / vmax),
            ("v_cap", f"{pd.v_cap_mps:4.2f} m/s", pd.v_cap_mps / vmax, style.ACCENT, None),
            ("R_cert", f"{pd.r_cert_m:4.1f} m", pd.r_cert_m / defaults.CAM_MAX_RANGE_M, style.ACCENT, None),
            ("q", f"{pd.q:4.2f}", pd.q, style.mode_color(pd.mode), None))
    y = y0 + 28
    for lab, val, frac, col, mk in rows:
        p.text(gx0, y, lab, 14, style.TEXT_2, "bold", "lm")
        p.text(gx1, y, val, 20, style.TEXT, "mono", "rm")
        bar(p, gx0, int(y + 16), gx1, int(y + 28), frac, col, marker=mk)
        y += 74
    p.text(gx0, y1 - 20, pd.reason or "", 14, style.TEXT_2, "mono", "lm")
