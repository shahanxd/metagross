"""Scene objects of a scenario: analytic ray-cast primitives + 2-D collision footprints.

Static objects (rocks, trees, bushes, logs) and dynamic obstacles (walker / box / boulder) are
converted into

* :class:`Primitive` -- an analytic solid (ellipsoid, vertical cylinder, horizontal cylinder,
  yawed box) used by the Tier-0 synthetic depth sensor for exact ray casts, and
* :class:`Footprint` -- a circle or yawed rectangle on the ground plane used by the referee
  (collision / clearance) and by the ground-truth hazard raster.

All coordinates are world frame (x east, y north, z up), metres, radians.

Ray casts are vectorised over N rays for one primitive. Ray directions are *not* normalised:
the sensor passes directions whose camera-z component is 1, so the returned ray parameter is
directly the camera depth Z (m). Misses return ``+inf``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from metagross.config import defaults
from metagross.contracts.scenario import OBJECT_SEM
from metagross.sim.geometry import point_segment_distance
from metagross.sim.terrain import Terrain

# Rays hitting a surface closer than this along the ray are ignored (camera inside an object).
_T_EPS = 1e-4
# Tree trunks are embedded this far below the terrain so no gap shows on slopes (m).
_TRUNK_EMBED_M = 0.2
# Dynamic boulders sit slightly sunk into the ground (fraction of their half height).
_BOULDER_SINK = 0.05


@dataclass
class Primitive:
    """One analytic solid. ``params`` depends on ``kind``:

    * ``ellipsoid``: (ax, ay, az) semi-axes, axis-aligned in the world frame.
    * ``vcyl``: (radius, z0, z1) vertical cylinder with bottom z0 and top z1 (capped).
    * ``hcyl``: (radius, half_length, yaw) horizontal cylinder whose axis points along yaw (capped).
    * ``obox``: (hx, hy, hz, yaw) box with half extents in its own frame, yawed about z.
    """

    kind: str
    center: np.ndarray  # (3,) world
    params: tuple[float, ...]
    bound_r: float  # bounding-sphere radius around ``center`` (m)
    sem: int = OBJECT_SEM


@dataclass
class Footprint:
    """Ground-plane footprint. ``kind`` is 'circle' (uses r) or 'rect' (uses hl, hw, yaw)."""

    kind: str
    cx: float
    cy: float
    r: float = 0.0
    hl: float = 0.0
    hw: float = 0.0
    yaw: float = 0.0
    height: float = 0.0  # protrusion above local ground (m)
    lethal: bool = True
    label: str = ""

    @property
    def bound_r(self) -> float:
        return self.r if self.kind == "circle" else math.hypot(self.hl, self.hw)


# --------------------------------------------------------------------------- static objects
def parse_static_objects(scenario: dict, terrain: Terrain) -> tuple[list[Primitive], list[Footprint]]:
    """Convert ``scenario['objects']`` into ray-cast primitives and ground footprints.

    A footprint is *lethal* when the object protrudes more than the platform ground clearance
    (``defaults.STEP_LETHAL_M``); smaller rocks are driven over and are not collisions.
    """
    prims: list[Primitive] = []
    feet: list[Footprint] = []
    clearance = defaults.STEP_LETHAL_M
    for k, ob in enumerate(scenario.get("objects", [])):
        typ = ob["type"]
        label = f"{typ}_{k}"
        if typ == "rock":
            x, y, z = (float(v) for v in ob["xyz"])
            r = float(ob["radius"])
            az = r * float(ob["squash"])
            prims.append(Primitive("ellipsoid", np.array([x, y, z]), (r, r, az), max(r, az)))
            g = float(terrain.height_at(x, y))
            protr = z + az - g
            # Horizontal radius of the ellipsoid where it meets the local ground plane.
            u = np.clip((g - z) / az, -1.0, 1.0)
            r_ground = r * math.sqrt(max(1.0 - u * u, 0.0)) if g > z else r
            feet.append(Footprint("circle", x, y, r=r_ground, height=protr, lethal=protr > clearance, label=label))
        elif typ == "tree":
            x, y = (float(v) for v in ob["xy"])
            g = float(terrain.height_at(x, y))
            tr, h, cr = float(ob["trunk_r"]), float(ob["height"]), float(ob["canopy_r"])
            zc = g + h - cr
            z0 = g - _TRUNK_EMBED_M
            prims.append(Primitive("vcyl", np.array([x, y, 0.5 * (z0 + zc)]), (tr, z0, zc), math.hypot(tr, 0.5 * (zc - z0))))
            prims.append(Primitive("ellipsoid", np.array([x, y, zc]), (cr, cr, cr), cr))
            feet.append(Footprint("circle", x, y, r=tr, height=h, lethal=True, label=label))
        elif typ == "bush":
            x, y = (float(v) for v in ob["xy"])
            g = float(terrain.height_at(x, y))
            r, h = float(ob["radius"]), float(ob["height"])
            prims.append(Primitive("ellipsoid", np.array([x, y, g + 0.5 * h]), (r, r, 0.5 * h), max(r, 0.5 * h)))
            feet.append(Footprint("circle", x, y, r=r, height=h, lethal=h > clearance, label=label))
        elif typ == "log":
            x, y = (float(v) for v in ob["xy"])
            g = float(terrain.height_at(x, y))
            yaw, L, r = float(ob["yaw"]), float(ob["length"]), float(ob["radius"])
            prims.append(Primitive("hcyl", np.array([x, y, g + r]), (r, 0.5 * L, yaw), math.hypot(0.5 * L, r)))
            feet.append(Footprint("rect", x, y, hl=0.5 * L, hw=r, yaw=yaw, height=2 * r, lethal=2 * r > clearance, label=label))
        else:
            raise ValueError(f"unknown object type {typ!r}")
    return prims, feet


# --------------------------------------------------------------------------- dynamic objects
@dataclass
class DynamicObject:
    """Scripted obstacle that starts moving along ``path`` once the vehicle comes within
    ``trigger_dist_m`` of the path segment, then stops at the path end.

    The trigger distance is measured from the vehicle body origin to the closest point of the
    straight path segment (the segment crosses the route, so this is ~the distance to the
    crossing point while approaching).
    """

    index: int
    type: str
    size: tuple[float, float, float]
    trigger_dist_m: float
    p0: np.ndarray
    p1: np.ndarray
    speed: float
    t_trigger: float | None = None
    _len: float = field(init=False)
    _dir: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        d = self.p1 - self.p0
        self._len = float(np.linalg.norm(d))
        self._dir = d / max(self._len, 1e-9)

    @classmethod
    def from_dict(cls, index: int, d: dict) -> "DynamicObject":
        return cls(index, d["type"], tuple(float(v) for v in d["size"]), float(d["trigger_dist_m"]),
                   np.asarray(d["path"][0], float), np.asarray(d["path"][1], float), float(d["speed"]))

    @property
    def yaw(self) -> float:
        return math.atan2(self._dir[1], self._dir[0])

    def maybe_trigger(self, t: float, vehicle_xy: np.ndarray) -> bool:
        """Arm the object if the vehicle is close enough; returns True on the triggering call."""
        if self.t_trigger is not None:
            return False
        if float(point_segment_distance(np.asarray(vehicle_xy, float), self.p0, self.p1)) <= self.trigger_dist_m:
            self.t_trigger = t
            return True
        return False

    def xy_at(self, t: float) -> np.ndarray:
        if self.t_trigger is None or t <= self.t_trigger:
            return self.p0.copy()
        s = min(self.speed * (t - self.t_trigger), self._len)
        return self.p0 + s * self._dir

    def moving_at(self, t: float) -> bool:
        return self.t_trigger is not None and t > self.t_trigger and self.speed * (t - self.t_trigger) < self._len

    def primitive_at(self, t: float, terrain: Terrain) -> Primitive:
        x, y = self.xy_at(t)
        g = float(terrain.height_at(x, y))
        sx, sy, sz = self.size
        if self.type == "walker":
            r = 0.5 * max(sx, sy)
            return Primitive("vcyl", np.array([x, y, g + 0.5 * sz]), (r, g - 0.05, g + sz), math.hypot(r, 0.5 * sz))
        if self.type == "box":
            return Primitive("obox", np.array([x, y, g + 0.5 * sz]), (0.5 * sx, 0.5 * sy, 0.5 * sz, self.yaw),
                             math.sqrt(sx * sx + sy * sy + sz * sz) * 0.5)
        if self.type == "boulder":
            rh = 0.5 * max(sx, sy)
            return Primitive("ellipsoid", np.array([x, y, g + 0.5 * sz * (1.0 - _BOULDER_SINK)]), (rh, rh, 0.5 * sz),
                             max(rh, 0.5 * sz))
        raise ValueError(f"unknown dynamic type {self.type!r}")

    def footprint_at(self, t: float) -> Footprint:
        x, y = self.xy_at(t)
        sx, sy, sz = self.size
        label = f"dyn_{self.type}_{self.index}"
        if self.type == "box":
            return Footprint("rect", float(x), float(y), hl=0.5 * sx, hw=0.5 * sy, yaw=self.yaw, height=sz, label=label)
        return Footprint("circle", float(x), float(y), r=0.5 * max(sx, sy), height=sz, label=label)

    def state_dict(self, t: float, terrain: Terrain) -> dict:
        """Renderer-facing state (see RendererProto.render_stereo 'dynamic')."""
        x, y = self.xy_at(t)
        return {"id": self.index, "type": self.type, "size": list(self.size), "xyz": [float(x), float(y), float(terrain.height_at(x, y))],
                "yaw": self.yaw, "moving": self.moving_at(t), "triggered": self.t_trigger is not None}


# --------------------------------------------------------------------------- ray casts
def _nearest_positive(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Smallest root > eps of each (t1 <= t2) pair, +inf if none."""
    return np.where(t1 > _T_EPS, t1, np.where(t2 > _T_EPS, t2, np.inf))


def ray_ellipsoid(o: np.ndarray, d: np.ndarray, c: np.ndarray, axes: tuple[float, float, float]) -> np.ndarray:
    s = np.asarray(axes, float)
    oc = (o - c) / s
    dd = d / s
    A = np.einsum("ij,ij->i", dd, dd)
    B = 2.0 * (dd @ oc)
    C = float(oc @ oc) - 1.0
    disc = B * B - 4.0 * A * C
    hit = disc >= 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t1 = (-B - sq) / (2.0 * A)
    t2 = (-B + sq) / (2.0 * A)
    return np.where(hit, _nearest_positive(t1, t2), np.inf)


def ray_vcylinder(o: np.ndarray, d: np.ndarray, c: np.ndarray, radius: float, z0: float, z1: float) -> np.ndarray:
    ox, oy = o[0] - c[0], o[1] - c[1]
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    A = dx * dx + dy * dy
    B = 2.0 * (ox * dx + oy * dy)
    C = ox * ox + oy * oy - radius * radius
    disc = B * B - 4.0 * A * C
    ok = (disc >= 0.0) & (A > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    A_safe = np.where(A > 1e-12, A, 1.0)
    t1 = (-B - sq) / (2.0 * A_safe)
    t2 = (-B + sq) / (2.0 * A_safe)
    z_1 = o[2] + t1 * dz
    z_2 = o[2] + t2 * dz
    t1 = np.where(ok & (z_1 >= z0) & (z_1 <= z1) & (t1 > _T_EPS), t1, np.inf)
    t2 = np.where(ok & (z_2 >= z0) & (z_2 <= z1) & (t2 > _T_EPS), t2, np.inf)
    best = np.minimum(t1, t2)
    # Caps (top and bottom discs).
    dz_safe = np.where(np.abs(dz) > 1e-12, dz, 1e-12)
    for zc in (z0, z1):
        tc = (zc - o[2]) / dz_safe
        px = ox + tc * dx
        py = oy + tc * dy
        cap = (tc > _T_EPS) & (px * px + py * py <= radius * radius)
        best = np.where(cap & (tc < best), tc, best)
    return best


def ray_hcylinder(o: np.ndarray, d: np.ndarray, c: np.ndarray, radius: float, half_len: float, yaw: float) -> np.ndarray:
    u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    w = o - c
    wu = float(w @ u)
    du = d @ u
    w_perp = w - wu * u
    d_perp = d - du[:, None] * u[None, :]
    A = np.einsum("ij,ij->i", d_perp, d_perp)
    B = 2.0 * (d_perp @ w_perp)
    C = float(w_perp @ w_perp) - radius * radius
    disc = B * B - 4.0 * A * C
    ok = (disc >= 0.0) & (A > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    A_safe = np.where(A > 1e-12, A, 1.0)
    best = np.full(len(d), np.inf)
    for sgn in (-1.0, 1.0):
        t = (-B + sgn * sq) / (2.0 * A_safe)
        s = wu + t * du
        valid = ok & (t > _T_EPS) & (np.abs(s) <= half_len)
        best = np.where(valid & (t < best), t, best)
    du_safe = np.where(np.abs(du) > 1e-12, du, 1e-12)
    for s_cap in (-half_len, half_len):
        tc = (s_cap - wu) / du_safe
        p = w_perp[None, :] + tc[:, None] * d_perp
        cap = (tc > _T_EPS) & (np.einsum("ij,ij->i", p, p) <= radius * radius)
        best = np.where(cap & (tc < best), tc, best)
    return best


def ray_obox(o: np.ndarray, d: np.ndarray, c: np.ndarray, half: tuple[float, float, float], yaw: float) -> np.ndarray:
    cyaw, syaw = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[cyaw, -syaw, 0.0], [syaw, cyaw, 0.0], [0.0, 0.0, 1.0]])
    ol = R @ (o - c)
    dl = d @ R.T
    h = np.asarray(half, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(dl) > 1e-12, dl, 1e-12)
        ta = (-h[None, :] - ol[None, :]) * inv
        tb = (h[None, :] - ol[None, :]) * inv
    tnear = np.max(np.minimum(ta, tb), axis=1)
    tfar = np.min(np.maximum(ta, tb), axis=1)
    hit = (tnear <= tfar) & (tfar > _T_EPS)
    return np.where(hit, np.where(tnear > _T_EPS, tnear, tfar), np.inf)


def raycast_primitive(p: Primitive, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Ray parameter of the first hit of rays ``o + t d`` (d: (N, 3)) with primitive ``p``."""
    if p.kind == "ellipsoid":
        return ray_ellipsoid(o, d, p.center, p.params)  # type: ignore[arg-type]
    if p.kind == "vcyl":
        r, z0, z1 = p.params
        return ray_vcylinder(o, d, p.center, r, z0, z1)
    if p.kind == "hcyl":
        r, hl, yaw = p.params
        return ray_hcylinder(o, d, p.center, r, hl, yaw)
    if p.kind == "obox":
        hx, hy, hz, yaw = p.params
        return ray_obox(o, d, p.center, (hx, hy, hz), yaw)
    raise ValueError(p.kind)


# --------------------------------------------------------------------------- footprints
def rect_corners(fp: Footprint) -> np.ndarray:
    c, s = math.cos(fp.yaw), math.sin(fp.yaw)
    local = np.array([[fp.hl, fp.hw], [-fp.hl, fp.hw], [-fp.hl, -fp.hw], [fp.hl, -fp.hw]])
    return local @ np.array([[c, s], [-s, c]]) + np.array([fp.cx, fp.cy])


def circle_rect_distance(cx: np.ndarray, cy: np.ndarray, r: np.ndarray, rect: Footprint) -> np.ndarray:
    """Signed distance (m) between circles and one yawed rectangle (negative = overlap)."""
    c, s = math.cos(rect.yaw), math.sin(rect.yaw)
    dx = np.asarray(cx) - rect.cx
    dy = np.asarray(cy) - rect.cy
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    qx = np.abs(lx) - rect.hl
    qy = np.abs(ly) - rect.hw
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - np.asarray(r)


def rect_rect_distance(a: Footprint, b: Footprint) -> float:
    """Signed distance (m) between two yawed rectangles (negative = penetration depth, SAT)."""
    ca, cb = rect_corners(a), rect_corners(b)
    axes = []
    for fp in (a, b):
        axes.append(np.array([math.cos(fp.yaw), math.sin(fp.yaw)]))
        axes.append(np.array([-math.sin(fp.yaw), math.cos(fp.yaw)]))
    min_overlap = np.inf
    separated = False
    for ax in axes:
        pa, pb = ca @ ax, cb @ ax
        overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if overlap <= 0.0:
            separated = True
            break
        min_overlap = min(min_overlap, overlap)
    if not separated:
        return -float(min_overlap)
    best = np.inf
    for P, Q in ((ca, cb), (cb, ca)):
        a0 = Q
        b0 = np.roll(Q, -1, axis=0)
        d = point_segment_distance(P[:, None, :], a0[None, :, :], b0[None, :, :])
        best = min(best, float(d.min()))
    return best
