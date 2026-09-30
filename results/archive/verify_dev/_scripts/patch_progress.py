from pathlib import Path

p = Path(r"D:\Downloads\sih again\metagross\metagross\autonomy\safety\supervisor.py")
s = p.read_text(encoding="utf-8")
rep = [
("""STOP_AND_LOOK           3 s without 0.3 m of progress    rotate +45, -45, back to 0 deg in
                                                         place to certify adjacent ground""",
"""STOP_AND_LOOK           3 s without 0.3 m of progress    rotate +45, -45, back to 0 deg in
                        toward the goal (cost-to-go, or  place to certify adjacent ground
                        distance when there is no route)"""),
("""        self._anchor: Optional[tuple[float, float, float]] = None  # (t, x, y)""",
"""        self._anchor: Optional[tuple[float, float, str]] = None  # (t, progress metric, metric kind)"""),
("""        frame_gap_s: float = 0.0,
        immobilised_hint: bool = False,
    ) -> SupervisorDecision:
        \"\"\"Advance the state machine. ``q`` in [0,1]; pose A-frame; speeds [m/s]; gap [s];
        ``immobilised_hint``: the localiser's wheel-vs-VO immobilisation flag.\"\"\"""",
"""        frame_gap_s: float = 0.0,
        immobilised_hint: bool = False,
        progress_metric: Optional[float] = None,
        metric_kind: str = "dist",
    ) -> SupervisorDecision:
        \"\"\"Advance the state machine. ``q`` in [0,1]; pose A-frame; speeds [m/s]; gap [s];
        ``immobilised_hint``: the localiser's wheel-vs-VO immobilisation flag;
        ``progress_metric``: how far the goal still is (global cost-to-go ~ metres, kind "ctg");
        defaults to the straight-line distance to the goal (kind "dist").\"\"\""""),
("""        # STOP_AND_LOOK
        if self._look is not None:
            return self._look_step(t, pose, inflate)
        if self._anchor is None or math.hypot(pose[0] - self._anchor[1], pose[1] - self._anchor[2]) >= p.progress_min_m:
            if self._anchor is not None:
                self._look_cycles = 0
            self._anchor = (t, pose[0], pose[1])
        elif t - self._anchor[0] >= p.no_progress_s:""",
"""        # STOP_AND_LOOK: progress means the goal got closer, not merely that the wheels turned
        finite = progress_metric is not None and math.isfinite(progress_metric)
        metric = float(progress_metric) if finite else d_goal
        kind = metric_kind if finite else "dist"
        self._metric = (metric, kind)
        if self._look is not None:
            return self._look_step(t, pose, inflate)
        if self._anchor is None or self._anchor[2] != kind:
            self._anchor = (t, metric, kind)
        elif metric <= self._anchor[1] - p.progress_min_m:
            self._anchor = (t, metric, kind)
            self._look_cycles = 0
        elif metric > self._anchor[1]:
            self._anchor = (self._anchor[0], metric, kind)  # route got longer (new obstacle): re-baseline, same clock
        elif t - self._anchor[0] >= p.no_progress_s:"""),
("""                speed, reason = 0.0, f"DEGRADED hop-pause q={q:.2f}"
                self._anchor = (t, pose[0], pose[1])  # pauses are not lack of progress""",
"""                speed, reason = 0.0, f"DEGRADED hop-pause q={q:.2f}"
                self._anchor = (t, metric, kind)  # pauses are not lack of progress"""),
("""        self._look = None
        self._look_cycles += 1
        self._anchor = (t, pose[0], pose[1])""",
"""        self._look = None
        self._look_cycles += 1
        self._anchor = (t, *self._metric)"""),
("""        self._degraded_since: Optional[float] = None
        self.transitions""", """        self._degraded_since: Optional[float] = None
        self._metric: tuple[float, str] = (math.inf, "dist")
        self.transitions"""),
]
for a, b in rep:
    assert a in s, a[:70]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")

p = Path(r"D:\Downloads\sih again\metagross\metagross\autonomy\node.py")
s = p.read_text(encoding="utf-8")
a = """        # 4. supervisor (mode, speed factor, inflation)
        t0 = time.perf_counter()
        dec = self.supervisor.update(t, q, pose, self.mixer.v, self._speed_meas, gap, immobilised)
        tm["supervisor"] = (time.perf_counter() - t0) * 1e3

        # 5. costmap
        t0 = time.perf_counter()
        nominal_health = (q > Q_NOMINAL) if cfg["use_health"] else True
        maps = build_costmap(self.map, t, nominal_health, self.cm_params.scaled(dec.inflate_scale))
        self.last_maps = maps
        tm["costmap"] = (time.perf_counter() - t0) * 1e3

        # 6. global planner
        t0 = time.perf_counter()
        goal = self.supervisor.goal_xy
        gplan = self.global_planner.update(t, maps, (pose[0], pose[1]), goal)
        tm["global"] = (time.perf_counter() - t0) * 1e3
"""
b = """        # 4. costmap (inflation scale from the supervisor's health level of the previous tick)
        t0 = time.perf_counter()
        nominal_health = (q > Q_NOMINAL) if cfg["use_health"] else True
        maps = build_costmap(self.map, t, nominal_health, self.cm_params.scaled(self._inflate_scale))
        self.last_maps = maps
        tm["costmap"] = (time.perf_counter() - t0) * 1e3

        # 5. global planner
        t0 = time.perf_counter()
        goal = self.supervisor.goal_xy
        gplan = self.global_planner.update(t, maps, (pose[0], pose[1]), goal)
        tm["global"] = (time.perf_counter() - t0) * 1e3

        # 6. supervisor (mode, speed factor, overrides); progress = cost-to-go when a route exists
        t0 = time.perf_counter()
        dec = self.supervisor.update(t, q, pose, self.mixer.v, self._speed_meas, gap, immobilised,
                                     progress_metric=gplan.ctg_vehicle if gplan.route_ok else None, metric_kind="ctg")
        self._inflate_scale = dec.inflate_scale
        tm["supervisor"] = (time.perf_counter() - t0) * 1e3
"""
assert a in s
s = s.replace(a, b)
a2 = """    RollingMap.integrate               -> A-frame seen-ground memory
    Supervisor.update                  -> drive mode, speed factor, cost inflation, overrides
    build_costmap                      -> inflated costs, certification, clearance
    GlobalPlanner.update (1 Hz / cut)  -> cost-to-go field, descent path, lookahead"""
b2 = """    RollingMap.integrate               -> A-frame seen-ground memory
    build_costmap                      -> inflated costs, certification, clearance
    GlobalPlanner.update (1 Hz / cut)  -> cost-to-go field, descent path, lookahead
    Supervisor.update                  -> drive mode, speed factor, cost inflation, overrides"""
assert a2 in s
s = s.replace(a2, b2)
a3 = """    SkidSteerMixer                     -> wheel rad/s (WheelCmd)
"""
b3 = """    SkidSteerMixer                     -> wheel rad/s (WheelCmd)

The costmap uses the supervisor's cost-inflation scale from the previous tick, because the
supervisor needs the fresh cost-to-go for its progress monitor and so runs after the planner.
"""
assert a3 in s
s = s.replace(a3, b3, 1)
s = s.replace("""        self._apron_done = False
        self.last_maps""", """        self._apron_done = False
        self._inflate_scale = 1.0
        self.last_maps""")
s = s.replace('TIMING_KEYS = ("localizer", "perception", "map", "supervisor", "costmap", "global", "governor", "mppi", "gate_mixer")',
              'TIMING_KEYS = ("localizer", "perception", "map", "costmap", "global", "supervisor", "governor", "mppi", "gate_mixer")')
p.write_text(s, encoding="utf-8")
print("ok")
