from pathlib import Path

# ---------------- global planner: report info-driven CTG jump at the vehicle
p = Path(r"D:\Downloads\sih again\metagross\metagross\autonomy\planning\global_planner.py")
s = p.read_text(encoding="utf-8")
rep = [
("""    t_computed: float = -math.inf
    compute_ms: float = 0.0
""", """    t_computed: float = -math.inf
    compute_ms: float = 0.0
    ctg_jump: float = 0.0  # on a recompute: new minus old CTG at the same vehicle position (new map info)
"""),
("""        if due:
            self._recompute(t, steps, geo, goal_xy)
        assert self.plan is not None
        self._trace(pose_xy)
        return self.plan""", """        old = self.plan
        if due:
            self._recompute(t, steps, geo, goal_xy)
        assert self.plan is not None
        self._trace(pose_xy)
        if due and old is not None and old.route_ok and self.plan.route_ok and old.goal_xy == self.plan.goal_xy:
            prev = float(old.ctg_at(np.array([pose_xy[0]]), np.array([pose_xy[1]]))[0])
            self.plan.ctg_jump = self.plan.ctg_vehicle - prev
        return self.plan"""),
]
for a, b in rep:
    assert a in s, a[:60]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")

# ---------------- supervisor: shift anchor only by info-driven jumps
p = Path(r"D:\Downloads\sih again\metagross\metagross\autonomy\safety\supervisor.py")
s = p.read_text(encoding="utf-8")
rep = [
("""        progress_metric: Optional[float] = None,
        metric_kind: str = "dist",
    ) -> SupervisorDecision:""", """        progress_metric: Optional[float] = None,
        metric_kind: str = "dist",
        metric_shift: float = 0.0,
    ) -> SupervisorDecision:"""),
("""        defaults to the straight-line distance to the goal (kind "dist").\"\"\"""",
 """        defaults to the straight-line distance to the goal (kind "dist"); ``metric_shift``: increase
        of the metric caused by new map information (a replan lengthening the route), which moves
        the progress baseline. Driving away from the goal never moves it.\"\"\""""),
("""        elif metric > self._anchor[1]:
            self._anchor = (self._anchor[0], metric, kind)  # route got longer (new obstacle): re-baseline, same clock
        elif t - self._anchor[0] >= p.no_progress_s:""",
 """        elif t - self._anchor[0] >= p.no_progress_s:"""),
("""        if self._anchor is None or self._anchor[2] != kind:
            self._anchor = (t, metric, kind)
        elif metric <= self._anchor[1] - p.progress_min_m:""",
 """        if self._anchor is not None and self._anchor[2] == kind and metric_shift > 0.0:
            self._anchor = (self._anchor[0], self._anchor[1] + metric_shift, kind)  # route got longer: same clock
        if self._anchor is None or self._anchor[2] != kind:
            self._anchor = (t, metric, kind)
        elif metric <= self._anchor[1] - p.progress_min_m:"""),
]
for a, b in rep:
    assert a in s, a[:60]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")

# ---------------- node: pass the jump
p = Path(r"D:\Downloads\sih again\metagross\metagross\autonomy\node.py")
s = p.read_text(encoding="utf-8")
a = """                                     progress_metric=gplan.ctg_vehicle if gplan.route_ok else None, metric_kind="ctg")"""
b = """                                     progress_metric=gplan.ctg_vehicle if gplan.route_ok else None, metric_kind="ctg",
                                     metric_shift=max(gplan.ctg_jump, 0.0) if gplan.t_computed == t else 0.0)"""
assert a in s
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
