from pathlib import Path
p = Path(r"D:\Downloads\sih again\metagross\tests\test_node_process.py")
s = p.read_text(encoding="utf-8")
a = "    assert all(abs(c[1].omega_l_rad_s) < 1e-9 for c in cmds)  # blind stub perception: never leaves the apron\n"
b = ("    lim = VEHICLE.max_wheel_rad_s + 1e-9\n"
     "    assert all(np.isfinite(c[1].omega_l_rad_s) and abs(c[1].omega_l_rad_s) <= lim and abs(c[1].omega_r_rad_s) <= lim for c in cmds)\n"
     "    assert all(c[1].compute_ms > 0 for c in cmds)\n")
assert a in s
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")

p = Path(r"D:\Downloads\sih again\metagross\tests\test_node_closed_loop.py")
s = p.read_text(encoding="utf-8")
a = "def test_typical_stack_ablation_enters_ditch():"
b = '''def test_blind_vehicle_never_leaves_launch_apron():
    """Unknown is never free: with a perception that sees nothing, the vehicle may only use
    the certified launch apron and must stop inside it."""
    from metagross.autonomy.node_stubs import StubBlindPerception
    from metagross.autonomy.planning.rolling_map import LAUNCH_APRON_M

    world = ToyWorld()
    stack = AutonomyStack(perception=StubBlindPerception(), localizer=PerfectOdomLocalizer(world))
    stack.reset(MissionSpec("blind", (14.0, 0.0), 2.0, 30.0), stereo_calibration(), VEHICLE, {})
    reach = []
    for k in range(100):
        s = world.s
        frame = SensorFrame(t=0.2 * k, seq=k, left_rgb=None, right_gray=None, wheel_angle_l_rad=s.wl, wheel_angle_r_rad=s.wr,
                            gyro_z_rps=s.w)
        cmd, _, _ = stack.step(frame)
        world.step(cmd.omega_l_rad_s, cmd.omega_r_rad_s, 0.2)
        c, sn = math.cos(world.s.yaw), math.sin(world.s.yaw)
        reach.append(max(math.hypot(world.s.x + d * c, world.s.y + d * sn) for d in (-0.4, 0.4)))
    assert max(reach) <= LAUNCH_APRON_M
    assert abs(world.s.v) < 0.05


def test_typical_stack_ablation_enters_ditch():'''
assert a in s
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
