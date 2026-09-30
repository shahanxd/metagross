import sys, time, cProfile, pstats, io
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
from metagross.sim.render.bridge import ThreeRenderer
with ThreeRenderer() as r:
    p = r._page
    p.evaluate("""() => { const u = new Uint8Array(262143); for (let i=0;i<u.length;i++) u[i]=(i*2654435761)>>>24; window.__s = u.toBase64(); }""")
    for _ in range(3): p.evaluate("window.__s")
    pr = cProfile.Profile(); t = time.perf_counter(); pr.enable()
    for _ in range(20): p.evaluate("window.__s")
    pr.disable(); print("per call ms", (time.perf_counter() - t) / 20 * 1e3)
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12); print(s.getvalue()[:3500])
