import sys, time, base64, statistics
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
import numpy as np
from metagross.sim.render.bridge import ThreeRenderer

def tm(fn, n=12):
    fn(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(ts), 1)

with ThreeRenderer() as r:
    p = r._page
    p.evaluate("""() => { const u = new Uint8Array(1050000); for (let i=0;i<u.length;i++) u[i]=(i*2654435761)>>>24; window.__big = u.toBase64(); }""")
    N = 1400000
    for k in (1, 2, 4, 8, 16):
        c = N // k
        print(f"{k} separate evaluates of {c}", tm(lambda: [p.evaluate(f"window.__big.slice({i*c}, {(i+1)*c})") for i in range(k)]))
    for k in (4, 16):
        c = N // k
        print(f"one evaluate, array of {k} strs", tm(lambda: p.evaluate(f"Array.from({{length:{k}}}, (_, i) => window.__big.slice(i*{c}, (i+1)*{c}))")))
    # latin1 string (1 char per byte) instead of base64
    p.evaluate("""() => { const u = new Uint8Array(1050000); for (let i=0;i<u.length;i++) u[i]=(i*2654435761)>>>24; let s=''; for (let i=0;i<u.length;i+=32768) s += String.fromCharCode.apply(null, u.subarray(i, i+32768)); window.__l1 = s; }""")
    print("latin1 1.05M chars", tm(lambda: p.evaluate("window.__l1")))
    # number array
    print("typed array direct 262144 u8", tm(lambda: p.evaluate("new Uint8Array(262144)")))
