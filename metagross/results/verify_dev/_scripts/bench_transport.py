import sys, time, base64, statistics
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
import numpy as np
from metagross.sim.render.bridge import ThreeRenderer

def tm(fn, n=15):
    fn(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(ts), 1)

with ThreeRenderer() as r:
    p = r._page
    p.evaluate("""() => { window.__s = {}; for (const n of [0, 350000, 1400000]) { const u = new Uint8Array(n*3/4|0); for (let i=0;i<u.length;i++) u[i]=(i*2654435761)>>>24; window.__s[n] = u.toBase64(); window.__u = u; } }""")
    print("noop", tm(lambda: p.evaluate("1")))
    for n in (0, 350000, 1400000):
        print("str", n, tm(lambda: p.evaluate(f"window.__s[{n}]")))
    print("two strs 1.05M+0.35M", tm(lambda: p.evaluate("({a: window.__s[1400000].slice(0, 1050000), b: window.__s[350000]})")))
    cdp = p.context.new_cdp_session(p)
    print("cdp str 1.4M", tm(lambda: cdp.send("Runtime.evaluate", {"expression": "window.__s[1400000]", "returnByValue": True})))
    # b64 decode cost
    s = p.evaluate("window.__s[1400000]")
    print("py b64decode 1.4M", tm(lambda: np.frombuffer(base64.b64decode(s), np.uint8)))
    # binary via route POST
    got = {}
    def push(route):
        got["b"] = route.request.post_data_buffer
        route.fulfill(status=200, body="ok")
    p.route("http://metagross.local/__push", push)
    js = """() => { const x = new XMLHttpRequest(); x.open('POST', 'http://metagross.local/__push', false); x.send(window.__u); return x.status; }"""
    print("xhr push status", p.evaluate(js))
    print("xhr push 1.05MB", tm(lambda: p.evaluate(js)), len(got.get("b", b"")))
