import sys, time, statistics, base64
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
from metagross.sim.render.bridge import ThreeRenderer

def tm(fn, n=12):
    fn(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(ts), 1)

with ThreeRenderer() as r:
    p = r._page
    p.evaluate("""() => { const u = new Uint8Array(640000); for (let i=0;i<u.length;i++) u[i]=(i*2654435761)>>>24; window.__u = u; const s = u.toBase64(); window.__c = []; for (let i=0;i<s.length;i+=349524) window.__c.push(s.slice(i,i+349524)); }""")
    def chunks():
        n = p.evaluate("window.__c.length"); "".join(p.evaluate(f"window.__c[{i}]") for i in range(n))
    print("evaluate chunks 640KB", tm(chunks))
    cdp = p.context.new_cdp_session(p)
    def ioread():
        o = cdp.send("Runtime.evaluate", {"expression": "new Blob([window.__u])"})
        uuid = cdp.send("IO.resolveBlob", {"objectId": o["result"]["objectId"]})["uuid"]
        h = "blob:" + uuid; parts = []
        while True:
            rr = cdp.send("IO.read", {"handle": h, "size": 1 << 22})
            parts.append(base64.b64decode(rr["data"]) if rr.get("base64Encoded") else rr["data"].encode("latin1"))
            if rr.get("eof"): break
        cdp.send("IO.close", {"handle": h}); cdp.send("Runtime.releaseObject", {"objectId": o["result"]["objectId"]})
        data = b"".join(parts); assert len(data) == 640000, len(data)
    try:
        print("CDP IO.read blob 640KB", tm(ioread))
    except Exception as e:
        print("ioread failed", e)
