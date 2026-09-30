import sys, time, statistics, threading, queue
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
sys.path.insert(0, r"D:\Downloads\sih again\metagross")
from metagross.sim.render import bridge
from metagross.sim.render.bridge import ThreeRenderer

def tm(fn, n=12):
    fn(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(ts), 1)

Q = queue.Queue()
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"]); Q.put(self.rfile.read(n))
        self.send_response(200); self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Private-Network", "true"); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Private-Network", "true"); self.send_header("Access-Control-Allow-Headers", "*"); self.send_header("Access-Control-Allow-Methods", "POST"); self.end_headers()
    def log_message(self, *a): pass
srv = ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

bridge.CHROME_ARGS = bridge.CHROME_ARGS + ("--disable-features=LocalNetworkAccessChecks,BlockInsecurePrivateNetworkRequests,PrivateNetworkAccessRespectPreflightResults",)
with ThreeRenderer() as r:
    p = r._page
    p.evaluate("""() => { const u = new Uint8Array(1024000); for (let i=0;i<u.length;i++) u[i]=(i*2654435761)>>>24; window.__u = u; }""")
    js = f"""() => {{ try {{ const x = new XMLHttpRequest(); x.open('POST', 'http://127.0.0.1:{port}/f', false); x.send(window.__u); return x.status; }} catch (e) {{ return String(e); }} }}"""
    print("xhr loopback status", p.evaluate(js))
    if not Q.empty():
        print("got", len(Q.get()))
        def one():
            p.evaluate(js); Q.get(timeout=5)
        print("xhr loopback 1.02MB", tm(one))
    # download path
    djs = """() => { const b = new Blob([window.__u]); const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'f.bin'; document.body.appendChild(a); a.click(); a.remove(); return 1; }"""
    def dl():
        with p.expect_download() as d:
            p.evaluate(djs)
        path = d.value.path(); data = open(path, 'rb').read(); assert len(data) == 1024000
    try:
        print("download 1.02MB", tm(dl, n=6))
    except Exception as e:
        print("download failed", e)
srv.shutdown()
