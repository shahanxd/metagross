import os, time, json, sys, urllib.request, statistics
import numpy as np, cv2, torch, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "bw"); os.makedirs(W, exist_ok=True)
torch.set_num_threads(4)
res = {"cpu_count": os.cpu_count(), "torch": torch.__version__, "ort": ort.__version__, "cv2": cv2.__version__}

def fetch(url, name):
    p = os.path.join(W, name)
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p)
    return p

def timeit(fn, n=20, warm=3):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 1), round(min(ts), 1)

def ort_sess(path, threads=4):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

def bench_onnx(path, shape, key, n=20):
    s = ort_sess(path); inp = s.get_inputs()[0].name
    x = np.random.rand(*shape).astype(np.float32)
    med, mn = timeit(lambda: s.run(None, {inp: x}), n=n)
    res[key] = {"median_ms": med, "min_ms": mn, "shape": shape}
    print(key, res[key], flush=True)

# 1. SegFormer-B0 ADE 512
try:
    from transformers import SegformerForSemanticSegmentation
    m = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").eval()
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        res["segformer_b0_512_torch"] = dict(zip(["median_ms", "min_ms"], timeit(lambda: m(pixel_values=x), n=10)))
    print("segformer torch", res["segformer_b0_512_torch"], flush=True)
    p = os.path.join(W, "segformer_b0_512.onnx")
    if not os.path.exists(p):
        class Wrap(torch.nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x): return s.m(pixel_values=x).logits
        torch.onnx.export(Wrap(m), x, p, opset_version=17, input_names=["x"], output_names=["y"], dynamo=False)
    bench_onnx(p, (1, 3, 512, 512), "segformer_b0_512_ort")
    x2 = torch.rand(1, 3, 384, 384)
    p2 = os.path.join(W, "segformer_b0_384.onnx")
    if not os.path.exists(p2):
        torch.onnx.export(Wrap(m), x2, p2, opset_version=17, input_names=["x"], output_names=["y"], dynamo=False)
    bench_onnx(p2, (1, 3, 384, 384), "segformer_b0_384_ort")
except Exception as e:
    res["segformer_err"] = repr(e); print("segformer err", e, flush=True)

# 2. LR-ASPP MobileNetV3-Large
try:
    import torchvision
    lm = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights="DEFAULT").eval()
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        res["lraspp_512_torch"] = dict(zip(["median_ms", "min_ms"], timeit(lambda: lm(x), n=10)))
    print("lraspp torch", res["lraspp_512_torch"], flush=True)
    p = os.path.join(W, "lraspp_512.onnx")
    if not os.path.exists(p):
        class W2(torch.nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x): return s.m(x)["out"]
        torch.onnx.export(W2(lm), x, p, opset_version=17, input_names=["x"], output_names=["y"], dynamo=False)
    bench_onnx(p, (1, 3, 512, 512), "lraspp_512_ort")
except Exception as e:
    res["lraspp_err"] = repr(e); print("lraspp err", e, flush=True)

# 3. YOLOX nano / tiny ONNX (official release assets)
for nm, sz in [("yolox_nano", 416), ("yolox_tiny", 416), ("yolox_s", 640)]:
    try:
        p = fetch(f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{nm}.onnx", nm + ".onnx")
        bench_onnx(p, (1, 3, sz, sz), nm + f"_{sz}_ort")
    except Exception as e:
        res[nm + "_err"] = repr(e); print(nm, "err", e, flush=True)

# 4. Depth Anything V2 Small ONNX (onnx-community)
try:
    p = fetch("https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx", "dav2s.onnx")
    for s_ in (518, 364, 266):
        bench_onnx(p, (1, 3, s_, s_), f"depth_anything_v2_small_{s_}_ort", n=8)
except Exception as e:
    res["dav2_err"] = repr(e); print("dav2 err", e, flush=True)

# 5. OpenCV SGBM + ORB on a real KITTI 04 stereo pair (HF mirror)
try:
    L = cv2.imread(fetch("https://huggingface.co/datasets/yujie2696/kitti_odometry_04/resolve/main/image_0/000100.png", "k04_L.png"), 0)
    R = cv2.imread(fetch("https://huggingface.co/datasets/yujie2696/kitti_odometry_04/resolve/main/image_1/000100.png", "k04_R.png"), 0)
    res["kitti_img_shape"] = L.shape
    for scale, nd in [(1.0, 128), (0.5, 64)]:
        l = cv2.resize(L, None, fx=scale, fy=scale); r = cv2.resize(R, None, fx=scale, fy=scale)
        for mode, mname in [(cv2.STEREO_SGBM_MODE_SGBM_3WAY, "3way"), (cv2.STEREO_SGBM_MODE_HH4, "hh4"), (cv2.STEREO_SGBM_MODE_SGBM, "sgbm")]:
            sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=5, P1=8 * 25, P2=32 * 25, mode=mode)
            res[f"sgbm_{mname}_{l.shape[1]}x{l.shape[0]}_nd{nd}"] = dict(zip(["median_ms", "min_ms"], timeit(lambda: sg.compute(l, r), n=10)))
        bm = cv2.StereoBM_create(numDisparities=nd, blockSize=15)
        res[f"bm_{l.shape[1]}x{l.shape[0]}_nd{nd}"] = dict(zip(["median_ms", "min_ms"], timeit(lambda: bm.compute(l, r), n=10)))
    orb = cv2.ORB_create(2000)
    res["orb2000_detect_compute_ms"] = dict(zip(["median_ms", "min_ms"], timeit(lambda: orb.detectAndCompute(L, None), n=10)))
    kp1, d1 = orb.detectAndCompute(L, None); kp2, d2 = orb.detectAndCompute(R, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    res["orb2000_knnmatch_ms"] = dict(zip(["median_ms", "min_ms"], timeit(lambda: bf.knnMatch(d1, d2, k=2), n=10)))
    gft = lambda: cv2.goodFeaturesToTrack(L, 1500, 0.01, 7)
    res["gftt1500_ms"] = dict(zip(["median_ms", "min_ms"], timeit(gft, n=10)))
    pts = cv2.goodFeaturesToTrack(L, 1500, 0.01, 7)
    res["klt_1500_ms"] = dict(zip(["median_ms", "min_ms"], timeit(lambda: cv2.calcOpticalFlowPyrLK(L, R, pts, None), n=10)))
    print({k: v for k, v in res.items() if k.startswith(("sgbm", "bm_", "orb", "gftt", "klt"))}, flush=True)
except Exception as e:
    res["cv_err"] = repr(e); print("cv err", e, flush=True)

json.dump(res, open(os.path.join(HERE, "bench_results.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
