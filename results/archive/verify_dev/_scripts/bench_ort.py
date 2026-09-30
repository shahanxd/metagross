import os, time, json, urllib.request, statistics
import numpy as np, cv2, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "bw"); os.makedirs(W, exist_ok=True)
res = {"cpu_count": os.cpu_count(), "ort": ort.__version__, "cv2": cv2.__version__}

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
    return {"median_ms": round(statistics.median(ts), 1), "min_ms": round(min(ts), 1)}

def ort_sess(path, threads=4):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

def bench_onnx(path, shape, key, n=20):
    try:
        s = ort_sess(path); inp = s.get_inputs()[0].name
        x = np.random.rand(*shape).astype(np.float32)
        res[key] = dict(timeit(lambda: s.run(None, {inp: x}), n=n), shape=list(shape))
    except Exception as e:
        res[key] = {"err": repr(e)[:300]}
    print(key, res[key], flush=True)

# SegFormer-B0 (Xenova ONNX exports of nvidia checkpoints)
p = fetch("https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model.onnx", "segb0_ade.onnx")
for s_ in (512, 384, 256):
    bench_onnx(p, (1, 3, s_, s_), f"segformer_b0_ade_fp32_{s_}")
pq = fetch("https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx", "segb0_ade_q.onnx")
bench_onnx(pq, (1, 3, 512, 512), "segformer_b0_ade_int8_512")
bench_onnx(pq, (1, 3, 384, 384), "segformer_b0_ade_int8_384")

# YOLOX official ONNX
for nm, sz in [("yolox_nano", 416), ("yolox_tiny", 416), ("yolox_s", 640)]:
    try:
        pp = fetch(f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{nm}.onnx", nm + ".onnx")
        bench_onnx(pp, (1, 3, sz, sz), f"{nm}_{sz}")
    except Exception as e:
        res[nm] = {"err": repr(e)[:300]}; print(nm, e)

# Depth Anything V2 Small (onnx-community)
pd = fetch("https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx", "dav2s.onnx")
for s_ in (518, 364, 252):
    bench_onnx(pd, (1, 3, s_, s_), f"depth_anything_v2_small_fp32_{s_}", n=8)
pdq = fetch("https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_int8.onnx", "dav2s_int8.onnx")
bench_onnx(pdq, (1, 3, 364, 364), "depth_anything_v2_small_int8_364", n=8)

# OpenCV stereo + features on a real KITTI 04 pair
L = cv2.imread(fetch("https://huggingface.co/datasets/yujie2696/kitti_odometry_04/resolve/main/image_0/000100.png", "k04_L.png"), 0)
R = cv2.imread(fetch("https://huggingface.co/datasets/yujie2696/kitti_odometry_04/resolve/main/image_1/000100.png", "k04_R.png"), 0)
res["kitti_img_shape"] = list(L.shape)
cv2.setNumThreads(4)
for scale, nd in [(1.0, 128), (0.5, 64)]:
    l = cv2.resize(L, None, fx=scale, fy=scale); r = cv2.resize(R, None, fx=scale, fy=scale)
    tag = f"{l.shape[1]}x{l.shape[0]}_nd{nd}"
    for mode, mname in [(cv2.STEREO_SGBM_MODE_SGBM_3WAY, "3way"), (cv2.STEREO_SGBM_MODE_HH4, "hh4"), (cv2.STEREO_SGBM_MODE_SGBM, "5path")]:
        sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=5, P1=8 * 25, P2=32 * 25, mode=mode)
        res[f"sgbm_{mname}_{tag}"] = timeit(lambda: sg.compute(l, r), n=10)
    bm = cv2.StereoBM_create(numDisparities=nd, blockSize=15)
    res[f"stereobm_{tag}"] = timeit(lambda: bm.compute(l, r), n=10)
orb = cv2.ORB_create(2000)
res["orb2000_detect_compute_full"] = timeit(lambda: orb.detectAndCompute(L, None), n=10)
kp1, d1 = orb.detectAndCompute(L, None); kp2, d2 = orb.detectAndCompute(R, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
res["orb2000_bf_knnmatch"] = timeit(lambda: bf.knnMatch(d1, d2, k=2), n=10)
res["gftt1500_full"] = timeit(lambda: cv2.goodFeaturesToTrack(L, 1500, 0.01, 7), n=10)
pts = cv2.goodFeaturesToTrack(L, 1500, 0.01, 7)
res["klt_pyrLK_1500_full"] = timeit(lambda: cv2.calcOpticalFlowPyrLK(L, R, pts, None), n=10)
for k, v in res.items():
    if k.startswith(("sgbm", "stereobm", "orb", "gftt", "klt")): print(k, v)
json.dump(res, open(os.path.join(HERE, "bench_ort_results.json"), "w"), indent=1)
print("DONE")
