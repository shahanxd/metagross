import os, time, json, statistics, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torchvision, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "bw")
torch.set_num_threads(4)
res = {"torch": torch.__version__, "threads": torch.get_num_threads()}

def timeit(fn, n=10, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": round(statistics.median(ts), 1), "min_ms": round(min(ts), 1)}

# LR-ASPP MobileNetV3-Large (torchvision COCO-VOC weights)
lm = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights="DEFAULT").eval()
x = torch.rand(1, 3, 512, 512)
with torch.no_grad():
    res["lraspp_torch_512_infer"] = timeit(lambda: lm(x))
class W2(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m(x)["out"]
p = os.path.join(W, "lraspp_512.onnx")
if not os.path.exists(p):
    torch.onnx.export(W2(lm), x, p, opset_version=17, input_names=["x"], output_names=["y"], dynamo=False)
so = ort.SessionOptions(); so.intra_op_num_threads = 4
s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
xn = x.numpy()
res["lraspp_ort_512_infer"] = timeit(lambda: s.run(None, {"x": xn}), n=20)
print(res, flush=True)

# CPU training-step timing (fwd+bwd+step), batch 8
def train_step_timer(model, fwd, bs, sz, ncls):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=6e-5)
    xb = torch.rand(bs, 3, sz, sz); yb = torch.randint(0, ncls, (bs, sz, sz))
    def step():
        opt.zero_grad(); out = fwd(model, xb)
        out = torch.nn.functional.interpolate(out, size=(sz, sz), mode="bilinear", align_corners=False)
        loss = torch.nn.functional.cross_entropy(out, yb); loss.backward(); opt.step()
    return timeit(step, n=3, warm=1)

lt = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights="DEFAULT")
lt.classifier.low_classifier = torch.nn.Conv2d(40, 6, 1); lt.classifier.high_classifier = torch.nn.Conv2d(128, 6, 1)
res["lraspp_cpu_trainstep_bs8_384"] = train_step_timer(lt, lambda m, x: m(x)["out"], 8, 384, 6)
print("lraspp train", res["lraspp_cpu_trainstep_bs8_384"], flush=True)

from transformers import SegformerForSemanticSegmentation
sm = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=6, ignore_mismatched_sizes=True)
res["segformer_b0_params_M"] = round(sum(p.numel() for p in sm.parameters()) / 1e6, 2)
res["segformer_cpu_trainstep_bs8_384"] = train_step_timer(sm, lambda m, x: m(pixel_values=x).logits, 8, 384, 6)
print("segformer train", res["segformer_cpu_trainstep_bs8_384"], flush=True)
sm.eval()
with torch.no_grad():
    res["segformer_torch_512_infer"] = timeit(lambda: sm(pixel_values=torch.rand(1, 3, 512, 512)))
json.dump(res, open(os.path.join(HERE, "bench_torch_results.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
